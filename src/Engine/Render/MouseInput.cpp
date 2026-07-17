#include "Engine/Render/MouseInput.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace Engine::Render {

    bool MouseEvent::IsDown(MouseButton queryButton) const {
        return (pressedButtons & MouseInput::ButtonMask(queryButton)) != 0u;
    }

    bool MouseEvent::HasModifier(MouseModifierBits modifier) const {
        return (modifiers & static_cast<MouseModifierFlags>(modifier)) != 0u;
    }

    MouseInput::ListenerId MouseInput::AddListener(MouseEventType type,
                                                   Callback callback,
                                                   int priority) {
        if (!callback)
            throw std::runtime_error("MouseInput::AddListener received empty callback");

        const ListenerId id = m_nextListenerId++;
        m_listeners.emplace(id, Listener{id, type, priority, std::move(callback)});
        return id;
    }

    bool MouseInput::RemoveListener(ListenerId id) {
        return m_listeners.erase(id) != 0;
    }

    void MouseInput::ClearListeners() {
        m_listeners.clear();
    }

    void MouseInput::OnMouseMove(double x,
                                 double y,
                                 MouseModifierFlags modifiers,
                                 double timestampSeconds) {
        const double previousX = m_hasPosition ? m_x : x;
        const double previousY = m_hasPosition ? m_y : y;
        m_x = x;
        m_y = y;
        m_hasPosition = true;

        Dispatch(MakeEvent(MouseEventType::Move,
                           MouseButton::None,
                           x, y,
                           previousX, previousY,
                           0.0, 0.0,
                           modifiers,
                           timestampSeconds));

        const double deltaX = x - previousX;
        const double deltaY = y - previousY;
        if (deltaX == 0.0 && deltaY == 0.0)
            return;

        for (uint32_t buttonValue = static_cast<uint32_t>(MouseButton::Left);
             buttonValue <= static_cast<uint32_t>(MouseButton::Button5);
             ++buttonValue) {
            MouseButton button = static_cast<MouseButton>(buttonValue);
            ButtonState *state = StateFor(button);
            if (!state || !state->down)
                continue;

            if (!state->dragging) {
                state->dragging = true;
                Dispatch(MakeEvent(MouseEventType::DragBegin,
                                   button,
                                   x, y,
                                   previousX, previousY,
                                   0.0, 0.0,
                                   modifiers,
                                   timestampSeconds));
            }

            Dispatch(MakeEvent(MouseEventType::Drag,
                               button,
                               x, y,
                               previousX, previousY,
                               0.0, 0.0,
                               modifiers,
                               timestampSeconds));
        }
    }

    void MouseInput::OnButton(MouseButton button,
                              bool pressed,
                              double x,
                              double y,
                              MouseModifierFlags modifiers,
                              double timestampSeconds) {
        const double previousX = m_hasPosition ? m_x : x;
        const double previousY = m_hasPosition ? m_y : y;
        m_x = x;
        m_y = y;
        m_hasPosition = true;

        ButtonState *state = StateFor(button);
        if (state) {
            if (pressed) {
                state->down = true;
                state->dragging = false;
                state->dragStartX = x;
                state->dragStartY = y;
                m_pressedButtons |= ButtonMask(button);
            } else {
                if (state->dragging) {
                    Dispatch(MakeEvent(MouseEventType::DragEnd,
                                       button,
                                       x, y,
                                       previousX, previousY,
                                       0.0, 0.0,
                                       modifiers,
                                       timestampSeconds));
                }
                state->down = false;
                state->dragging = false;
                m_pressedButtons &= ~ButtonMask(button);
            }
        }

        Dispatch(MakeEvent(pressed ? MouseEventType::ButtonDown : MouseEventType::ButtonUp,
                           button,
                           x, y,
                           previousX, previousY,
                           0.0, 0.0,
                           modifiers,
                           timestampSeconds));
    }

    void MouseInput::OnScroll(double x,
                              double y,
                              double scrollX,
                              double scrollY,
                              MouseModifierFlags modifiers,
                              double timestampSeconds) {
        const double previousX = m_hasPosition ? m_x : x;
        const double previousY = m_hasPosition ? m_y : y;
        m_x = x;
        m_y = y;
        m_hasPosition = true;

        Dispatch(MakeEvent(MouseEventType::Scroll,
                           MouseButton::None,
                           x, y,
                           previousX, previousY,
                           scrollX, scrollY,
                           modifiers,
                           timestampSeconds));
    }

    void MouseInput::OnCursorEnter(double x,
                                   double y,
                                   MouseModifierFlags modifiers,
                                   double timestampSeconds) {
        const double previousX = m_hasPosition ? m_x : x;
        const double previousY = m_hasPosition ? m_y : y;
        m_x = x;
        m_y = y;
        m_hasPosition = true;

        Dispatch(MakeEvent(MouseEventType::Enter,
                           MouseButton::None,
                           x, y,
                           previousX, previousY,
                           0.0, 0.0,
                           modifiers,
                           timestampSeconds));
    }

    void MouseInput::OnCursorLeave(double x,
                                   double y,
                                   MouseModifierFlags modifiers,
                                   double timestampSeconds) {
        const double previousX = m_hasPosition ? m_x : x;
        const double previousY = m_hasPosition ? m_y : y;
        m_x = x;
        m_y = y;

        Dispatch(MakeEvent(MouseEventType::Leave,
                           MouseButton::None,
                           x, y,
                           previousX, previousY,
                           0.0, 0.0,
                           modifiers,
                           timestampSeconds));
    }

    bool MouseInput::IsButtonDown(MouseButton button) const {
        return (m_pressedButtons & ButtonMask(button)) != 0u;
    }

    MouseButtonFlags MouseInput::ButtonMask(MouseButton button) {
        switch (button) {
            case MouseButton::Left:
                return 1u << 0u;
            case MouseButton::Right:
                return 1u << 1u;
            case MouseButton::Middle:
                return 1u << 2u;
            case MouseButton::Button4:
                return 1u << 3u;
            case MouseButton::Button5:
                return 1u << 4u;
            default:
                return 0u;
        }
    }

    void MouseInput::Dispatch(MouseEvent event) {
        std::vector<Listener> listeners;
        listeners.reserve(m_listeners.size());
        for (const auto &entry: m_listeners) {
            const Listener &listener = entry.second;
            if (listener.type == MouseEventType::Any || listener.type == event.type)
                listeners.push_back(listener);
        }

        std::sort(listeners.begin(), listeners.end(),
                  [](const Listener &a, const Listener &b) {
                      if (a.priority != b.priority)
                          return a.priority > b.priority;
                      return a.id < b.id;
                  });

        for (const Listener &listener: listeners) {
            listener.callback(event);
            if (event.handled)
                break;
        }
    }

    MouseEvent MouseInput::MakeEvent(MouseEventType type,
                                     MouseButton button,
                                     double x,
                                     double y,
                                     double previousX,
                                     double previousY,
                                     double scrollX,
                                     double scrollY,
                                     MouseModifierFlags modifiers,
                                     double timestampSeconds) const {
        MouseEvent event{};
        event.type = type;
        event.button = button;
        event.x = x;
        event.y = y;
        event.previousX = previousX;
        event.previousY = previousY;
        event.deltaX = x - previousX;
        event.deltaY = y - previousY;
        event.scrollX = scrollX;
        event.scrollY = scrollY;
        event.modifiers = modifiers;
        event.timestampSeconds = timestampSeconds;
        event.pressedButtons = m_pressedButtons;

        const ButtonState *state = StateFor(button);
        if (state) {
            event.dragStartX = state->dragStartX;
            event.dragStartY = state->dragStartY;
        } else {
            event.dragStartX = x;
            event.dragStartY = y;
        }
        return event;
    }

    MouseInput::ButtonState *MouseInput::StateFor(MouseButton button) {
        const size_t index = ButtonIndex(button);
        if (index >= m_buttons.size())
            return nullptr;
        return &m_buttons[index];
    }

    const MouseInput::ButtonState *MouseInput::StateFor(MouseButton button) const {
        const size_t index = ButtonIndex(button);
        if (index >= m_buttons.size())
            return nullptr;
        return &m_buttons[index];
    }

    size_t MouseInput::ButtonIndex(MouseButton button) {
        switch (button) {
            case MouseButton::Left:
                return 0;
            case MouseButton::Right:
                return 1;
            case MouseButton::Middle:
                return 2;
            case MouseButton::Button4:
                return 3;
            case MouseButton::Button5:
                return 4;
            default:
                return static_cast<size_t>(-1);
        }
    }

    MouseListenerGroup::MouseListenerGroup(MouseInput &input)
        : m_input(&input) {
    }

    MouseListenerGroup::~MouseListenerGroup() {
        Clear();
    }

    MouseListenerGroup::MouseListenerGroup(MouseListenerGroup &&other) noexcept
        : m_input(other.m_input),
          m_listenerIds(std::move(other.m_listenerIds)) {
        other.m_input = nullptr;
    }

    MouseListenerGroup &MouseListenerGroup::operator=(MouseListenerGroup &&other) noexcept {
        if (this == &other)
            return *this;

        Clear();
        m_input = other.m_input;
        m_listenerIds = std::move(other.m_listenerIds);
        other.m_input = nullptr;
        return *this;
    }

    MouseInput::ListenerId MouseListenerGroup::Add(MouseEventType type,
                                                   MouseInput::Callback callback,
                                                   int priority) {
        if (!m_input)
            throw std::runtime_error("MouseListenerGroup requires a MouseInput");

        MouseInput::ListenerId id = m_input->AddListener(type, std::move(callback), priority);
        m_listenerIds.push_back(id);
        return id;
    }

    void MouseListenerGroup::Clear() {
        if (m_input) {
            for (MouseInput::ListenerId id: m_listenerIds)
                m_input->RemoveListener(id);
        }
        m_listenerIds.clear();
    }

} // namespace Engine::Render

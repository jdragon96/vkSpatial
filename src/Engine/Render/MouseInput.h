#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace Engine::Render {

    enum class MouseButton : uint32_t {
        None = 0,
        Left = 1,
        Right = 2,
        Middle = 3,
        Button4 = 4,
        Button5 = 5,
        Other = 255,
    };

    enum class MouseEventType : uint32_t {
        Any = 0,
        Move,
        ButtonDown,
        ButtonUp,
        DragBegin,
        Drag,
        DragEnd,
        Scroll,
        Enter,
        Leave,
    };

    enum MouseModifierBits : uint32_t {
        MouseModifierShift = 1u << 0u,
        MouseModifierControl = 1u << 1u,
        MouseModifierAlt = 1u << 2u,
        MouseModifierSuper = 1u << 3u,
    };

    using MouseModifierFlags = uint32_t;
    using MouseButtonFlags = uint32_t;

    struct MouseEvent {
        MouseEventType type = MouseEventType::Move;
        MouseButton button = MouseButton::None;
        double x = 0.0;
        double y = 0.0;
        double previousX = 0.0;
        double previousY = 0.0;
        double deltaX = 0.0;
        double deltaY = 0.0;
        double scrollX = 0.0;
        double scrollY = 0.0;
        double dragStartX = 0.0;
        double dragStartY = 0.0;
        double timestampSeconds = 0.0;
        MouseModifierFlags modifiers = 0;
        MouseButtonFlags pressedButtons = 0;
        bool handled = false;

        bool IsDown(MouseButton queryButton) const;
        bool HasModifier(MouseModifierBits modifier) const;
    };

    class MouseInput {
    public:
        using ListenerId = uint64_t;
        using Callback = std::function<void(MouseEvent &)>;

        ListenerId AddListener(MouseEventType type,
                               Callback callback,
                               int priority = 0);
        bool RemoveListener(ListenerId id);
        void ClearListeners();

        void OnMouseMove(double x,
                         double y,
                         MouseModifierFlags modifiers = 0,
                         double timestampSeconds = 0.0);
        void OnButton(MouseButton button,
                      bool pressed,
                      double x,
                      double y,
                      MouseModifierFlags modifiers = 0,
                      double timestampSeconds = 0.0);
        void OnScroll(double x,
                      double y,
                      double scrollX,
                      double scrollY,
                      MouseModifierFlags modifiers = 0,
                      double timestampSeconds = 0.0);
        void OnCursorEnter(double x,
                           double y,
                           MouseModifierFlags modifiers = 0,
                           double timestampSeconds = 0.0);
        void OnCursorLeave(double x,
                           double y,
                           MouseModifierFlags modifiers = 0,
                           double timestampSeconds = 0.0);

        bool IsButtonDown(MouseButton button) const;
        MouseButtonFlags PressedButtons() const { return m_pressedButtons; }
        double X() const { return m_x; }
        double Y() const { return m_y; }
        bool HasPosition() const { return m_hasPosition; }

        static MouseButtonFlags ButtonMask(MouseButton button);

    private:
        struct Listener {
            ListenerId id = 0;
            MouseEventType type = MouseEventType::Any;
            int priority = 0;
            Callback callback;
        };

        struct ButtonState {
            bool down = false;
            bool dragging = false;
            double dragStartX = 0.0;
            double dragStartY = 0.0;
        };

        std::unordered_map<ListenerId, Listener> m_listeners;
        std::array<ButtonState, 5> m_buttons{};
        ListenerId m_nextListenerId = 1;
        double m_x = 0.0;
        double m_y = 0.0;
        bool m_hasPosition = false;
        MouseButtonFlags m_pressedButtons = 0;

        void Dispatch(MouseEvent event);
        MouseEvent MakeEvent(MouseEventType type,
                             MouseButton button,
                             double x,
                             double y,
                             double previousX,
                             double previousY,
                             double scrollX,
                             double scrollY,
                             MouseModifierFlags modifiers,
                             double timestampSeconds) const;
        ButtonState *StateFor(MouseButton button);
        const ButtonState *StateFor(MouseButton button) const;
        static size_t ButtonIndex(MouseButton button);
    };

    class MouseListenerGroup {
    public:
        MouseListenerGroup() = default;
        explicit MouseListenerGroup(MouseInput &input);
        ~MouseListenerGroup();

        MouseListenerGroup(const MouseListenerGroup &) = delete;
        MouseListenerGroup &operator=(const MouseListenerGroup &) = delete;

        MouseListenerGroup(MouseListenerGroup &&other) noexcept;
        MouseListenerGroup &operator=(MouseListenerGroup &&other) noexcept;

        MouseInput::ListenerId Add(MouseEventType type,
                                   MouseInput::Callback callback,
                                   int priority = 0);
        void Clear();
        bool Empty() const { return m_listenerIds.empty(); }

    private:
        MouseInput *m_input = nullptr;
        std::vector<MouseInput::ListenerId> m_listenerIds;
    };

} // namespace Engine::Render

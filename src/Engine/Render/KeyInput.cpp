#include "Engine/Render/KeyInput.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace Engine::Render {

    KeyInput::ListenerId KeyInput::AddListener(KeyEventType type,
                                               Callback callback,
                                               int priority) {
        if (!callback)
            throw std::runtime_error("KeyInput::AddListener received empty callback");

        const ListenerId id = m_nextListenerId++;
        m_listeners.emplace(id, Listener{id, type, priority, std::move(callback)});
        return id;
    }

    bool KeyInput::RemoveListener(ListenerId id) {
        return m_listeners.erase(id) != 0;
    }

    void KeyInput::ClearListeners() {
        m_listeners.clear();
    }

    void KeyInput::OnKey(KeyCode keyCode,
                         KeyEventType type,
                         uint32_t modifiers,
                         double timestampSeconds) {
        if (type == KeyEventType::Press)
            m_keyStates[keyCode] = true;
        else if (type == KeyEventType::Release)
            m_keyStates[keyCode] = false;

        KeyEvent event{};
        event.type = type;
        event.keyCode = keyCode;
        event.modifiers = modifiers;
        event.timestampSeconds = timestampSeconds;

        Dispatch(event);
    }

    bool KeyInput::IsKeyDown(KeyCode keyCode) const {
        const auto it = m_keyStates.find(keyCode);
        return it != m_keyStates.end() && it->second;
    }

    void KeyInput::Dispatch(KeyEvent event) {
        std::vector<Listener> listeners;
        listeners.reserve(m_listeners.size());
        for (const auto &entry: m_listeners) {
            const Listener &listener = entry.second;
            if (listener.type == KeyEventType::Any || listener.type == event.type)
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

    KeyListenerGroup::KeyListenerGroup(KeyInput &input)
        : m_input(&input) {
    }

    KeyListenerGroup::~KeyListenerGroup() {
        Clear();
    }

    KeyListenerGroup::KeyListenerGroup(KeyListenerGroup &&other) noexcept
        : m_input(other.m_input),
          m_listenerIds(std::move(other.m_listenerIds)) {
        other.m_input = nullptr;
    }

    KeyListenerGroup &KeyListenerGroup::operator=(KeyListenerGroup &&other) noexcept {
        if (this == &other)
            return *this;

        Clear();
        m_input = other.m_input;
        m_listenerIds = std::move(other.m_listenerIds);
        other.m_input = nullptr;
        return *this;
    }

    KeyInput::ListenerId KeyListenerGroup::Add(KeyEventType type,
                                               KeyInput::Callback callback,
                                               int priority) {
        if (!m_input)
            throw std::runtime_error("KeyListenerGroup requires a KeyInput");

        KeyInput::ListenerId id = m_input->AddListener(type, std::move(callback), priority);
        m_listenerIds.push_back(id);
        return id;
    }

    void KeyListenerGroup::Clear() {
        if (m_input) {
            for (KeyInput::ListenerId id: m_listenerIds)
                m_input->RemoveListener(id);
        }
        m_listenerIds.clear();
    }

} // namespace Engine::Render

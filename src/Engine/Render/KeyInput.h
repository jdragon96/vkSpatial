#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace Engine::Render {

    enum class KeyEventType : uint32_t {
        Any = 0,
        Press,
        Release,
        Repeat,
    };

    // Common keys only; anything a window backend can't map to one of these becomes
    // Unknown (see GlfwWindow in Task 5).
    enum class KeyCode : uint32_t {
        Unknown = 0,
        Escape, Space, Enter, Tab, Backspace,
        A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        Left, Right, Up, Down,
        LeftShift, LeftControl, LeftAlt, LeftSuper,
    };

    struct KeyEvent {
        KeyEventType type = KeyEventType::Press;
        KeyCode keyCode = KeyCode::Unknown;
        uint32_t modifiers = 0;
        double timestampSeconds = 0.0;
        bool handled = false;
    };

    class KeyInput {
    public:
        using ListenerId = uint64_t;
        using Callback = std::function<void(KeyEvent &)>;

        ListenerId AddListener(KeyEventType type,
                               Callback callback,
                               int priority = 0);
        bool RemoveListener(ListenerId id);
        void ClearListeners();

        void OnKey(KeyCode keyCode,
                   KeyEventType type,
                   uint32_t modifiers = 0,
                   double timestampSeconds = 0.0);

        bool IsKeyDown(KeyCode keyCode) const;

    private:
        struct Listener {
            ListenerId id = 0;
            KeyEventType type = KeyEventType::Any;
            int priority = 0;
            Callback callback;
        };

        std::unordered_map<ListenerId, Listener> m_listeners;
        std::unordered_map<KeyCode, bool> m_keyStates;
        ListenerId m_nextListenerId = 1;

        void Dispatch(KeyEvent event);
    };

    class KeyListenerGroup {
    public:
        KeyListenerGroup() = default;
        explicit KeyListenerGroup(KeyInput &input);
        ~KeyListenerGroup();

        KeyListenerGroup(const KeyListenerGroup &) = delete;
        KeyListenerGroup &operator=(const KeyListenerGroup &) = delete;

        KeyListenerGroup(KeyListenerGroup &&other) noexcept;
        KeyListenerGroup &operator=(KeyListenerGroup &&other) noexcept;

        KeyInput::ListenerId Add(KeyEventType type,
                                 KeyInput::Callback callback,
                                 int priority = 0);
        void Clear();
        bool Empty() const { return m_listenerIds.empty(); }

    private:
        KeyInput *m_input = nullptr;
        std::vector<KeyInput::ListenerId> m_listenerIds;
    };

} // namespace Engine::Render

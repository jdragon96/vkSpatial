#include "Engine/Render/GlfwWindow.h"

#include <GLFW/glfw3.h>

#include <stdexcept>

namespace Engine::Render {

    namespace {

        uint32_t QueryMouseModifiers(GLFWwindow *window) {
            uint32_t flags = 0;
            if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
                flags |= MouseModifierShift;
            if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS)
                flags |= MouseModifierControl;
            if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS)
                flags |= MouseModifierAlt;
            if (glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS)
                flags |= MouseModifierSuper;
            return flags;
        }

        MouseButton ToMouseButton(int glfwButton) {
            switch (glfwButton) {
                case GLFW_MOUSE_BUTTON_LEFT: return MouseButton::Left;
                case GLFW_MOUSE_BUTTON_RIGHT: return MouseButton::Right;
                case GLFW_MOUSE_BUTTON_MIDDLE: return MouseButton::Middle;
                case GLFW_MOUSE_BUTTON_4: return MouseButton::Button4;
                case GLFW_MOUSE_BUTTON_5: return MouseButton::Button5;
                default: return MouseButton::Other;
            }
        }

        KeyCode ToKeyCode(int glfwKey) {
            switch (glfwKey) {
                case GLFW_KEY_ESCAPE: return KeyCode::Escape;
                case GLFW_KEY_SPACE: return KeyCode::Space;
                case GLFW_KEY_ENTER: return KeyCode::Enter;
                case GLFW_KEY_TAB: return KeyCode::Tab;
                case GLFW_KEY_BACKSPACE: return KeyCode::Backspace;
                case GLFW_KEY_LEFT: return KeyCode::Left;
                case GLFW_KEY_RIGHT: return KeyCode::Right;
                case GLFW_KEY_UP: return KeyCode::Up;
                case GLFW_KEY_DOWN: return KeyCode::Down;
                case GLFW_KEY_LEFT_SHIFT: return KeyCode::LeftShift;
                case GLFW_KEY_LEFT_CONTROL: return KeyCode::LeftControl;
                case GLFW_KEY_LEFT_ALT: return KeyCode::LeftAlt;
                case GLFW_KEY_LEFT_SUPER: return KeyCode::LeftSuper;
                default:
                    break;
            }
            if (glfwKey >= GLFW_KEY_A && glfwKey <= GLFW_KEY_Z)
                return static_cast<KeyCode>(static_cast<uint32_t>(KeyCode::A) + (glfwKey - GLFW_KEY_A));
            if (glfwKey >= GLFW_KEY_0 && glfwKey <= GLFW_KEY_9)
                return static_cast<KeyCode>(static_cast<uint32_t>(KeyCode::Num0) + (glfwKey - GLFW_KEY_0));
            return KeyCode::Unknown;
        }

        KeyEventType ToKeyEventType(int action) {
            switch (action) {
                case GLFW_PRESS: return KeyEventType::Press;
                case GLFW_RELEASE: return KeyEventType::Release;
                case GLFW_REPEAT: return KeyEventType::Repeat;
                default: return KeyEventType::Press;
            }
        }

    } // namespace

    GlfwWindow::GlfwWindow(const WindowDescriptor &descriptor) {
        if (!glfwInit())
            throw std::runtime_error("GlfwWindow: glfwInit failed");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        m_window = glfwCreateWindow(static_cast<int>(descriptor.width),
                                    static_cast<int>(descriptor.height),
                                    descriptor.title.c_str(), nullptr, nullptr);
        if (!m_window) {
            glfwTerminate();
            throw std::runtime_error("GlfwWindow: glfwCreateWindow failed");
        }

        glfwSetWindowUserPointer(m_window, this);
        glfwSetCursorPosCallback(m_window, CursorPosCallback);
        glfwSetMouseButtonCallback(m_window, MouseButtonCallback);
        glfwSetScrollCallback(m_window, ScrollCallback);
        glfwSetCursorEnterCallback(m_window, CursorEnterCallback);
        glfwSetKeyCallback(m_window, KeyCallback);
    }

    GlfwWindow::~GlfwWindow() {
        if (m_window)
            glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    bool GlfwWindow::ShouldClose() const {
        return glfwWindowShouldClose(m_window);
    }

    void GlfwWindow::RequestClose() {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }

    void GlfwWindow::PollEvents() {
        glfwPollEvents();
    }

    VkExtent2D GlfwWindow::FramebufferSize() const {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(m_window, &width, &height);
        return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    }

    std::vector<const char *> GlfwWindow::RequiredInstanceExtensions() const {
        uint32_t count = 0;
        const char **extensions = glfwGetRequiredInstanceExtensions(&count);
        if (!extensions || count == 0)
            throw std::runtime_error("GlfwWindow: GLFW did not provide Vulkan extensions");
        return std::vector<const char *>(extensions, extensions + count);
    }

    VkSurfaceKHR GlfwWindow::CreateSurface(VkInstance instance) const {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (glfwCreateWindowSurface(instance, m_window, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error("GlfwWindow: failed to create window surface");
        return surface;
    }

    void GlfwWindow::CursorPosCallback(GLFWwindow *window, double x, double y) {
        auto *self = static_cast<GlfwWindow *>(glfwGetWindowUserPointer(window));
        if (!self)
            return;
        self->m_mouseInput.OnMouseMove(x, y, QueryMouseModifiers(window), glfwGetTime());
    }

    void GlfwWindow::MouseButtonCallback(GLFWwindow *window, int button, int action, int) {
        auto *self = static_cast<GlfwWindow *>(glfwGetWindowUserPointer(window));
        if (!self)
            return;
        double x = 0.0;
        double y = 0.0;
        glfwGetCursorPos(window, &x, &y);
        self->m_mouseInput.OnButton(ToMouseButton(button), action == GLFW_PRESS, x, y,
                                    QueryMouseModifiers(window), glfwGetTime());
    }

    void GlfwWindow::ScrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
        auto *self = static_cast<GlfwWindow *>(glfwGetWindowUserPointer(window));
        if (!self)
            return;
        double x = 0.0;
        double y = 0.0;
        glfwGetCursorPos(window, &x, &y);
        self->m_mouseInput.OnScroll(x, y, xoffset, yoffset, QueryMouseModifiers(window), glfwGetTime());
    }

    void GlfwWindow::CursorEnterCallback(GLFWwindow *window, int entered) {
        auto *self = static_cast<GlfwWindow *>(glfwGetWindowUserPointer(window));
        if (!self)
            return;
        double x = 0.0;
        double y = 0.0;
        glfwGetCursorPos(window, &x, &y);
        if (entered)
            self->m_mouseInput.OnCursorEnter(x, y, QueryMouseModifiers(window), glfwGetTime());
        else
            self->m_mouseInput.OnCursorLeave(x, y, QueryMouseModifiers(window), glfwGetTime());
    }

    void GlfwWindow::KeyCallback(GLFWwindow *window, int key, int, int action, int mods) {
        auto *self = static_cast<GlfwWindow *>(glfwGetWindowUserPointer(window));
        if (!self)
            return;
        self->m_keyInput.OnKey(ToKeyCode(key), ToKeyEventType(action),
                               static_cast<uint32_t>(mods), glfwGetTime());
    }

} // namespace Engine::Render

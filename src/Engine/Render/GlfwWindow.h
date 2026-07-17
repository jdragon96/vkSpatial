#pragma once

#include "Engine/Render/Window.h"

struct GLFWwindow;

namespace Engine::Render {

    class GlfwWindow : public Window {
    public:
        explicit GlfwWindow(const WindowDescriptor &descriptor);
        ~GlfwWindow() override;

        bool ShouldClose() const override;
        void RequestClose() override;
        void PollEvents() override;
        VkExtent2D FramebufferSize() const override;

        std::vector<const char *> RequiredInstanceExtensions() const override;
        VkSurfaceKHR CreateSurface(VkInstance instance) const override;

    private:
        GLFWwindow *m_window = nullptr;

        static void CursorPosCallback(GLFWwindow *window, double x, double y);
        static void MouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
        static void ScrollCallback(GLFWwindow *window, double xoffset, double yoffset);
        static void CursorEnterCallback(GLFWwindow *window, int entered);
        static void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    };

} // namespace Engine::Render

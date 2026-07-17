#pragma once

#include "Engine/Render/KeyInput.h"
#include "Engine/Render/MouseInput.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Render {

    enum class WindowBackend {
        GLFW,
    };

    struct WindowDescriptor {
        uint32_t width = 1280;
        uint32_t height = 720;
        std::string title = "Engine::Render";
    };

    class Window {
    public:
        using UniquePtr = std::unique_ptr<Window>;

        static UniquePtr Create(WindowBackend backend, const WindowDescriptor &descriptor);
        virtual ~Window() = default;

        Window(const Window &) = delete;
        Window &operator=(const Window &) = delete;

        virtual bool ShouldClose() const = 0;
        virtual void RequestClose() = 0;
        virtual void PollEvents() = 0;
        virtual VkExtent2D FramebufferSize() const = 0;

        // Used internally by Application when constructing the Context — app code does
        // not call these directly.
        virtual std::vector<const char *> RequiredInstanceExtensions() const = 0;
        virtual VkSurfaceKHR CreateSurface(VkInstance instance) const = 0;

        MouseInput &Mouse() { return m_mouseInput; }
        KeyInput &Keys() { return m_keyInput; }

    protected:
        Window() = default;

        MouseInput m_mouseInput;
        KeyInput m_keyInput;
    };

} // namespace Engine::Render

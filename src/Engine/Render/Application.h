#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Render/Renderer.h"
#include "Engine/Render/SwapChain.h"
#include "Engine/Render/View.h"
#include "Engine/Render/Window.h"

#include <memory>

namespace Engine::Render {

    struct ApplicationDescriptor {
        WindowBackend backend = WindowBackend::GLFW;
        WindowDescriptor window;
        VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    };

    class Application {
    public:
        explicit Application(const ApplicationDescriptor &descriptor = {});
        ~Application();

        Application(const Application &) = delete;
        Application &operator=(const Application &) = delete;

        Engine::Core::Context &GetContext() { return *m_context; }
        SwapChain &GetSwapChain() { return *m_swapChain; }
        Renderer &GetRenderer() { return *m_renderer; }
        Window &GetWindow() { return *m_window; }
        View &GetView() { return m_view; }

        // Loops until GetWindow().ShouldClose(). Polls events, skips frames while
        // minimized, and calls Renderer::BeginFrame/Render/EndFrame each iteration. If
        // anything throws mid-loop, waits for the device to idle before rethrowing (since
        // Context::~Context() does not itself wait for the device to idle).
        void Run();

    private:
        Window::UniquePtr m_window;
        std::unique_ptr<Engine::Core::Context> m_context;
        std::unique_ptr<SwapChain> m_swapChain;
        std::unique_ptr<Renderer> m_renderer;
        View m_view;
    };

} // namespace Engine::Render

#include "Engine/Render/Application.h"

namespace Engine::Render {

    Application::Application(const ApplicationDescriptor &descriptor) {
        m_window = Window::Create(descriptor.backend, descriptor.window);

        Window *windowPtr = m_window.get();
        m_context = std::make_unique<Engine::Core::Context>(
                true,
                windowPtr->RequiredInstanceExtensions(),
                [windowPtr](VkInstance instance) { return windowPtr->CreateSurface(instance); });

        const VkExtent2D framebufferSize = m_window->FramebufferSize();
        SwapChainDescriptor swapChainDescriptor{};
        swapChainDescriptor.width = framebufferSize.width;
        swapChainDescriptor.height = framebufferSize.height;
        m_swapChain = std::make_unique<SwapChain>(*m_context, swapChainDescriptor);

        m_renderer = std::make_unique<Renderer>(*m_context, *m_swapChain, descriptor.depthFormat);
    }

    Application::~Application() = default;

    void Application::Run() {
        try {
            while (!m_window->ShouldClose()) {
                m_window->PollEvents();

                const VkExtent2D size = m_window->FramebufferSize();
                if (size.width == 0 || size.height == 0)
                    continue;

                if (!m_renderer->BeginFrame(size.width, size.height))
                    continue;
                m_renderer->Render(m_view);
                m_renderer->EndFrame();
            }
        } catch (...) {
            vkDeviceWaitIdle(m_context->device);
            throw;
        }
        vkDeviceWaitIdle(m_context->device);
    }

} // namespace Engine::Render

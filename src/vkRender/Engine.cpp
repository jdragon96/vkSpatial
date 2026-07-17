#include "vkRender/Engine.h"

#include <stdexcept>

namespace vkRender {

    Engine::Engine(vkCommon::VkContext *context)
        : m_context(context) {
        if (!m_context)
            throw std::runtime_error("Engine requires a valid VkContext");
    }

    Renderer::UniquePtr Engine::CreateRenderer() const {
        return std::make_unique<Renderer>(m_context);
    }

    SwapChain::UniquePtr Engine::CreateSwapChain(const SwapChainDescriptor &descriptor) const {
        return std::make_unique<SwapChain>(m_context, descriptor);
    }

    Scene::UniquePtr Engine::CreateScene() const {
        return std::make_unique<Scene>();
    }

    View::UniquePtr Engine::CreateView() const {
        return std::make_unique<View>();
    }

    Camera::UniquePtr Engine::CreateCamera() const {
        return std::make_unique<Camera>();
    }

    RenderGraph::UniquePtr Engine::CreateRenderGraph() const {
        return std::make_unique<RenderGraph>();
    }

    std::unique_ptr<MouseInput> Engine::CreateMouseInput() const {
        return std::make_unique<MouseInput>();
    }

    std::unique_ptr<KeyInput> Engine::CreateKeyInput() const {
        return std::make_unique<KeyInput>();
    }

} // namespace vkRender

#pragma once

#include "vkRender/Camera.h"
#include "vkRender/KeyInput.h"
#include "vkRender/MouseInput.h"
#include "vkRender/RenderGraph.h"
#include "vkRender/Renderer.h"
#include "vkRender/Scene.h"
#include "vkRender/SwapChain.h"
#include "vkRender/View.h"

#include "vkCommon/vkContext.h"

#include <memory>

namespace vkRender {

    class Engine {
    public:
        using UniquePtr = std::unique_ptr<Engine>;

        static UniquePtr Create(vkCommon::VkContext *context) {
            return std::make_unique<Engine>(context);
        }

        explicit Engine(vkCommon::VkContext *context);
        ~Engine() = default;

        Engine(const Engine &) = delete;
        Engine &operator=(const Engine &) = delete;

        Renderer::UniquePtr CreateRenderer() const;
        SwapChain::UniquePtr CreateSwapChain(const SwapChainDescriptor &descriptor = {}) const;
        Scene::UniquePtr CreateScene() const;
        View::UniquePtr CreateView() const;
        Camera::UniquePtr CreateCamera() const;
        RenderGraph::UniquePtr CreateRenderGraph() const;
        std::unique_ptr<MouseInput> CreateMouseInput() const;
        std::unique_ptr<KeyInput> CreateKeyInput() const;

        vkCommon::VkContext &Context() const { return *m_context; }

    private:
        vkCommon::VkContext *m_context = nullptr;
    };

} // namespace vkRender

#pragma once

#include "Engine/Render/RenderTypes.h"

#include <cstddef>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Core {
    class Context;
    class Image;
} // namespace Engine::Core

namespace Engine::Render {

    class SwapChain;
    class View;

    struct RenderContext {
        Engine::Core::Context *context = nullptr;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        SwapChain *swapChain = nullptr;
        Engine::Core::Image *depthImage = nullptr;
        View *view = nullptr;
        uint32_t imageIndex = 0;
        FrameInfo frame;
    };

    class RenderPass {
    public:
        virtual ~RenderPass() = default;

        virtual const char *Name() const = 0;
        virtual void Execute(RenderContext &context) = 0;
    };

    class RenderGraph {
    public:
        using UniquePtr = std::unique_ptr<RenderGraph>;

        RenderGraph &AddPass(std::unique_ptr<RenderPass> pass);
        void Clear();
        void Execute(RenderContext &context) const;
        bool Empty() const { return m_passes.empty(); }
        size_t PassCount() const { return m_passes.size(); }

    private:
        std::vector<std::unique_ptr<RenderPass>> m_passes;
    };

} // namespace Engine::Render

#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Core/Image.h"
#include "Engine/Render/RenderGraph.h"
#include "Engine/Render/SwapChain.h"
#include "Engine/Render/View.h"

#include <vulkan/vulkan.h>

namespace Engine::Render {

    class Renderer {
    public:
        explicit Renderer(Engine::Core::Context &context,
                          SwapChain &swapChain,
                          VkFormat depthFormat = VK_FORMAT_D32_SFLOAT);
        ~Renderer();

        Renderer(const Renderer &) = delete;
        Renderer &operator=(const Renderer &) = delete;

        // Takes the app's current framebuffer size every frame. If it differs from the
        // swapchain's extent, recreates the swapchain and depth image and returns false
        // (skip this frame) rather than proceeding. If AcquireNextImage reports
        // OUT_OF_DATE, does the same. The caller never calls SwapChain::Recreate() itself.
        // On success, unconditionally transitions the swapchain color image and the owned
        // depth image to ATTACHMENT_OPTIMAL, so EndFrame's transition back to PRESENT_SRC
        // is always paired correctly regardless of whether Render() executes a graph.
        bool BeginFrame(uint32_t width, uint32_t height);

        // No-op if view.GetRenderGraph() is null or empty. Otherwise builds a RenderContext
        // and calls graph->Execute(context). The color/depth image transitions to
        // ATTACHMENT_OPTIMAL happen unconditionally in BeginFrame(), not here.
        void Render(View &view);

        // Transitions the color image back to PRESENT_SRC, ends and submits the command
        // buffer, and presents. Tolerates OUT_OF_DATE/SUBOPTIMAL from Present without
        // recreating (the next BeginFrame call handles that, since it always has a fresh
        // width/height).
        void EndFrame();

    private:
        Engine::Core::Context *m_context = nullptr;
        SwapChain *m_swapChain = nullptr;
        VkFormat m_depthFormat;
        Engine::Core::Image m_depthImage;

        VkCommandPool m_commandPool = VK_NULL_HANDLE;
        VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
        VkSemaphore m_imageAvailable = VK_NULL_HANDLE;
        VkSemaphore m_renderFinished = VK_NULL_HANDLE;
        VkFence m_inFlightFence = VK_NULL_HANDLE;

        uint32_t m_imageIndex = 0;
        bool m_frameActive = false;
        FrameInfo m_frameInfo;

        void RecreateDepthImage();
        void TransitionForRendering();
        void cleanup();
    };

} // namespace Engine::Render

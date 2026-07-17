#pragma once

#include "vkRender/RenderTypes.h"

#include "vkCommon/vkContext.h"
#include "vkCommon/vkGPUMemory.h"

#include <memory>
#include <vulkan/vulkan.h>

namespace vkRender {

    class SwapChain;
    class View;

    class Renderer {
    public:
        using UniquePtr = std::unique_ptr<Renderer>;

        explicit Renderer(vkCommon::VkContext *context);
        ~Renderer();

        Renderer(const Renderer &) = delete;
        Renderer &operator=(const Renderer &) = delete;

        bool BeginFrame(SwapChain &swapChain);
        void Render(View &view);
        void CopyBufferToSwapChain(VkBuffer srcBuffer, VkDeviceSize sizeBytes);
        void CopyBufferToSwapChain(vkCommon::vkGPUMemory &pixelBuffer);
        void EndFrame();

        bool NeedsSwapChainRecreate() const { return m_needsSwapChainRecreate; }
        void ClearSwapChainRecreateFlag() { m_needsSwapChainRecreate = false; }
        VkCommandBuffer CommandBuffer() const { return m_commandBuffer; }
        uint32_t CurrentImageIndex() const { return m_imageIndex; }
        const FrameInfo &GetFrameInfo() const { return m_frameInfo; }

    private:
        vkCommon::VkContext *m_context = nullptr;
        VkCommandPool m_commandPool = VK_NULL_HANDLE;
        VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
        VkSemaphore m_imageAvailable = VK_NULL_HANDLE;
        VkSemaphore m_renderFinished = VK_NULL_HANDLE;
        VkFence m_inFlightFence = VK_NULL_HANDLE;
        SwapChain *m_activeSwapChain = nullptr;
        uint32_t m_imageIndex = 0;
        bool m_frameActive = false;
        bool m_needsSwapChainRecreate = false;
        FrameInfo m_frameInfo;

        void CreateFrameObjects();
        void DestroyFrameObjects();
        void TransitionSwapImage(VkImage image,
                                 VkImageLayout oldLayout,
                                 VkImageLayout newLayout,
                                 VkPipelineStageFlags srcStage,
                                 VkPipelineStageFlags dstStage,
                                 VkAccessFlags srcAccess,
                                 VkAccessFlags dstAccess);
    };

} // namespace vkRender

#include "vkRender/Renderer.h"

#include "vkRender/RenderGraph.h"
#include "vkRender/SwapChain.h"
#include "vkRender/View.h"

#include <stdexcept>

namespace vkRender {

    Renderer::Renderer(vkCommon::VkContext *context)
        : m_context(context) {
        if (!m_context)
            throw std::runtime_error("Renderer requires a valid VkContext");
        if (m_context->graphicsQueue == VK_NULL_HANDLE)
            throw std::runtime_error("Renderer requires VkContext::graphicsQueue");

        CreateFrameObjects();
    }

    Renderer::~Renderer() {
        DestroyFrameObjects();
    }

    bool Renderer::BeginFrame(SwapChain &swapChain) {
        if (m_frameActive)
            throw std::runtime_error("Renderer::BeginFrame called while frame is active");

        vkWaitForFences(m_context->device, 1, &m_inFlightFence, VK_TRUE, UINT64_MAX);

        const VkResult acquireResult =
                swapChain.AcquireNextImage(m_imageAvailable, VK_NULL_HANDLE, &m_imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            m_needsSwapChainRecreate = true;
            return false;
        }
        if (acquireResult == VK_SUBOPTIMAL_KHR)
            m_needsSwapChainRecreate = true;
        else if (acquireResult != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to acquire swapchain image");

        vkResetFences(m_context->device, 1, &m_inFlightFence);
        vkResetCommandBuffer(m_commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_commandBuffer, &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to begin command buffer");

        m_activeSwapChain = &swapChain;
        m_frameActive = true;
        return true;
    }

    void Renderer::Render(View &view) {
        if (!m_frameActive)
            throw std::runtime_error("Renderer::Render called outside BeginFrame/EndFrame");

        RenderGraph *graph = view.GetRenderGraph();
        if (!graph || graph->Empty())
            return;

        RenderContext renderContext{};
        renderContext.context = m_context;
        renderContext.commandBuffer = m_commandBuffer;
        renderContext.swapChain = m_activeSwapChain;
        renderContext.view = &view;
        renderContext.imageIndex = m_imageIndex;
        renderContext.frame = m_frameInfo;
        graph->Execute(renderContext);
    }

    void Renderer::CopyBufferToSwapChain(VkBuffer srcBuffer, VkDeviceSize sizeBytes) {
        if (!m_frameActive || !m_activeSwapChain)
            throw std::runtime_error("Renderer::CopyBufferToSwapChain called outside active frame");
        if (srcBuffer == VK_NULL_HANDLE)
            throw std::runtime_error("Renderer::CopyBufferToSwapChain received null buffer");

        const VkExtent2D extent = m_activeSwapChain->Extent();
        const VkDeviceSize requiredBytes =
                static_cast<VkDeviceSize>(extent.width) *
                static_cast<VkDeviceSize>(extent.height) * 4u;
        if (sizeBytes < requiredBytes)
            throw std::runtime_error("Renderer::CopyBufferToSwapChain buffer is too small");

        VkBufferMemoryBarrier bufferBarrier{};
        bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufferBarrier.buffer = srcBuffer;
        bufferBarrier.offset = 0;
        bufferBarrier.size = requiredBytes;
        vkCmdPipelineBarrier(m_commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &bufferBarrier, 0, nullptr);

        VkImage swapImage = m_activeSwapChain->Image(m_imageIndex);
        TransitionSwapImage(swapImage,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0,
                            VK_ACCESS_TRANSFER_WRITE_BIT);

        VkBufferImageCopy copyRegion{};
        copyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copyRegion.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyBufferToImage(m_commandBuffer, srcBuffer, swapImage,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &copyRegion);

        TransitionSwapImage(swapImage,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            VK_ACCESS_TRANSFER_WRITE_BIT,
                            0);
    }

    void Renderer::CopyBufferToSwapChain(vkCommon::vkGPUMemory &pixelBuffer) {
        CopyBufferToSwapChain(pixelBuffer.GetBuffer(),
                              static_cast<VkDeviceSize>(pixelBuffer.GetSize()));
    }

    void Renderer::EndFrame() {
        if (!m_frameActive)
            throw std::runtime_error("Renderer::EndFrame called without BeginFrame");

        if (vkEndCommandBuffer(m_commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to end command buffer");

        const VkPipelineStageFlags waitStages =
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                VK_PIPELINE_STAGE_TRANSFER_BIT;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &m_imageAvailable;
        submitInfo.pWaitDstStageMask = &waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &m_renderFinished;

        if (vkQueueSubmit(m_context->graphicsQueue, 1,
                          &submitInfo, m_inFlightFence) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to submit frame");

        const VkResult presentResult =
                m_activeSwapChain->Present(m_imageIndex, m_renderFinished);
        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
            presentResult == VK_SUBOPTIMAL_KHR)
            m_needsSwapChainRecreate = true;
        else if (presentResult != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to present frame");

        m_frameActive = false;
        m_activeSwapChain = nullptr;
        ++m_frameInfo.frameIndex;
    }

    void Renderer::CreateFrameObjects() {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = m_context->graphicsFamily;
        if (vkCreateCommandPool(m_context->device, &poolInfo,
                                nullptr, &m_commandPool) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to create command pool");

        VkCommandBufferAllocateInfo commandBufferInfo{};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferInfo.commandPool = m_commandPool;
        commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(m_context->device, &commandBufferInfo,
                                     &m_commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to allocate command buffer");

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(m_context->device, &semaphoreInfo,
                              nullptr, &m_imageAvailable) != VK_SUCCESS ||
            vkCreateSemaphore(m_context->device, &semaphoreInfo,
                              nullptr, &m_renderFinished) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to create semaphores");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(m_context->device, &fenceInfo,
                          nullptr, &m_inFlightFence) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to create fence");
    }

    void Renderer::DestroyFrameObjects() {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            return;

        vkDeviceWaitIdle(m_context->device);

        if (m_inFlightFence != VK_NULL_HANDLE)
            vkDestroyFence(m_context->device, m_inFlightFence, nullptr);
        if (m_renderFinished != VK_NULL_HANDLE)
            vkDestroySemaphore(m_context->device, m_renderFinished, nullptr);
        if (m_imageAvailable != VK_NULL_HANDLE)
            vkDestroySemaphore(m_context->device, m_imageAvailable, nullptr);
        if (m_commandPool != VK_NULL_HANDLE)
            vkDestroyCommandPool(m_context->device, m_commandPool, nullptr);

        m_inFlightFence = VK_NULL_HANDLE;
        m_renderFinished = VK_NULL_HANDLE;
        m_imageAvailable = VK_NULL_HANDLE;
        m_commandPool = VK_NULL_HANDLE;
        m_commandBuffer = VK_NULL_HANDLE;
    }

    void Renderer::TransitionSwapImage(VkImage image,
                                       VkImageLayout oldLayout,
                                       VkImageLayout newLayout,
                                       VkPipelineStageFlags srcStage,
                                       VkPipelineStageFlags dstStage,
                                       VkAccessFlags srcAccess,
                                       VkAccessFlags dstAccess) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(m_commandBuffer, srcStage, dstStage,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

} // namespace vkRender

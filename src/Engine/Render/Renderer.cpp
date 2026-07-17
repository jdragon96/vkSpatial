#include "Engine/Render/Renderer.h"

#include <stdexcept>

namespace Engine::Render {

    Renderer::Renderer(Engine::Core::Context &context, SwapChain &swapChain, VkFormat depthFormat)
        : m_context(&context),
          m_swapChain(&swapChain),
          m_depthFormat(depthFormat),
          m_depthImage(context) {
        if (context.graphicsQueue == VK_NULL_HANDLE)
            throw std::runtime_error("Renderer requires Context constructed with enablePresent=true");

        try {
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex = context.graphicsFamily;
            if (vkCreateCommandPool(context.device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS)
                throw std::runtime_error("Renderer failed to create command pool");

            VkCommandBufferAllocateInfo cmdAllocInfo{};
            cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdAllocInfo.commandPool = m_commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(context.device, &cmdAllocInfo, &m_commandBuffer) != VK_SUCCESS)
                throw std::runtime_error("Renderer failed to allocate command buffer");

            VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            if (vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &m_imageAvailable) != VK_SUCCESS ||
                vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &m_renderFinished) != VK_SUCCESS)
                throw std::runtime_error("Renderer failed to create semaphores");

            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            if (vkCreateFence(context.device, &fenceInfo, nullptr, &m_inFlightFence) != VK_SUCCESS)
                throw std::runtime_error("Renderer failed to create fence");

            m_depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(swapChain.Extent(), depthFormat));
        } catch (...) {
            cleanup();
            throw;
        }
    }

    Renderer::~Renderer() {
        cleanup();
    }

    void Renderer::cleanup() {
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

    void Renderer::RecreateDepthImage() {
        m_depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(m_swapChain->Extent(), m_depthFormat));
    }

    bool Renderer::BeginFrame(uint32_t width, uint32_t height) {
        if (m_frameActive)
            throw std::runtime_error("Renderer::BeginFrame called while frame is active");

        const VkExtent2D currentExtent = m_swapChain->Extent();
        if (width != currentExtent.width || height != currentExtent.height) {
            vkDeviceWaitIdle(m_context->device);
            m_swapChain->Recreate(width, height);
            RecreateDepthImage();
            return false;
        }

        vkWaitForFences(m_context->device, 1, &m_inFlightFence, VK_TRUE, UINT64_MAX);

        const VkResult acquireResult =
                m_swapChain->AcquireNextImage(m_imageAvailable, VK_NULL_HANDLE, &m_imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            vkDeviceWaitIdle(m_context->device);
            m_swapChain->Recreate(width, height);
            RecreateDepthImage();
            return false;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Renderer failed to acquire swapchain image");

        vkResetFences(m_context->device, 1, &m_inFlightFence);
        vkResetCommandBuffer(m_commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_commandBuffer, &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to begin command buffer");

        TransitionForRendering();

        m_frameActive = true;
        return true;
    }

    void Renderer::TransitionForRendering() {
        VkImage swapImage = m_swapChain->Image(m_imageIndex);
        Engine::Core::Image::TransitionLayout(
                m_commandBuffer, swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        m_depthImage.TransitionLayout(
                m_commandBuffer, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
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
        renderContext.swapChain = m_swapChain;
        renderContext.depthImage = &m_depthImage;
        renderContext.view = &view;
        renderContext.imageIndex = m_imageIndex;
        renderContext.frame = m_frameInfo;
        graph->Execute(renderContext);
    }

    void Renderer::EndFrame() {
        if (!m_frameActive)
            throw std::runtime_error("Renderer::EndFrame called without BeginFrame");

        VkImage swapImage = m_swapChain->Image(m_imageIndex);
        Engine::Core::Image::TransitionLayout(
                m_commandBuffer, swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0);

        if (vkEndCommandBuffer(m_commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to end command buffer");

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &m_imageAvailable;
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &m_renderFinished;

        if (vkQueueSubmit(m_context->graphicsQueue, 1, &submitInfo, m_inFlightFence) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to submit frame");

        const VkResult presentResult = m_swapChain->Present(m_imageIndex, m_renderFinished);
        if (presentResult != VK_SUCCESS &&
            presentResult != VK_SUBOPTIMAL_KHR &&
            presentResult != VK_ERROR_OUT_OF_DATE_KHR) {
            throw std::runtime_error("Renderer failed to present frame");
        }

        m_frameActive = false;
        ++m_frameInfo.frameIndex;
    }

} // namespace Engine::Render

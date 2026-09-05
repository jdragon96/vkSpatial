#include "Engine/Compute/CommandBatch.h"

#include <mutex>
#include <stdexcept>

namespace Engine::Compute {

    CommandBatch::CommandBatch(Engine::Core::Context &context, Engine::Core::QueueRole role)
        : m_context(context) {
        if (role == Engine::Core::QueueRole::Compute) {
            m_queue = context.computeQueue;
            m_pool = context.cmdPool;
        } else {
            m_queue = context.graphicsQueue;
            m_pool = context.graphicsCmdPool;
        }
        if (m_queue == VK_NULL_HANDLE || m_pool == VK_NULL_HANDLE)
            throw std::runtime_error("CommandBatch: requested QueueRole is unavailable on this Context");
    }

    CommandBatch::~CommandBatch() {
        if (m_cmd != VK_NULL_HANDLE && !m_submitted)
            vkFreeCommandBuffers(m_context.device, m_pool, 1, &m_cmd);
    }

    void CommandBatch::ensureBegun() {
        if (m_cmd != VK_NULL_HANDLE) return;

        // First recorded command: take the pool for the rest of this batch's life.
        m_submissionLock = std::unique_lock<std::recursive_mutex>(m_context.submissionMutex);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(m_context.device, &allocInfo, &m_cmd) != VK_SUCCESS)
            throw std::runtime_error("CommandBatch: failed to allocate command buffer");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_cmd, &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("CommandBatch: failed to begin command buffer");
    }

    CommandBatch &CommandBatch::Dispatch(Engine::Core::ComputePipeline &pipe,
                                         uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        ensureBegun();
        pipe.RecordDispatch(m_cmd, gridX, gridY, gridZ);
        return *this;
    }

    CommandBatch &CommandBatch::DispatchElements(Engine::Core::ComputePipeline &pipe,
                                                 uint32_t numElements) {
        const uint32_t local = pipe.GetLocalSize().width;
        if (local == 0)
            throw std::runtime_error("CommandBatch::DispatchElements: pipeline local_size.x=0");
        return Dispatch(pipe, (numElements + local - 1) / local, 1, 1);
    }

    CommandBatch &CommandBatch::CopyBuffer(VkBuffer src, VkBuffer dst,
                                           const std::vector<VkBufferCopy> &regions) {
        if (regions.empty()) return *this;
        ensureBegun();
        vkCmdCopyBuffer(m_cmd, src, dst, static_cast<uint32_t>(regions.size()), regions.data());
        return *this;
    }

    CommandBatch &CommandBatch::CopyBufferToImage(VkBuffer source,
                                                  Engine::Core::Image &image,
                                                  VkImageLayout finalLayout) {
        if (!image.Valid())
            throw std::runtime_error("CommandBatch::CopyBufferToImage: the image was never created");
        ensureBegun();

        const VkExtent3D extent = image.Extent();

        image.TransitionLayout(m_cmd,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0,
                               VK_ACCESS_TRANSFER_WRITE_BIT);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = image.AspectMask();
        region.imageSubresource.layerCount = 1;
        region.imageExtent = extent;
        vkCmdCopyBufferToImage(m_cmd,
                               source,
                               image.Handle(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1,
                               &region);

        image.TransitionLayout(m_cmd,
                               finalLayout,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_ACCESS_SHADER_READ_BIT);
        return *this;
    }

    CommandBatch &CommandBatch::FillBuffer(VkBuffer buffer, VkDeviceSize offset,
                                           VkDeviceSize size, uint32_t data) {
        ensureBegun();
        vkCmdFillBuffer(m_cmd, buffer, offset, size, data);
        return *this;
    }

    CommandBatch &CommandBatch::Barrier() {
        ensureBegun();
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        const VkPipelineStageFlags stages =
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        vkCmdPipelineBarrier(m_cmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        return *this;
    }

    void CommandBatch::Submit() {
        if (m_submitted)
            throw std::runtime_error("CommandBatch: already submitted");
        if (m_cmd == VK_NULL_HANDLE) { // nothing recorded
            m_submitted = true;
            return;
        }
        if (vkEndCommandBuffer(m_cmd) != VK_SUCCESS)
            throw std::runtime_error("CommandBatch: failed to end command buffer");

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_cmd;
        if (vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
            throw std::runtime_error("CommandBatch: failed to submit");
        vkQueueWaitIdle(m_queue);

        vkFreeCommandBuffers(m_context.device, m_pool, 1, &m_cmd);
        m_cmd = VK_NULL_HANDLE;
        m_submitted = true;
        m_submissionLock.unlock();
    }

} // namespace Engine::Compute

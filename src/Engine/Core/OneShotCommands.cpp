#include "Engine/Core/OneShotCommands.h"

#include <mutex>

#include <stdexcept>

namespace Engine::Core {

    void SubmitOneShot(Context &context,
                       QueueRole role,
                       const std::function<void(VkCommandBuffer)> &record) {
    // Held for the whole allocate -> record -> submit -> wait -> free sequence: the pool needs
    // external synchronisation for every one of those, not only the allocation.
    std::lock_guard<std::recursive_mutex> submissionLock(context.submissionMutex);

        VkQueue queue = (role == QueueRole::Compute) ? context.computeQueue : context.graphicsQueue;
        VkCommandPool pool = (role == QueueRole::Compute) ? context.cmdPool : context.graphicsCmdPool;

        if (queue == VK_NULL_HANDLE || pool == VK_NULL_HANDLE)
            throw std::runtime_error(
                    "SubmitOneShot: requested QueueRole is unavailable on this Context "
                    "(Graphics requires enablePresent=true)");

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(context.device, &allocInfo, &cmd) != VK_SUCCESS)
            throw std::runtime_error("SubmitOneShot: failed to allocate command buffer");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
            vkFreeCommandBuffers(context.device, pool, 1, &cmd);
            throw std::runtime_error("SubmitOneShot: failed to begin command buffer");
        }

        try {
            record(cmd);
        } catch (...) {
            vkFreeCommandBuffers(context.device, pool, 1, &cmd);
            throw;
        }

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            vkFreeCommandBuffers(context.device, pool, 1, &cmd);
            throw std::runtime_error("SubmitOneShot: failed to end command buffer");
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            vkFreeCommandBuffers(context.device, pool, 1, &cmd);
            throw std::runtime_error("SubmitOneShot: failed to submit command buffer");
        }

        vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(context.device, pool, 1, &cmd);
    }

} // namespace Engine::Core

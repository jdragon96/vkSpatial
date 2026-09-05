#pragma once

#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Image.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/OneShotCommands.h"

#include <mutex>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Compute {

    // Records a run of compute dispatches and buffer transfers into ONE command buffer
    // and submits it once (a single vkQueueWaitIdle), instead of one submit per op.
    //
    // Constraint: do NOT Dispatch the SAME ComputePipeline object twice within one batch.
    // Each pipeline owns a single descriptor set; re-binding it before Submit would rewrite
    // the set the first dispatch still references. Use separate pipeline objects or batches.
    class CommandBatch {
    public:
        explicit CommandBatch(Engine::Core::Context &context,
                              Engine::Core::QueueRole role = Engine::Core::QueueRole::Compute);
        ~CommandBatch();

        CommandBatch(const CommandBatch &) = delete;
        CommandBatch &operator=(const CommandBatch &) = delete;

        CommandBatch &Dispatch(Engine::Core::ComputePipeline &pipe,
                               uint32_t gridX, uint32_t gridY = 1, uint32_t gridZ = 1);
        CommandBatch &DispatchElements(Engine::Core::ComputePipeline &pipe, uint32_t numElements);
        CommandBatch &CopyBuffer(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy> &regions);

        // Fills a sampled image from a host-visible staging buffer, recording the two layout
        // transitions around the copy. The image is left in `finalLayout`, which is where a
        // compute kernel sampling it needs it -- so the whole upload is one call inside the same
        // batch as the dispatch that reads it, rather than a separate blocking submit per frame.
        CommandBatch &CopyBufferToImage(VkBuffer source,
                                        Engine::Core::Image &image,
                                        VkImageLayout finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        CommandBatch &FillBuffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, uint32_t data);
        // Conservative global compute+transfer read/write barrier between dependent ops.
        CommandBatch &Barrier();

        // End recording, submit on the batch's queue, wait idle, free the command buffer.
        // Single-use: a batch cannot be submitted twice.
        void Submit();

    private:
        Engine::Core::Context &m_context;
        VkQueue m_queue = VK_NULL_HANDLE;
        VkCommandPool m_pool = VK_NULL_HANDLE;
        VkCommandBuffer m_cmd = VK_NULL_HANDLE;

        // Taken on the first recorded command and released by Submit(). The whole record-to-submit
        // span has to be covered, because the command pool needs external synchronisation for
        // vkBeginCommandBuffer and every vkCmd* as well as for allocation -- see
        // Engine::Core::Context::submissionMutex. A member rather than a local so ~CommandBatch's
        // body, which frees an unsubmitted buffer, still holds it.
        std::unique_lock<std::recursive_mutex> m_submissionLock;
        bool m_submitted = false;

        void ensureBegun();
    };

} // namespace Engine::Compute

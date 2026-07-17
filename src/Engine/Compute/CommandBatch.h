#pragma once

#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/OneShotCommands.h"

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
        bool m_submitted = false;

        void ensureBegun();
    };

} // namespace Engine::Compute

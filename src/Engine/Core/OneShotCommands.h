#pragma once

#include "Engine/Core/Context.h"

#include <functional>
#include <vulkan/vulkan.h>

namespace Engine::Core {

    enum class QueueRole {
        Compute,
        Graphics,
    };

    // Allocates a transient command buffer from the command pool matching `role`
    // (context.cmdPool for Compute, context.graphicsCmdPool for Graphics), begins it,
    // invokes `record(cmd)` to fill it in, then ends, submits, and blocks until the
    // corresponding queue is idle before freeing the command buffer. Throws
    // std::runtime_error on any Vulkan failure or if QueueRole::Graphics is requested
    // on a Context that wasn't constructed with enablePresent=true.
    void SubmitOneShot(Context &context,
                       QueueRole role,
                       const std::function<void(VkCommandBuffer)> &record);

} // namespace Engine::Core

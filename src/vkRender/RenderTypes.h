#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

namespace vkRender {

    struct Extent {
        uint32_t width = 0;
        uint32_t height = 0;

        bool Empty() const { return width == 0 || height == 0; }
    };

    struct Viewport {
        int32_t x = 0;
        int32_t y = 0;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    struct ClearOptions {
        bool clearColor = true;
        bool clearDepth = true;
        bool clearStencil = false;
        float color[4] = {0.02f, 0.02f, 0.025f, 1.0f};
        float depth = 1.0f;
        uint32_t stencil = 0;
    };

    struct FrameInfo {
        uint64_t frameIndex = 0;
        double elapsedSeconds = 0.0;
        float deltaSeconds = 0.0f;
    };

    struct SwapChainDescriptor {
        uint32_t width = 0;
        uint32_t height = 0;
        VkFormat preferredFormat = VK_FORMAT_B8G8R8A8_UNORM;
        VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
        VkImageUsageFlags imageUsage =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    };

} // namespace vkRender

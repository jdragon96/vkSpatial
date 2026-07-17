#pragma once

#include "vkCommon/vkContext.h"
#include "vkRender/Image.h"

#include <string>
#include <vulkan/vulkan.h>

namespace vkRender {

    class Capture {
    public:
        // Copies `image` (currently in `currentLayout`, e.g. VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        // to an RGBA PNG file at `path`. Blocks until the GPU copy completes and restores
        // `currentLayout` on `image` afterwards. `image` must have been created with
        // VK_IMAGE_USAGE_TRANSFER_SRC_BIT. Supports VK_FORMAT_{B,R}8G8R8A8_{UNORM,SRGB} only.
        static void CaptureToPNG(vkCommon::VkContext *context,
                                 VkImage image,
                                 VkFormat format,
                                 VkExtent2D extent,
                                 VkImageLayout currentLayout,
                                 const std::string &path);

        // Copies all pixels from `src` to `dst` using vkCmdCopyImage. Both images must have
        // matching format and extent. `src` needs VK_IMAGE_USAGE_TRANSFER_SRC_BIT and `dst`
        // needs VK_IMAGE_USAGE_TRANSFER_DST_BIT.
        static void CopyImageToImage(vkCommon::VkContext *context,
                                     Image &src,
                                     Image &dst);

        // Records an image -> buffer readback copy into an existing command buffer.
        // The image is transitioned to TRANSFER_SRC_OPTIMAL for the copy, then to
        // `finalLayout`. The buffer receives a TRANSFER_WRITE -> HOST_READ barrier.
        static void RecordImageToBufferCopy(VkCommandBuffer cmd,
                                            VkImage image,
                                            VkImageLayout currentLayout,
                                            VkImageLayout finalLayout,
                                            VkBuffer dstBuffer,
                                            VkDeviceSize dstBufferSize,
                                            VkExtent2D extent,
                                            VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);

        // Builds "<directory>/screenshot_YYYYMMDD_HHMMSS.png", creating <directory> if needed.
        static std::string TimestampedPath(const std::string &directory = "screenshots");
    };

    using Screenshot = Capture;

} // namespace vkRender

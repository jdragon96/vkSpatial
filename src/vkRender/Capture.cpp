#include "vkRender/Capture.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vkRender {
    namespace {

        bool RequiresRedBlueSwap(VkFormat format) {
            return format == VK_FORMAT_B8G8R8A8_UNORM ||
                   format == VK_FORMAT_B8G8R8A8_SRGB;
        }

        bool IsSupportedFormat(VkFormat format) {
            return format == VK_FORMAT_B8G8R8A8_UNORM ||
                   format == VK_FORMAT_B8G8R8A8_SRGB ||
                   format == VK_FORMAT_R8G8B8A8_UNORM ||
                   format == VK_FORMAT_R8G8B8A8_SRGB;
        }

        uint32_t FindMemoryType(VkPhysicalDevice physicalDevice,
                                uint32_t typeFilter,
                                VkMemoryPropertyFlags properties) {
            VkPhysicalDeviceMemoryProperties memProperties;
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
            for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
                if ((typeFilter & (1u << i)) &&
                    (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                    return i;
            }
            throw std::runtime_error("Capture: no suitable memory type for staging buffer");
        }

        void TransitionImageLayout(VkCommandBuffer cmd,
                                   VkImage image,
                                   VkImageAspectFlags aspectMask,
                                   VkImageLayout oldLayout,
                                   VkImageLayout newLayout) {
            Image::TransitionLayout(cmd, image, aspectMask,
                                    oldLayout, newLayout,
                                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                    VK_ACCESS_MEMORY_WRITE_BIT,
                                    VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        }

        VkCommandPool CreateTransientCommandPool(vkCommon::VkContext *context) {
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            poolInfo.queueFamilyIndex = context->graphicsFamily;

            VkCommandPool commandPool = VK_NULL_HANDLE;
            if (vkCreateCommandPool(context->device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
                throw std::runtime_error("Capture: failed to create command pool");
            return commandPool;
        }

        VkCommandBuffer AllocateCommandBuffer(VkDevice device, VkCommandPool commandPool) {
            VkCommandBufferAllocateInfo cmdAllocInfo{};
            cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;

            VkCommandBuffer cmd = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd) != VK_SUCCESS)
                throw std::runtime_error("Capture: failed to allocate command buffer");
            return cmd;
        }

        void BeginOneTimeCommands(VkCommandBuffer cmd) {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
                throw std::runtime_error("Capture: failed to begin command buffer");
        }

        void EndSubmitAndWait(vkCommon::VkContext *context, VkCommandBuffer cmd) {
            if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
                throw std::runtime_error("Capture: failed to end command buffer");

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            if (vkQueueSubmit(context->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
                throw std::runtime_error("Capture: failed to submit command buffer");
            vkQueueWaitIdle(context->graphicsQueue);
        }

    } // namespace

    void Capture::CaptureToPNG(vkCommon::VkContext *context,
                               VkImage image,
                               VkFormat format,
                               VkExtent2D extent,
                               VkImageLayout currentLayout,
                               const std::string &path) {
        if (!context)
            throw std::runtime_error("Capture::CaptureToPNG requires a valid VkContext");
        if (!IsSupportedFormat(format))
            throw std::runtime_error("Capture::CaptureToPNG received an unsupported swapchain format");

        vkQueueWaitIdle(context->graphicsQueue);

        const VkDeviceSize bufferSize = static_cast<VkDeviceSize>(extent.width) * extent.height * 4u;
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        if (vkCreateBuffer(context->device, &bufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS)
            throw std::runtime_error("Capture: failed to create staging buffer");

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(context->device, stagingBuffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = FindMemoryType(
                context->physDevice, memRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        if (vkAllocateMemory(context->device, &allocInfo, nullptr, &stagingMemory) != VK_SUCCESS) {
            vkDestroyBuffer(context->device, stagingBuffer, nullptr);
            throw std::runtime_error("Capture: failed to allocate staging memory");
        }

        if (vkBindBufferMemory(context->device, stagingBuffer, stagingMemory, 0) != VK_SUCCESS) {
            vkFreeMemory(context->device, stagingMemory, nullptr);
            vkDestroyBuffer(context->device, stagingBuffer, nullptr);
            throw std::runtime_error("Capture: failed to bind staging buffer memory");
        }

        VkCommandPool commandPool = VK_NULL_HANDLE;
        try {
            commandPool = CreateTransientCommandPool(context);
        } catch (...) {
            vkFreeMemory(context->device, stagingMemory, nullptr);
            vkDestroyBuffer(context->device, stagingBuffer, nullptr);
            throw;
        }

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        try {
            cmd = AllocateCommandBuffer(context->device, commandPool);
            BeginOneTimeCommands(cmd);
        } catch (...) {
            vkDestroyCommandPool(context->device, commandPool, nullptr);
            vkFreeMemory(context->device, stagingMemory, nullptr);
            vkDestroyBuffer(context->device, stagingBuffer, nullptr);
            throw;
        }

        RecordImageToBufferCopy(cmd,
                                image,
                                currentLayout,
                                currentLayout,
                                stagingBuffer,
                                bufferSize,
                                extent);

        try {
            EndSubmitAndWait(context, cmd);
        } catch (...) {
            vkDestroyCommandPool(context->device, commandPool, nullptr);
            vkFreeMemory(context->device, stagingMemory, nullptr);
            vkDestroyBuffer(context->device, stagingBuffer, nullptr);
            throw;
        }

        std::vector<uint8_t> pixels(static_cast<size_t>(bufferSize));
        void *mapped = nullptr;
        if (vkMapMemory(context->device, stagingMemory, 0, bufferSize, 0, &mapped) != VK_SUCCESS) {
            vkFreeCommandBuffers(context->device, commandPool, 1, &cmd);
            vkDestroyCommandPool(context->device, commandPool, nullptr);
            vkFreeMemory(context->device, stagingMemory, nullptr);
            vkDestroyBuffer(context->device, stagingBuffer, nullptr);
            throw std::runtime_error("Capture: failed to map staging memory");
        }
        std::memcpy(pixels.data(), mapped, static_cast<size_t>(bufferSize));
        vkUnmapMemory(context->device, stagingMemory);

        if (RequiresRedBlueSwap(format)) {
            for (size_t i = 0; i + 2 < pixels.size(); i += 4)
                std::swap(pixels[i], pixels[i + 2]);
        }

        vkFreeCommandBuffers(context->device, commandPool, 1, &cmd);
        vkDestroyCommandPool(context->device, commandPool, nullptr);
        vkFreeMemory(context->device, stagingMemory, nullptr);
        vkDestroyBuffer(context->device, stagingBuffer, nullptr);

        const int stride = static_cast<int>(extent.width) * 4;
        if (!stbi_write_png(
                    path.c_str(),
                    static_cast<int>(extent.width),
                    static_cast<int>(extent.height),
                    4, pixels.data(), stride))
            throw std::runtime_error("Capture: failed to write PNG file: " + path);
    }

    void Capture::CopyImageToImage(vkCommon::VkContext *context,
                                   Image &src,
                                   Image &dst) {
        if (!context)
            throw std::runtime_error("Capture::CopyImageToImage requires a valid VkContext");
        if (!src.Valid() || !dst.Valid())
            throw std::runtime_error("Capture::CopyImageToImage requires valid src and dst images");
        if (src.Format() != dst.Format())
            throw std::runtime_error("Capture::CopyImageToImage requires matching image formats");
        if (src.Extent().width != dst.Extent().width ||
            src.Extent().height != dst.Extent().height ||
            src.Extent().depth != dst.Extent().depth)
            throw std::runtime_error("Capture::CopyImageToImage requires matching image extents");
        if ((src.Descriptor().usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0)
            throw std::runtime_error("Capture::CopyImageToImage src needs VK_IMAGE_USAGE_TRANSFER_SRC_BIT");
        if ((dst.Descriptor().usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0)
            throw std::runtime_error("Capture::CopyImageToImage dst needs VK_IMAGE_USAGE_TRANSFER_DST_BIT");
        if (src.AspectMask() != dst.AspectMask())
            throw std::runtime_error("Capture::CopyImageToImage requires matching image aspect masks");

        vkQueueWaitIdle(context->graphicsQueue);

        VkCommandPool commandPool = CreateTransientCommandPool(context);
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        try {
            cmd = AllocateCommandBuffer(context->device, commandPool);
            BeginOneTimeCommands(cmd);

            const VkImageLayout srcLayout = src.CurrentLayout();
            const VkImageLayout dstLayout = dst.CurrentLayout();
            src.TransitionLayout(cmd,
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_ACCESS_MEMORY_WRITE_BIT,
                                 VK_ACCESS_TRANSFER_READ_BIT);
            dst.TransitionLayout(cmd,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                                 VK_ACCESS_TRANSFER_WRITE_BIT);

            VkImageCopy region{};
            region.srcSubresource = {src.AspectMask(), 0, 0, 1};
            region.dstSubresource = {dst.AspectMask(), 0, 0, 1};
            region.extent = src.Extent();
            vkCmdCopyImage(cmd,
                           src.Handle(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dst.Handle(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &region);

            src.TransitionLayout(cmd,
                                 srcLayout,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_ACCESS_TRANSFER_READ_BIT,
                                 VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
            dst.TransitionLayout(cmd,
                                 dstLayout,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);

            EndSubmitAndWait(context, cmd);
        } catch (...) {
            if (cmd != VK_NULL_HANDLE)
                vkFreeCommandBuffers(context->device, commandPool, 1, &cmd);
            vkDestroyCommandPool(context->device, commandPool, nullptr);
            throw;
        }

        vkFreeCommandBuffers(context->device, commandPool, 1, &cmd);
        vkDestroyCommandPool(context->device, commandPool, nullptr);
    }

    void Capture::RecordImageToBufferCopy(VkCommandBuffer cmd,
                                          VkImage image,
                                          VkImageLayout currentLayout,
                                          VkImageLayout finalLayout,
                                          VkBuffer dstBuffer,
                                          VkDeviceSize dstBufferSize,
                                          VkExtent2D extent,
                                          VkImageAspectFlags aspectMask) {
        TransitionImageLayout(cmd, image, aspectMask,
                              currentLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {aspectMask, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(cmd,
                               image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               dstBuffer,
                               1,
                               &region);

        VkBufferMemoryBarrier readbackBarrier{};
        readbackBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        readbackBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        readbackBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        readbackBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        readbackBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        readbackBarrier.buffer = dstBuffer;
        readbackBarrier.offset = 0;
        readbackBarrier.size = dstBufferSize;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0,
                             0,
                             nullptr,
                             1,
                             &readbackBarrier,
                             0,
                             nullptr);

        TransitionImageLayout(cmd, image, aspectMask,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, finalLayout);
    }

    std::string Capture::TimestampedPath(const std::string &directory) {
        std::filesystem::create_directories(directory);

        const auto now = std::chrono::system_clock::now();
        const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
        std::tm localTime{};
#if defined(_WIN32)
        localtime_s(&localTime, &nowTime);
#else
        localtime_r(&nowTime, &localTime);
#endif

        std::ostringstream oss;
        oss << "screenshot_"
            << std::put_time(&localTime, "%Y%m%d_%H%M%S")
            << ".png";

        std::filesystem::path result = std::filesystem::path(directory) / oss.str();
        return result.string();
    }

} // namespace vkRender

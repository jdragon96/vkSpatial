#pragma once

#include "vkCommon/vkContext.h"

#include <memory>
#include <vulkan/vulkan.h>

namespace vkRender {

    struct ImageDescriptor {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t depth = 1;
        uint32_t mipLevels = 1;
        uint32_t arrayLayers = 1;

        VkFormat format = VK_FORMAT_UNDEFINED;
        VkImageUsageFlags usage = 0;
        VkImageAspectFlags aspectMask = 0;
        VkImageType imageType = VK_IMAGE_TYPE_2D;
        VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
        VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        bool createView = true;

        static ImageDescriptor Depth2D(VkExtent2D extent,
                                       VkFormat format = VK_FORMAT_D32_SFLOAT);
        static ImageDescriptor Color2D(VkExtent2D extent,
                                       VkFormat format,
                                       VkImageUsageFlags usage =
                                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                               VK_IMAGE_USAGE_SAMPLED_BIT);
    };

    class Image {
    public:
        using UniquePtr = std::unique_ptr<Image>;

        explicit Image(vkCommon::VkContext *context);
        Image(vkCommon::VkContext *context, const ImageDescriptor &descriptor);
        ~Image();

        Image(const Image &) = delete;
        Image &operator=(const Image &) = delete;
        Image(Image &&rhs) noexcept;
        Image &operator=(Image &&rhs) noexcept;

        Image &Create(const ImageDescriptor &descriptor);
        void Destroy();

        bool Valid() const { return m_image != VK_NULL_HANDLE; }
        bool HasView() const { return m_view != VK_NULL_HANDLE; }
        bool Matches(VkExtent2D extent, VkFormat format = VK_FORMAT_UNDEFINED) const;

        VkImage Handle() const { return m_image; }
        VkDeviceMemory Memory() const { return m_memory; }
        VkImageView View() const { return m_view; }
        VkFormat Format() const { return m_descriptor.format; }
        VkImageAspectFlags AspectMask() const { return m_descriptor.aspectMask; }
        VkImageLayout CurrentLayout() const { return m_currentLayout; }
        VkExtent3D Extent() const {
            return {m_descriptor.width, m_descriptor.height, m_descriptor.depth};
        }
        VkExtent2D Extent2D() const {
            return {m_descriptor.width, m_descriptor.height};
        }
        const ImageDescriptor &Descriptor() const { return m_descriptor; }

        // Records a barrier from this image's tracked current layout to newLayout
        // and updates the tracked layout; no-op if already in newLayout.
        void TransitionLayout(VkCommandBuffer cmd,
                              VkImageLayout newLayout,
                              VkPipelineStageFlags srcStage,
                              VkPipelineStageFlags dstStage,
                              VkAccessFlags srcAccess,
                              VkAccessFlags dstAccess);

        // Raw-handle variant for images not owned by an Image instance (e.g. swapchain images).
        static void TransitionLayout(VkCommandBuffer cmd,
                                     VkImage image,
                                     VkImageAspectFlags aspectMask,
                                     VkImageLayout oldLayout,
                                     VkImageLayout newLayout,
                                     VkPipelineStageFlags srcStage,
                                     VkPipelineStageFlags dstStage,
                                     VkAccessFlags srcAccess,
                                     VkAccessFlags dstAccess);

    private:
        vkCommon::VkContext *m_context = nullptr;
        VkImage m_image = VK_NULL_HANDLE;
        VkDeviceMemory m_memory = VK_NULL_HANDLE;
        VkImageView m_view = VK_NULL_HANDLE;
        ImageDescriptor m_descriptor{};
        VkImageLayout m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        uint32_t FindMemoryType(uint32_t typeFilter,
                                VkMemoryPropertyFlags properties) const;
        void CreateImageObject();
        void AllocateAndBindMemory();
        void CreateImageView();
        void MoveFrom(Image &&rhs) noexcept;
    };

} // namespace vkRender

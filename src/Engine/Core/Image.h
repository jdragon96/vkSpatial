#pragma once

#include "Engine/Core/Context.h"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace Engine::Core {

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
        explicit Image(Context &context);
        Image(Context &context, const ImageDescriptor &descriptor);
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
        VkImageView View() const { return m_view; }
        VkFormat Format() const { return m_descriptor.format; }
        VkImageAspectFlags AspectMask() const { return m_descriptor.aspectMask; }
        VkImageLayout CurrentLayout() const { return m_currentLayout; }
        VkExtent3D Extent() const { return {m_descriptor.width, m_descriptor.height, m_descriptor.depth}; }
        VkExtent2D Extent2D() const { return {m_descriptor.width, m_descriptor.height}; }
        const ImageDescriptor &Descriptor() const { return m_descriptor; }

        void TransitionLayout(VkCommandBuffer cmd,
                              VkImageLayout newLayout,
                              VkPipelineStageFlags srcStage,
                              VkPipelineStageFlags dstStage,
                              VkAccessFlags srcAccess,
                              VkAccessFlags dstAccess);

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
        Context *m_context = nullptr;
        VkImage m_image = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;
        VkImageView m_view = VK_NULL_HANDLE;
        ImageDescriptor m_descriptor{};
        VkImageLayout m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        void createImageAndAllocation();
        void createImageView();
        void moveFrom(Image &&rhs) noexcept;
    };

} // namespace Engine::Core

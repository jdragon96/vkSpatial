#include "vkRender/Image.h"

#include <stdexcept>
#include <utility>

namespace vkRender {

    ImageDescriptor ImageDescriptor::Depth2D(VkExtent2D extent,
                                             VkFormat format) {
        ImageDescriptor descriptor{};
        descriptor.width = extent.width;
        descriptor.height = extent.height;
        descriptor.format = format;
        descriptor.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        descriptor.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        return descriptor;
    }

    ImageDescriptor ImageDescriptor::Color2D(VkExtent2D extent,
                                             VkFormat format,
                                             VkImageUsageFlags usage) {
        ImageDescriptor descriptor{};
        descriptor.width = extent.width;
        descriptor.height = extent.height;
        descriptor.format = format;
        descriptor.usage = usage;
        descriptor.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        return descriptor;
    }

    Image::Image(vkCommon::VkContext *context)
        : m_context(context) {
        if (!m_context)
            throw std::runtime_error("Image requires a valid VkContext");
    }

    Image::Image(vkCommon::VkContext *context,
                 const ImageDescriptor &descriptor)
        : Image(context) {
        Create(descriptor);
    }

    Image::~Image() {
        Destroy();
    }

    Image::Image(Image &&rhs) noexcept {
        MoveFrom(std::move(rhs));
    }

    Image &Image::operator=(Image &&rhs) noexcept {
        if (this != &rhs) {
            Destroy();
            MoveFrom(std::move(rhs));
        }
        return *this;
    }

    Image &Image::Create(const ImageDescriptor &descriptor) {
        if (descriptor.width == 0 || descriptor.height == 0 || descriptor.depth == 0)
            throw std::runtime_error("Image::Create requires a non-empty extent");
        if (descriptor.format == VK_FORMAT_UNDEFINED)
            throw std::runtime_error("Image::Create requires a valid VkFormat");
        if (descriptor.usage == 0)
            throw std::runtime_error("Image::Create requires VkImageUsageFlags");
        if (descriptor.createView && descriptor.aspectMask == 0)
            throw std::runtime_error("Image::Create view requires aspectMask");

        Destroy();
        m_descriptor = descriptor;

        try {
            CreateImageObject();
            AllocateAndBindMemory();
            if (m_descriptor.createView)
                CreateImageView();
        } catch (...) {
            Destroy();
            throw;
        }

        m_currentLayout = m_descriptor.initialLayout;

        return *this;
    }

    void Image::Destroy() {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            return;

        if (m_view != VK_NULL_HANDLE)
            vkDestroyImageView(m_context->device, m_view, nullptr);
        if (m_image != VK_NULL_HANDLE)
            vkDestroyImage(m_context->device, m_image, nullptr);
        if (m_memory != VK_NULL_HANDLE)
            vkFreeMemory(m_context->device, m_memory, nullptr);

        m_view = VK_NULL_HANDLE;
        m_image = VK_NULL_HANDLE;
        m_memory = VK_NULL_HANDLE;
        m_descriptor = {};
        m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void Image::TransitionLayout(VkCommandBuffer cmd,
                                 VkImageLayout newLayout,
                                 VkPipelineStageFlags srcStage,
                                 VkPipelineStageFlags dstStage,
                                 VkAccessFlags srcAccess,
                                 VkAccessFlags dstAccess) {
        if (m_currentLayout == newLayout)
            return;

        TransitionLayout(cmd, m_image, m_descriptor.aspectMask,
                         m_currentLayout, newLayout,
                         srcStage, dstStage, srcAccess, dstAccess);
        m_currentLayout = newLayout;
    }

    void Image::TransitionLayout(VkCommandBuffer cmd,
                                 VkImage image,
                                 VkImageAspectFlags aspectMask,
                                 VkImageLayout oldLayout,
                                 VkImageLayout newLayout,
                                 VkPipelineStageFlags srcStage,
                                 VkPipelineStageFlags dstStage,
                                 VkAccessFlags srcAccess,
                                 VkAccessFlags dstAccess) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {aspectMask, 0, 1, 0, 1};
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(cmd,
                             srcStage,
                             dstStage,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &barrier);
    }

    bool Image::Matches(VkExtent2D extent, VkFormat format) const {
        if (!Valid())
            return false;
        if (m_descriptor.width != extent.width ||
            m_descriptor.height != extent.height)
            return false;
        return format == VK_FORMAT_UNDEFINED || m_descriptor.format == format;
    }

    uint32_t Image::FindMemoryType(uint32_t typeFilter,
                                   VkMemoryPropertyFlags properties) const {
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(m_context->physDevice, &memoryProperties);
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1u << i)) &&
                (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        throw std::runtime_error("Image: no suitable memory type");
    }

    void Image::CreateImageObject() {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = m_descriptor.imageType;
        imageInfo.format = m_descriptor.format;
        imageInfo.extent = {
                m_descriptor.width,
                m_descriptor.height,
                m_descriptor.depth,
        };
        imageInfo.mipLevels = m_descriptor.mipLevels;
        imageInfo.arrayLayers = m_descriptor.arrayLayers;
        imageInfo.samples = m_descriptor.samples;
        imageInfo.tiling = m_descriptor.tiling;
        imageInfo.usage = m_descriptor.usage;
        imageInfo.sharingMode = m_descriptor.sharingMode;
        imageInfo.initialLayout = m_descriptor.initialLayout;

        if (vkCreateImage(m_context->device, &imageInfo, nullptr, &m_image) != VK_SUCCESS)
            throw std::runtime_error("Image: failed to create VkImage");
    }

    void Image::AllocateAndBindMemory() {
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(m_context->device, m_image, &requirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = requirements.size;
        allocInfo.memoryTypeIndex = FindMemoryType(
                requirements.memoryTypeBits,
                m_descriptor.memoryProperties);

        if (vkAllocateMemory(m_context->device, &allocInfo, nullptr, &m_memory) != VK_SUCCESS)
            throw std::runtime_error("Image: failed to allocate VkDeviceMemory");

        if (vkBindImageMemory(m_context->device, m_image, m_memory, 0) != VK_SUCCESS)
            throw std::runtime_error("Image: failed to bind VkImage memory");
    }

    void Image::CreateImageView() {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_image;
        viewInfo.viewType = m_descriptor.viewType;
        viewInfo.format = m_descriptor.format;
        viewInfo.subresourceRange.aspectMask = m_descriptor.aspectMask;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = m_descriptor.mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = m_descriptor.arrayLayers;

        if (vkCreateImageView(m_context->device, &viewInfo, nullptr, &m_view) != VK_SUCCESS)
            throw std::runtime_error("Image: failed to create VkImageView");
    }

    void Image::MoveFrom(Image &&rhs) noexcept {
        m_context = rhs.m_context;
        m_image = rhs.m_image;
        m_memory = rhs.m_memory;
        m_view = rhs.m_view;
        m_descriptor = rhs.m_descriptor;

        rhs.m_context = nullptr;
        rhs.m_image = VK_NULL_HANDLE;
        rhs.m_memory = VK_NULL_HANDLE;
        rhs.m_view = VK_NULL_HANDLE;
        rhs.m_descriptor = {};
    }

} // namespace vkRender

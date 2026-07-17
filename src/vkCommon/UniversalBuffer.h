#pragma once

#include "utilities/Math.h"
#include "utilities/VulkanUtilities.h"
#include "vkCommon/vkContext.h"

#include <memory>
#include <vulkan/vulkan.h>

// vkCmdCopyImageToBuffer
// VkImage 버퍼에서 de-tiling 과정을 거쳐 VkBuffer 로 복사하는데 사용되는 유틸리티 클래스

class LinearBuffer {
public:
    void FromImage(vkCommon::VkContext *context,
                   VkCommandBuffer cmd,
                   VkImage srcImage,
                   VkImageLayout currentLayout,
                   VkExtent2D extent,
                   VkImageAspectFlags aspectMask) {

        Ensure(context, static_cast<VkDeviceSize>(extent.width) * extent.height * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    ~LinearBuffer() {
        Destroy(m_context);
    }

private:
    void Destroy(vkCommon::VkContext *context) {
        if (m_bufferHandle != VK_NULL_HANDLE)
            vkDestroyBuffer(context->device, m_bufferHandle, nullptr);
        if (m_memoryHandle != VK_NULL_HANDLE)
            vkFreeMemory(context->device, m_memoryHandle, nullptr);
        m_bufferHandle = VK_NULL_HANDLE;
        m_memoryHandle = VK_NULL_HANDLE;
        m_size = 0;
    }

    void Ensure(vkCommon::VkContext *context, VkDeviceSize size, VkBufferUsageFlags usage) {
        if (m_bufferHandle != VK_NULL_HANDLE && m_size == size)
            return;

        Destroy(context);

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(context->device, &bufferInfo, nullptr, &m_bufferHandle) != VK_SUCCESS)
            throw std::runtime_error("LinearBuffer: failed to create buffer");

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(context->device, m_bufferHandle, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;

        auto result = VulkanUtilities::FindMemoryType(
                context->physDevice,
                memRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (result.has_value()) {
            allocInfo.memoryTypeIndex = result.value();
        } else {
            throw std::runtime_error("LinearBuffer: failed to find suitable memory type");
        }

        if (vkAllocateMemory(context->device, &allocInfo, nullptr, &m_memoryHandle) != VK_SUCCESS) {
            vkDestroyBuffer(context->device, m_bufferHandle, nullptr);
            throw std::runtime_error("LinearBuffer: failed to allocate memory");
        }

        if (vkBindBufferMemory(context->device, m_bufferHandle, m_memoryHandle, 0) != VK_SUCCESS) {
            vkFreeMemory(context->device, m_memoryHandle, nullptr);
            vkDestroyBuffer(context->device, m_bufferHandle, nullptr);
            throw std::runtime_error("LinearBuffer: failed to bind buffer memory");
        }

        m_size = size;
    }

private:
    VkBuffer m_bufferHandle = VK_NULL_HANDLE;
    VkDeviceMemory m_memoryHandle = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
};
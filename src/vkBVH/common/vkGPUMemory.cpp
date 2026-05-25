#include "vkBVH/common/vkGPUMemory.h"

#include <cstring>
#include <stdexcept>

vkGPUMemory::vkGPUMemory(VkDevice device, VkPhysicalDevice physicalDevice,
                         VkBufferUsageFlags extraUsage)
    : m_device(device), m_physicalDevice(physicalDevice),
      m_extraUsage(extraUsage) {}

vkGPUMemory::~vkGPUMemory() { Clear(); }

bool vkGPUMemory::Allocate(uint32_t bytes) {
    if (m_buffer != VK_NULL_HANDLE) Clear();

    m_size = bytes;

    VkBufferUsageFlags usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            m_extraUsage;

    return createBuffer(bytes, usage,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        m_buffer, m_memory);
}

bool vkGPUMemory::Clear() {
    if (m_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device, m_buffer, nullptr);
        m_buffer = VK_NULL_HANDLE;
    }
    if (m_memory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_memory, nullptr);
        m_memory = VK_NULL_HANDLE;
    }
    m_size = 0;
    return true;
}

// ── 데이터 전송 ───────────────────────────────────────────────────────────────

bool vkGPUMemory::Upload(const void *data, uint32_t bytes,
                         VkQueue queue, VkCommandPool cmdPool) {
    if (!m_buffer || bytes > m_size) return false;

    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    if (!createBuffer(bytes,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      stagingBuf, stagingMem))
        return false;

    {
        vkScopedMemory mapped(m_device, stagingMem, 0, bytes);
        std::memcpy(mapped.data(), data, bytes);
    } // 스코프 종료 시 자동 Unmap

    submitCopy(stagingBuf, m_buffer, bytes, queue, cmdPool);

    vkDestroyBuffer(m_device, stagingBuf, nullptr);
    vkFreeMemory(m_device, stagingMem, nullptr);
    return true;
}

bool vkGPUMemory::Download(void *data, uint32_t bytes,
                           VkQueue queue, VkCommandPool cmdPool) {
    if (!m_buffer || bytes > m_size) return false;

    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    if (!createBuffer(bytes,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      stagingBuf, stagingMem))
        return false;

    submitCopy(m_buffer, stagingBuf, bytes, queue, cmdPool);

    {
        vkScopedMemory mapped(m_device, stagingMem, 0, bytes);
        std::memcpy(data, mapped.data(), bytes);
    } // 스코프 종료 시 자동 Unmap

    vkDestroyBuffer(m_device, stagingBuf, nullptr);
    vkFreeMemory(m_device, stagingMem, nullptr);
    return true;
}

vkScopedMemory vkGPUMemory::MapScoped(VkDeviceSize offset, VkDeviceSize size) {
    return vkScopedMemory(m_device, m_memory, offset, size);
}

uint32_t vkGPUMemory::findMemoryType(uint32_t typeFilter,
                                     VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((typeFilter & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    throw std::runtime_error("vkGPUMemory: no suitable memory type");
}

bool vkGPUMemory::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags memProps,
                               VkBuffer &outBuf, VkDeviceMemory &outMem) const {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(m_device, &bi, nullptr, &outBuf) != VK_SUCCESS)
        return false;

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(m_device, outBuf, &mr);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, memProps);

    if (vkAllocateMemory(m_device, &ai, nullptr, &outMem) != VK_SUCCESS) {
        vkDestroyBuffer(m_device, outBuf, nullptr);
        outBuf = VK_NULL_HANDLE;
        return false;
    }

    vkBindBufferMemory(m_device, outBuf, outMem, 0);
    return true;
}

void vkGPUMemory::submitCopy(VkBuffer src, VkBuffer dst, VkDeviceSize size,
                             VkQueue queue, VkCommandPool cmdPool) const {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = cmdPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(m_device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkBufferCopy region{.size = size};
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(m_device, cmdPool, 1, &cmd);
}

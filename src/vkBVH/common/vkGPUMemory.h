#pragma once

#include "vkBVH/common/vkScopedMemory.h"

#include <vulkan/vulkan.h>

class vkGPUMemory {


public:
    vkGPUMemory(VkDevice device, VkPhysicalDevice physicalDevice,
                VkBufferUsageFlags extraUsage = 0);
    ~vkGPUMemory();

    virtual bool Allocate(uint32_t bytes);

    virtual bool Clear();

    bool Upload(const void *data, uint32_t bytes, VkQueue queue, VkCommandPool cmdPool);

    bool Download(void *data, uint32_t bytes, VkQueue queue, VkCommandPool cmdPool);

    vkScopedMemory MapScoped(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

    VkBuffer GetBuffer() const { return m_buffer; }

    VkDeviceMemory GetMemory() const { return m_memory; }

    uint32_t GetSize() const { return m_size; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    uint32_t m_size = 0;
    VkBufferUsageFlags m_extraUsage = 0;

    uint32_t findMemoryType(uint32_t typeFilter,
                            VkMemoryPropertyFlags properties) const;

    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags memProps,
                      VkBuffer &outBuf, VkDeviceMemory &outMem) const;

    void submitCopy(VkBuffer src, VkBuffer dst, VkDeviceSize size,
                    VkQueue queue, VkCommandPool cmdPool) const;
};

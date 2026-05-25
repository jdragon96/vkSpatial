#include "vkSpatial/common/vkScopedMemory.h"

namespace vkCommon {

    vkScopedMemory::vkScopedMemory(VkDevice device,
                                   VkDeviceMemory memory,
                                   VkDeviceSize offset,
                                   VkDeviceSize size)
        : m_device(device),
          m_memory(memory) {
        vkMapMemory(m_device, m_memory, offset, size, 0, &m_ptr);
    }

    vkScopedMemory::vkScopedMemory(vkScopedMemory &&o) noexcept
        : m_device(o.m_device),
          m_memory(o.m_memory),
          m_ptr(o.m_ptr) {
        o.m_ptr = nullptr;
    }

    vkScopedMemory::~vkScopedMemory() {
        if (m_ptr) vkUnmapMemory(m_device, m_memory);
    }

} // namespace vkCommon
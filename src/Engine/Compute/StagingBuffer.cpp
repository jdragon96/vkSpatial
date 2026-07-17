#include "Engine/Compute/StagingBuffer.h"

#include <stdexcept>

namespace Engine::Compute {

    StagingBuffer::StagingBuffer(Engine::Core::Context &context, VkDeviceSize bytes,
                                 VkBufferUsageFlags usage)
        : m_context(context), m_size(bytes) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bytes;
        bufferInfo.usage = usage;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo allocationInfo{};
        if (vmaCreateBuffer(m_context.allocator, &bufferInfo, &allocInfo,
                            &m_buffer, &m_allocation, &allocationInfo) != VK_SUCCESS)
            throw std::runtime_error("StagingBuffer: failed to allocate");
        m_mapped = allocationInfo.pMappedData;
    }

    StagingBuffer::~StagingBuffer() {
        if (m_buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(m_context.allocator, m_buffer, m_allocation);
    }

} // namespace Engine::Compute

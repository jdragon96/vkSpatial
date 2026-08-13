#include "Engine/Core/Buffer.h"

#include <cstring>
#include <stdexcept>

namespace Engine::Core {

    Buffer::Buffer(Context &context, VkBufferUsageFlags extraUsage)
        : m_context(context), m_extraUsage(extraUsage) {}

    Buffer::~Buffer() {
        free();
    }

    void Buffer::free() {
        if (m_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_context.allocator, m_buffer, m_allocation);
            m_buffer = VK_NULL_HANDLE;
            m_allocation = VK_NULL_HANDLE;
        }
        m_size = 0;
        m_mapped = nullptr;
    }

    void Buffer::Allocate(uint32_t bytes) {
        free();

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bytes;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           m_extraUsage;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

        if (vmaCreateBuffer(m_context.allocator, &bufferInfo, &allocInfo,
                            &m_buffer, &m_allocation, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Buffer: failed to allocate");

        m_size = bytes;
    }

    void Buffer::AllocateHostVisible(uint32_t bytes) {
        free();

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bytes;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           m_extraUsage;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo info{};
        if (vmaCreateBuffer(m_context.allocator, &bufferInfo, &allocInfo, &m_buffer, &m_allocation,
                            &info) != VK_SUCCESS)
            throw std::runtime_error("Buffer: failed to allocate host-visible");

        m_mapped = info.pMappedData;
        m_size = bytes;
    }

    void Buffer::AllocateHostVisibleReadback(uint32_t bytes) {
        free();

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bytes;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | m_extraUsage;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        // RANDOM (not SEQUENTIAL_WRITE) so the mapping is readable — the CPU reads GPU writes back.
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo info{};
        if (vmaCreateBuffer(m_context.allocator, &bufferInfo, &allocInfo, &m_buffer, &m_allocation,
                            &info) != VK_SUCCESS)
            throw std::runtime_error("Buffer: failed to allocate host-visible readback");

        m_mapped = info.pMappedData;
        m_size = bytes;
    }
    void Buffer::MakeVisibleToGPU(uint32_t bytes) const {
        if (m_allocation != VK_NULL_HANDLE)
            vmaFlushAllocation(m_context.allocator, m_allocation, 0, bytes);
    }
    void Buffer::MakeVisibleToCPU(uint32_t bytes) const {
        if (m_allocation != VK_NULL_HANDLE)
            vmaInvalidateAllocation(m_context.allocator, m_allocation, 0, bytes);
    }

    void Buffer::Upload(const void *data, uint32_t bytes, QueueRole role) {
        if (m_buffer == VK_NULL_HANDLE || bytes > m_size)
            throw std::runtime_error("Buffer::Upload: not allocated or bytes exceeds capacity");

        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = bytes;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                 VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingAllocation = VK_NULL_HANDLE;
        VmaAllocationInfo stagingAllocationInfo{};
        if (vmaCreateBuffer(m_context.allocator, &stagingInfo, &stagingAllocInfo,
                            &stagingBuffer, &stagingAllocation, &stagingAllocationInfo) != VK_SUCCESS)
            throw std::runtime_error("Buffer::Upload: failed to create staging buffer");

        std::memcpy(stagingAllocationInfo.pMappedData, data, bytes);

        VkBuffer dstBuffer = m_buffer;
        try {
            if (vmaFlushAllocation(m_context.allocator, stagingAllocation, 0, bytes) != VK_SUCCESS)
                throw std::runtime_error("Buffer::Upload: failed to flush staging allocation");
            SubmitOneShot(m_context, role, [&](VkCommandBuffer cmd) {
                VkBufferCopy region{};
                region.size = bytes;
                vkCmdCopyBuffer(cmd, stagingBuffer, dstBuffer, 1, &region);
            });
        } catch (...) {
            vmaDestroyBuffer(m_context.allocator, stagingBuffer, stagingAllocation);
            throw;
        }

        vmaDestroyBuffer(m_context.allocator, stagingBuffer, stagingAllocation);
    }

    void Buffer::Download(void *data, uint32_t bytes, QueueRole role) {
        if (m_buffer == VK_NULL_HANDLE || bytes > m_size)
            throw std::runtime_error("Buffer::Download: not allocated or bytes exceeds capacity");

        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = bytes;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                 VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingAllocation = VK_NULL_HANDLE;
        VmaAllocationInfo stagingAllocationInfo{};
        if (vmaCreateBuffer(m_context.allocator, &stagingInfo, &stagingAllocInfo,
                            &stagingBuffer, &stagingAllocation, &stagingAllocationInfo) != VK_SUCCESS)
            throw std::runtime_error("Buffer::Download: failed to create staging buffer");

        VkBuffer srcBuffer = m_buffer;
        try {
            SubmitOneShot(m_context, role, [&](VkCommandBuffer cmd) {
                VkBufferCopy region{};
                region.size = bytes;
                vkCmdCopyBuffer(cmd, srcBuffer, stagingBuffer, 1, &region);
            });
            if (vmaInvalidateAllocation(m_context.allocator, stagingAllocation, 0, bytes) != VK_SUCCESS)
                throw std::runtime_error("Buffer::Download: failed to invalidate staging allocation");
        } catch (...) {
            vmaDestroyBuffer(m_context.allocator, stagingBuffer, stagingAllocation);
            throw;
        }

        std::memcpy(data, stagingAllocationInfo.pMappedData, bytes);

        vmaDestroyBuffer(m_context.allocator, stagingBuffer, stagingAllocation);
    }

} // namespace Engine::Core

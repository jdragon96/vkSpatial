#pragma once

#include "Engine/Core/Context.h"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace Engine::Compute {

    // A persistently-mapped host-visible VMA buffer for staging transfers. Unlike
    // Engine::Core::Buffer (device-local, transient staging created per Upload/Download),
    // this is allocated once and its Mapped() pointer stays valid, so it can back many
    // batched copies without re-allocating each frame.
    class StagingBuffer {
    public:
        StagingBuffer(Engine::Core::Context &context, VkDeviceSize bytes, VkBufferUsageFlags usage);
        ~StagingBuffer();

        StagingBuffer(const StagingBuffer &) = delete;
        StagingBuffer &operator=(const StagingBuffer &) = delete;

        void *Mapped() const { return m_mapped; }
        VkBuffer Handle() const { return m_buffer; }
        VkDeviceSize Size() const { return m_size; }

    private:
        Engine::Core::Context &m_context;
        VkBuffer m_buffer = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;
        void *m_mapped = nullptr;
        VkDeviceSize m_size = 0;
    };

} // namespace Engine::Compute

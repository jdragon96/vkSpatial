#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Core/OneShotCommands.h"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace Engine::Core {

    class Buffer {
    public:
        explicit Buffer(Context &context, VkBufferUsageFlags extraUsage = 0);
        ~Buffer();

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;

        // (Re)allocates a device-local buffer of `bytes` size. Any prior allocation is freed
        // first. Throws std::runtime_error on failure.
        void Allocate(uint32_t bytes);

        // Uploads `bytes` from `data` via a temporary host-visible staging buffer and a
        // one-shot GPU copy on `role`'s queue. Throws if `bytes` exceeds the current
        // allocation.
        void Upload(const void *data, uint32_t bytes, QueueRole role = QueueRole::Compute);

        // Downloads `bytes` into `data` via a temporary host-visible staging buffer and a
        // one-shot GPU copy on `role`'s queue. Throws if `bytes` exceeds the current
        // allocation.
        void Download(void *data, uint32_t bytes, QueueRole role = QueueRole::Compute);

        VkBuffer Handle() const { return m_buffer; }
        uint32_t Size() const { return m_size; }

    private:
        Context &m_context;
        VkBufferUsageFlags m_extraUsage;
        VkBuffer m_buffer = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;
        uint32_t m_size = 0;

        void free();
    };

} // namespace Engine::Core

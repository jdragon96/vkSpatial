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

        // Allocates a HOST_VISIBLE (and, on UMA, DEVICE_LOCAL) buffer that stays persistently
        // mapped: MappedPtr() returns a CPU pointer you can memcpy into, and the same buffer is a
        // valid storage buffer for shaders (zero-copy upload — no staging, no submit). Any prior
        // allocation is freed first. Throws std::runtime_error on failure.
        void AllocateHostVisible(uint32_t bytes);

        // Like AllocateHostVisible, but the mapping also allows READING back GPU writes (host-random
        // access). Use for a buffer a shader WRITES and the CPU then reads via MappedPtr — pair with
        // InvalidateMapped() before the read. Zero-copy readback (no staging, no download copy).
        void AllocateHostVisibleReadback(uint32_t bytes);

        // Persistent mapped pointer for AllocateHostVisible* buffers; nullptr after plain Allocate.
        void *MappedPtr() const { return m_mapped; }

        // Flush `bytes` of host writes to the device (no-op on HOST_COHERENT memory; always safe).
        void FlushMapped(uint32_t bytes) const;

        // Invalidate `bytes` of the mapping so a subsequent CPU read sees the device's writes (no-op
        // on HOST_COHERENT memory; always safe). Call after the writing dispatch completes.
        void InvalidateMapped(uint32_t bytes) const;

        VkBuffer Handle() const { return m_buffer; }
        uint32_t Size() const { return m_size; }

    private:
        Context &m_context;
        VkBufferUsageFlags m_extraUsage;
        VkBuffer m_buffer = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;
        uint32_t m_size = 0;
        void *m_mapped = nullptr;

        void free();
    };

} // namespace Engine::Core

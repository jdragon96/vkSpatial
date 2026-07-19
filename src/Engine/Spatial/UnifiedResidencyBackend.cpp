#include "Engine/Spatial/UnifiedResidencyBackend.h"

#include <cstring>
#include <stdexcept>

namespace Engine::Spatial {

    UnifiedResidencyBackend::MappedBuffer
    UnifiedResidencyBackend::allocCoherent(uint32_t bytes, VkBufferUsageFlags usage) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes;
        bi.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        // Ask for host access + persistent map; on UMA VMA returns a DEVICE_LOCAL|HOST_VISIBLE heap.
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        MappedBuffer b{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(m_ctx->allocator, &bi, &ai, &b.buffer, &b.alloc, &info) != VK_SUCCESS)
            throw std::runtime_error("UnifiedResidencyBackend: coherent alloc failed");
        b.mapped = info.pMappedData; // non-null because MAPPED_BIT was set
        // VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT only *prefers* HOST_COHERENT (it may
        // pick HOST_CACHED, which on some UMA devices is host-visible-but-not-coherent). Query
        // what we actually got so flushIfNeeded knows whether an explicit flush is required.
        VkMemoryPropertyFlags props = 0;
        vmaGetAllocationMemoryProperties(m_ctx->allocator, b.alloc, &props);
        b.coherent = (props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        std::memset(b.mapped, 0, bytes);
        return b;
    }

    void UnifiedResidencyBackend::freeBuffer(MappedBuffer &b) {
        if (b.buffer) vmaDestroyBuffer(m_ctx->allocator, b.buffer, b.alloc);
        b = {};
    }

    void UnifiedResidencyBackend::flushIfNeeded(const MappedBuffer &b) const {
        // No-op on HOST_COHERENT memory: the next vkQueueSubmit makes prior host writes
        // visible to the device automatically. Only non-coherent memory needs the flush.
        if (!b.coherent)
            vmaFlushAllocation(m_ctx->allocator, b.alloc, 0, VK_WHOLE_SIZE);
    }

    UnifiedResidencyBackend::~UnifiedResidencyBackend() {
        if (m_ctx) { freeBuffer(m_pool); freeBuffer(m_indexGrid); freeBuffer(m_meta); }
    }

    void UnifiedResidencyBackend::Build(Engine::Core::Context &ctx, uint32_t initialCapacity) {
        m_ctx = &ctx;
        m_capacity = initialCapacity;
        m_pool = allocCoherent(uint32_t(sizeof(GpuTsdfVoxel)) * kVoxelsPerGroup * m_capacity, 0);
        m_indexGrid = allocCoherent(uint32_t(sizeof(uint32_t)) * kIndexGridCells, 0);
        m_meta = allocCoherent(uint32_t(sizeof(ActiveGroupMeta)) * m_capacity, 0);
    }

    uint32_t UnifiedResidencyBackend::offsetOf(const Eigen::Vector3i &g, uint8_t dir) const {
        Eigen::Vector3i l = g - m_localBase;
        if (l.x() < 0 || l.y() < 0 || l.z() < 0 ||
            l.x() >= int(kLocalGroupGrid) || l.y() >= int(kLocalGroupGrid) || l.z() >= int(kLocalGroupGrid))
            return kInvalidPoolIndex;
        return IndexGridOffset(uint32_t(l.x()), uint32_t(l.y()), uint32_t(l.z()), dir);
    }

    void UnifiedResidencyBackend::BeginFrame(const Eigen::Vector3i &localBase) {
        m_localBase = localBase;
        m_stats = {};
        // Relabel: reset indexGrid, then re-register every resident slot that falls in the window.
        auto *grid = static_cast<uint32_t *>(m_indexGrid.mapped);
        for (uint32_t i = 0; i < kIndexGridCells; ++i) grid[i] = kInvalidPoolIndex;
        auto *meta = static_cast<ActiveGroupMeta *>(m_meta.mapped);
        for (const auto &kv : m_slotOf) {
            const DirectionalGroupKey &k = kv.first;
            uint32_t slot = kv.second;
            uint32_t cell = offsetOf(Eigen::Vector3i(k.gx, k.gy, k.gz), k.direction);
            if (cell != kInvalidPoolIndex) {
                grid[cell] = slot;
                meta[slot] = ActiveGroupMeta{k.gx, k.gy, k.gz,
                    PackMeta(k.direction, SlotState::ResidentClean, /*dirty=*/false, /*valid=*/true)};
                ++m_stats.residentCount;
                // Every slot re-registered here was already resident from a prior frame (no
                // GPU/host round trip needed) — that is exactly the "reusable" classification
                // Streaming reports via classify; on UMA it coincides with residentCount.
                ++m_stats.reusableCount;
            }
        }
        // The indexGrid reset + relabel and the meta rewrites above are CPU writes the GPU
        // integrate/extract kernels will read this frame; make them visible before any submit.
        flushIfNeeded(m_indexGrid);
        flushIfNeeded(m_meta);
    }

    void UnifiedResidencyBackend::EnsureResident(const std::vector<DirectionalGroupKey> &required) {
        auto *grid = static_cast<uint32_t *>(m_indexGrid.mapped);
        auto *meta = static_cast<ActiveGroupMeta *>(m_meta.mapped);
        auto *pool = static_cast<GpuTsdfVoxel *>(m_pool.mapped);
        for (const DirectionalGroupKey &k : required) {
            uint32_t cell = offsetOf(Eigen::Vector3i(k.gx, k.gy, k.gz), k.direction);
            if (cell == kInvalidPoolIndex)
                throw std::runtime_error("UnifiedResidencyBackend: required key outside window");
            auto it = m_slotOf.find(k);
            uint32_t slot;
            if (it == m_slotOf.end()) {
                if (m_slotOf.size() >= m_capacity)
                    throw std::runtime_error("UnifiedResidencyBackend: pool exhausted (grow not yet impl)");
                slot = uint32_t(m_slotOf.size());
                m_slotOf.emplace(k, slot);
                std::memset(pool + size_t(slot) * kVoxelsPerGroup, 0,
                            sizeof(GpuTsdfVoxel) * kVoxelsPerGroup); // zero-fill on first touch
                // Count keys first made resident this frame (BeginFrame already counted the
                // prior-resident slots it re-registered), so residentCount == total resident,
                // matching StreamingResidencyBackend::RecordResidency's residentIndex.size().
                ++m_stats.residentCount;
            } else {
                slot = it->second;
            }
            grid[cell] = slot;
            meta[slot] = ActiveGroupMeta{k.gx, k.gy, k.gz,
                PackMeta(k.direction, SlotState::ResidentClean, false, true)};
        }
        // Zero-copy: no H2D bytes, nothing "missing".
        m_stats.h2dBytes = 0;
        m_stats.missingCount = 0;
        // Pool zero-fills, indexGrid registrations and meta writes above are all CPU writes
        // the GPU integrate/extract kernels read; flush them before DirectionalTSDF submits.
        flushIfNeeded(m_pool);
        flushIfNeeded(m_indexGrid);
        flushIfNeeded(m_meta);
    }

    std::vector<uint32_t> UnifiedResidencyBackend::DebugDownloadIndexGrid() {
        auto *grid = static_cast<uint32_t *>(m_indexGrid.mapped);
        return std::vector<uint32_t>(grid, grid + kIndexGridCells);
    }

    uint32_t UnifiedResidencyBackend::DebugQueryPoolIndex(const DirectionalGroupKey &key) {
        auto it = m_slotOf.find(key);
        return it == m_slotOf.end() ? kInvalidPoolIndex : it->second;
    }

    DirectionalHostStore::Group UnifiedResidencyBackend::DebugDownloadGroupVoxels(const DirectionalGroupKey &key) {
        DirectionalHostStore::Group out{}; // value/weight form
        auto it = m_slotOf.find(key);
        if (it == m_slotOf.end()) return out;
        auto *pool = static_cast<GpuTsdfVoxel *>(m_pool.mapped);
        const GpuTsdfVoxel *g = pool + size_t(it->second) * kVoxelsPerGroup;
        for (uint32_t i = 0; i < kVoxelsPerGroup; ++i) {
            out[i].weight = float(g[i].sumW) / float(kTsdfFixedScale);
            out[i].value = g[i].sumW ? float(g[i].sumDW) / float(g[i].sumW) : 0.0f;
        }
        return out;
    }

} // namespace Engine::Spatial

#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "TSDF/Backends/Residency/IResidencyBackend.h"

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <vk_mem_alloc.h>

namespace TSDF {

    // UMA backend: one coherent DEVICE_LOCAL|HOST_VISIBLE pool holds the whole model, so
    // residency is CPU-side slot bookkeeping + indexGrid relabel with no GPU copies.
    class UnifiedResidencyBackend : public IResidencyBackend {
    public:
        ~UnifiedResidencyBackend() override;
        void Build(Engine::Core::Context &ctx, uint32_t initialCapacity);

        void BeginFrame(const Eigen::Vector3i &localBase) override;
        void EnsureResident(const std::vector<DirectionalGroupKey> &required) override;
        void EndFrame() override {}

        // Expanded-interface methods (Part A). On UMA the batched path is the same CPU
        // bookkeeping as EnsureResident and records nothing into the command batch.
        uint32_t RecordResidency(const std::vector<DirectionalGroupKey> &required,
                                 Engine::Compute::CommandBatch &batch) override {
            EnsureResident(required);
            (void)batch;
            return uint32_t(required.size());
        }
        const std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> &
        ResidentIndex() const override { return m_slotOf; }

        VkBuffer IndexGridBuffer() const override { return m_indexGrid.buffer; }
        VkBuffer PoolVoxelBuffer() const override { return m_pool.buffer; }
        VkBuffer MetaBuffer() const override { return m_meta.buffer; }
        uint32_t PoolCapacity() const override { return m_capacity; }
        Eigen::Vector3i LocalBase() const override { return m_localBase; }
        bool IsUnified() const override { return true; }
        bool SupportsZeroCopyCpuAccess() const override { return true; }
        DirectionalHostStore &HostStore() override { return m_storeView; }
        ResidencyStats FrameStats() const override { return m_stats; }
        std::vector<uint32_t> DebugDownloadIndexGrid() override;
        uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &key) override;
        DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &key) override;

    private:
        struct MappedBuffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            void *mapped = nullptr;
            bool coherent = true; // actual HOST_COHERENT-ness of the backing memory (queried in Build)
        };
        MappedBuffer allocCoherent(VkDeviceSize bytes, VkBufferUsageFlags usage);
        void freeBuffer(MappedBuffer &b);
        // Make CPU writes through `b.mapped` visible to the GPU before a queue submit. On
        // HOST_COHERENT memory (the M4 Max UMA case) the submit's implicit host-write
        // visibility already covers it, so this is a no-op; on HOST_CACHED-but-not-coherent
        // memory (which VMA may still hand back for HOST_ACCESS_RANDOM) an explicit flush is
        // required by the Vulkan spec. Called after every CPU write to pool/indexGrid/meta.
        void flushIfNeeded(const MappedBuffer &b) const;
        uint32_t offsetOf(const Eigen::Vector3i &g, uint8_t dir) const; // indexGrid cell, or kInvalidPoolIndex if outside window

        Engine::Core::Context *m_ctx = nullptr;
        uint32_t m_capacity = 0;
        Eigen::Vector3i m_localBase = Eigen::Vector3i::Zero();
        MappedBuffer m_pool;       // GpuTsdfVoxel[capacity*512]
        MappedBuffer m_indexGrid;  // uint32[kIndexGridCells]
        MappedBuffer m_meta;       // ActiveGroupMeta[capacity]
        std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> m_slotOf;
        DirectionalHostStore m_storeView; // unused authoritative-store placeholder for interface parity
        ResidencyStats m_stats;
    };

} // namespace TSDF

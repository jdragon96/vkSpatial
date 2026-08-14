#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Compute/StagingBuffer.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "TSDF/Backends/Residency/IResidencyBackend.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace TSDF {

    // Streaming (discrete/PCIe) residency backend: an active GPU pool that caches a
    // window of directional groups over the authoritative host store, with
    // classify/register/write-back machinery driven entirely on the GPU
    // (docs/superpowers/specs/2026-07-17-directional-tsdf-design.md). This is the
    // residency half of what used to be DirectionalTSDF before the IResidencyBackend
    // extraction (Task 2/3 of the crossplatform-directional-tsdf-uma plan) — method
    // bodies were moved verbatim, not reimplemented.
    class StreamingResidencyBackend : public IResidencyBackend {
    public:
        void Build(Engine::Core::Context &ctx, uint32_t poolCapacity);

        void BeginFrame(const Eigen::Vector3i &localBase) override;
        void EnsureResident(const std::vector<DirectionalGroupKey> &required) override;
        void EndFrame() override;

        // Stages missing groups into persistent staging and records upload copies +
        // combined (reusable + missing) register dispatch into `batch`. Does NOT submit.
        // Returns the number of slots registered.
        uint32_t RecordResidency(const std::vector<DirectionalGroupKey> &required,
                                 Engine::Compute::CommandBatch &batch) override;
        const std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> &
        ResidentIndex() const override { return m_residentIndex; }

        VkBuffer IndexGridBuffer() const override { return m_indexGrid->Handle(); }
        VkBuffer PoolVoxelBuffer() const override { return m_poolVoxels->Handle(); }
        VkBuffer MetaBuffer() const override { return m_metaBuffer->Handle(); }
        uint32_t PoolCapacity() const override { return m_poolCapacity; }
        Eigen::Vector3i LocalBase() const override { return m_localBase; }
        bool IsUnified() const override { return false; }
        bool SupportsZeroCopyCpuAccess() const override { return false; }
        DirectionalHostStore &HostStore() override { return m_hostStore; }
        ResidencyStats FrameStats() const override { return m_stats; }
        std::vector<uint32_t> DebugDownloadIndexGrid() override;
        uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &key) override;
        DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &key) override;

    private:
        // Recomputes m_stats.overlapRatio from the required/missing counts accumulated by
        // RecordResidency this frame. Called internally at the end of RecordResidency (and
        // by EnsureResident, which calls RecordResidency); not part of IResidencyBackend —
        // the core reads the result via FrameStats().overlapRatio only.
        void updateOverlapRatio();

        Engine::Core::Context *m_ctx = nullptr;
        uint32_t m_poolCapacity = 0;

        DirectionalHostStore m_hostStore;
        Eigen::Vector3i m_localBase = Eigen::Vector3i::Zero();

        std::unique_ptr<Engine::Core::Buffer> m_indexGrid;      // uint32[kIndexGridCells]
        std::unique_ptr<Engine::Core::Buffer> m_poolVoxels;     // GpuTsdfVoxel[poolCapacity*512]
        std::unique_ptr<Engine::Core::Buffer> m_metaBuffer;     // ActiveGroupMeta[poolCapacity]
        std::unique_ptr<Engine::Core::Buffer> m_slotListBuffer; // uint32[poolCapacity]
        std::unique_ptr<Engine::Core::Buffer> m_reusableList;   // uint32[poolCapacity]
        std::unique_ptr<Engine::Core::Buffer> m_cleanFreeList;  // uint32[poolCapacity]
        std::unique_ptr<Engine::Core::Buffer> m_writeBackList;  // uint32[poolCapacity]
        std::unique_ptr<Engine::Core::Buffer> m_countsBuffer;   // uint32[3]
        std::unique_ptr<Engine::Core::ComputePipeline> m_registerKernel;
        std::unique_ptr<Engine::Core::ComputePipeline> m_classifyKernel;

        // Persistent host-visible staging (Phase 5): allocated once in Build, reused every
        // frame to back batched copies instead of per-call transient staging.
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageCounts;    // DST, 3*u32
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageLists;     // DST, poolCapacity*u32 (reusable/writeBack)
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageCleanFree; // DST, poolCapacity*u32
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageGroups;    // SRC|DST, kStageGroupCap*groupBytes
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageMeta;      // SRC|DST, poolCapacity*sizeof(ActiveGroupMeta)
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageSlotList;  // SRC, poolCapacity*u32
        std::vector<uint32_t> m_reusableSlots; // reusable pool slots downloaded in BeginFrame

        // CPU mirror of slot occupancy: which key each slot currently holds.
        std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> m_residentIndex;
        std::vector<DirectionalGroupKey> m_slotKeys;
        std::vector<uint32_t> m_freeSlots; // rebuilt from CleanFreeList every BeginFrame
        std::unordered_set<DirectionalGroupKey, DirectionalGroupKeyHash> m_requiredThisFrame;

        ResidencyStats m_stats;

        void fillIndexGridInvalid();
    };

} // namespace TSDF

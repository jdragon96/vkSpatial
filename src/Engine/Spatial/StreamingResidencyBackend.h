#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Compute/StagingBuffer.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/IResidencyBackend.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Engine::Spatial {

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

        // --- Phase 1 extensions (not part of IResidencyBackend) ---
        // DirectionalTSDF::Integrate folds residency recording into the SAME CommandBatch
        // as the point-upload + integrate dispatch (Phase 5 batching optimization that
        // predates this refactor); IResidencyBackend::EnsureResident owns its own batch and
        // submit, so it can't be reused for that combined path. DirectionalTSDF hardcodes
        // this concrete type in Task 3 and calls these directly. Revisit if/when a
        // batching-aware interface method is worth adding (Task 5+).
        //
        // Stages missing groups into persistent staging and records upload copies +
        // combined (reusable + missing) register dispatch into `batch`. Does NOT submit.
        // Returns the number of slots registered.
        uint32_t recordResidency(const std::vector<DirectionalGroupKey> &required,
                                 Engine::Compute::CommandBatch &batch);
        void updateOverlapRatio();

        // gpuSubmits/classify-pass counts aren't part of ResidencyStats (Task 1 interface);
        // DirectionalTSDF reads them here to reconstruct its own Stats/ClassifyCounts.
        uint32_t GpuSubmits() const { return m_gpuSubmits; }
        uint32_t LastReusableCount() const { return m_lastReusable; }
        uint32_t LastCleanFreeCount() const { return m_lastCleanFree; }
        uint32_t LastWriteBackCount() const { return m_lastWriteBack; }

        // CPU mirror of slot occupancy (key -> pool slot) for every group resident this
        // frame. DirectionalTSDF::Integrate walks this to find the pool slots overlapping
        // the recompute mask for extraction; IResidencyBackend has no iteration accessor.
        const std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> &
        ResidentIndex() const { return m_residentIndex; }

    private:
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
        uint32_t m_gpuSubmits = 0;
        uint32_t m_lastReusable = 0;
        uint32_t m_lastCleanFree = 0;
        uint32_t m_lastWriteBack = 0;

        void fillIndexGridInvalid();
    };

} // namespace Engine::Spatial

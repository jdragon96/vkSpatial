#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalHostStore.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"

#include <Eigen/Core>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Engine::Spatial {

    // Directional TSDF with a host/GPU streaming cache
    // (docs/superpowers/specs/2026-07-17-directional-tsdf-design.md).
    //
    // Phase 1 scope: BeginFrame uses full-reload semantics (every slot freed each frame);
    // EnsureResident uploads every requested group from the host store and registers it in
    // the indexGrid. Device-side classification/reuse (Phase 2), integration/extraction
    // (Phase 3) and dirty write-back (Phase 4) come later.
    class DirectionalTSDF {
    public:
        struct Stats {
            uint32_t residentCount = 0;
            uint32_t missingCount = 0;
            uint32_t writeBackCount = 0;
            uint32_t h2dBytes = 0;
            uint32_t d2hBytes = 0;
            float overlapRatio = 0.0f;
        };

        void Build(Engine::Core::Context &ctx,
                   float voxelSize = 0.1f,
                   float truncation = 0.3f,
                   uint32_t poolCapacity = 32768);

        // Starts a frame: recomputes the local base (window centred on the hint, snapped to
        // the group grid) and resets the indexGrid to kInvalidPoolIndex.
        void BeginFrame(const Eigen::Vector3f &aabbCenterHint);

        // Makes the given groups resident: fetches each from the host store (zero-filled on
        // first touch), uploads into free pool slots, and registers them in the indexGrid.
        // Keys already resident this frame are skipped. Throws if a key lies outside the
        // current local window or the pool is exhausted.
        void EnsureResident(const std::vector<DirectionalGroupKey> &required);

        DirectionalHostStore &HostStore() { return m_hostStore; }
        Eigen::Vector3i LocalBase() const { return m_localBase; }
        float VoxelSize() const { return m_voxelSize; }
        float Truncation() const { return m_truncation; }
        float GroupWorldSize() const { return m_voxelSize * float(kGroupDim); }
        uint32_t PoolCapacity() const { return m_poolCapacity; }
        Stats LastFrameStats() const { return m_stats; }

        // Test/debug helpers — synchronous GPU downloads, not for per-frame use.
        std::vector<uint32_t> DebugDownloadIndexGrid();
        uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &key);
        DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &key);

    private:
        Engine::Core::Context *m_ctx = nullptr;
        float m_voxelSize = 0.1f;
        float m_truncation = 0.3f;
        uint32_t m_poolCapacity = 0;

        DirectionalHostStore m_hostStore;
        Eigen::Vector3i m_localBase = Eigen::Vector3i::Zero();

        std::unique_ptr<Engine::Core::Buffer> m_indexGrid;      // uint32[kIndexGridCells]
        std::unique_ptr<Engine::Core::Buffer> m_poolVoxels;     // GpuTsdfVoxel[poolCapacity*512]
        std::unique_ptr<Engine::Core::Buffer> m_metaBuffer;     // ActiveGroupMeta[poolCapacity]
        std::unique_ptr<Engine::Core::Buffer> m_slotListBuffer; // uint32[poolCapacity]
        std::unique_ptr<Engine::Core::ComputePipeline> m_registerKernel;

        // CPU mirror of slot occupancy: which key each slot currently holds.
        std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> m_residentIndex;
        std::vector<DirectionalGroupKey> m_slotKeys;
        uint32_t m_nextFreeSlot = 0;

        Stats m_stats;

        void fillIndexGridInvalid();
        Eigen::Vector3i quantizeLocalBase(const Eigen::Vector3f &center) const;
    };

} // namespace Engine::Spatial

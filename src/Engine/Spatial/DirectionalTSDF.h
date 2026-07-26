#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Compute/StagingBuffer.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalHostStore.h"
#include "Engine/Spatial/DirectionalIntegrationQuality.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"
#include "Engine/Spatial/IResidencyBackend.h"
#include "Engine/Spatial/OrientedPointCloud.h"

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace Engine::Spatial {

    // Directional TSDF with a host/GPU streaming cache
    // (docs/superpowers/specs/2026-07-17-directional-tsdf-design.md).
    //
    // Phase 2 scope: BeginFrame classifies the active pool against the new local base on
    // the GPU (ReusableList / CleanFreeList / WriteBackList), re-registers reusable slots
    // into the indexGrid, and EnsureResident uploads only genuinely missing groups using
    // CleanFreeList slots. Integration/extraction (Phase 3) and dirty write-back (Phase 4)
    // come later; WriteBackList is produced but not yet consumed.
    //
    // Residency (where a group lives / how it becomes device-addressable) is delegated to
    // an IResidencyBackend (see IResidencyBackend.h); this class owns only the TSDF
    // algorithm (integrate/extract/merge) and binds its kernels to the backend's buffers.
    class DirectionalTSDF {
    public:
        struct Stats {
            uint32_t residentCount = 0;
            uint32_t missingCount = 0;
            uint32_t writeBackCount = 0;
            uint32_t h2dBytes = 0;
            uint32_t d2hBytes = 0;
            float overlapRatio = 0.0f;
            // Stage timings (ms). Valid because every GPU submission is synchronous.
            float beginFrameMs = 0.0f;
            float ensureResidentMs = 0.0f;
            float integrateMs = 0.0f;
            float extractMs = 0.0f;
            float mergeMs = 0.0f;
            uint32_t gpuSubmits = 0; // queue submissions this frame (Phase 5 batching metric)
        };

        struct ClassifyCounts {
            uint32_t reusable = 0;
            uint32_t cleanFree = 0;
            uint32_t writeBack = 0;
        };

        void Build(Engine::Core::Context &ctx,
                   float voxelSize = 0.1f,
                   float truncation = 0.3f,
                   uint32_t poolCapacity = 32768,
                   uint32_t maxPoints = 1u << 15,
                   uint32_t maxCandidates = 1u << 16,
                   ResidencyMode residency = ResidencyMode::Streaming);

        // Configures the multi-direction write-set / integration quality (opt-in; default
        // {maxDirections=1, dirExponent=4, viewAngleWeight=false} reproduces the original
        // single-dominant-axis behavior exactly).
        void SetIntegrationQuality(const IntegrationQuality &q) { m_quality = q; }

        // Extraction mode: 0 = legacy axis-crossing average, 1 = FD gradient projection,
        // 2 = stored-gradient (normalize(sumN) normal + isosurface projection). Default 0.
        void SetExtractMode(uint32_t mode) { m_extractMode = mode; }

        // Starts a frame: recomputes the local base (window centred on the hint, snapped to
        // the group grid) and resets the indexGrid to kInvalidPoolIndex.
        void BeginFrame(const Eigen::Vector3f &aabbCenterHint);

        // Makes the given groups resident: fetches each from the host store (zero-filled on
        // first touch), uploads into free pool slots, and registers them in the indexGrid.
        // Keys already resident this frame are skipped. Throws if a key lies outside the
        // current local window or the pool is exhausted.
        void EnsureResident(const std::vector<DirectionalGroupKey> &required);

        // Full frame pipeline (design doc §5): BeginFrame → write-set/halo residency →
        // GPU integration → extraction over the recompute mask → candidate merge →
        // old-point replacement. points/normals must be the same length; at most
        // maxPoints samples are used.
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos,
                       const Eigen::Vector3f &aabbCenterHint);

        const std::vector<ExtractedPoint> &PointCloud() const { return m_pointCloud; }
        void ExportPointCloud(const std::string &path) const; // ASCII PLY with normals

        // Oriented surface point cloud (shared feature currency; e.g. FPFH): position + normal
        // copied from the last extraction. Decouples feature code from the ExtractedPoint layout.
        OrientedPointCloud ExtractOrientedCloud() const {
            OrientedPointCloud c;
            c.points.reserve(m_pointCloud.size());
            c.normals.reserve(m_pointCloud.size());
            for (const ExtractedPoint &p : m_pointCloud) {
                c.points.push_back(p.position);
                c.normals.push_back(p.normal);
            }
            return c;
        }

        DirectionalHostStore &HostStore() { return m_backend->HostStore(); }
        Eigen::Vector3i LocalBase() const { return m_backend->LocalBase(); }
        float VoxelSize() const { return m_voxelSize; }
        float Truncation() const { return m_truncation; }
        float GroupWorldSize() const { return m_voxelSize * float(kGroupDim); }
        uint32_t PoolCapacity() const { return m_backend->PoolCapacity(); }
        Stats LastFrameStats() const { return m_stats; }

        // Test/diagnostic seam: mutable backend handle (tests/diagnostics only), exposing host
        // store, resident index, and lifecycle calls (BeginFrame/EnsureResident/etc). Mirrors the
        // backend's own Debug* methods; not for per-frame use. Non-const (rather than the
        // const-ref alternative) so callers can reach HostStore() without a const_cast —
        // HostStore() itself is non-const on IResidencyBackend.
        IResidencyBackend &DebugBackend() { return *m_backend; }

        // Result of the most recent BeginFrame classification (test/debug).
        ClassifyCounts DebugLastClassifyCounts() const { return m_lastCounts; }

        // Test/debug helpers — synchronous GPU downloads, not for per-frame use.
        std::vector<uint32_t> DebugDownloadIndexGrid() { return m_backend->DebugDownloadIndexGrid(); }
        uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &key) { return m_backend->DebugQueryPoolIndex(key); }
        DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &key) {
            return m_backend->DebugDownloadGroupVoxels(key);
        }

    private:
        Engine::Core::Context *m_ctx = nullptr;
        float m_voxelSize = 0.1f;
        float m_truncation = 0.3f;
        IntegrationQuality m_quality;
        uint32_t m_extractMode = 0;

        std::unique_ptr<IResidencyBackend> m_backend;

        std::unique_ptr<Engine::Core::Buffer> m_pointBuffer;      // PointSample[maxPoints]
        std::unique_ptr<Engine::Core::Buffer> m_candidateBuffer;  // DirectionalCandidate[maxCandidates]
        std::unique_ptr<Engine::Core::Buffer> m_candidateCounter; // uint32
        // Recompute-group slot list bound to the extract kernel. Kept on the core side
        // (rather than reusing the backend's internal register-list buffer) because
        // IResidencyBackend only exposes indexGrid/poolVoxels/meta, not a scratch slot-list
        // buffer; the two lists are populated at different points in the frame and never
        // conflict, so this is a plain size-for-size split of the old shared buffer.
        std::unique_ptr<Engine::Core::Buffer> m_groupSlotListBuffer; // uint32[poolCapacity]
        std::unique_ptr<Engine::Core::ComputePipeline> m_integrateKernel;
        std::unique_ptr<Engine::Core::ComputePipeline> m_extractKernel;
        uint32_t m_maxPoints = 0;
        uint32_t m_maxCandidates = 0;
        std::vector<ExtractedPoint> m_pointCloud;

        // Persistent host-visible staging (Phase 5): allocated once in Build, reused every
        // frame to back batched copies instead of per-call transient staging.
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stagePoints;         // SRC, maxPoints*6*f32
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageCandidates;     // DST, maxCandidates*sizeof(DirectionalCandidate)
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageGroupSlotList;  // SRC, poolCapacity*u32
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageCandidateCount; // DST, sizeof(uint32_t)

        Stats m_stats;
        ClassifyCounts m_lastCounts;

        Eigen::Vector3i quantizeLocalBase(const Eigen::Vector3f &center) const;
        std::vector<ExtractedPoint> mergeCandidates(
                const std::vector<DirectionalCandidate> &candidates) const;
    };

} // namespace Engine::Spatial

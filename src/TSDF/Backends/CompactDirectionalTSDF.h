#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "TSDF/Backends/DirectionalIntegrationQuality.h"
#include "Engine/Core/OrientedPointCloud.h"

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Spatial {

    // Per-(voxel,direction) hash entry -- the compact-directional analogue of SimpleTSDF's
    // TSDFEntry. 24 bytes, keyed on (voxel, direction) so it stores DirectionalTSDF's 6
    // direction layers WITHOUT the 8^3=512-voxel blocks: only occupied (voxel,dir) pairs
    // consume storage. sumNx/sumNy/sumNz accumulate a stored gradient (surface normal) for
    // point-to-plane extraction (was a 4-byte `pad` field prior to Compact Best-TSDF v1).
    // Layout must match DirEntry in the compact_directional_* shaders.
    struct DirEntry {
        uint32_t key;   // packDirKey(voxel, dir); 0xFFFFFFFF = empty
        int32_t sumDW;  // sum(value_i * w_i) * 10000, value = clamp(sdf/trunc, -1, 1)
        uint32_t sumW;  // sum(w_i)          * 10000
        int32_t sumNx;  // sum(n_i * w_i)    * 10000  (stored gradient; was pad)
        int32_t sumNy;
        int32_t sumNz;
    }; // 24 bytes. Layout must match DirEntry in compact_directional_{integrate,extract}.comp.
    static_assert(sizeof(DirEntry) == 24, "DirEntry must be 24 bytes");
    static_assert(offsetof(DirEntry, sumNx) == 12);

    // Per-(voxel,direction) readback for variance-adaptive-resolution experiments (mirrors
    // SimpleTSDF::VoxelStat / DownloadVoxels): world-space voxel centre, the direction layer
    // (0..5, see topK/directional_tsdf_integrate.comp's axis encoding), and the recovered
    // TSDF value/weight. See CompactDirectionalTSDF::DownloadEntries.
    struct CompactEntry {
        Eigen::Vector3f center;
        uint32_t direction;
        float tsdf;
        float weight;
        Eigen::Vector3f normal;   // v1: normalize(sumN); zero if degenerate
    };

    // DirectionalTSDF accuracy at ~SimpleTSDF memory: a per-voxel flat hash keyed by
    // (voxel, direction), integrated/extracted with DirectionalTSDF's directional logic
    // (topK dominant directions, view-angle weight, per-direction zero crossings) but stored
    // like SimpleTSDF (one 16-byte entry per occupied key, ray-march band only). Mirrors
    // SimpleTSDF's class shape (Build/Integrate/ExtractPointCloud/FilledCount/Reset).
    //
    // 32-bit key => a 512^3-voxel window (9 bits/axis, settable ORIGIN via Build's
    // windowMinCorner -- see below). Scenes that don't fit in one 512^3 window need tiling
    // (multiple CompactDirectionalTSDF instances) or the streaming DirectionalTSDF; this GPU
    // reports shaderBufferInt64Atomics=false, so a wider (e.g. 64-bit) key isn't available --
    // atomicCompSwap-based findOrInsert requires a 32-bit-atomic-sized key.
    class CompactDirectionalTSDF {
    public:
        explicit CompactDirectionalTSDF();

        // windowMinCorner: world-space minimum corner of the movable 512^3-voxel hash window.
        // Internally floored to a voxel-space origin (m_originVoxel = floor(windowMinCorner /
        // voxelSize)); voxels with (worldVoxel - m_originVoxel) outside [0,511] per axis fall
        // outside the window and are skipped (integrate) / never found (extract neighbour
        // probe). The default (-25.6,-25.6,-25.6) reproduces the ORIGINAL fixed +256-bias key
        // exactly at the original default voxelSize (0.1): floor(-25.6/0.1) = -256 per axis, an
        // origin-relative pack of a voxel v -> lv = v-(-256) = v+256, identical to the old
        // hard-coded bias. Passing a different windowMinCorner re-centers the window anywhere
        // in space (e.g. a scene far from the world origin) without changing the key format.
        void Build(Engine::Core::Context &ctx,
                   float voxelSize = 0.1f,
                   float truncation = 0.3f,
                   uint32_t hashCapacity = 1u << 20,
                   uint32_t maxPoints = 1u << 15,
                   const Eigen::Vector3f &windowMinCorner = Eigen::Vector3f(-25.6f, -25.6f, -25.6f));

        // Voxel-space origin of the movable window (see Build's windowMinCorner). Exposed for
        // tests/diagnostics.
        Eigen::Vector3i OriginVoxel() const { return m_originVoxel; }

        void SetIntegrationQuality(const IntegrationQuality &q) { m_quality = q; }

        // v1: integrate SDF form. true = point-to-plane (removes grazing bias, default),
        // false = projective ray distance. Flows into integrate PC g_pointToPlane.
        void SetPointToPlane(bool on) { m_pointToPlane = on; }

        // Integrate a normal-carrying point cloud observed from cameraPos. Normals drive both
        // the dominant-direction selection (topK) and the view-angle weight. points and normals
        // are parallel arrays (both clamped to maxPoints).
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero());

        // Dispatch the extract kernel (one thread per hash slot) and download the emitted
        // per-direction surface candidates into an OrientedPointCloud.
        //
        // merge=false (default): RAW candidates, one point per emitted (voxel,direction)
        // crossing -- byte-identical to the pre-merge behavior.
        // merge=true: candidates are clustered on the CPU (ported from
        // DirectionalTSDF::mergeCandidates, see MergeCandidates below) to dedup redundant
        // candidates into averaged points while preserving sharp corners (a hard normal-angle
        // split prevents merging across a corner).
        OrientedPointCloud ExtractPointCloud(uint32_t maxCandidates = 1u << 19,
                                             bool merge = false) const;

        uint32_t FilledCount() const;

        // Downloads the whole hash buffer and unpacks every occupied, sufficiently-observed
        // entry (key != EMPTY_KEY, sumW >= TSDF_SCALE/2 -- same occupancy gate as
        // compact_directional_extract.comp) into a CompactEntry: world-space voxel centre,
        // direction layer, recovered TSDF value, and weight. Used by the Compact-Directional x
        // variance-adaptive-resolution benchmark (per-voxel/per-direction memory accounting).
        std::vector<CompactEntry> DownloadEntries() const;

        // Cluster raw {position,normal} candidates into deduped points (ported from
        // DirectionalTSDF::mergeCandidates): buckets by voxel floor(p/voxelSize); within a
        // bucket, a candidate merges into a cluster only within posThresh (0.6*voxelSize) +
        // cosThresh (30 deg) of its mean, with a hard 60 deg strong-split so corners keep
        // separate points. Public static utility -- reusable on ANY oriented point set (e.g.
        // the variance-adaptive combine's assembled cloud), not just this instance's extraction.
        static OrientedPointCloud MergeCandidates(const std::vector<Eigen::Vector3f> &points,
                                                  const std::vector<Eigen::Vector3f> &normals,
                                                  float voxelSize);

        void Reset();

    private:
        Engine::Core::Context *m_ctx = nullptr;
        float m_voxelSize = 0.1f;
        float m_truncation = 0.3f;
        uint32_t m_hashCapacity = 0;
        uint32_t m_maxPoints = 0;
        Eigen::Vector3i m_originVoxel = Eigen::Vector3i::Constant(-256); // see Build's windowMinCorner
        IntegrationQuality m_quality; // defaults: single dominant direction, no view weight
        bool m_pointToPlane = true; // v1 default: point-to-plane

        std::unique_ptr<Engine::Core::Buffer> m_hashBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_pointBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_normalBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_statBuffer;
        std::unique_ptr<Engine::Core::ComputePipeline> m_kernel;
    };

} // namespace Engine::Spatial

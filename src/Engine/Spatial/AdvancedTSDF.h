#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalIntegrationQuality.h"
#include "Engine/Spatial/OrientedPointCloud.h"

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace Engine::Spatial {

    // Per-(voxel,direction) hash entry for AdvancedTSDF. 24 bytes: distance/weight
    // accumulators + a stored-gradient (observed normal) accumulator. Layout must match
    // DirEntry in advanced_tsdf_{integrate,extract}.vert.glsl exactly.
    struct AdvDirEntry {
        uint32_t key;   // packDirKey(voxel, dir); 0xFFFFFFFF = empty
        int32_t sumDW;  // Σ tsdf · w · 10000
        uint32_t sumW;  // Σ w · 10000
        int32_t sumNx;  // Σ n · w · 10000  (stored gradient; normalized at extraction)
        int32_t sumNy;
        int32_t sumNz;
    };
    static_assert(sizeof(AdvDirEntry) == 24, "AdvDirEntry must be 24 bytes");
    static_assert(offsetof(AdvDirEntry, sumNx) == 12);

    // Per-(voxel,direction) readback: world-space voxel centre + recovered tsdf/weight and the
    // denoised stored-gradient normal (normalize(sumN); zero if degenerate).
    struct AdvancedEntry {
        Eigen::Vector3f center;
        uint32_t direction;
        float tsdf;
        float weight;
        Eigen::Vector3f normal;
    };

    // AdvancedTSDF — the best-of-all directional TSDF distilled from this repo's measurements:
    //   - compact per-(voxel,direction) flat hash: block-DirectionalTSDF accuracy at roughly an
    //     order of magnitude less memory (no 512-voxel block waste),
    //   - point-to-plane integration (near-exact on flat surfaces, better edges — measured),
    //   - stored-gradient mode-3 extraction (denoised normals + legacy sub-voxel zero-crossing
    //     position),
    // implemented over the user-style advanced_tsdf_{integrate,extract}.vert.glsl shaders.
    //
    // 32-bit key => a movable 512^3-voxel window. Unlike CompactDirectionalTSDF's fixed-corner
    // default (only correct at voxelSize=0.1), the default window here is CENTRED on the world
    // origin at ANY voxelSize (originVoxel = -256). Pass an explicit windowMinCorner to
    // re-centre elsewhere. Scenes larger than one 512^3 window need tiling.
    class AdvancedTSDF {
    public:
        AdvancedTSDF() = default;

        void Build(Engine::Core::Context &ctx,
                   float voxelSize = 0.01f,
                   float truncation = 0.03f,
                   uint32_t hashCapacity = 1u << 20,
                   uint32_t maxPoints = 1u << 15,
                   const Eigen::Vector3f &windowMinCorner =
                           Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()));

        // Voxel-space origin of the movable window (exposed for tests/diagnostics).
        Eigen::Vector3i OriginVoxel() const { return m_originVoxel; }

        void SetIntegrationQuality(const IntegrationQuality &q) { m_quality = q; }

        // Integrate SDF form. true (default) = point-to-plane (removes grazing bias, measured
        // best); false = projective ray distance. Flows into the integrate push-constant.
        void SetPointToPlane(bool on) { m_pointToPlane = on; }

        // Integrate a normal-carrying point cloud observed from cameraPos. Normals drive the
        // dominant-direction selection, the view-angle weight, and the stored gradient.
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero());

        // Mode-3 hybrid extraction → oriented point cloud. merge=true clusters/dedups the raw
        // candidates on the CPU (corner-preserving), via MergeCandidates.
        OrientedPointCloud ExtractPointCloud(uint32_t maxCandidates = 1u << 19,
                                             bool merge = false) const;

        uint32_t FilledCount() const;

        // Unpack every occupied, sufficiently-observed entry (world centre, direction, tsdf,
        // weight, stored-gradient normal).
        std::vector<AdvancedEntry> DownloadEntries() const;

        void Reset();

        // Corner-preserving cluster/dedup of raw {position,normal} candidates (shared utility).
        static OrientedPointCloud MergeCandidates(const std::vector<Eigen::Vector3f> &points,
                                                  const std::vector<Eigen::Vector3f> &normals,
                                                  float voxelSize);

    private:
        Engine::Core::Context *m_ctx = nullptr;
        float m_voxelSize = 0.01f;
        float m_truncation = 0.03f;
        uint32_t m_hashCapacity = 0;
        uint32_t m_maxPoints = 0;
        Eigen::Vector3i m_originVoxel = Eigen::Vector3i::Constant(-256); // centred default
        IntegrationQuality m_quality;
        bool m_pointToPlane = true; // measured-best default

        std::unique_ptr<Engine::Core::Buffer> m_hashBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_pointBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_normalBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_statBuffer;
        std::unique_ptr<Engine::Core::ComputePipeline> m_kernel;
    };

} // namespace Engine::Spatial

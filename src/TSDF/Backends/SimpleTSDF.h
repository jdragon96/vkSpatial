#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/OrientedPointCloud.h"

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace TSDF {

    // Fixed-point scale for TSDF values stored in the hash table.
    // sumDW / sumW recovers the float SDF in metres.
    static constexpr int32_t TSDF_FIXED_SCALE = 10000;

    struct TSDFEntry {
        uint32_t key;   // packed 10-bit coord per axis; 0xFFFFFFFF = empty
        int32_t sumDW;  // sum(d_i * w_i) * TSDF_FIXED_SCALE
        uint32_t sumW;  // sum(w_i)       * TSDF_FIXED_SCALE
        uint32_t sumD2; // sum(d_i^2)     * TSDF_FIXED_SCALE -- online per-voxel variance (MrHash)
    };

    // Per-voxel readback for variance-adaptive-resolution experiments (MrHash foundation):
    // world-space centre, recovered TSDF value/weight, and the online variance sigma^2
    // (world-units^2) computed from sumDW/sumW/sumD2 -- see SimpleTSDF::DownloadVoxels.
    struct VoxelStat {
        Eigen::Vector3f center;
        float tsdf;
        float weight;
        float variance;
    };

    class SimpleTSDF {
    public:
        explicit SimpleTSDF();

        // Build the GPU structures.
        // truncation: TSDF truncation band in world units (same as voxelSize unit).
        void Build(Engine::Core::Context &ctx,
                   float voxelSize = 0.1f,
                   float truncation = 0.3f,
                   uint32_t hashCapacity = 1u << 20,
                   uint32_t maxPoints = 1u << 12);

        // Integrate a point cloud observed from cameraPos.
        // cameraPos defaults to the origin (suitable for pre-transformed clouds).
        // Unweighted (w=1 per observation) -- does NOT read normals.
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero());

        // Integrate with per-observation normal view-angle weighting:
        //   w = max(0, dot(normal, -rayDir))
        // (the same view-angle confidence DirectionalTSDF uses). Down-weights grazing/
        // oblique observations, reducing the projective-SDF bias on flat surfaces seen
        // obliquely -- no extra memory over the unweighted path (same TSDFEntry layout).
        // points and normals must be the same length (parallel arrays; both clamped to
        // maxPoints).
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero());

        void Reset();

        // Export voxel centres of all occupied slots as a PLY point cloud.
        void ExportMesh(const std::string &path) const;

        // Extract the zero-level isosurface via Marching Cubes and write a PLY mesh.
        void ExportMC(const std::string &path, uint32_t maxTris = 500000u) const;

        // Oriented surface point cloud (shared feature currency; e.g. FPFH): Marching-Cubes
        // vertices welded to unique positions, with per-vertex area-weighted normals computed
        // from the triangles (the volume stores no normals).
        Engine::Core::OrientedPointCloud ExtractPointCloud(uint32_t maxTris = 500000u) const;

        uint32_t FilledCount() const;

        // Downloads the whole hash table and unpacks every occupied, sufficiently-observed
        // voxel (same MIN_WEIGHT occupancy gate as the Marching Cubes kernel) into a
        // VoxelStat: world-space centre, TSDF value, weight, and online variance. Used by
        // variance-adaptive-resolution validation (no multi-resolution storage yet).
        std::vector<VoxelStat> DownloadVoxels() const;

    private:
        Engine::Core::Context *m_ctx = nullptr;
        float m_voxelSize = 0.1f;
        float m_truncation = 0.3f;
        uint32_t m_hashCapacity = 0;
        uint32_t m_maxPoints = 0;

        std::unique_ptr<Engine::Core::Buffer> m_hashBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_pointBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_normalBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_statBuffer;
        std::unique_ptr<Engine::Core::ComputePipeline> m_kernel;

        // Runs Marching Cubes and returns triangle vertices (3 consecutive per triangle).
        std::vector<Eigen::Vector3f> downloadMCVertices(uint32_t maxTris) const;
    };

} // namespace TSDF

#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace Engine::Spatial {

    // Fixed-point scale for TSDF values stored in the hash table.
    // sumDW / sumW recovers the float SDF in metres.
    static constexpr int32_t TSDF_FIXED_SCALE = 10000;

    struct TSDFEntry {
        uint32_t key;  // packed 10-bit coord per axis; 0xFFFFFFFF = empty
        int32_t sumDW; // sum(d_i * w_i) * TSDF_FIXED_SCALE
        uint32_t sumW; // sum(w_i)       * TSDF_FIXED_SCALE
        uint32_t pad;  // alignment
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
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero());

        void Reset();

        // Export voxel centres of all occupied slots as a PLY point cloud.
        void ExportMesh(const std::string &path) const;

        // Extract the zero-level isosurface via Marching Cubes and write a PLY mesh.
        void ExportMC(const std::string &path, uint32_t maxTris = 500000u) const;

        uint32_t FilledCount() const;

    private:
        Engine::Core::Context *m_ctx = nullptr;
        float m_voxelSize = 0.1f;
        float m_truncation = 0.3f;
        uint32_t m_hashCapacity = 0;
        uint32_t m_maxPoints = 0;

        std::unique_ptr<Engine::Core::Buffer> m_hashBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_pointBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_statBuffer;
        std::unique_ptr<Engine::Core::ComputePipeline> m_kernel;
    };

} // namespace Engine::Spatial

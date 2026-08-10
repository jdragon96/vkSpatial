#pragma once
#include "Engine/Spatial/Extraction/SurfaceMesh.h"
#include "Engine/Spatial/SimpleTSDF.h"
#include <Eigen/Core>
#include <vector>
namespace Engine::Spatial {
    struct MixedVoxel { Eigen::Vector3f center; float tsdf; float weight; float size; uint8_t level; };
    using AdaptiveMesh = Engine::Spatial::Extraction::SurfaceMesh;
    class AdaptiveVoxelGrid {
    public:
        void Build(Engine::Core::Context& ctx, float fineVoxelSize, float truncation,
                   uint32_t hashCapacity = 1u<<20, uint32_t maxPoints = 1u<<17);
        void Integrate(const std::vector<Eigen::Vector3f>& points,
                       const Eigen::Vector3f& cameraPos = Eigen::Vector3f::Zero());
        void Reset();
        void SetVarianceThreshold(float sigma2);      // Task 2
        void SetVariancePercentile(float p);          // Task 2
        void SetMinOccupancy(uint32_t n);             // Task 2
        std::vector<MixedVoxel> DownloadMixedVoxels();// Task 2
        size_t FineCount() const;                     // Task 1: fine voxels
        size_t CoarseCount() const;                   // Task 2
        AdaptiveMesh ExtractMesh();                   // Task 4
    private:
        SimpleTSDF m_fine;
        float m_h = 0.05f, m_trunc = 0.15f;
        float m_threshold = -1.0f;   // <0 => percentile mode
        float m_percentile = 0.5f;
        uint32_t m_minOcc = 4;
        void buildMixed();           // Task 2
        std::vector<MixedVoxel> m_mixed;   // Task 2 cache
        bool m_mixedDirty = true;
        size_t m_fineCount = 0, m_coarseCount = 0;
    };
}

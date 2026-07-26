#include "Engine/Spatial/AdaptiveVoxelGrid.h"

namespace Engine::Spatial {

    void AdaptiveVoxelGrid::Build(Engine::Core::Context &ctx, float fineVoxelSize, float truncation,
                                  uint32_t hashCapacity, uint32_t maxPoints) {
        m_h = fineVoxelSize;
        m_trunc = truncation;
        m_fine.Build(ctx, fineVoxelSize, truncation, hashCapacity, maxPoints);
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::Integrate(const std::vector<Eigen::Vector3f> &points,
                                      const Eigen::Vector3f &cameraPos) {
        m_fine.Integrate(points, cameraPos);
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::Reset() {
        m_fine.Reset();
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::SetVarianceThreshold(float /*sigma2*/) {
        // Task 2
    }

    void AdaptiveVoxelGrid::SetVariancePercentile(float /*p*/) {
        // Task 2
    }

    void AdaptiveVoxelGrid::SetMinOccupancy(uint32_t /*n*/) {
        // Task 2
    }

    std::vector<MixedVoxel> AdaptiveVoxelGrid::DownloadMixedVoxels() {
        // Task 2
        return {};
    }

    size_t AdaptiveVoxelGrid::FineCount() const {
        // Task 1: fine passthrough. Task 2 replaces with the mixed-grid fine count.
        return size_t(m_fine.FilledCount());
    }

    size_t AdaptiveVoxelGrid::CoarseCount() const {
        // Task 2
        return 0;
    }

    AdaptiveMesh AdaptiveVoxelGrid::ExtractMesh() {
        // Task 4
        return {};
    }

    void AdaptiveVoxelGrid::buildMixed() {
        // Task 2
    }

} // namespace Engine::Spatial

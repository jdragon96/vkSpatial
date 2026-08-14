#include "TSDF/Memory/SubmapStrategy.h"

namespace TSDF {

    void SubmapStrategy::Build(Engine::Core::Context &context, const VolumeParams &params) {
        m_context = &context;
        // windowMinCorner is deliberately unused: like TiledAdvancedTSDF, SubmapAdvancedTSDF is
        // tiled internally (both its base and detail levels), so there is no single window corner
        // to place -- each tile derives its own origin from the tile grid.
        m_tsdf.Build(context, params.voxelSize, params.truncation, kBlockVoxels,
                     kDetailPointsPerVoxel, params.hashCapacity, params.maxPointsPerFrame);
    }

    void SubmapStrategy::Reset() {
        if (m_context == nullptr) return;
        m_tsdf.Reset();
    }

    void SubmapStrategy::Configure(const IntegrationOptions &options) {
        // Forward every option, exactly as the sibling strategies do -- SubmapAdvancedTSDF fans
        // each setter out to both its base and detail levels. A strategy that quietly dropped
        // sweep-relevant options would make an A/B run lie: the same parameter set would mean
        // something different for `submap` than for `flat`/`tile`.
        m_tsdf.SetIntegrationQuality(options.quality);
        m_tsdf.SetPointToPlane(options.pointToPlane);
        m_tsdf.SetConfidenceWeight(options.confidenceWeight);
        m_tsdf.SetHermitePosition(options.hermitePosition);
        m_tsdf.SetCurrentFrame(options.currentFrame);
    }

    void SubmapStrategy::Record(const std::vector<Eigen::Vector3f> &points,
                                const std::vector<Eigen::Vector3f> &normals,
                                const Eigen::Vector3f &cameraPosition,
                                Engine::Compute::CommandBatch & /*batch*/) {
        if (m_context == nullptr) return;
        // Deviation documented in the header: this call opens and submits its own batch.
        // IntegrateGPU, not Integrate: Integrate reaches AdvancedTSDF::RecordIntegrate per tile,
        // which CLAMPS to maxPointsPerFrame and silently drops the overflow -- a truncated cloud
        // would under-report occupiedEntryCount and bias the comparison toward submap. IntegrateGPU
        // self-submits exactly as Integrate did, so the disclosed deviation is unchanged; only the
        // clamp is gone. Its cost trade-off is documented in the header.
        m_tsdf.IntegrateGPU(points, normals, cameraPosition);
    }

    void SubmapStrategy::Download(std::vector<TSDF::AdvancedEntry> &out) const {
        if (m_context == nullptr) {
            out.clear();
            return;
        }
        m_tsdf.DownloadEntries(out);
    }

    VolumeStats SubmapStrategy::Stats() const {
        VolumeStats stats;
        if (m_context == nullptr) return stats;
        stats.occupiedEntryCount = m_tsdf.FilledCount();
        stats.slotCapacity = m_tsdf.SlotCapacity();
        stats.deviceMemoryBytes = stats.slotCapacity * kBytesPerHashSlot;
        stats.tableCount = m_tsdf.BaseTileCount() + m_tsdf.DetailTileCount();
        stats.insertFailureCount = m_tsdf.InsertFailureCount();
        stats.growCount = m_tsdf.GrowCount();
        return stats;
    }

} // namespace TSDF

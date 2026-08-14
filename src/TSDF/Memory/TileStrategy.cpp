#include "TSDF/Memory/TileStrategy.h"

namespace TSDF {

    void TileStrategy::Build(Engine::Core::Context &context, const VolumeParams &params) {
        m_context = &context;
        // windowMinCorner is deliberately unused: a tiled layout derives each tile's origin from
        // the tile grid, so there is no single window corner to place.
        m_tsdf.Build(context, params.voxelSize, params.truncation, params.hashCapacity,
                     params.maxPointsPerFrame);
    }

    void TileStrategy::Reset() {
        if (m_context == nullptr) return;
        m_tsdf.Reset();
    }

    // The one method that deliberately carries no `m_context == nullptr` guard: every wrapped
    // setter is a plain field write that a later Build preserves, so Configure-before-Build is a
    // working, useful sequence. A guard would silently discard those settings instead.
    // NOTE: set options BEFORE the first integration -- see the warning in TileStrategy.h.
    void TileStrategy::Configure(const IntegrationOptions &options) {
        m_tsdf.SetIntegrationQuality(options.quality);
        m_tsdf.SetPointToPlane(options.pointToPlane);
        m_tsdf.SetConfidenceWeight(options.confidenceWeight);
        m_tsdf.SetHermitePosition(options.hermitePosition);
        m_tsdf.SetCurrentFrame(options.currentFrame);
    }

    void TileStrategy::Record(const std::vector<Eigen::Vector3f> &points,
                              const std::vector<Eigen::Vector3f> &normals,
                              const Eigen::Vector3f &cameraPosition,
                              Engine::Compute::CommandBatch &batch) {
        if (m_context == nullptr) return;
        // Routes points to tiles and records every tile's dispatch into the one batch -- that
        // fusion is the whole reason Record is the interface primitive. The GPU form, not the
        // batched Integrate: Integrate reaches AdvancedTSDF::RecordIntegrate, which CLAMPS each
        // tile's cloud to maxPointsPerFrame and silently drops the rest. A truncated cloud
        // under-reports occupiedEntryCount in exactly the direction that flatters tiling, so the
        // measurement this class exists for would be biased. RecordIntegrateGPU grows instead.
        m_tsdf.RecordIntegrateGPU(points, normals, cameraPosition, batch);
    }

    void TileStrategy::Download(std::vector<TSDF::AdvancedEntry> &out) const {
        if (m_context == nullptr) {
            out.clear();
            return;
        }
        m_tsdf.DownloadEntries(out);
    }

    VolumeStats TileStrategy::Stats() const {
        VolumeStats stats;
        if (m_context == nullptr) return stats;
        stats.occupiedEntryCount = m_tsdf.FilledCount();
        stats.slotCapacity = m_tsdf.SlotCapacity();
        stats.deviceMemoryBytes = stats.slotCapacity * kBytesPerHashSlot;
        stats.tableCount = m_tsdf.TileCount();
        stats.insertFailureCount = m_tsdf.InsertFailureCount();
        stats.growCount = m_tsdf.GrowCount();
        return stats;
    }

} // namespace TSDF

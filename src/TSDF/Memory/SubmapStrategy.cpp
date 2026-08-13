#include "TSDF/Memory/SubmapStrategy.h"

namespace TSDF {

    void SubmapStrategy::Build(Engine::Core::Context &context, const VolumeParams &params) {
        m_context = &context;
        m_tsdf.Build(context, params.voxelSize, params.truncation, kBlockVoxels,
                     kDetailPointsPerVoxel, params.hashCapacity, params.maxPointsPerFrame);
    }

    void SubmapStrategy::Reset() {
        if (m_context == nullptr) return;
        m_tsdf.Reset();
    }

    void SubmapStrategy::Configure(const IntegrationOptions &options) {
        // Only the options this backend exposes; the interface allows a strategy to ignore the
        // rest rather than fail, so one parameter set can drive every strategy in a sweep.
        m_tsdf.SetIntegrationQuality(options.quality);
        m_tsdf.SetCurrentFrame(options.currentFrame);
    }

    void SubmapStrategy::Record(const std::vector<Eigen::Vector3f> &points,
                                const std::vector<Eigen::Vector3f> &normals,
                                const Eigen::Vector3f &cameraPosition,
                                Engine::Compute::CommandBatch & /*batch*/) {
        if (m_context == nullptr) return;
        // Deviation documented in the header: this call opens and submits its own batch.
        m_tsdf.Integrate(points, normals, cameraPosition);
    }

    void SubmapStrategy::Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const {
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
        return stats;
    }

} // namespace TSDF

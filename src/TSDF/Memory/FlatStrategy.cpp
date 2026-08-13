#include "TSDF/Memory/FlatStrategy.h"

namespace TSDF {

    void FlatStrategy::Build(Engine::Core::Context &context, const VolumeParams &params) {
        m_context = &context;
        m_tsdf.Build(context, params.voxelSize, params.truncation, params.hashCapacity,
                     params.maxPointsPerFrame, params.windowMinCorner);
    }

    void FlatStrategy::Reset() {
        if (m_context == nullptr) return;
        m_tsdf.Reset();
    }

    void FlatStrategy::Configure(const IntegrationOptions &options) {
        m_tsdf.SetIntegrationQuality(options.quality);
        m_tsdf.SetPointToPlane(options.pointToPlane);
        m_tsdf.SetConfidenceWeight(options.confidenceWeight);
        m_tsdf.SetHermitePosition(options.hermitePosition);
        m_tsdf.SetCurrentFrame(options.currentFrame);
    }

    void FlatStrategy::Record(const std::vector<Eigen::Vector3f> &points,
                              const std::vector<Eigen::Vector3f> &normals,
                              const Eigen::Vector3f &cameraPosition,
                              Engine::Compute::CommandBatch &batch) {
        if (m_context == nullptr) return;
        // GPU form: grows the upload buffers to the whole frame instead of clamping, so a
        // comparison run never silently drops points on a large frame.
        m_tsdf.RecordIntegrateGPU(points, normals, cameraPosition, batch);
    }

    void FlatStrategy::Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const {
        if (m_context == nullptr) {
            out.clear();
            return;
        }
        out = m_tsdf.DownloadEntries();
    }

    VolumeStats FlatStrategy::Stats() const {
        VolumeStats stats;
        if (m_context == nullptr) return stats; // FilledCount would dereference an unbuilt buffer
        stats.occupiedEntryCount = m_tsdf.FilledCount();
        stats.slotCapacity = m_tsdf.HashCapacity();
        stats.deviceMemoryBytes = stats.slotCapacity * kBytesPerHashSlot;
        stats.tableCount = 1;
        // insertFailureCount and growCount stay 0 until the hash axis lands in the next plan.
        return stats;
    }

} // namespace TSDF

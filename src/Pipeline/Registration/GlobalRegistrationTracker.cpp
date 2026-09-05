#include "Pipeline/Registration/GlobalRegistrationTracker.h"

namespace Pipeline {

    TrackingResult GlobalRegistrationTracker::Track(const Frame &frame, const ModelSnapshot *model,
                                                    const Eigen::Isometry3f &priorPose) {
        TrackingResult r;
        r.pose = priorPose;
        if (model == nullptr || model->entries.empty() || frame.pts.empty()) return r;

        // The default RegistrationConfig is mm-scale; every derived radius (downsample cell, FPFH,
        // RANSAC inlier threshold, Ceres loss) hangs off voxelSize, so on a metre-scale map the
        // whole pipeline collapses unless it is re-anchored to the map resolution — same reason
        // GpuIcpTracker scales maxCorrDist off model->voxel.
        Registration::RegistrationConfig cfg = m_cfg;
        if (model->voxel > 0.0f) cfg.voxelSize = model->voxel;

        Registration::PointCloud src, tgt;
        src.points = frame.pts;
        src.normals = frame.nrm;
        tgt.points.reserve(model->entries.size());
        tgt.normals.reserve(model->entries.size());
        const float truncationDistance = model->truncationDistance > 0.0f ? model->truncationDistance : 0.0f;
        for (const TSDFVoxel &e: model->entries) {
            // Sub-voxel surface point, not the raw voxel center: a voxel sits up to half a
            // truncation band away from the surface it describes.
            tgt.points.push_back(e.center - e.tsdf * truncationDistance * e.normal);
            tgt.normals.push_back(e.normal);
        }
        const Registration::RegistrationResult reg =
                Registration::Estimate(src, tgt, cfg);
        r.pose = Eigen::Isometry3f(reg.T);
        r.fitness = reg.fitness;
        r.inliers = reg.numInliers;
        r.rmse = reg.rmse;
        r.valid = reg.valid;
        // A failed solve against an EXISTING map must never be classified NoModel: ShouldFuse
        // fuses NoModel frames (there is normally nothing to corrupt), and that would push this
        // solve's garbage pose into the map.
        r.failure = reg.valid ? ETrackFailure::None
                              : (reg.numInliers < 3 ? ETrackFailure::TooFewInliers
                                                    : ETrackFailure::LowOverlap);
        return r;
    }

} // namespace Pipeline

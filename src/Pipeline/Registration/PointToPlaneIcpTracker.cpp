#include "Pipeline/Registration/PointToPlaneIcpTracker.h"

namespace Pipeline {

    TrackingResult PointToPlaneIcpTracker::Track(const Frame &frame,
                                                 const ModelSnapshot *model,
                                                 const Eigen::Isometry3f &priorPose) {
        TrackingResult r;
        r.pose = priorPose;
        if (model == nullptr || model->entries.empty() || frame.pts.empty()) return r;

        const float truncationDistance = model->truncationDistance > 0.0f ? model->truncationDistance : 0.0f;
        Engine::Registration::PointCloud tgt;
        tgt.points.reserve(model->entries.size());
        tgt.normals.reserve(model->entries.size());
        for (const TSDFVoxel &entry: model->entries) {
            const Eigen::Vector3f surfacePoint = entry.center - entry.tsdf * truncationDistance * entry.normal;
            tgt.points.push_back(surfacePoint);
            tgt.normals.push_back(entry.normal);
        }

        Engine::Registration::SortTargetIntoCanonicalOrder(tgt);

        Engine::Registration::RegistrationParam params = m_params;
        if (model->voxel > 0.0f) {
            params.maxCorrDist = 2.0f * model->voxel;
            params.huberScale = model->voxel;
        }
        if (params.maxStepMeters <= 0.0f)
            params.maxStepMeters = Engine::Registration::kDefaultTrackerMaxStepMeters;
        const Engine::Registration::RegistrationResult icp =
                Engine::Registration::AlignPointToPlaneIcp(
                        frame.pts,
                        frame.nrm,
                        tgt,
                        priorPose.matrix(),
                        params);

        // Same physical step gate as GpuIcpTracker -- the two trackers must judge a solve alike.
        const float stepFromPriorMeters =
                (Eigen::Isometry3f(icp.T).translation() - priorPose.translation()).norm();
        if (params.maxStepMeters > 0.0f && stepFromPriorMeters > params.maxStepMeters) {
            r.pose = priorPose;
            r.fitness = icp.fitness;
            r.inliers = icp.numInliers;
            r.rmse = icp.rmse;
            r.valid = false;
            r.failure = ETrackFailure::ImplausibleMotion;
            return r;
        }

        r.pose = Eigen::Isometry3f(icp.T);
        r.fitness = icp.fitness;
        r.inliers = icp.numInliers;
        r.rmse = icp.rmse;
        r.valid = icp.valid;
        if (r.valid) r.failure = ETrackFailure::None;
        return r;
    }

} // namespace Pipeline

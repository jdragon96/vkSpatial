#include "Pipeline/Registration/PointToPlaneIcpTracker.h"

#include <algorithm>
#include <cstddef>

namespace Pipeline {

    TrackingResult PointToPlaneIcpTracker::Track(const Frame &frame,
                                                 const ModelSnapshot *model,
                                                 const Eigen::Isometry3f &priorPose) {
        TrackingResult result;
        result.pose = priorPose;
        if (model == nullptr || model->entries.empty() || frame.pts.empty()) return result;

        const float truncationDistance = model->truncationDistance > 0.0f ? model->truncationDistance : 0.0f;
        Registration::PointCloud target;
        target.points.reserve(model->entries.size());
        target.normals.reserve(model->entries.size());
        for (const TSDFVoxel &entry: model->entries) {
            const Eigen::Vector3f surfacePoint = entry.center - entry.tsdf * truncationDistance * entry.normal;
            target.points.push_back(surfacePoint);
            target.normals.push_back(entry.normal);
        }

        Registration::SortTargetIntoCanonicalOrder(target);

        // Defaults mirror GpuIcpTracker exactly -- the two trackers must judge a solve alike.
        Registration::RegistrationParam params = m_params;
        if (params.minFitness <= 0.0f) params.minFitness = 0.4f;
        if (params.maxStepMeters <= 0.0f)
            params.maxStepMeters = std::max(Registration::kDefaultTrackerMaxStepMeters,
                                            Registration::kDefaultTrackerMaxStepVoxels * model->voxel);
        if (model->voxel > 0.0f) {
            params.maxCorrDist = 2.0f * model->voxel;
            params.huberScale = model->voxel;
        }
        const Registration::RegistrationResult icp =
                Registration::AlignPointToPlaneIcp(
                        frame.pts,
                        frame.nrm,
                        target,
                        priorPose.matrix(),
                        params);

        // Same physical step gate as GpuIcpTracker -- the two trackers must judge a solve alike.
        const float stepFromPriorMeters =
                (Eigen::Isometry3f(icp.T).translation() - priorPose.translation()).norm();
        if (params.maxStepMeters > 0.0f && stepFromPriorMeters > params.maxStepMeters) {
            result.pose = priorPose;
            result.fitness = icp.fitness;
            result.inliers = icp.numInliers;
            result.rmse = icp.rmse;
            result.valid = false;
            result.failure = ETrackFailure::ImplausibleMotion;
            return result;
        }

        result.pose = Eigen::Isometry3f(icp.T);
        result.fitness = icp.fitness;
        result.inliers = icp.numInliers;
        result.rmse = icp.rmse;
        // AlignPointToPlaneIcp itself gates only on minInliers; the fitness share is judged here,
        // with the same classification as GpuIcpTracker. Leaving a failed solve at the default
        // NoModel would make it fusible (ShouldFuse), corrupting the next frame's alignment target.
        result.valid = icp.valid && icp.fitness >= params.minFitness;
        result.failure = result.valid ? ETrackFailure::None
                                      : (icp.numInliers < std::size_t(params.minInliers)
                                                 ? ETrackFailure::TooFewInliers
                                                 : ETrackFailure::LowOverlap);
        return result;
    }

} // namespace Pipeline

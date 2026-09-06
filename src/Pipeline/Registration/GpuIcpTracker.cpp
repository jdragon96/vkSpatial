#include "Pipeline/Registration/GpuIcpTracker.h"
#include "Common/PointCloud.h"
#include "Registration/RegistrationParam.h"
#include "Registration/RegistrationResult.h"

#include <cstddef>

namespace Pipeline {

    TrackingResult GpuIcpTracker::Track(const Frame &frame,
                                        const ModelSnapshot *model,
                                        const Eigen::Isometry3f &priorPose) {
        TrackingResult result;
        result.pose = priorPose;
        result.failure = ETrackFailure::NoModel;
        if (model == nullptr || model->entries.empty() || frame.pts.empty()) return result;
        if (!m_ctx) {
            m_ctx = std::make_unique<Engine::Core::Context>();
            m_gpu = std::make_unique<GpuPointToPlaneIcp>(*m_ctx);
        }

        // 1. Find the frame's world-space bounding box under the prior pose
        Eigen::Vector3f minBound = priorPose * frame.pts[0];
        Eigen::Vector3f maxBound = priorPose * frame.pts[0];
        for (const auto &point: frame.pts) {
            const Eigen::Vector3f worldPoint = priorPose * point;
            minBound = minBound.cwiseMin(worldPoint);
            maxBound = maxBound.cwiseMax(worldPoint);
        }

        // 2. Resolve the solve parameters against the map resolution
        Registration::RegistrationParam params = m_params;
        if (params.minFitness <= 0.0f) params.minFitness = 0.4f;
        if (params.maxStepMeters <= 0.0f) {
            // Default gate scales with the map resolution (see Registration::kDefaultTrackerMaxStepVoxels);
            // an explicit SetMaxStepMeters is absolute and skips this.
            params.maxStepMeters = std::max(Registration::kDefaultTrackerMaxStepMeters,
                                            Registration::kDefaultTrackerMaxStepVoxels * model->voxel);
        }
        if (model->voxel > 0.0f) {
            params.maxCorrDist = 2.0f * model->voxel;
            params.huberScale = model->voxel;
        }

        // 3. Expand the bounding box by the correspondence distance
        const float margin = params.maxCorrDist;
        minBound.array() -= margin;
        maxBound.array() += margin;
        const float truncationDistance = model->truncationDistance > 0.0f ? model->truncationDistance : 0.0f;
        Common::PointCloud target;
        target.points.reserve(model->entries.size());
        target.normals.reserve(model->entries.size());

        // 4. Collect the target surface points
        ForEachEntryInBox(*model, minBound, maxBound, [&](const TSDFVoxel &entry) {
            const Eigen::Vector3f surfacePoint = entry.center - entry.tsdf * truncationDistance * entry.normal;
            target.points.push_back(surfacePoint);
            target.normals.push_back(entry.normal);
        });
        if (target.points.size() < 3) {
            result.failure = ETrackFailure::NoLocalTarget;
            return result;
        }

        // 5. Sort canonical order
        target.SortTargetIntoCanonicalOrder();

        // 6. Do ICP
        const Registration::RegistrationResult icp = m_gpu->Solve(
                frame.pts,
                frame.nrm,
                target,
                priorPose.matrix(),
                params);
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
        result.valid = icp.valid;
        result.failure = icp.valid ? ETrackFailure::None
                                   : (icp.numInliers < std::size_t(params.minInliers)
                                              ? ETrackFailure::TooFewInliers
                                              : ETrackFailure::LowOverlap);
        return result;
    }

} // namespace Pipeline

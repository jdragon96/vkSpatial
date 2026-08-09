#include "Engine/Pipeline/Registration/PointToPlaneIcpTracker.h"

namespace Engine::Pipeline {

    TrackingResult PointToPlaneIcpTracker::Track(const Frame &frame,
                                                 const ModelSnapshot *model,
                                                 const Eigen::Isometry3f &priorPose) {
        TrackingResult r;
        r.pose = priorPose;
        if (model == nullptr || model->entries.empty() || frame.pts.empty()) return r;

        // entry.tsdf is a sub-voxel signed distance (normalized to truncation units) from the quantized
        // voxel center to the true surface; center - tsdf*truncationDistance*normal recovers that
        // sub-voxel surface point instead of handing the tracker the quantized voxel center, which would
        // otherwise leave a truncationDistance/2-scale offset for the solved pose to absorb.
        const float truncationDistance = model->truncationDistance > 0.0f ? model->truncationDistance : 0.0f;
        Engine::Registration::PointCloud tgt;
        tgt.points.reserve(model->entries.size());
        tgt.normals.reserve(model->entries.size());
        for (const Engine::Spatial::AdvancedEntry &entry: model->entries) {
            const Eigen::Vector3f surfacePoint = entry.center - entry.tsdf * truncationDistance * entry.normal;
            tgt.points.push_back(surfacePoint);
            tgt.normals.push_back(entry.normal);
        }

        // minCorrespondenceDistance is deliberately left at its default (0) here, so Tier-3
        // coarse-to-fine annealing (AnnealIcpIteration, RegistrationTypes.h) is DORMANT -- only
        // Tiers 1-2 (sub-voxel target + Huber/normal-rejection robustness) are active in production.
        // To enable annealing, set minCorrespondenceDistance > 0 AND widen maxCorrDist for a large
        // convergence basin (the annealed schedule narrows the per-iteration filter down FROM
        // maxCorrDist, so a tight maxCorrDist leaves nothing to anneal). The cost of doing so: a
        // wider maxCorrDist grows the GPU LocalGrid cell count / memory (nCells = (extent/cell)^3,
        // cell == maxCorrDist) -- exactly why the live gate below is kept at 2*voxel instead.
        Engine::Registration::RegistrationParam params = m_params;
        if (model->voxel > 0.0f) {
            params.maxCorrDist = 2.0f * model->voxel;
            params.huberScale = model->voxel;
        }
        const Engine::Registration::RegistrationResult icp =
                Engine::Registration::AlignPointToPlaneIcp(
                        frame.pts,
                        frame.nrm,
                        tgt,
                        priorPose.matrix(),
                        params);
        r.pose = Eigen::Isometry3f(icp.T);
        r.fitness = icp.fitness;
        r.inliers = icp.numInliers;
        r.rmse = icp.rmse;
        r.valid = icp.valid;
        return r;
    }

} // namespace Engine::Pipeline

#include "Engine/Pipeline/Registration/GpuIcpTracker.h"

namespace Engine::Pipeline {

    TrackingResult GpuIcpTracker::Track(const Frame &frame,
                                        const ModelSnapshot *model,
                                        const Eigen::Isometry3f &priorPose) {
        TrackingResult r;
        r.pose = priorPose;
        if (model == nullptr || model->entries.empty() || frame.pts.empty()) return r;
        if (!m_ctx) {
            m_ctx = std::make_unique<Engine::Core::Context>();
            m_gpu = std::make_unique<GpuPointToPlaneIcp>(*m_ctx); // lazy, on the ICP thread
        }

        // Crop the model to the source AABB + margin so upload/grid stay local. frame.pts is
        // SENSOR-LOCAL but model->entries[].center is WORLD-frame (IntegrationThread integrates
        // tf.pose * tf.frame.pts[i]) -- transform each source point by priorPose before folding
        // it into the AABB, otherwise the crop only overlaps the model when priorPose ~= Identity
        // and silently excludes everything (< 3 survivors -> prior pose returned unchanged) once
        // the camera has actually moved.
        Eigen::Vector3f mn = priorPose * frame.pts[0], mx = priorPose * frame.pts[0];
        for (const auto &p: frame.pts) {
            const Eigen::Vector3f w = priorPose * p;
            mn = mn.cwiseMin(w);
            mx = mx.cwiseMax(w);
        }
        // Scale the correspondence distance to the map voxel: a fixed 0.1m against a 0.5m map is
        // both too tight to find correspondences AND makes the dense LocalGrid cell (= maxCorrDist)
        // 5x finer than the voxel, exploding nCells = (extent/cell)^3 (a 190m scene at 0.1m ~=
        // 1.6e9 cells -> a multi-GB bucketStart per frame). 2x voxel keeps the grid ~voxel-res.
        //
        // minCorrespondenceDistance is deliberately left at its default (0) here, so Tier-3
        // coarse-to-fine annealing (AnnealIcpIteration, RegistrationTypes.h) is DORMANT -- only
        // Tiers 1-2 (sub-voxel target + Huber/normal-rejection robustness) are active in production.
        // To enable annealing, set minCorrespondenceDistance > 0 AND widen maxCorrDist for a large
        // convergence basin (the annealed schedule narrows the per-iteration filter down FROM
        // maxCorrDist, so a tight maxCorrDist leaves nothing to anneal). The cost of doing so: a
        // wider maxCorrDist grows the GPU LocalGrid cell count / memory (see the nCells blow-up
        // above) -- exactly why the live gate above is kept at 2*voxel instead.
        Engine::Registration::RegistrationParam params = m_params;
        if (model->voxel > 0.0f) {
            params.maxCorrDist = 2.0f * model->voxel;
            params.huberScale = model->voxel;
        }
        const float m = params.maxCorrDist;
        mn.array() -= m;
        mx.array() += m;
        // entry.tsdf is a sub-voxel signed distance (normalized to truncation units) from the quantized
        // voxel center to the true surface; center - tsdf*truncationDistance*normal recovers that
        // sub-voxel surface point instead of handing the tracker the quantized voxel center, which would
        // otherwise leave a truncationDistance/2-scale offset for the solved pose to absorb.
        const float truncationDistance = model->truncationDistance > 0.0f ? model->truncationDistance : 0.0f;
        Engine::Registration::PointCloud tgt;
        tgt.points.reserve(model->entries.size());
        tgt.normals.reserve(model->entries.size());
        for (const Engine::Spatial::AdvancedEntry &entry: model->entries)
            if ((entry.center.array() >= mn.array()).all() && (entry.center.array() <= mx.array()).all()) {
                const Eigen::Vector3f surfacePoint = entry.center - entry.tsdf * truncationDistance * entry.normal;
                tgt.points.push_back(surfacePoint);
                tgt.normals.push_back(entry.normal);
            }
        if (tgt.points.size() < 3) return r; // nothing local to align to -> keep prior

        const Engine::Registration::RegistrationResult icp =
                m_gpu->Solve(frame.pts, frame.nrm, tgt, priorPose.matrix(), params);
        r.pose = Eigen::Isometry3f(icp.T);
        r.fitness = icp.fitness;
        r.inliers = icp.numInliers;
        r.rmse = icp.rmse;
        r.valid = icp.valid;
        return r;
    }

} // namespace Engine::Pipeline

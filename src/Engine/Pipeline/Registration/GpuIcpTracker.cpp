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
        Engine::Registration::RegistrationParam params = m_params;
        if (model->voxel > 0.0f) params.maxCorrDist = 2.0f * model->voxel;
        const float m = params.maxCorrDist;
        mn.array() -= m;
        mx.array() += m;
        Engine::Registration::PointCloud tgt;
        tgt.points.reserve(model->entries.size());
        tgt.normals.reserve(model->entries.size());
        for (const Engine::Spatial::AdvancedEntry &e: model->entries)
            if ((e.center.array() >= mn.array()).all() && (e.center.array() <= mx.array()).all()) {
                tgt.points.push_back(e.center);
                tgt.normals.push_back(e.normal);
            }
        if (tgt.points.size() < 3) return r; // nothing local to align to -> keep prior

        const Engine::Registration::RegistrationResult icp =
                m_gpu->Solve(frame.pts, tgt, priorPose.matrix(), params);
        r.pose = Eigen::Isometry3f(icp.T);
        r.fitness = icp.fitness;
        r.inliers = icp.numInliers;
        r.valid = icp.valid;
        return r;
    }

} // namespace Engine::Pipeline

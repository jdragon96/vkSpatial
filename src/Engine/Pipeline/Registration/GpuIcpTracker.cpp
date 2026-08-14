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

        Eigen::Vector3f mn = priorPose * frame.pts[0], mx = priorPose * frame.pts[0];
        for (const auto &p: frame.pts) {
            const Eigen::Vector3f w = priorPose * p;
            mn = mn.cwiseMin(w);
            mx = mx.cwiseMax(w);
        }

        Engine::Registration::RegistrationParam params = m_params;
        if (model->voxel > 0.0f) {
            params.maxCorrDist = 2.0f * model->voxel;
            params.huberScale = model->voxel;
        }

        const float m = params.maxCorrDist;
        mn.array() -= m;
        mx.array() += m;
        const float truncationDistance = model->truncationDistance > 0.0f ? model->truncationDistance : 0.0f;
        Engine::Registration::PointCloud tgt;
        tgt.points.reserve(model->entries.size());
        tgt.normals.reserve(model->entries.size());
        for (const TSDF::AdvancedEntry &entry: model->entries)
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

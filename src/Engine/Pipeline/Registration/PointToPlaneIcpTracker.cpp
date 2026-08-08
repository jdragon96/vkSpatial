#include "Engine/Pipeline/Registration/PointToPlaneIcpTracker.h"

namespace Engine::Pipeline {

    TrackingResult PointToPlaneIcpTracker::Track(const Frame &frame,
                                                 const ModelSnapshot *model,
                                                 const Eigen::Isometry3f &priorPose) {
        TrackingResult r;
        r.pose = priorPose;
        if (model == nullptr || model->entries.empty() || frame.pts.empty()) return r;

        Engine::Registration::PointCloud tgt;
        tgt.points.reserve(model->entries.size());
        tgt.normals.reserve(model->entries.size());
        for (const Engine::Spatial::AdvancedEntry &e: model->entries) {
            tgt.points.push_back(e.center);
            tgt.normals.push_back(e.normal);
        }

        Engine::Registration::RegistrationParam params = m_params;
        if (model->voxel > 0.0f) params.maxCorrDist = 2.0f * model->voxel;
        const Engine::Registration::RegistrationResult icp =
                Engine::Registration::AlignPointToPlaneIcp(
                        frame.pts,
                        tgt,
                        priorPose.matrix(),
                        params);
        r.pose = Eigen::Isometry3f(icp.T);
        r.fitness = icp.fitness;
        r.inliers = icp.numInliers;
        r.valid = icp.valid;
        return r;
    }

} // namespace Engine::Pipeline

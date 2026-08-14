#include "Engine/Pipeline/Registration/GlobalRegistrationTracker.h"

namespace Engine::Pipeline {

    TrackingResult GlobalRegistrationTracker::Track(const Frame &frame, const ModelSnapshot *model,
                                                     const Eigen::Isometry3f &priorPose) {
        TrackingResult r;
        r.pose = priorPose;
        if (model == nullptr || model->entries.empty() || frame.pts.empty()) return r;

        Engine::Registration::PointCloud src, tgt;
        src.points = frame.pts;
        src.normals = frame.nrm;
        tgt.points.reserve(model->entries.size());
        tgt.normals.reserve(model->entries.size());
        for (const TSDF::AdvancedEntry &e: model->entries) {
            tgt.points.push_back(e.center);
            tgt.normals.push_back(e.normal);
        }
        const Engine::Registration::RegistrationResult reg =
                Engine::Registration::Estimate(src, tgt, m_cfg);
        r.pose = Eigen::Isometry3f(reg.T);
        r.fitness = reg.fitness;
        r.inliers = reg.numInliers;
        r.rmse = reg.rmse;
        r.valid = reg.valid;
        return r;
    }

} // namespace Engine::Pipeline

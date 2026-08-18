#include "Pipeline/Registration/GpuIcpTracker.h"

#include <cstddef>

namespace Pipeline {

    TrackingResult GpuIcpTracker::Track(const Frame &frame,
                                        const ModelSnapshot *model,
                                        const Eigen::Isometry3f &priorPose) {
        TrackingResult r;
        r.pose = priorPose;
        r.failure = ETrackFailure::NoModel;
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
        // A depth frame sweeping a growing map legitimately contains new territory, so demanding
        // most of it match would reject good tracks. This value separates "aligned against real
        // overlap" from "latched onto a handful of stray correspondences", which is what the bare
        // minInliers floor lets through. Swept on a 477-frame D435 capture (map voxel 0.05):
        // 0.2 -> 166k voxels, 0.4 -> 50k, 0.6 -> 67k, against a 49k identity reference.
        if (params.minFitness <= 0.0f) params.minFitness = 0.4f;
        if (params.maxStepMeters <= 0.0f)
            params.maxStepMeters = Engine::Registration::kDefaultTrackerMaxStepMeters;
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
        for (const TSDFVoxel &entry: model->entries)
            if ((entry.center.array() >= mn.array()).all() && (entry.center.array() <= mx.array()).all()) {
                const Eigen::Vector3f surfacePoint = entry.center - entry.tsdf * truncationDistance * entry.normal;
                tgt.points.push_back(surfacePoint);
                tgt.normals.push_back(entry.normal);
            }
        if (tgt.points.size() < 3) {
            r.failure = ETrackFailure::NoLocalTarget;
            return r; // nothing local to align to -> keep prior
        }
        SortTargetIntoCanonicalOrder(tgt);

        const Engine::Registration::RegistrationResult icp =
                m_gpu->Solve(frame.pts, frame.nrm, tgt, priorPose.matrix(), params);

        // Physical step gate, checked BEFORE the fitness verdict: a solve that moved further from
        // the prior than a hand-held camera can in one frame is a mis-convergence regardless of how
        // many correspondences endorse it (see RegistrationParam::maxStepMeters -- the measured
        // failure passed the fitness gate at 0.571). The prior is handed back untouched.
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
        r.failure = icp.valid ? ETrackFailure::None
                              : (icp.numInliers < std::size_t(params.minInliers)
                                         ? ETrackFailure::TooFewInliers
                                         : ETrackFailure::LowOverlap);
        return r;
    }

} // namespace Pipeline

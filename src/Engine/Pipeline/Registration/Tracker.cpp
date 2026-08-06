#include "Engine/Pipeline/Registration/Tracker.h"

#include "Engine/Core/Context.h"
#include "Engine/Pipeline/Registration/GpuIcp.h"
#include "Engine/Registration/GlobalRegistration.h"
#include "Engine/Registration/Icp.h"
#include "Engine/Registration/RegistrationTypes.h"
#include "Engine/Spatial/AdvancedTSDF.h" // Engine::Spatial::AdvancedEntry

#include <memory>

namespace Engine::Pipeline {

    namespace {

        // Frames are already world-registered (e.g. object_scan_viewer output): pose = identity. Also
        // the bootstrap command when no model exists yet.
        class IdentityTracker : public Tracker {
        public:
            const char *Name() const override { return "identity"; }
            TrackingResult Track(const Frame &, const ModelSnapshot *,
                                 const Eigen::Isometry3f &) override {
                TrackingResult r;
                r.pose = Eigen::Isometry3f::Identity();
                r.fitness = 1.0f;
                r.valid = true;
                return r;
            }
        };

        // Local point-to-plane ICP against the latest model's occupied voxels (centres + normals).
        class PointToPlaneIcpTracker : public Tracker {
        public:
            explicit PointToPlaneIcpTracker(Engine::Registration::RegistrationParam params = {})
                : m_params(params) {}
            const char *Name() const override { return "icp-cpu"; }

            TrackingResult Track(const Frame &frame,
                                 const ModelSnapshot *model,
                                 const Eigen::Isometry3f &priorPose) override {
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
                const Engine::Registration::RegistrationResult icp =
                        Engine::Registration::AlignPointToPlaneIcp(frame.pts, tgt, priorPose.matrix(),
                                                                   m_params);
                r.pose = Eigen::Isometry3f(icp.T);
                r.fitness = icp.fitness;
                r.inliers = icp.numInliers;
                r.valid = icp.valid;
                return r;
            }

        private:
            Engine::Registration::RegistrationParam m_params;
        };

        // GPU point-to-plane ICP against the latest model's occupied voxels, cropped to the source
        // frame's AABB + maxCorrDist margin so the upload + LocalGrid stay local (not O(full model)).
        // The Context + GpuPointToPlaneIcp are created lazily on the FIRST Track() call, which runs on
        // the ICP (RegistrationThread) thread -- mirrors how IntegrationThread creates its own Context
        // inside its own Run(). Result: two live GPU contexts at runtime (this one + Integration's).
        class GpuIcpTracker : public Tracker {
        public:
            const char *Name() const override { return "icp"; }

            TrackingResult Track(const Frame &frame,
                                 const ModelSnapshot *model,
                                 const Eigen::Isometry3f &priorPose) override {
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
                const float m = m_params.maxCorrDist;
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
                        m_gpu->Solve(frame.pts, tgt, priorPose.matrix(), m_params);
                r.pose = Eigen::Isometry3f(icp.T);
                r.fitness = icp.fitness;
                r.inliers = icp.numInliers;
                r.valid = icp.valid;
                return r;
            }

        private:
            Engine::Registration::RegistrationParam m_params;
            std::unique_ptr<Engine::Core::Context> m_ctx;
            std::unique_ptr<GpuPointToPlaneIcp> m_gpu;
        };

        // Prior-free global registration (FPFH + RANSAC + Ceres) — (re)localisation / A/B baseline.
        class GlobalRegistrationTracker : public Tracker {
        public:
            explicit GlobalRegistrationTracker(Engine::Registration::RegistrationConfig cfg = {})
                : m_cfg(cfg) {}
            const char *Name() const override { return "global"; }

            TrackingResult Track(const Frame &frame, const ModelSnapshot *model,
                                 const Eigen::Isometry3f &priorPose) override {
                TrackingResult r;
                r.pose = priorPose;
                if (model == nullptr || model->entries.empty() || frame.pts.empty()) return r;

                Engine::Registration::PointCloud src, tgt;
                src.points = frame.pts;
                src.normals = frame.nrm;
                tgt.points.reserve(model->entries.size());
                tgt.normals.reserve(model->entries.size());
                for (const Engine::Spatial::AdvancedEntry &e: model->entries) {
                    tgt.points.push_back(e.center);
                    tgt.normals.push_back(e.normal);
                }
                const Engine::Registration::RegistrationResult reg =
                        Engine::Registration::Estimate(src, tgt, m_cfg);
                r.pose = Eigen::Isometry3f(reg.T);
                r.fitness = reg.fitness;
                r.inliers = reg.numInliers;
                r.valid = reg.valid;
                return r;
            }

        private:
            Engine::Registration::RegistrationConfig m_cfg;
        };

    } // namespace

    TrackerRegistry TrackerRegistry::Default() {
        TrackerRegistry reg;
        reg.Register("identity", [] { return std::make_unique<IdentityTracker>(); });
        reg.Register("icp", [] { return std::make_unique<GpuIcpTracker>(); });
        reg.Register("icp-cpu", [] { return std::make_unique<PointToPlaneIcpTracker>(); });
        reg.Register("global", [] { return std::make_unique<GlobalRegistrationTracker>(); });
        return reg;
    }

} // namespace Engine::Pipeline

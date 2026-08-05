#include "Engine/Pipeline/ICP/Alignment.h"

#include "Engine/Registration/GlobalRegistration.h"
#include "Engine/Registration/Icp.h"
#include "Engine/Registration/RegistrationTypes.h"
#include "Engine/Spatial/AdvancedTSDF.h" // Engine::Spatial::AdvancedEntry

#include <memory>

namespace Engine::Pipeline {

    namespace {

        // Frames are already world-registered (e.g. object_scan_viewer output): pose = identity. Also
        // the bootstrap command when no model exists yet.
        class IdentityAlignment : public AlignmentCommand {
        public:
            const char *Name() const override { return "identity"; }
            AlignmentResult Execute(const Frame &, const ModelSnapshot *,
                                    const Eigen::Isometry3f &) override {
                AlignmentResult r;
                r.pose = Eigen::Isometry3f::Identity();
                r.fitness = 1.0f;
                r.valid = true;
                return r;
            }
        };

        // Local point-to-plane ICP against the latest model's occupied voxels (centres + normals).
        class PointToPlaneIcpAlignment : public AlignmentCommand {
        public:
            explicit PointToPlaneIcpAlignment(Engine::Registration::IcpParams params = {})
                : m_params(params) {}
            const char *Name() const override { return "icp"; }

            AlignmentResult Execute(const Frame &frame, const ModelSnapshot *model,
                                    const Eigen::Isometry3f &priorPose) override {
                AlignmentResult r;
                r.pose = priorPose; // fall back to the prior when there is nothing to align to yet
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
            Engine::Registration::IcpParams m_params;
        };

        // Prior-free global registration (FPFH + RANSAC + Ceres) — (re)localisation / A/B baseline.
        class GlobalRegistrationAlignment : public AlignmentCommand {
        public:
            explicit GlobalRegistrationAlignment(Engine::Registration::RegistrationConfig cfg = {})
                : m_cfg(cfg) {}
            const char *Name() const override { return "global"; }

            AlignmentResult Execute(const Frame &frame, const ModelSnapshot *model,
                                    const Eigen::Isometry3f &priorPose) override {
                AlignmentResult r;
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

    AlignmentRegistry AlignmentRegistry::Default() {
        AlignmentRegistry reg;
        reg.Register("identity", [] { return std::make_unique<IdentityAlignment>(); });
        reg.Register("icp", [] { return std::make_unique<PointToPlaneIcpAlignment>(); });
        reg.Register("global", [] { return std::make_unique<GlobalRegistrationAlignment>(); });
        return reg;
    }

} // namespace Engine::Pipeline

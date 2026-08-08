#pragma once

#include "PipelineTypes.h"

#include "Engine/Pipeline/Registration/GlobalRegistration.h"
#include "Engine/Pipeline/Registration/PointToPlaneIcp.h"
#include "Engine/Features/RegistrationTypes.h"
#include "Engine/Spatial/AdvancedTSDF.h" // AdvancedEntry

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Command Pattern for frame alignment (pose estimation): each registration algorithm is an
// AlignmentCommand with a uniform Execute, registered by name in AlignmentRegistry for runtime A/B
// selection. The Track stage holds one command.
namespace pipeline {

    struct AlignmentResult {
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity(); // sensor -> world
        float fitness = 0.0f;
        std::size_t inliers = 0;
        bool valid = false;
    };

    class AlignmentCommand {
    public:
        virtual ~AlignmentCommand() = default;
        virtual const char *Name() const = 0;
        // Align `frame` to `model` (null before the first map), seeded by `priorPose`. Pure CPU.
        virtual AlignmentResult Execute(const Frame &frame, const ModelSnapshot *model,
                                        const Eigen::Isometry3f &priorPose) = 0;
    };

    // Frames are already world-registered (e.g. object_scan_viewer output): pose = identity. Also the
    // bootstrap command when no model exists yet.
    class IdentityAlignment : public AlignmentCommand {
    public:
        const char *Name() const override { return "identity"; }
        AlignmentResult Execute(const Frame &, const ModelSnapshot *, const Eigen::Isometry3f &) override {
            AlignmentResult r;
            r.pose = Eigen::Isometry3f::Identity();
            r.fitness = 1.0f;
            r.valid = true;
            return r;
        }
    };

    // Real local point-to-plane ICP against the latest model's occupied voxels (centres + normals).
    class PointToPlaneIcpAlignment : public AlignmentCommand {
    public:
        explicit PointToPlaneIcpAlignment(Engine::Registration::RegistrationParam params = {})
            : m_params(params) {}
        const char *Name() const override { return "icp"; }

        AlignmentResult Execute(const Frame &frame, const ModelSnapshot *model,
                                const Eigen::Isometry3f &priorPose) override {
            AlignmentResult r;
            r.pose = priorPose; // fall back to the prior when there is nothing to align against yet
            if (model == nullptr || model->entries.empty() || frame.pts.empty()) return r;

            Engine::Registration::PointCloud tgt;
            tgt.points.reserve(model->entries.size());
            tgt.normals.reserve(model->entries.size());
            for (const Engine::Spatial::AdvancedEntry &e : model->entries) {
                tgt.points.push_back(e.center);
                tgt.normals.push_back(e.normal);
            }
            const Engine::Registration::RegistrationResult icp = Engine::Registration::AlignPointToPlaneIcp(
                    frame.pts, tgt, priorPose.matrix(), m_params);
            r.pose = Eigen::Isometry3f(icp.T);
            r.fitness = icp.fitness;
            r.inliers = icp.numInliers;
            r.valid = icp.valid;
            return r;
        }

    private:
        Engine::Registration::RegistrationParam m_params;
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
            for (const Engine::Spatial::AdvancedEntry &e : model->entries) {
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

    // name -> factory. Lets the app/debugger pick and swap algorithms at runtime.
    class AlignmentRegistry {
    public:
        using Factory = std::function<std::unique_ptr<AlignmentCommand>()>;

        void Register(const std::string &name, Factory factory) {
            m_factories[name] = std::move(factory);
        }
        std::unique_ptr<AlignmentCommand> Create(const std::string &name) const {
            const auto it = m_factories.find(name);
            return it == m_factories.end() ? nullptr : it->second();
        }
        bool Has(const std::string &name) const { return m_factories.count(name) != 0; }

        // A registry pre-populated with the built-in commands.
        static AlignmentRegistry Default() {
            AlignmentRegistry reg;
            reg.Register("identity", [] { return std::make_unique<IdentityAlignment>(); });
            reg.Register("icp", [] { return std::make_unique<PointToPlaneIcpAlignment>(); });
            reg.Register("global", [] { return std::make_unique<GlobalRegistrationAlignment>(); });
            return reg;
        }

    private:
        std::unordered_map<std::string, Factory> m_factories;
    };

} // namespace pipeline

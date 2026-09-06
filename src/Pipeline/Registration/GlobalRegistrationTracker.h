#pragma once

#include "Registration/Frontend/GlobalRegistrationPipeline.h"
#include "Registration/RegistrationTypes.h"
#include "Pipeline/Registration/Tracker.h"
#include "Registration/RegistrationConfig.h"
#include "TSDF/Backends/TSDFBackend.h" // TSDFVoxel

namespace Pipeline {

    // Prior-free global registration (FPFH + RANSAC + Ceres) — (re)localisation / A/B baseline.
    class GlobalRegistrationTracker : public Tracker {
    public:
        explicit GlobalRegistrationTracker(Registration::RegistrationConfig cfg = {})
            : m_cfg(cfg) {}
        const char *Name() const override { return "global"; }

        TrackingResult Track(const Frame &frame, const ModelSnapshot *model,
                             const Eigen::Isometry3f &priorPose) override;

    private:
        Registration::RegistrationConfig m_cfg;
    };

} // namespace Pipeline

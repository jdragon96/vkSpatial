#pragma once

#include "Engine/Pipeline/Registration/Tracker.h"
#include "Engine/Pipeline/Registration/GlobalRegistration.h"
#include "Engine/Features/RegistrationTypes.h"
#include "Engine/Spatial/AdvancedTSDF.h" // Engine::Spatial::AdvancedEntry

namespace Engine::Pipeline {

    // Prior-free global registration (FPFH + RANSAC + Ceres) — (re)localisation / A/B baseline.
    class GlobalRegistrationTracker : public Tracker {
    public:
        explicit GlobalRegistrationTracker(Engine::Registration::RegistrationConfig cfg = {})
            : m_cfg(cfg) {}
        const char *Name() const override { return "global"; }

        TrackingResult Track(const Frame &frame, const ModelSnapshot *model,
                             const Eigen::Isometry3f &priorPose) override;

    private:
        Engine::Registration::RegistrationConfig m_cfg;
    };

} // namespace Engine::Pipeline

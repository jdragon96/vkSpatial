#pragma once

#include "Engine/Core/Context.h"
#include "Pipeline/Registration/GpuPointToPlaneIcp.h"
#include "Pipeline/Registration/RegistrationTypes.h"
#include "Pipeline/Registration/Tracker.h"
#include "TSDF/Backends/TSDFBackend.h" // TSDFVoxel

#include <memory>

namespace Pipeline {

    class GpuIcpTracker : public Tracker {
    public:
        void SetMinFitness(float f) { m_params.minFitness = f; }
        // Physical single-step bound; 0 restores the tracker default (kDefaultTrackerMaxStepMeters).
        void SetMaxStepMeters(float metres) { m_params.maxStepMeters = metres; }

        const char *Name() const override { return "icp"; }

        TrackingResult Track(const Frame &frame,
                             const ModelSnapshot *model,
                             const Eigen::Isometry3f &priorPose) override;

    private:
        Engine::Registration::RegistrationParam m_params;
        std::unique_ptr<Engine::Core::Context> m_ctx;
        std::unique_ptr<GpuPointToPlaneIcp> m_gpu;
    };

} // namespace Pipeline

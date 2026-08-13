#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Pipeline/Registration/GpuPointToPlaneIcp.h"
#include "Engine/Pipeline/Registration/RegistrationTypes.h"
#include "Engine/Pipeline/Registration/Tracker.h"
#include "Engine/Spatial/AdvancedTSDF.h" // Engine::Spatial::AdvancedEntry

#include <memory>

namespace Engine::Pipeline {

    class GpuIcpTracker : public Tracker {
    public:
        const char *Name() const override { return "icp"; }

        TrackingResult Track(const Frame &frame,
                             const ModelSnapshot *model,
                             const Eigen::Isometry3f &priorPose) override;

    private:
        Engine::Registration::RegistrationParam m_params;
        std::unique_ptr<Engine::Core::Context> m_ctx;
        std::unique_ptr<GpuPointToPlaneIcp> m_gpu;
    };

} // namespace Engine::Pipeline

#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Pipeline/Registration/GpuPointToPlaneIcp.h"
#include "Engine/Pipeline/Registration/RegistrationTypes.h"
#include "Engine/Pipeline/Registration/Tracker.h"
#include "Engine/Spatial/AdvancedTSDF.h" // Engine::Spatial::AdvancedEntry

#include <memory>

namespace Engine::Pipeline {

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
                             const Eigen::Isometry3f &priorPose) override;

    private:
        Engine::Registration::RegistrationParam m_params;
        std::unique_ptr<Engine::Core::Context> m_ctx;
        std::unique_ptr<GpuPointToPlaneIcp> m_gpu;
    };

} // namespace Engine::Pipeline

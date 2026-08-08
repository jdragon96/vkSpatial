#pragma once

#include "Engine/Pipeline/Registration/Tracker.h"
#include "Engine/Pipeline/Registration/PointToPlaneIcp.h"
#include "Engine/Spatial/AdvancedTSDF.h" // Engine::Spatial::AdvancedEntry

namespace Engine::Pipeline {

    // Local point-to-plane ICP against the latest model's occupied voxels (centres + normals).
    class PointToPlaneIcpTracker : public Tracker {
    public:
        explicit PointToPlaneIcpTracker(Engine::Registration::RegistrationParam params = {})
            : m_params(params) {}
        const char *Name() const override { return "icp-cpu"; }

        TrackingResult Track(const Frame &frame,
                             const ModelSnapshot *model,
                             const Eigen::Isometry3f &priorPose) override;

    private:
        Engine::Registration::RegistrationParam m_params;
    };

} // namespace Engine::Pipeline

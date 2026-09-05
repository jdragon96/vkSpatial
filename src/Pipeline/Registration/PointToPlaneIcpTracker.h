#pragma once

#include "LocalRegistration/Algorithm/PointToPlaneIcp.h"
#include "Features/RegistrationTypes.h"
#include "Pipeline/Registration/Tracker.h"
#include "TSDF/Backends/TSDFBackend.h" // TSDFVoxel

namespace Pipeline {

    // Local point-to-plane ICP against the latest model's occupied voxels (centres + normals).
    class PointToPlaneIcpTracker : public Tracker {
    public:
        explicit PointToPlaneIcpTracker(Registration::RegistrationParam params = {})
            : m_params(params) {}
        const char *Name() const override { return "icp-cpu"; }

        TrackingResult Track(const Frame &frame,
                             const ModelSnapshot *model,
                             const Eigen::Isometry3f &priorPose) override;

    private:
        Registration::RegistrationParam m_params;
    };

} // namespace Pipeline

#pragma once

#include "Registration/Frontend/PointToPlaneIcp.h"
#include "Registration/RegistrationTypes.h"
#include "Pipeline/Registration/Tracker.h"
#include "Registration/RegistrationParam.h"
#include "TSDF/Backends/TSDFBackend.h" // TSDFVoxel

namespace Pipeline {

    // Local point-to-plane ICP against the latest model's occupied voxels (centres + normals).
    class PointToPlaneIcpTracker : public Tracker {
    public:
        explicit PointToPlaneIcpTracker(Registration::RegistrationParam params = {})
            : m_params(params) {}
        const char *Name() const override { return "icp-cpu"; }

        // Same "0 keeps the default" rule as GpuIcpTracker, so the two solvers stay comparable
        // under one set of flags.
        void Configure(const Registration::RegistrationParam &params) override {
            if (params.maxCorrDist > 0.0f) m_params.maxCorrDist = params.maxCorrDist;
            if (params.minFitness > 0.0f) m_params.minFitness = params.minFitness;
            if (params.maxStepMeters > 0.0f) m_params.maxStepMeters = params.maxStepMeters;
            if (params.minInliers > 0) m_params.minInliers = params.minInliers;
        }

        TrackingResult Track(const Frame &frame,
                             const ModelSnapshot *model,
                             const Eigen::Isometry3f &priorPose) override;

    private:
        Registration::RegistrationParam m_params;
    };

} // namespace Pipeline

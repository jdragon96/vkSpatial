#pragma once

#include "Engine/Core/Context.h"
#include "Registration/Frontend/GpuPointToPlaneIcp.h"
#include "Registration/RegistrationTypes.h"
#include "Pipeline/Registration/Tracker.h"
#include "Registration/RegistrationParam.h"
#include "TSDF/Backends/TSDFBackend.h" // TSDFVoxel

#include <memory>

namespace Pipeline {

    class GpuIcpTracker : public Tracker {
    public:
        void SetMinFitness(float f) { m_params.minFitness = f; }
        // Physical single-step bound; 0 restores the tracker default (Registration::kDefaultTrackerMaxStepMeters).
        void SetMaxStepMeters(float metres) { m_params.maxStepMeters = metres; }
        // 0 leaves Track's own voxel-derived value in place. Widening this grows the GPU LocalGrid
        // cell count cubically, so a viewer exposing it should keep it near a few map voxels.
        void SetMaxCorrespondenceDistance(float metres) { m_params.maxCorrDist = metres; }
        void SetMinInliers(int count) { m_params.minInliers = count; }
        const char *Name() const override { return "icp"; }

        TrackingResult Track(const Frame &frame,
                             const ModelSnapshot *model,
                             const Eigen::Isometry3f &priorPose) override;

    private:
        Registration::RegistrationParam m_params;
        std::unique_ptr<Engine::Core::Context> m_ctx;
        std::unique_ptr<GpuPointToPlaneIcp> m_gpu;
    };

} // namespace Pipeline

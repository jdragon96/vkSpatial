#pragma once

namespace Registration {

    // Local (ICP) registration's knobs. The gates are set by the CALLER -- Pipeline's trackers
    // scale maxCorrDist and maxStepMeters to the map voxel -- not by the solver.
    struct RegistrationParam {
        int maxIters = 20;
        float maxCorrDist = 0.1f;
        int minInliers = 10;
        float minFitness = 0.0f;
        // stop when the incremental update norm drops below this
        float convEps = 1e-6f;
        float huberScale = 0.05f; // robust-weight knee (world units; caller sets ~voxel)
        float normalCompatibilityCosine = 0.5f;
        float maxStepMeters = 0.0f;
        float minCorrespondenceDistance = 0.0f;
    };

    inline constexpr float kDefaultTrackerMaxStepMeters = 0.08f;

    inline constexpr float kDefaultTrackerMaxStepVoxels = 1.6f;

} // namespace Registration

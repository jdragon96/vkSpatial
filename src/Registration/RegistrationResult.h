#pragma once

#include <Eigen/Core>

#include <cstddef>

namespace Registration {

    // What any registration -- global or local, CPU or GPU -- answers with.
    //
    // The backend's pairwise answer is Registration::Backend::PairwiseRegistrationResult -- a
    // different struct, deliberately no longer sharing this name.
    struct RegistrationResult {
        bool valid = false;
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        std::size_t numInliers = 0;
        float fitness = 0.0f;
        float rmse = 0.0f; // sqrt(mean squared point-to-plane residual) over inliers, final iteration; 0 if none
    };

} // namespace Registration

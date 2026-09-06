#pragma once

// The ICP annealing internals. This module's other vocabulary now sits one type per file beside
// this one (RegistrationConfig.h / RegistrationParam.h / RegistrationResult.h); PointCloud, the one
// structure every library here speaks, moved to src/Common.

#include "Registration/RegistrationParam.h" // AnnealIcpIteration reads one

#include <Eigen/Core>

#include <cmath>

namespace Registration {

    using Fpfh33 = Eigen::Matrix<float, 33, 1>;

    struct Correspondence {
        int srcIdx, tgtIdx;
    };

    struct AnnealedIcpIterationParams {
        float maxCorrespondenceDistance;
        float huberScale;
    };

    inline AnnealedIcpIterationParams AnnealIcpIteration(const RegistrationParam &params, int iter) {
        if (params.minCorrespondenceDistance <= 0.0f || params.minCorrespondenceDistance >= params.maxCorrDist)
            return {params.maxCorrDist, params.huberScale};
        const float ratio = params.maxIters > 1 ? float(iter) / float(params.maxIters - 1) : 0.0f;
        const float currentMaxCorrespondenceDistance =
                params.maxCorrDist *
                std::pow(params.minCorrespondenceDistance / params.maxCorrDist, ratio);
        const float currentHuberScale =
                params.huberScale * (currentMaxCorrespondenceDistance / params.maxCorrDist);
        return {currentMaxCorrespondenceDistance, currentHuberScale};
    }

} // namespace Registration

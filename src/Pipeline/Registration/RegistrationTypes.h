#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <vector>

namespace Engine::Registration {

    // Pipeline configuration. mm-scale; caller overrides voxelSize per data.
    //
    // NOTE: C++ default member initializers cannot reference another member, so only
    // *gains* are stored here as plain constant defaults. Physical values are derived
    // in the pipeline stages from voxelSize, e.g.:
    //   normalRadius    = normalRadiusGain * voxelSize
    //   fpfhRadius      = fpfhRadiusGain   * voxelSize
    //   ransacInlierThr = ransacInlierGain * voxelSize
    //   ceresLossScale  = ceresLossGain    * voxelSize
    struct RegistrationConfig {
        float voxelSize = 5.0f;
        float normalRadiusGain = 3.0f;
        float fpfhRadiusGain = 5.0f;
        int numMaxCorr = 5000;
        float ransacInlierGain = 2.0f;
        int ransacIters = 5000;
        float ceresLossGain = 1.0f;
    };

    struct RegistrationResult {
        bool valid = false;
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        size_t numInliers = 0;
        float fitness = 0.0f;
        float rmse = 0.0f; // sqrt(mean squared point-to-plane residual) over inliers, final iteration; 0 if none
    };

    struct RegistrationParam {
        int maxIters = 20;
        // correspondence gate (world units); set to the data scale. Doubles as the COARSEST
        // (widest) distance when coarse-to-fine annealing is on (see minCorrespondenceDistance) --
        // it is always the value the NN grid is built/cell-sized at (grid hoist: built ONCE per
        // Solve, never per iteration), so the widest value must be this field, never narrower.
        float maxCorrDist = 0.1f;
        int minInliers = 10;
        // stop when the incremental update norm drops below this
        float convEps = 1e-6f;
        float huberScale = 0.05f;               // robust-weight knee (world units; caller sets ~voxel)
        float normalCompatibilityCosine = 0.5f; // reject correspondence if sourceN·targetN < this (~60deg)
        // Coarse-to-fine correspondence-distance annealing. 0 (default) => OFF: every iteration uses
        // the fixed maxCorrDist above, unchanged pre-Task-5 behaviour. When > 0 (and < maxCorrDist),
        // the per-iteration DISTANCE FILTER shrinks geometrically from maxCorrDist (widest, iter 0 --
        // large convergence basin) down to this value (narrowest, final iteration -- sub-voxel
        // precision) via AnnealIcpIteration() below. The NN grid itself is still built ONCE, at the
        // widest (maxCorrDist) cell -- annealing narrows only the query radius, never rebuilds the
        // grid, so the existing per-solve hoist stays intact on both GPU and CPU.
        float minCorrespondenceDistance = 0.0f;
    };

    // Per-iteration coarse-to-fine schedule, shared verbatim by GpuPointToPlaneIcp::Solve (GPU) and
    // AlignPointToPlaneIcp (CPU) so both trackers anneal identically. `iter` in [0, params.maxIters).
    // huberScale is annealed by the SAME ratio as the distance gate, so the robust-weight knee
    // tightens in step with the correspondence filter instead of staying fixed while the gate narrows
    // around it.
    struct AnnealedIcpIterationParams {
        float maxCorrespondenceDistance;
        float huberScale;
    };

    inline AnnealedIcpIterationParams AnnealIcpIteration(const RegistrationParam &params, int iter) {
        if (params.minCorrespondenceDistance <= 0.0f || params.minCorrespondenceDistance >= params.maxCorrDist)
            return {params.maxCorrDist, params.huberScale}; // annealing off -> fixed gate (unchanged behaviour)
        // ratio in [0,1]: 0 at iter 0 (widest), 1 at the final iteration (narrowest). maxIters<=1 has
        // no "across iterations" to anneal over -- stay at the wide (bootstrap) end.
        const float ratio = params.maxIters > 1 ? float(iter) / float(params.maxIters - 1) : 0.0f;
        const float currentMaxCorrespondenceDistance =
                params.maxCorrDist *
                std::pow(params.minCorrespondenceDistance / params.maxCorrDist, ratio);
        const float currentHuberScale =
                params.huberScale * (currentMaxCorrespondenceDistance / params.maxCorrDist);
        return {currentMaxCorrespondenceDistance, currentHuberScale};
    }

    using Fpfh33 = Eigen::Matrix<float, 33, 1>;

    struct Correspondence {
        int srcIdx, tgtIdx;
    };

    // normals empty ⇒ absent.
    struct PointCloud {
        std::vector<Eigen::Vector3f> points, normals;
    };

} // namespace Engine::Registration

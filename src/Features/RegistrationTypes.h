#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
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
        // Minimum share of the SOURCE points that must find a correspondence for the solve to
        // count. minInliers alone is an absolute floor, and an absolute floor cannot judge a
        // solve: ten correspondences out of fifty thousand determine six degrees of freedom about
        // as well as noise does, yet they pass. A frame that has drifted off the map finds exactly
        // that handful, "solves" them, and reports a pose the caller then trusts.
        // 0 disables the gate (the historical behaviour).
        float minFitness = 0.0f;
        // stop when the incremental update norm drops below this
        float convEps = 1e-6f;
        float huberScale = 0.05f;               // robust-weight knee (world units; caller sets ~voxel)
        float normalCompatibilityCosine = 0.5f; // reject correspondence if sourceN·targetN < this (~60deg)
        // Physical single-step bound (world units): reject a solve whose translation from the prior
        // exceeds this. 0 (default) = off, the historical behaviour. The fitness gate cannot catch
        // every mis-convergence -- on the 477-frame capture/ recording a diverged solve claimed a
        // 0.225 m single-frame step at fitness 0.571, sailed past the 0.4 fitness gate, and its
        // fused data corrupted the map for every frame after. A 30 fps hand-held camera moves well
        // under 0.05 m per frame, so a larger claimed step is wrong by physics no matter how many
        // correspondences endorse it. Trackers default this to kDefaultTrackerMaxStepMeters below;
        // a relocalizing caller that legitimately expects large jumps sets it high or 0.
        float maxStepMeters = 0.0f;
        // Coarse-to-fine correspondence-distance annealing. 0 (default) => OFF: every iteration uses
        // the fixed maxCorrDist above, unchanged pre-Task-5 behaviour. When > 0 (and < maxCorrDist),
        // the per-iteration DISTANCE FILTER shrinks geometrically from maxCorrDist (widest, iter 0 --
        // large convergence basin) down to this value (narrowest, final iteration -- sub-voxel
        // precision) via AnnealIcpIteration() below. The NN grid itself is still built ONCE, at the
        // widest (maxCorrDist) cell -- annealing narrows only the query radius, never rebuilds the
        // grid, so the existing per-solve hoist stays intact on both GPU and CPU.
        float minCorrespondenceDistance = 0.0f;
    };

    // The trackers' default for RegistrationParam::maxStepMeters. Not 0.05 (the physical bound the
    // stats comments cite) but above it, so the gate only fires on solves that are clearly wrong --
    // a brisk hand-held jerk at 30 fps stays under this; the measured mis-convergences it exists to
    // stop claimed 0.09-0.23 m.
    inline constexpr float kDefaultTrackerMaxStepMeters = 0.08f;

    // The same tuned gate expressed in map voxels (0.08 m at the capture/ voxel of 0.05 m). The
    // absolute constant only means "hand-held camera, metres"; on maps whose units make the voxel
    // comparable to 0.08 (scanData ~mm: voxel 5.7; scan_out: voxel 0.5) solve jitter alone exceeds
    // it and the gate rejects nearly every frame. Trackers therefore default the gate to
    // max(kDefaultTrackerMaxStepMeters, kDefaultTrackerMaxStepVoxels * voxel) — bit-identical on
    // capture/-scale maps, proportionate on coarser ones. An explicit caller-set maxStepMeters is
    // always absolute.
    inline constexpr float kDefaultTrackerMaxStepVoxels = 1.6f;

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

    struct PointCloud {
        std::vector<Eigen::Vector3f> points, normals;
    };

    // Canonical(좌표 기준) Order: X -> Y -> Z 순서로 비교하여 정렬
    inline void SortTargetIntoCanonicalOrder(PointCloud &target) {
        const std::size_t n = target.points.size();
        if (target.normals.size() != n) return;

        struct PointWithNormal {
            Eigen::Vector3f point, normal;
        };
        std::vector<PointWithNormal> records(n);
        for (std::size_t i = 0; i < n; ++i) records[i] = {target.points[i], target.normals[i]};
        std::sort(records.begin(), records.end(), [](const PointWithNormal &a, const PointWithNormal &b) {
            if (a.point.x() != b.point.x()) return a.point.x() < b.point.x();
            if (a.point.y() != b.point.y()) return a.point.y() < b.point.y();
            return a.point.z() < b.point.z();
        });
        for (std::size_t i = 0; i < n; ++i) {
            target.points[i] = records[i].point;
            target.normals[i] = records[i].normal;
        }
    }

} // namespace Engine::Registration

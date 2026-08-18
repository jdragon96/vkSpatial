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

    // Put a target cloud into a canonical (lexicographic) order, so an ICP solve depends on the SET
    // of target points and not on the order they arrived in.
    //
    // The order really does vary: a target built from ModelSnapshot::entries inherits the TSDF
    // compaction kernel's output order, and that kernel appends via `atomicAdd(g_count, 1u)` -- i.e.
    // thread-completion order, different on every run. Two things then leak that order into the
    // result. Solve() sums the target centroid to shift into the centred frame, and float addition is
    // not associative, so a 100k-point centroid moves in its last bits; every residual is then
    // quantised through `int(round(x * SCALE))`, so a last-bit shift flips a share of the fixed-point
    // contributions. And LocalGrid::Nearest breaks exact distance ties by whichever candidate it
    // visited first. Neither is large per frame, but each frame's pose seeds the next frame's map,
    // which is the next frame's alignment target -- so it compounds. Sorting removes both.
    inline void SortTargetIntoCanonicalOrder(PointCloud &target) {
        const std::size_t n = target.points.size();
        if (target.normals.size() != n) return; // malformed; Solve rejects it anyway
        std::vector<std::size_t> order(n);
        for (std::size_t i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&target](std::size_t a, std::size_t b) {
            const Eigen::Vector3f &p = target.points[a], &q = target.points[b];
            if (p.x() != q.x()) return p.x() < q.x();
            if (p.y() != q.y()) return p.y() < q.y();
            return p.z() < q.z();
        });
        std::vector<Eigen::Vector3f> sortedPoints(n), sortedNormals(n);
        for (std::size_t i = 0; i < n; ++i) {
            sortedPoints[i] = target.points[order[i]];
            sortedNormals[i] = target.normals[order[i]];
        }
        target.points.swap(sortedPoints);
        target.normals.swap(sortedNormals);
    }

} // namespace Engine::Registration

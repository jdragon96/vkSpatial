#pragma once

#include "Engine/Features/RegistrationTypes.h"

#include <vector>

namespace Engine::Registration {

    // Closed-form rigid (rotation + translation, NO scaling) alignment that maps
    // src[i] -> dst[i] in a least-squares sense, via Eigen's umeyama() with
    // with_scaling=false. `src` and `dst` must be the same (non-zero) size.
    //
    // Returns Identity if src/dst are empty, size-mismatched, or contain fewer than 3
    // points (3 non-collinear points are the minimum for a well-posed rigid fit).
    Eigen::Matrix4f SolveRigidUmeyama(const std::vector<Eigen::Vector3f> &src, const std::vector<Eigen::Vector3f> &dst);

    // Full coarse global-registration pipeline, src -> tgt:
    //   1. DownsampleVoxel both clouds at cfg.voxelSize.
    //   2. ComputeFpfh on both (normalRadius = normalRadiusGain*voxelSize,
    //      fpfhRadius = fpfhRadiusGain*voxelSize).
    //   3. MatchFeatures (src->tgt descriptor-space nearest neighbour, ratio test).
    //   4. RANSAC over cfg.ransacIters: sample 3 correspondences, SolveRigidUmeyama on
    //      those 3, count inliers over ALL correspondences with threshold
    //      ransacInlierGain*voxelSize, keep the max-inlier transform. Final
    //      SolveRigidUmeyama refit on that transform's inlier set.
    //
    // result.valid iff numInliers >= 3 and fitness (numInliers / numCorrespondences)
    // exceeds a minimum-fitness floor. NO Ceres refinement here (that's a later stage
    // layered on top of this coarse estimate).
    RegistrationResult EstimateRansac(const PointCloud &src, const PointCloud &tgt, const RegistrationConfig &cfg);

    // Full pipeline: EstimateRansac() for a coarse T, then a Ceres robust SE(3) refine over
    // the coarse estimate's inlier correspondences:
    //   - Parameters: quaternion q[4] (ceres::QuaternionManifold) + translation t[3], seeded
    //     from the coarse T's rotation/translation.
    //   - One 3-vector residual per inlier correspondence: r = R(q)*p_src_i + t - p_tgt_i,
    //     via ceres::AutoDiffCostFunction, wrapped in
    //     ceres::CauchyLoss(cfg.ceresLossGain*cfg.voxelSize) for M-estimation robustness to
    //     any residual outliers RANSAC's inlier threshold didn't fully screen out.
    //   - Solved with ceres::Solver (dense QR, SILENT logging).
    //
    // If the coarse estimate is !valid, returns it unchanged (does not refine garbage).
    // Otherwise composes the refined T and recounts inliers/fitness against the same
    // correspondence set the coarse pipeline matched.
    RegistrationResult Estimate(const PointCloud &src, const PointCloud &tgt, const RegistrationConfig &cfg);

} // namespace Engine::Registration

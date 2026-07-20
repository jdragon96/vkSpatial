#pragma once

// Geometric loop-closure detection = point-cloud registration.
//
// Given two oriented surface point clouds that may observe the same region
// (e.g. a scanner passing over the same teeth twice), estimate the rigid
// transform that aligns them and decide whether they really overlap. The
// output relative pose + confidence becomes a loop-closure edge in the pose
// graph (see PoseGraph.h).
//
// Pipeline (all CPU, reusing the frontend's FPFH + neighbour query):
//   1. FPFH descriptor per point on each cloud.
//   2. Feature-space nearest-neighbour correspondences (source -> target).
//   3. RANSAC over 3-point samples (Umeyama minimal solve) with an edge-length
//      pre-check; keep the transform with the most spatial inliers.
//   4. Point-to-point ICP refinement over the inliers.
//   5. Accept only if inlier support passes a threshold.
//
// Intraoral scanning has a narrow FOV and repetitive teeth, so visual bag-of-words
// place recognition is weak; this geometric matcher is the recommended detector.

#include "Engine/Backend/Lie.h"
#include "Engine/Spatial/OrientedPointCloud.h"

namespace Engine::Backend {

    struct RegistrationConfig {
        float fpfhRadius = 0.25f;     // FPFH neighbourhood (~2.5x voxel); match to data scale
        float inlierThreshold = 0.05f; // world units: max residual for a spatial inlier
        int ransacIterations = 4000;
        float edgeLengthTol = 0.9f;   // triangle edge-length consistency for a RANSAC sample
        int icpIterations = 25;       // point-to-point ICP refinement steps
        int minInliers = 25;          // reject the loop below this many inliers
        double minFitness = 0.3;      // reject below inliers / min(|src|,|tgt|)
        unsigned seed = 1;            // deterministic RANSAC
    };

    struct RegistrationResult {
        // Rigid transform mapping SOURCE points into the TARGET frame:
        //   p_target ~= T_source_to_target * p_source.
        SE3 T_source_to_target = SE3::Identity();
        int inliers = 0;
        double fitness = 0.0;    // inliers / min(|src|,|tgt|)
        double inlierRmse = 0.0; // RMS residual over inliers (world units)
        bool success = false;    // passed inlier/fitness thresholds
    };

    // Registers source onto target. Clouds must carry unit normals (FPFH needs them).
    RegistrationResult RegisterPointClouds(const Engine::Spatial::OrientedPointCloud &source,
                                           const Engine::Spatial::OrientedPointCloud &target,
                                           const RegistrationConfig &cfg = RegistrationConfig{});

} // namespace Engine::Backend

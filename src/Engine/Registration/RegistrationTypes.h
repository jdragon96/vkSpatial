#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

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
    };

    using Fpfh33 = Eigen::Matrix<float, 33, 1>;

    struct Correspondence {
        int srcIdx, tgtIdx;
    };

    // normals empty ⇒ absent.
    struct PointCloud {
        std::vector<Eigen::Vector3f> points, normals;
    };

} // namespace Engine::Registration

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
        float rmse = 0.0f; // sqrt(mean squared point-to-plane residual) over inliers, final iteration; 0 if none
    };

    struct RegistrationParam {
        int maxIters = 20;
        // correspondence gate (world units); set to the data scale
        float maxCorrDist = 0.1f;
        int minInliers = 10;
        // stop when the incremental update norm drops below this
        float convEps = 1e-6f;
        float huberScale = 0.05f;               // robust-weight knee (world units; caller sets ~voxel)
        float normalCompatibilityCosine = 0.5f; // reject correspondence if sourceN·targetN < this (~60deg)
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

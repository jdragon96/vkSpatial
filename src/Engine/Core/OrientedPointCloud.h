#pragma once

// Common currency for surface features (FPFH, future descriptors): a set of surface points
// with matching unit normals. Both SimpleTSDF and DirectionalTSDF produce one of these, and
// feature modules consume only this type — so they stay decoupled from any TSDF variant.

#include <Eigen/Core>
#include <cstddef>
#include <vector>

namespace Engine::Spatial {

    struct OrientedPointCloud {
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> normals; // parallel to points, expected unit length

        size_t size() const { return points.size(); }
        bool empty() const { return points.empty(); }
    };

} // namespace Engine::Spatial

#pragma once
#include <Eigen/Core>
#include <vector>
namespace Engine::Spatial::Extraction {
    struct SurfaceMesh {
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> triangles;   // vertex indices
        std::vector<Eigen::Vector3f> normals;      // per-vertex
    };
}

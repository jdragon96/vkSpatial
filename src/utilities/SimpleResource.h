#pragma once

#include <Eigen/Dense>
#include <vector>

struct Primitives {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;
    std::vector<uint32_t> indices;
};

class SimpleResource {
public:
    static Primitives CreateCube(float size = 1.0f);
};
#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace vkMath {

    using Vec3 = Eigen::Vector3f;
    using Mat4 = Eigen::Matrix4f;
    using Quat = Eigen::Quaternionf;

    Mat4 Translation(float x, float y, float z);
    Mat4 RotationX(float radians);
    Mat4 RotationY(float radians);

    // Vulkan clip space (Y-down, depth range [0, 1]) right-handed perspective projection.
    Mat4 Perspective(float fovYRadians, float aspect, float nearPlane, float farPlane);

    // Vulkan clip space (Y-down, depth range [0, 1]) right-handed orthographic projection.
    Mat4 Orthographic(float left, float right, float bottom, float top,
                      float nearPlane, float farPlane);

    Mat4 LookAt(const Vec3 &eye, const Vec3 &target, const Vec3 &up);

    // Maps a 2D cursor position within [0,width]x[0,height] onto a unit arcball/hemisphere.
    Vec3 MapToArcball(double cursorX, double cursorY, int width, int height);

} // namespace vkMath

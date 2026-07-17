#include "utilities/Math.h"

#include <cmath>

namespace vkMath {

    Mat4 Translation(float x, float y, float z) {
        Mat4 out = Mat4::Identity();
        out(0, 3) = x;
        out(1, 3) = y;
        out(2, 3) = z;
        return out;
    }

    Mat4 RotationX(float radians) {
        Mat4 out = Mat4::Identity();
        const float c = std::cos(radians);
        const float s = std::sin(radians);
        out(1, 1) = c;
        out(2, 1) = s;
        out(1, 2) = -s;
        out(2, 2) = c;
        return out;
    }

    Mat4 RotationY(float radians) {
        Mat4 out = Mat4::Identity();
        const float c = std::cos(radians);
        const float s = std::sin(radians);
        out(0, 0) = c;
        out(2, 0) = -s;
        out(0, 2) = s;
        out(2, 2) = c;
        return out;
    }

    Mat4 Perspective(float fovYRadians, float aspect, float nearPlane, float farPlane) {
        Mat4 out = Mat4::Zero();
        const float f = 1.0f / std::tan(fovYRadians * 0.5f);
        out(0, 0) = f / aspect;
        out(1, 1) = -f;
        out(2, 2) = farPlane / (nearPlane - farPlane);
        out(3, 2) = -1.0f;
        out(2, 3) = (farPlane * nearPlane) / (nearPlane - farPlane);
        return out;
    }

    Mat4 Orthographic(float left, float right, float bottom, float top,
                      float nearPlane, float farPlane) {
        Mat4 out = Mat4::Identity();
        out(0, 0) = 2.0f / (right - left);
        out(1, 1) = 2.0f / (top - bottom);
        out(2, 2) = 1.0f / (nearPlane - farPlane);
        out(0, 3) = -(right + left) / (right - left);
        out(1, 3) = -(top + bottom) / (top - bottom);
        out(2, 3) = nearPlane / (nearPlane - farPlane);
        return out;
    }

    Mat4 LookAt(const Vec3 &eye, const Vec3 &target, const Vec3 &up) {
        const Vec3 forward = (target - eye).normalized();
        const Vec3 crossFU = forward.cross(up);
        const Vec3 right = crossFU.norm() < 1e-6f
                                    ? Vec3(1.0f, 0.0f, 0.0f)
                                    : crossFU.normalized();
        const Vec3 realUp = right.cross(forward);

        Mat4 out = Mat4::Identity();
        out(0, 0) = right.x();
        out(0, 1) = right.y();
        out(0, 2) = right.z();
        out(0, 3) = -right.dot(eye);

        out(1, 0) = realUp.x();
        out(1, 1) = realUp.y();
        out(1, 2) = realUp.z();
        out(1, 3) = -realUp.dot(eye);

        out(2, 0) = -forward.x();
        out(2, 1) = -forward.y();
        out(2, 2) = -forward.z();
        out(2, 3) = forward.dot(eye);

        return out;
    }

    Vec3 MapToArcball(double cursorX, double cursorY, int width, int height) {
        const float x = static_cast<float>((2.0 * cursorX - width) / width);
        const float y = static_cast<float>((height - 2.0 * cursorY) / height);
        const float len2 = x * x + y * y;
        if (len2 <= 1.0f)
            return Vec3(x, y, std::sqrt(1.0f - len2));

        const float invLen = 1.0f / std::sqrt(len2);
        return Vec3(x * invLen, y * invLen, 0.0f);
    }

} // namespace vkMath

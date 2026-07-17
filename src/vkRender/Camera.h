#pragma once

#include <cmath>
#include <memory>

namespace vkRender {

    struct Vec3 {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };

    struct Mat4 {
        float m[16] = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f};
    };

    class Camera {
    public:
        using UniquePtr = std::unique_ptr<Camera>;

        enum class Projection {
            Perspective,
            Orthographic,
        };

        void SetPerspective(float fovYRadians,
                            float aspect,
                            float nearPlane,
                            float farPlane) {
            m_projectionType = Projection::Perspective;
            m_nearPlane = nearPlane;
            m_farPlane = farPlane;

            const float f = 1.0f / std::tan(fovYRadians * 0.5f);
            m_projection = {};
            m_projection.m[0] = f / aspect;
            m_projection.m[5] = -f;
            m_projection.m[10] = farPlane / (nearPlane - farPlane);
            m_projection.m[11] = -1.0f;
            m_projection.m[14] = (farPlane * nearPlane) / (nearPlane - farPlane);
            m_projection.m[15] = 0.0f;
        }

        void SetOrthographic(float left,
                             float right,
                             float bottom,
                             float top,
                             float nearPlane,
                             float farPlane) {
            m_projectionType = Projection::Orthographic;
            m_nearPlane = nearPlane;
            m_farPlane = farPlane;

            m_projection = {};
            m_projection.m[0] = 2.0f / (right - left);
            m_projection.m[5] = 2.0f / (bottom - top);
            m_projection.m[10] = 1.0f / (nearPlane - farPlane);
            m_projection.m[12] = -(right + left) / (right - left);
            m_projection.m[13] = -(bottom + top) / (bottom - top);
            m_projection.m[14] = nearPlane / (nearPlane - farPlane);
            m_projection.m[15] = 1.0f;
        }

        void LookAt(Vec3 eye, Vec3 target, Vec3 up = {0.0f, 1.0f, 0.0f}) {
            m_eye = eye;
            m_target = target;
            m_up = up;
            RecomputeView();
        }

        Projection GetProjectionType() const { return m_projectionType; }
        float GetNearPlane() const { return m_nearPlane; }
        float GetFarPlane() const { return m_farPlane; }
        Vec3 GetEye() const { return m_eye; }
        Vec3 GetTarget() const { return m_target; }
        const Mat4 &GetViewMatrix() const { return m_view; }
        const Mat4 &GetProjectionMatrix() const { return m_projection; }

    private:
        Projection m_projectionType = Projection::Perspective;
        float m_nearPlane = 0.01f;
        float m_farPlane = 1000.0f;
        Vec3 m_eye = {0.0f, 0.0f, 1.0f};
        Vec3 m_target = {0.0f, 0.0f, 0.0f};
        Vec3 m_up = {0.0f, 1.0f, 0.0f};
        Mat4 m_view;
        Mat4 m_projection;

        static Vec3 Sub(Vec3 a, Vec3 b) {
            return {a.x - b.x, a.y - b.y, a.z - b.z};
        }

        static float Dot(Vec3 a, Vec3 b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        static Vec3 Cross(Vec3 a, Vec3 b) {
            return {
                    a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x};
        }

        static Vec3 Normalize(Vec3 v) {
            const float len = std::sqrt(Dot(v, v));
            if (len <= 1e-6f) return {0.0f, 0.0f, 0.0f};
            return {v.x / len, v.y / len, v.z / len};
        }

        void RecomputeView() {
            Vec3 forward = Normalize(Sub(m_target, m_eye));
            Vec3 right = Normalize(Cross(forward, m_up));
            if (Dot(right, right) <= 1e-6f)
                right = {1.0f, 0.0f, 0.0f};
            Vec3 up = Cross(right, forward);

            m_view = {};
            m_view.m[0] = right.x;
            m_view.m[4] = right.y;
            m_view.m[8] = right.z;
            m_view.m[12] = -Dot(right, m_eye);

            m_view.m[1] = up.x;
            m_view.m[5] = up.y;
            m_view.m[9] = up.z;
            m_view.m[13] = -Dot(up, m_eye);

            m_view.m[2] = -forward.x;
            m_view.m[6] = -forward.y;
            m_view.m[10] = -forward.z;
            m_view.m[14] = Dot(forward, m_eye);

            m_view.m[15] = 1.0f;
        }
    };

} // namespace vkRender

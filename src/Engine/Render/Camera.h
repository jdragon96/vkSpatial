#pragma once

#include "utilities/Math.h"

#include <cmath>
#include <memory>

namespace Engine::Render {

    class Camera {
    public:
        using UniquePtr = std::unique_ptr<Camera>;

        enum class Projection {
            Perspective,
            Orthographic,
        };

        void SetPerspective(float fovYRadians, float aspect, float nearPlane, float farPlane) {
            m_projectionType = Projection::Perspective;
            m_nearPlane = nearPlane;
            m_farPlane = farPlane;
            m_projection = vkMath::Perspective(fovYRadians, aspect, nearPlane, farPlane);
        }

        void SetOrthographic(float left, float right, float bottom, float top,
                             float nearPlane, float farPlane) {
            m_projectionType = Projection::Orthographic;
            m_nearPlane = nearPlane;
            m_farPlane = farPlane;
            m_projection = vkMath::Orthographic(left, right, bottom, top, nearPlane, farPlane);
        }

        void LookAt(vkMath::Vec3 eye, vkMath::Vec3 target, vkMath::Vec3 up = {0.0f, 1.0f, 0.0f}) {
            m_eye = eye;
            m_target = target;
            m_up = up;
            m_view = vkMath::LookAt(eye, target, up);
            syncOrbitFromView();
        }

        void SetTarget(vkMath::Vec3 target) {
            m_target = target;
            updateOrbitView();
        }

        void SetDistance(float distance) {
            m_distance = sanitizeDistance(distance);
            updateOrbitView();
        }

        void SetOrientation(vkMath::Quat orientation) {
            m_orientation = orientation.normalized();
            updateOrbitView();
        }

        void SetOrbit(vkMath::Vec3 target, float distance,
                      vkMath::Quat orientation = vkMath::Quat::Identity()) {
            m_target = target;
            m_distance = sanitizeDistance(distance);
            m_orientation = orientation.normalized();
            updateOrbitView();
        }

        bool BeginTrackballDrag(double cursorX, double cursorY, int width, int height) {
            if (width <= 0 || height <= 0)
                return false;
            m_dragging = true;
            m_lastCursorX = cursorX;
            m_lastCursorY = cursorY;
            return true;
        }

        bool DragTrackball(double cursorX, double cursorY, int width, int height) {
            if (width <= 0 || height <= 0)
                return false;

            if (!m_dragging) {
                BeginTrackballDrag(cursorX, cursorY, width, height);
                return false;
            }

            const vkMath::Quat delta = vkMath::MapToArcball(
                    m_lastCursorX, m_lastCursorY, cursorX, cursorY, width, height);
            m_lastCursorX = cursorX;
            m_lastCursorY = cursorY;

            if (delta.vec().squaredNorm() <= 1e-8f)
                return false;

            m_orientation = (m_orientation * delta).normalized();
            updateOrbitView();
            return true;
        }

        void EndTrackballDrag() {
            m_dragging = false;
        }

        // Pan (translate) the orbit target in the current view plane by a screen-space drag,
        // grab-style: the world point under the cursor tracks the cursor. Speed scales with
        // orbit distance and vertical FOV, so it feels the same at any zoom. Perspective only:
        // Perspective() sets m_projection(1,1) = -1 / tan(fovY/2), so tan(fovY/2) = 1/|P(1,1)|.
        void Pan(double dxPixels, double dyPixels, int width, int height) {
            if (width <= 0 || height <= 0)
                return;
            const float tanHalfFovY = 1.0f / std::abs(m_projection(1, 1));
            const float worldPerPixel =
                    2.0f * m_distance * tanHalfFovY / static_cast<float>(height);
            const vkMath::Vec3 right =
                    (m_orientation * vkMath::Vec3(1.0f, 0.0f, 0.0f)).normalized();
            const vkMath::Vec3 up =
                    (m_orientation * vkMath::Vec3(0.0f, 1.0f, 0.0f)).normalized();
            m_target += right * (-static_cast<float>(dxPixels) * worldPerPixel)
                        + up * (static_cast<float>(dyPixels) * worldPerPixel);
            updateOrbitView();
        }

        Projection GetProjectionType() const { return m_projectionType; }
        float GetNearPlane() const { return m_nearPlane; }
        float GetFarPlane() const { return m_farPlane; }
        vkMath::Vec3 GetEye() const { return m_eye; }
        vkMath::Vec3 GetTarget() const { return m_target; }
        float GetDistance() const { return m_distance; }
        vkMath::Quat GetOrientation() const { return m_orientation; }
        bool IsTrackballDragging() const { return m_dragging; }
        const vkMath::Mat4 &GetViewMatrix() const { return m_view; }
        const vkMath::Mat4 &GetProjectionMatrix() const { return m_projection; }

    private:
        static constexpr float kMinDistance = 1e-4f;

        static float sanitizeDistance(float distance) {
            return distance > kMinDistance ? distance : kMinDistance;
        }

        void updateOrbitView() {
            const vkMath::Vec3 forward =
                    (m_orientation * vkMath::Vec3(0.0f, 0.0f, -1.0f)).normalized();
            m_up = (m_orientation * vkMath::Vec3(0.0f, 1.0f, 0.0f)).normalized();
            m_eye = m_target - forward * m_distance;
            m_view = vkMath::LookAt(m_eye, m_target, m_up);
        }

        void syncOrbitFromView() {
            const vkMath::Vec3 offset = m_eye - m_target;
            const float distance = offset.norm();
            if (distance <= kMinDistance) {
                m_distance = kMinDistance;
                m_orientation = vkMath::Quat::Identity();
                return;
            }

            m_distance = distance;
            const vkMath::Vec3 forward = (m_target - m_eye).normalized();
            vkMath::Vec3 right = forward.cross(m_up);
            if (right.squaredNorm() <= 1e-8f)
                right = m_orientation * vkMath::Vec3(1.0f, 0.0f, 0.0f);
            else
                right.normalize();

            const vkMath::Vec3 realUp = right.cross(forward).normalized();
            Eigen::Matrix3f basis = Eigen::Matrix3f::Identity();
            basis.col(0) = right;
            basis.col(1) = realUp;
            basis.col(2) = -forward;
            m_orientation = vkMath::Quat(basis).normalized();
            m_up = realUp;
        }

        Projection m_projectionType = Projection::Perspective;
        float m_nearPlane = 0.01f;
        float m_farPlane = 1000.0f;
        vkMath::Vec3 m_eye = {0.0f, 0.0f, 1.0f};
        vkMath::Vec3 m_target = {0.0f, 0.0f, 0.0f};
        float m_distance = 1.0f;
        vkMath::Quat m_orientation = vkMath::Quat::Identity();
        bool m_dragging = false;
        double m_lastCursorX = 0.0;
        double m_lastCursorY = 0.0;
        vkMath::Vec3 m_up = {0.0f, 1.0f, 0.0f};
        vkMath::Mat4 m_view = vkMath::Mat4::Identity();
        vkMath::Mat4 m_projection = vkMath::Mat4::Identity();
    };

} // namespace Engine::Render

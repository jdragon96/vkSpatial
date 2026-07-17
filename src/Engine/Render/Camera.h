#pragma once

#include "utilities/Math.h"

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
        }

        Projection GetProjectionType() const { return m_projectionType; }
        float GetNearPlane() const { return m_nearPlane; }
        float GetFarPlane() const { return m_farPlane; }
        vkMath::Vec3 GetEye() const { return m_eye; }
        vkMath::Vec3 GetTarget() const { return m_target; }
        const vkMath::Mat4 &GetViewMatrix() const { return m_view; }
        const vkMath::Mat4 &GetProjectionMatrix() const { return m_projection; }

    private:
        Projection m_projectionType = Projection::Perspective;
        float m_nearPlane = 0.01f;
        float m_farPlane = 1000.0f;
        vkMath::Vec3 m_eye = {0.0f, 0.0f, 1.0f};
        vkMath::Vec3 m_target = {0.0f, 0.0f, 0.0f};
        vkMath::Vec3 m_up = {0.0f, 1.0f, 0.0f};
        vkMath::Mat4 m_view = vkMath::Mat4::Identity();
        vkMath::Mat4 m_projection = vkMath::Mat4::Identity();
    };

} // namespace Engine::Render

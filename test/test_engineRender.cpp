#include <gtest/gtest.h>

#include "Engine/Render/Camera.h"
#include "Engine/Render/Scene.h"
#include "Engine/Render/View.h"

using namespace Engine::Render;

TEST(SceneTest, CreateEntityAssignsDistinctIds) {
    Scene scene;
    Entity a = scene.CreateEntity();
    Entity b = scene.CreateEntity();
    EXPECT_NE(a, b);
    EXPECT_TRUE(scene.Contains(a));
    EXPECT_TRUE(scene.Contains(b));
    EXPECT_EQ(scene.Entities().size(), 2u);
}

TEST(SceneTest, RemoveEntityDropsIt) {
    Scene scene;
    Entity a = scene.CreateEntity();
    scene.Remove(a);
    EXPECT_FALSE(scene.Contains(a));
}

TEST(CameraTest, PerspectiveProjectionMatchesVkMath) {
    Camera camera;
    const float fov = 60.0f * 3.14159265f / 180.0f;
    camera.SetPerspective(fov, 16.0f / 9.0f, 0.1f, 100.0f);
    const vkMath::Mat4 expected = vkMath::Perspective(fov, 16.0f / 9.0f, 0.1f, 100.0f);
    EXPECT_TRUE(camera.GetProjectionMatrix().isApprox(expected));
    EXPECT_EQ(camera.GetProjectionType(), Camera::Projection::Perspective);
}

TEST(CameraTest, LookAtMatchesVkMath) {
    Camera camera;
    const vkMath::Vec3 eye(0.0f, 0.0f, 5.0f);
    const vkMath::Vec3 target(0.0f, 0.0f, 0.0f);
    camera.LookAt(eye, target);
    const vkMath::Mat4 expected = vkMath::LookAt(eye, target, {0.0f, 1.0f, 0.0f});
    EXPECT_TRUE(camera.GetViewMatrix().isApprox(expected));
    EXPECT_EQ(camera.GetEye(), eye);
}

TEST(ViewTest, BindsSceneCameraAndGraph) {
    Scene scene;
    Camera camera;
    View view;
    view.SetScene(&scene);
    view.SetCamera(&camera);
    EXPECT_EQ(view.GetScene(), &scene);
    EXPECT_EQ(view.GetCamera(), &camera);
    EXPECT_EQ(view.GetRenderGraph(), nullptr);
}

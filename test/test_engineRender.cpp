#include <gtest/gtest.h>

#include "Engine/Render/Camera.h"
#include "Engine/Render/Scene.h"
#include "Engine/Render/View.h"
#include "Engine/Render/RenderGraph.h"
#include "Engine/Render/KeyInput.h"
#include "Engine/Render/MouseInput.h"

#include <memory>
#include <vector>

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

namespace {
    class RecordingPass : public RenderPass {
    public:
        RecordingPass(std::vector<int> &order, int id) : m_order(order), m_id(id) {}
        const char *Name() const override { return "RecordingPass"; }
        void Execute(RenderContext &) override { m_order.push_back(m_id); }

    private:
        std::vector<int> &m_order;
        int m_id;
    };
} // namespace

TEST(RenderGraphTest, ExecutesPassesInAddedOrder) {
    std::vector<int> order;
    RenderGraph graph;
    graph.AddPass(std::make_unique<RecordingPass>(order, 1));
    graph.AddPass(std::make_unique<RecordingPass>(order, 2));
    EXPECT_FALSE(graph.Empty());
    EXPECT_EQ(graph.PassCount(), 2u);

    RenderContext ctx{};
    graph.Execute(ctx);
    EXPECT_EQ(order, (std::vector<int>{1, 2}));
}

TEST(RenderGraphTest, AddNullPassThrows) {
    RenderGraph graph;
    EXPECT_THROW(graph.AddPass(nullptr), std::runtime_error);
}

TEST(MouseInputTest, ListenerReceivesMoveEvent) {
    MouseInput input;
    bool received = false;
    input.AddListener(MouseEventType::Move, [&](MouseEvent &e) {
        received = true;
        EXPECT_DOUBLE_EQ(e.x, 10.0);
    });
    input.OnMouseMove(10.0, 20.0);
    EXPECT_TRUE(received);
}

TEST(MouseInputTest, HigherPriorityListenerRunsFirstAndCanStopPropagation) {
    MouseInput input;
    std::vector<int> order;
    input.AddListener(MouseEventType::Move, [&](MouseEvent &e) {
        order.push_back(1);
        e.handled = true;
    }, 10);
    input.AddListener(MouseEventType::Move, [&](MouseEvent &) {
        order.push_back(2);
    }, 0);
    input.OnMouseMove(0.0, 0.0);
    EXPECT_EQ(order, (std::vector<int>{1}));
}

TEST(MouseListenerGroupTest, ClearsListenersOnDestruction) {
    MouseInput input;
    int callCount = 0;
    {
        MouseListenerGroup group(input);
        group.Add(MouseEventType::Move, [&](MouseEvent &) { ++callCount; });
        input.OnMouseMove(0.0, 0.0);
    }
    input.OnMouseMove(1.0, 1.0);
    EXPECT_EQ(callCount, 1);
}

TEST(KeyInputTest, ListenerReceivesPressAndTracksState) {
    KeyInput input;
    bool received = false;
    input.AddListener(KeyEventType::Press, [&](KeyEvent &e) {
        received = true;
        EXPECT_EQ(e.keyCode, KeyCode::Escape);
    });
    input.OnKey(KeyCode::Escape, KeyEventType::Press);
    EXPECT_TRUE(received);
    EXPECT_TRUE(input.IsKeyDown(KeyCode::Escape));

    input.OnKey(KeyCode::Escape, KeyEventType::Release);
    EXPECT_FALSE(input.IsKeyDown(KeyCode::Escape));
}

TEST(KeyListenerGroupTest, ClearsListenersOnDestruction) {
    KeyInput input;
    int callCount = 0;
    {
        KeyListenerGroup group(input);
        group.Add(KeyEventType::Press, [&](KeyEvent &) { ++callCount; });
        input.OnKey(KeyCode::A, KeyEventType::Press);
    }
    input.OnKey(KeyCode::A, KeyEventType::Press);
    EXPECT_EQ(callCount, 1);
}

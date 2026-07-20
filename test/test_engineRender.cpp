#include <gtest/gtest.h>

#include "Engine/Core/Descriptor.h"
#include "Engine/Core/Sampler.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GraphicsPipeline.h"
#include "Engine/Render/KeyInput.h"
#include "Engine/Render/MouseInput.h"
#include "Engine/Render/Object.h"
#include "Engine/Render/RenderGraph.h"
#include "Engine/Render/Scene.h"
#include "Engine/Render/View.h"

#include <memory>
#include <stdexcept>
#include <vector>

using namespace Engine::Render;

namespace {

    struct ObjectTestVertex {
        float position[3];
        float color[3];
    };

    struct TestTransformConstants {
        float data[32];
    };

    struct TestLightingConstants {
        float data[8];
    };

} // namespace

TEST(GraphicsPipelineDescriptorTest, PushConstantAllowsDisjointStagesAtSameOffset) {
    GraphicsPipelineDescriptor descriptor;
    descriptor.PushConstant<TestTransformConstants>(VK_SHADER_STAGE_VERTEX_BIT)
            .PushConstant<TestLightingConstants>(VK_SHADER_STAGE_FRAGMENT_BIT);

    ASSERT_EQ(descriptor.pushConstantRanges.size(), 2u);
    EXPECT_EQ(descriptor.pushConstantRanges[0].stageFlags, VK_SHADER_STAGE_VERTEX_BIT);
    EXPECT_EQ(descriptor.pushConstantRanges[0].offset, 0u);
    EXPECT_EQ(descriptor.pushConstantRanges[0].size, sizeof(TestTransformConstants));
    EXPECT_EQ(descriptor.pushConstantRanges[1].stageFlags, VK_SHADER_STAGE_FRAGMENT_BIT);
    EXPECT_EQ(descriptor.pushConstantRanges[1].offset, 0u);
    EXPECT_EQ(descriptor.pushConstantRanges[1].size, sizeof(TestLightingConstants));
}

TEST(GraphicsPipelineDescriptorTest, PushConstantRejectsOverlappingSharedStageRanges) {
    GraphicsPipelineDescriptor descriptor;
    descriptor.PushConstant(VK_SHADER_STAGE_VERTEX_BIT, 64, 0);

    EXPECT_THROW(
            descriptor.PushConstant(VK_SHADER_STAGE_VERTEX_BIT, 16, 32),
            std::runtime_error);
}

TEST(GraphicsPipelineDescriptorTest, PushConstantRejectsUnalignedRanges) {
    GraphicsPipelineDescriptor descriptor;

    EXPECT_THROW(
            descriptor.PushConstant(VK_SHADER_STAGE_VERTEX_BIT, 6, 0),
            std::runtime_error);
    EXPECT_THROW(
            descriptor.PushConstant(VK_SHADER_STAGE_VERTEX_BIT, 8, 2),
            std::runtime_error);
}

TEST(DescriptorBindingTest, CombinedImageSamplerBuildsFragmentBinding) {
    const Engine::Core::DescriptorBinding descriptor =
            Engine::Core::DescriptorBinding::CombinedImageSampler(
                    2,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    3);
    const VkDescriptorSetLayoutBinding binding = descriptor.Build();

    EXPECT_EQ(binding.binding, 2u);
    EXPECT_EQ(binding.descriptorType, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    EXPECT_EQ(binding.descriptorCount, 3u);
    EXPECT_EQ(binding.stageFlags, VK_SHADER_STAGE_FRAGMENT_BIT);
    EXPECT_EQ(binding.pImmutableSamplers, nullptr);
}

TEST(SamplerDescriptorTest, ShadowMapManualPCFUsesNearestClampedDepthSampling) {
    const Engine::Core::SamplerDescriptor descriptor =
            Engine::Core::SamplerDescriptor::ShadowMapManualPCF();
    const VkSamplerCreateInfo info = descriptor.Build();

    EXPECT_EQ(info.sType, VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO);
    EXPECT_EQ(info.magFilter, VK_FILTER_NEAREST);
    EXPECT_EQ(info.minFilter, VK_FILTER_NEAREST);
    EXPECT_EQ(info.mipmapMode, VK_SAMPLER_MIPMAP_MODE_NEAREST);
    EXPECT_EQ(info.addressModeU, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    EXPECT_EQ(info.addressModeV, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    EXPECT_EQ(info.addressModeW, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    EXPECT_EQ(info.compareEnable, VK_FALSE);
    EXPECT_EQ(info.compareOp, VK_COMPARE_OP_ALWAYS);
    EXPECT_FLOAT_EQ(info.maxLod, 1.0f);
}

TEST(SamplerDescriptorTest, ShadowMapCompareDefaultsToNearestHardwareDepthCompare) {
    const Engine::Core::SamplerDescriptor descriptor =
            Engine::Core::SamplerDescriptor::ShadowMapCompare();
    const VkSamplerCreateInfo info = descriptor.Build();

    EXPECT_EQ(info.magFilter, VK_FILTER_NEAREST);
    EXPECT_EQ(info.minFilter, VK_FILTER_NEAREST);
    EXPECT_EQ(info.compareEnable, VK_TRUE);
    EXPECT_EQ(info.compareOp, VK_COMPARE_OP_LESS_OR_EQUAL);
}

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

TEST(CameraTest, OrbitSettersUpdateViewState) {
    Camera camera;
    const vkMath::Vec3 target(1.0f, 2.0f, 3.0f);
    camera.SetTarget(target);
    camera.SetDistance(4.5f);

    EXPECT_TRUE(camera.GetTarget().isApprox(target));
    EXPECT_FLOAT_EQ(camera.GetDistance(), 4.5f);
    EXPECT_TRUE(camera.GetEye().isApprox(vkMath::Vec3(1.0f, 2.0f, 7.5f)));
}

TEST(CameraTest, OrbitOrientationProvidesStablePoleBasis) {
    Camera camera;
    camera.SetOrbit({0.0f, 0.0f, 0.0f}, 4.5f,
                    vkMath::Quat(Eigen::AngleAxisf(3.14159265f * 0.5f, vkMath::Vec3::UnitX())));

    EXPECT_TRUE(camera.GetEye().isApprox(vkMath::Vec3(0.0f, -4.5f, 0.0f), 1e-5f));
    EXPECT_TRUE(camera.GetViewMatrix().allFinite());
}

TEST(CameraTest, TrackballDragUpdatesOrientation) {
    Camera camera;
    camera.SetOrbit({0.0f, 0.0f, 0.0f}, 4.5f);
    ASSERT_TRUE(camera.BeginTrackballDrag(100.0, 100.0, 200, 200));

    EXPECT_TRUE(camera.DragTrackball(140.0, 100.0, 200, 200));
    EXPECT_FALSE(camera.GetOrientation().isApprox(vkMath::Quat::Identity()));
    EXPECT_TRUE(camera.GetViewMatrix().allFinite());
}

TEST(ObjectTest, OwnsVertexAndIndexData) {
    std::vector<ObjectTestVertex> vertices = {
            {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
            {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
            {{0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
    };
    std::vector<uint32_t> indices = {0, 1, 2};

    Object<ObjectTestVertex> object(vertices, indices);
    vertices.clear();
    indices.clear();

    EXPECT_FALSE(object.Empty());
    EXPECT_EQ(object.VertexCount(), 3u);
    EXPECT_EQ(object.IndexCount(), 3u);
    EXPECT_EQ(object.VertexByteSize(), 3u * sizeof(ObjectTestVertex));
    EXPECT_EQ(object.IndexByteSize(), 3u * sizeof(uint32_t));
    EXPECT_FLOAT_EQ(object.VertexData()[1].position[0], 1.0f);
    EXPECT_EQ(object.IndexData()[2], 2u);
    EXPECT_TRUE(object.Model().isApprox(vkMath::Mat4::Identity()));

    const vkMath::Mat4 model = vkMath::Translation(1.0f, 2.0f, 3.0f);
    object.SetModel(model);
    object.SetFirstIndex(7);
    EXPECT_TRUE(object.Model().isApprox(model));
    EXPECT_EQ(object.FirstIndex(), 7u);
}

TEST(ObjectTest, SetGeometryReplacesData) {
    Object<ObjectTestVertex> object;
    EXPECT_TRUE(object.Empty());

    object.SetGeometry(
            {{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}},
            {0});

    EXPECT_FALSE(object.Empty());
    EXPECT_EQ(object.VertexCount(), 1u);
    EXPECT_EQ(object.IndexCount(), 1u);
}

TEST(ObjectTest, PacksDefaultVertexObjectsForUpload) {
    Object<> first;
    const uint32_t a = first.AddVertex(Vertex({0.0f, 0.0f, 0.0f},
                                              {0.0f, 1.0f, 0.0f},
                                              {1.0f, 0.0f, 0.0f}));
    const uint32_t b = first.AddVertex(Vertex({1.0f, 0.0f, 0.0f},
                                              {0.0f, 1.0f, 0.0f},
                                              {0.0f, 1.0f, 0.0f}));
    const uint32_t c = first.AddVertex(Vertex({0.0f, 1.0f, 0.0f},
                                              {0.0f, 1.0f, 0.0f},
                                              {0.0f, 0.0f, 1.0f}));
    first.AddTriangle(a, b, c);

    Object<> second;
    const uint32_t d = second.AddVertex(Vertex({0.0f, 0.0f, 1.0f},
                                               {0.0f, 0.0f, 1.0f},
                                               {1.0f, 1.0f, 1.0f}));
    second.AddTriangle(d, d, d);

    std::vector<Object<>> objects = {first, second};
    const auto geometry = PackObjects(objects);

    EXPECT_EQ(geometry.VertexCount(), 4u);
    EXPECT_EQ(geometry.IndexCount(), 6u);
    EXPECT_EQ(objects[0].FirstIndex(), 0u);
    EXPECT_EQ(objects[1].FirstIndex(), 3u);
    EXPECT_EQ(geometry.indices[3], 3u);
    EXPECT_FLOAT_EQ(geometry.VertexData()[3].position[2], 1.0f);
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

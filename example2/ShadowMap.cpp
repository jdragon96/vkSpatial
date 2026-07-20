#include "Engine/Core/Descriptor.h"
#include "Engine/Core/Image.h"
#include "Engine/Core/Sampler.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GraphicsPipeline.h"
#include "Engine/Render/Object.h"
#include "Engine/Render/RenderGraph.h"
#include "Engine/Render/Rendering.h"
#include "Engine/Render/Scene.h"
#include "Engine/Render/SwapChain.h"
#include "Engine/Render/View.h"

#include "utilities/Math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

    constexpr float kPi = 3.14159265358979323846f;
    constexpr uint32_t kShadowMapSize = 2048;

    using RenderObject = Engine::Render::Object<>;
    using RenderVertex = Engine::Render::Vertex;

    struct PushConstants {
        vkMath::Mat4 model = vkMath::Mat4::Identity();
        vkMath::Mat4 viewProj = vkMath::Mat4::Identity();
        vkMath::Mat4 lightViewProj = vkMath::Mat4::Identity();
        float lightDir[4] = {};
    };

    class ShadowMapPass final : public Engine::Render::RenderPass {
    public:
        ShadowMapPass(Engine::Core::Context &context, VkFormat colorFormat, const std::string &shaderDir)
            : m_context(context),
              m_shadowTarget(context),
              m_shadowSampler(context),
              m_shadowDescriptorLayout(context),
              m_shadowDescriptorPool(context),
              m_shadowPipeline(context),
              m_scenePipeline(context) {
            CreateGeometry();
            CreateShadowResources();
            CreateDescriptorResources();
            CreatePipelines(colorFormat, shaderDir);
        }

        ~ShadowMapPass() override {
            m_scenePipeline.Destroy();
            m_shadowPipeline.Destroy();
            DestroyDescriptorResources();
            DestroyShadowResources();
        }

        const char *Name() const override { return "ShadowMapPass"; }

        void Execute(Engine::Render::RenderContext &ctx) override {
            if (!ctx.swapChain || !ctx.depthImage || !ctx.view || !ctx.view->GetCamera())
                throw std::runtime_error("ShadowMapPass requires swapchain, depth image, view and camera");

            const float timeSeconds = static_cast<float>(ctx.frame.frameIndex) / 60.0f;
            UpdateDrawModels(timeSeconds);

            const vkMath::Vec3 worldUp{0.0f, 1.0f, 0.0f};
            const vkMath::Vec3 lightPos{
                    std::sin(timeSeconds * 0.55f) * 3.8f,
                    5.2f,
                    std::cos(timeSeconds * 0.55f) * 2.6f};
            const vkMath::Vec3 lightTarget{0.0f, 0.55f, 0.0f};
            const vkMath::Vec3 lightDir = (lightTarget - lightPos).normalized();
            const vkMath::Mat4 lightView = vkMath::LookAt(lightPos, lightTarget, worldUp);
            const vkMath::Mat4 lightProj =
                    vkMath::Orthographic(-6.2f, 6.2f, -6.2f, 6.2f, 0.1f, 15.0f);
            const vkMath::Mat4 lightViewProj = lightProj * lightView;

            const Engine::Render::Camera *camera = ctx.view->GetCamera();
            const vkMath::Mat4 viewProj = camera->GetProjectionMatrix() * camera->GetViewMatrix();

            RecordShadowPass(ctx.commandBuffer, lightViewProj, lightDir);
            RecordScenePass(ctx, viewProj, lightViewProj, lightDir);
        }

    private:
        Engine::Core::Context &m_context;
        std::vector<RenderObject> m_objects;
        Engine::Core::Image m_shadowTarget;
        Engine::Core::Sampler m_shadowSampler;
        Engine::Core::DescriptorSetLayout m_shadowDescriptorLayout;
        Engine::Core::DescriptorPool m_shadowDescriptorPool;
        Engine::Core::DescriptorSet m_shadowDescriptorSet;
        Engine::Render::GraphicsPipeline m_shadowPipeline;
        Engine::Render::GraphicsPipeline m_scenePipeline;

        void CreateGeometry() {
            auto makePlane = [&]() {
                RenderObject plane;
                const vkMath::Vec3 color{0.58f, 0.58f, 0.60f};
                const vkMath::Vec3 normal{0.0f, 1.0f, 0.0f};
                const uint32_t v0 = plane.AddVertex(RenderVertex({-6.0f, 0.0f, -6.0f}, normal, color));
                const uint32_t v1 = plane.AddVertex(RenderVertex({6.0f, 0.0f, -6.0f}, normal, color));
                const uint32_t v2 = plane.AddVertex(RenderVertex({6.0f, 0.0f, 6.0f}, normal, color));
                const uint32_t v3 = plane.AddVertex(RenderVertex({-6.0f, 0.0f, 6.0f}, normal, color));
                plane.AddQuad(v0, v1, v2, v3);
                return plane;
            };

            auto makeCube = [&]() {
                RenderObject cube;
                const vkMath::Vec3 color{0.35f, 0.47f, 0.92f};
                const std::array<vkMath::Vec3, 8> p = {{
                        {-0.75f, -0.75f, -0.75f},
                        {0.75f, -0.75f, -0.75f},
                        {0.75f, 0.75f, -0.75f},
                        {-0.75f, 0.75f, -0.75f},
                        {-0.75f, -0.75f, 0.75f},
                        {0.75f, -0.75f, 0.75f},
                        {0.75f, 0.75f, 0.75f},
                        {-0.75f, 0.75f, 0.75f},
                }};

                auto face = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d, vkMath::Vec3 n) {
                    const uint32_t v0 = cube.AddVertex(RenderVertex(p[a], n, color));
                    const uint32_t v1 = cube.AddVertex(RenderVertex(p[b], n, color));
                    const uint32_t v2 = cube.AddVertex(RenderVertex(p[c], n, color));
                    const uint32_t v3 = cube.AddVertex(RenderVertex(p[d], n, color));
                    cube.AddQuad(v0, v1, v2, v3);
                };

                face(4, 5, 6, 7, {0.0f, 0.0f, 1.0f});
                face(1, 0, 3, 2, {0.0f, 0.0f, -1.0f});
                face(0, 4, 7, 3, {-1.0f, 0.0f, 0.0f});
                face(5, 1, 2, 6, {1.0f, 0.0f, 0.0f});
                face(3, 7, 6, 2, {0.0f, 1.0f, 0.0f});
                face(0, 1, 5, 4, {0.0f, -1.0f, 0.0f});

                cube.SetModel(vkMath::Translation(0.0f, 0.75f, 0.0f));
                return cube;
            };

            m_objects.push_back(makePlane());
            m_objects.push_back(makeCube());

            for (RenderObject &object: m_objects)
                object.Upload(m_context);
        }

        void UpdateDrawModels(float timeSeconds) {
            if (m_objects.size() < 2)
                return;

            const vkMath::Mat4 rotation = vkMath::RotationY(timeSeconds * 0.7f);
            const vkMath::Mat4 translate = vkMath::Translation(0.0f, 0.78f, 0.0f);
            m_objects[1].SetModel(translate * rotation);
        }

        void CreateShadowResources() {
            Engine::Core::ImageDescriptor descriptor =
                    Engine::Core::ImageDescriptor::Depth2D(
                            {kShadowMapSize, kShadowMapSize}, VK_FORMAT_D32_SFLOAT);
            descriptor.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
            m_shadowTarget.Create(descriptor);
            m_shadowSampler.Create(Engine::Core::SamplerDescriptor::ShadowMapManualPCF());
        }

        void DestroyShadowResources() {
            m_shadowSampler.Destroy();
            m_shadowTarget.Destroy();
        }

        void CreateDescriptorResources() {
            m_shadowDescriptorLayout.Create({
                    Engine::Core::DescriptorBinding::CombinedImageSampler(
                            0,
                            VK_SHADER_STAGE_FRAGMENT_BIT),
            });
            m_shadowDescriptorPool.Create(
                    {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}},
                    1);
            m_shadowDescriptorSet =
                    m_shadowDescriptorPool.Allocate(m_shadowDescriptorLayout.Handle());
            m_shadowDescriptorSet.UpdateCombinedImageSampler(
                    0,
                    m_shadowSampler.Handle(),
                    m_shadowTarget.View(),
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        void DestroyDescriptorResources() {
            m_shadowDescriptorSet.Reset();
            m_shadowDescriptorPool.Destroy();
            m_shadowDescriptorLayout.Destroy();
        }

        void CreatePipelines(VkFormat colorFormat, const std::string &shaderDir) {
            Engine::Render::GraphicsPipelineDescriptor shadowDescriptor;
            shadowDescriptor.VertexShader(shaderDir + "/realtime_shadow_depth.vert.spv")
                    .VertexBinding<RenderVertex>()
                    .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, position))
                    .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, normal))
                    .VertexAttribute(2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, color))
                    .DepthTarget(VK_FORMAT_D32_SFLOAT)
                    .DepthBias(1.2f, 1.8f)
                    .PushConstant<PushConstants>(
                            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
            m_shadowPipeline.Build(shadowDescriptor);

            Engine::Render::GraphicsPipelineDescriptor sceneDescriptor;
            sceneDescriptor.descriptorSetLayouts.push_back(m_shadowDescriptorLayout.Handle());
            sceneDescriptor.VertexShader(shaderDir + "/realtime_shadow_scene.vert.spv")
                    .FragmentShader(shaderDir + "/realtime_shadow_scene.frag.spv")
                    .VertexBinding<RenderVertex>()
                    .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, position))
                    .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, normal))
                    .VertexAttribute(2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, color))
                    .ColorTarget(colorFormat)
                    .DepthTarget(VK_FORMAT_D32_SFLOAT)
                    .PushConstant<PushConstants>(
                            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
            m_scenePipeline.Build(sceneDescriptor);
        }

        void RecordShadowPass(VkCommandBuffer cmd,
                              const vkMath::Mat4 &lightViewProj,
                              vkMath::Vec3 lightDir) {
            const bool firstUse = m_shadowTarget.CurrentLayout() == VK_IMAGE_LAYOUT_UNDEFINED;
            m_shadowTarget.TransitionLayout(
                    cmd,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    firstUse ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                    firstUse ? 0 : VK_ACCESS_SHADER_READ_BIT,
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            Engine::Render::RenderingDescriptor descriptor({kShadowMapSize, kShadowMapSize});
            descriptor.SetDepthAttachment(
                    Engine::Render::DepthAttachment(m_shadowTarget.View())
                            .Clear(1.0f)
                            .Store(true)
                            .Build());
            Engine::Render::RenderingScope rendering(cmd, descriptor);

            m_shadowPipeline.Bind(cmd);
            for (const RenderObject &object: m_objects) {
                PushConstants pc{};
                pc.model = object.Model();
                pc.lightViewProj = lightViewProj;
                pc.lightDir[0] = lightDir.x();
                pc.lightDir[1] = lightDir.y();
                pc.lightDir[2] = lightDir.z();
                m_shadowPipeline.PushConstants(
                        cmd,
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        pc);
                object.Render(cmd);
            }
            rendering.End();

            m_shadowTarget.TransitionLayout(
                    cmd,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
        }

        void RecordScenePass(Engine::Render::RenderContext &ctx,
                             const vkMath::Mat4 &viewProj,
                             const vkMath::Mat4 &lightViewProj,
                             vkMath::Vec3 lightDir) {
            Engine::Render::ClearOptions clear{};
            clear.color[0] = 0.045f;
            clear.color[1] = 0.047f;
            clear.color[2] = 0.055f;
            clear.color[3] = 1.0f;

            Engine::Render::RenderingDescriptor descriptor =
                    Engine::Render::RenderingDescriptor::ColorDepth(
                            ctx.swapChain->Extent(),
                            ctx.swapChain->ImageView(ctx.imageIndex),
                            ctx.depthImage->View(),
                            clear);
            Engine::Render::RenderingScope rendering(ctx.commandBuffer, descriptor);

            m_scenePipeline.Bind(ctx.commandBuffer);
            m_shadowDescriptorSet.Bind(ctx.commandBuffer, m_scenePipeline.Layout());

            for (const RenderObject &object: m_objects) {
                PushConstants pc{};
                pc.model = object.Model();
                pc.viewProj = viewProj;
                pc.lightViewProj = lightViewProj;
                pc.lightDir[0] = lightDir.x();
                pc.lightDir[1] = lightDir.y();
                pc.lightDir[2] = lightDir.z();
                m_scenePipeline.PushConstants(
                        ctx.commandBuffer,
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        pc);
                object.Render(ctx.commandBuffer);
            }
        }
    };

} // namespace

int main() {
    try {
        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {960, 720, "Engine::Render Shadow Map"};
        Engine::Render::Application app(descriptor);

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D extent = app.GetSwapChain().Extent();
        const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        camera.SetPerspective(55.0f * kPi / 180.0f, aspect, 0.1f, 80.0f);
        camera.LookAt({0.0f, 3.0f, 7.0f}, {0.0f, 0.75f, 0.0f});

        const std::string shaderDir = SHADOW_MAP_SHADER_DIR;
        Engine::Render::RenderGraph graph;
        graph.AddPass(std::make_unique<ShadowMapPass>(
                app.GetContext(),
                app.GetSwapChain().Format(),
                shaderDir));

        app.GetView().SetScene(&scene);
        app.GetView().SetCamera(&camera);
        app.GetView().SetRenderGraph(&graph);

        Engine::Render::MouseListenerGroup trackball(app.GetWindow().Mouse());
        trackball.Add(Engine::Render::MouseEventType::Drag, [&](Engine::Render::MouseEvent &e) {
            if (e.button != Engine::Render::MouseButton::Left)
                return;
            const VkExtent2D size = app.GetWindow().FramebufferSize();
            if (size.width == 0 || size.height == 0)
                return;

            if (!camera.IsTrackballDragging()) {
                camera.BeginTrackballDrag(
                        e.x, e.y,
                        static_cast<int>(size.width), static_cast<int>(size.height));
                return;
            }

            camera.DragTrackball(
                    e.x, e.y,
                    static_cast<int>(size.width), static_cast<int>(size.height));
            e.handled = true;
        });
        trackball.Add(Engine::Render::MouseEventType::ButtonUp, [&](Engine::Render::MouseEvent &e) {
            if (e.button == Engine::Render::MouseButton::Left)
                camera.EndTrackballDrag();
        });
        trackball.Add(Engine::Render::MouseEventType::Scroll, [&](Engine::Render::MouseEvent &e) {
            camera.SetDistance(
                    std::clamp(camera.GetDistance() * std::exp(static_cast<float>(-e.scrollY) * 0.08f),
                               2.0f, 12.0f));
            e.handled = true;
        });

        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape)
                app.GetWindow().RequestClose();
        });

        app.Run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

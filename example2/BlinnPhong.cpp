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
#include <vector>

#ifndef BLINN_PHONG_SHADER_DIR
#define BLINN_PHONG_SHADER_DIR "."
#endif

namespace {

    constexpr float kPi = 3.14159265358979323846f;

    using RenderObject = Engine::Render::Object<>;
    using RenderVertex = Engine::Render::Vertex;

    struct PushConstants {
        vkMath::Mat4 mvp = vkMath::Mat4::Identity();
        vkMath::Mat4 modelView = vkMath::Mat4::Identity();
    };

    struct BlinnPhongConstants {
        vkMath::Vec3 lightPosition;
        float shininess;
        vkMath::Vec3 lightColor;
        float specularStrength;
        vkMath::Vec3 cameraPosition;
        float dummy1;
    };

    RenderObject MakePlane() {
        RenderObject plane;
        const vkMath::Vec3 color{0.46f, 0.49f, 0.52f};
        const vkMath::Vec3 normal{0.0f, 1.0f, 0.0f};

        const uint32_t v0 = plane.AddVertex(RenderVertex({-6.0f, 0.0f, -6.0f}, normal, color));
        const uint32_t v1 = plane.AddVertex(RenderVertex({6.0f, 0.0f, -6.0f}, normal, color));
        const uint32_t v2 = plane.AddVertex(RenderVertex({6.0f, 0.0f, 6.0f}, normal, color));
        const uint32_t v3 = plane.AddVertex(RenderVertex({-6.0f, 0.0f, 6.0f}, normal, color));
        plane.AddQuad(v0, v1, v2, v3);

        return plane;
    }

    RenderObject MakeCube(float halfExtent, const vkMath::Vec3 &color) {
        RenderObject cube;
        const float h = halfExtent;
        const std::array<vkMath::Vec3, 8> p = {{
                {-h, -h, -h},
                {h, -h, -h},
                {h, h, -h},
                {-h, h, -h},
                {-h, -h, h},
                {h, -h, h},
                {h, h, h},
                {-h, h, h},
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

        return cube;
    }

    RenderObject MakeSphere(float radius,
                            uint32_t rings,
                            uint32_t segments,
                            const vkMath::Vec3 &color) {
        RenderObject sphere;

        for (uint32_t ring = 0; ring <= rings; ++ring) {
            const float v = static_cast<float>(ring) / static_cast<float>(rings);
            const float theta = v * kPi;
            const float y = std::cos(theta);
            const float ringRadius = std::sin(theta);

            for (uint32_t segment = 0; segment <= segments; ++segment) {
                const float u = static_cast<float>(segment) / static_cast<float>(segments);
                const float phi = u * kPi * 2.0f;
                const vkMath::Vec3 normal{
                        ringRadius * std::cos(phi),
                        y,
                        ringRadius * std::sin(phi),
                };
                sphere.AddVertex(RenderVertex(normal * radius, normal.normalized(), color));
            }
        }

        const uint32_t stride = segments + 1;
        for (uint32_t ring = 0; ring < rings; ++ring) {
            for (uint32_t segment = 0; segment < segments; ++segment) {
                const uint32_t i0 = ring * stride + segment;
                const uint32_t i1 = i0 + 1;
                const uint32_t i2 = (ring + 1) * stride + segment + 1;
                const uint32_t i3 = (ring + 1) * stride + segment;
                sphere.AddTriangle(i0, i3, i1);
                sphere.AddTriangle(i1, i3, i2);
            }
        }

        return sphere;
    }

    class BlinnPhongPass final : public Engine::Render::RenderPass {
    public:
        BlinnPhongPass(Engine::Core::Context &context,
                       VkFormat colorFormat,
                       const std::string &shaderDir)
            : m_context(context),
              m_pipeline(context) {
            CreateGeometry();
            CreatePipeline(colorFormat, shaderDir);
        }

        ~BlinnPhongPass() override {
            m_pipeline.Destroy();
        }

        const char *Name() const override { return "BlinnPhongPass"; }

        void Execute(Engine::Render::RenderContext &ctx) override {
            if (!ctx.swapChain || !ctx.depthImage || !ctx.view || !ctx.view->GetCamera())
                throw std::runtime_error("BlinnPhongPass requires swapchain, depth image, view and camera");

            const float timeSeconds = static_cast<float>(ctx.frame.frameIndex) / 60.0f;
            UpdateModels(timeSeconds);

            Engine::Render::ClearOptions clear{};
            clear.color[0] = 0.035f;
            clear.color[1] = 0.039f;
            clear.color[2] = 0.045f;
            clear.color[3] = 1.0f;

            Engine::Render::RenderingDescriptor descriptor =
                    Engine::Render::RenderingDescriptor::ColorDepth(
                            ctx.swapChain->Extent(),
                            ctx.swapChain->ImageView(ctx.imageIndex),
                            ctx.depthImage->View(),
                            clear);
            Engine::Render::RenderingScope rendering(ctx.commandBuffer, descriptor);

            const Engine::Render::Camera *camera = ctx.view->GetCamera();
            const vkMath::Mat4 view = camera->GetViewMatrix();
            const vkMath::Mat4 viewProj = camera->GetProjectionMatrix() * view;

            const vkMath::Vec3 worldLightPosition{
                    std::sin(timeSeconds * 0.45f) * 2.8f,
                    3.4f,
                    std::cos(timeSeconds * 0.45f) * 2.0f};
            const Eigen::Vector4f lightPosition =
                    view * Eigen::Vector4f(
                                   worldLightPosition.x(),
                                   worldLightPosition.y(),
                                   worldLightPosition.z(),
                                   1.0f);

            BlinnPhongConstants lighting{};
            lighting.lightPosition = vkMath::Vec3(lightPosition.x(), lightPosition.y(), lightPosition.z());
            lighting.shininess = 48.0f;
            lighting.lightColor = vkMath::Vec3(1.0f, 0.94f, 0.84f);
            lighting.specularStrength = 0.38f;
            lighting.cameraPosition = camera->GetEye();

            m_pipeline.Bind(ctx.commandBuffer);
            m_pipeline.PushConstants(ctx.commandBuffer, VK_SHADER_STAGE_FRAGMENT_BIT, lighting);
            for (const RenderObject &object: m_objects) {
                PushConstants pc{};
                pc.mvp = viewProj * object.Model();
                pc.modelView = view * object.Model();
                m_pipeline.PushConstants(
                        ctx.commandBuffer,
                        VK_SHADER_STAGE_VERTEX_BIT,
                        pc);
                object.Render(ctx.commandBuffer);
            }
        }

    private:
        Engine::Core::Context &m_context;
        std::vector<RenderObject> m_objects;
        Engine::Render::GraphicsPipeline m_pipeline;

        void CreateGeometry() {
            m_objects.push_back(MakePlane());
            m_objects.push_back(MakeCube(0.72f, {0.34f, 0.48f, 0.90f}));
            m_objects.push_back(MakeSphere(0.58f, 24, 32, {0.92f, 0.56f, 0.28f}));
            m_objects.push_back(MakeCube(0.36f, {0.30f, 0.78f, 0.58f}));

            for (RenderObject &object: m_objects)
                object.Upload(m_context);
        }

        void UpdateModels(float timeSeconds) {
            if (m_objects.size() < 4)
                return;

            m_objects[1].SetModel(
                    vkMath::Translation(-1.25f, 0.72f, 0.05f) *
                    vkMath::RotationY(timeSeconds * 0.75f));
            m_objects[2].SetModel(
                    vkMath::Translation(1.20f, 0.58f, 0.35f) *
                    vkMath::RotationY(-timeSeconds * 0.45f));
            m_objects[3].SetModel(
                    vkMath::Translation(0.0f, 0.36f, -1.35f) *
                    vkMath::RotationY(timeSeconds * 1.15f));
        }

        void CreatePipeline(VkFormat colorFormat, const std::string &shaderDir) {
            Engine::Render::GraphicsPipelineDescriptor descriptor;
            descriptor.VertexShader(shaderDir + "/BlinnPhong.vert.glsl.spv")
                    .FragmentShader(shaderDir + "/BlinnPhong.frag.glsl.spv")
                    .VertexBinding<RenderVertex>()
                    .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, position))
                    .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, normal))
                    .VertexAttribute(2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, color))
                    .ColorTarget(colorFormat)
                    .DepthTarget(VK_FORMAT_D32_SFLOAT)
                    .PushConstant<PushConstants>(VK_SHADER_STAGE_VERTEX_BIT)
                    .PushConstant<BlinnPhongConstants>(VK_SHADER_STAGE_FRAGMENT_BIT);

            m_pipeline.Build(descriptor);
        }
    };

} // namespace

int main() {
    try {
        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {960, 720, "Engine::Render Blinn-Phong"};
        Engine::Render::Application app(descriptor);

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D extent = app.GetSwapChain().Extent();
        const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        camera.SetPerspective(55.0f * kPi / 180.0f, aspect, 0.1f, 80.0f);
        camera.SetOrbit({0.0f, 0.55f, 0.0f}, 6.5f);

        const std::string shaderDir = BLINN_PHONG_SHADER_DIR;
        Engine::Render::RenderGraph graph;
        graph.AddPass(std::make_unique<BlinnPhongPass>(
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
                               2.5f, 12.0f));
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

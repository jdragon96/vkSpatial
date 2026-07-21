// Minimal windowed viewer for the DirectionalTSDF integrate/extract point-cloud pipeline
// (Task 1 of the interactive TSDF viewer plan: docs/superpowers/specs,
// .superpowers/sdd/task-1-brief.md). Builds the "plane" fixture also used by
// tsdf_slice_debug.cpp, integrates it into a DirectionalTSDF, and renders the raw input
// point cloud (white) with a trackball camera.
// Task 2 (.superpowers/sdd/task-2-brief.md) adds ImGuiPass, drawn as the LAST render-graph
// pass so its "load, don't clear" RenderingScope composites the panel on top of the points
// instead of erasing them. No extracted / slice point sets wired up yet (Task 3).
#include "ImGuiPass.h"
#include "PointCloudPass.h"
#include "tsdf_fixtures.h"

#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"
#include "Engine/Spatial/DirectionalTSDF.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    // Watches the render graph's frame counter and requests the window close once
    // `maxFrames` frames have been rendered, so `--frames N` gives the launch-smoke test an
    // exit-0-without-crashing signal instead of requiring a human to close the window.
    // maxFrames == 0 means "run until the window is closed" (no limit).
    class FrameLimiterPass : public Engine::Render::RenderPass {
    public:
        FrameLimiterPass(Engine::Render::Window &window, uint64_t maxFrames)
            : m_window(window), m_maxFrames(maxFrames) {}

        const char *Name() const override { return "FrameLimiterPass"; }

        void Execute(Engine::Render::RenderContext &ctx) override {
            if (m_maxFrames > 0 && ctx.frame.frameIndex + 1 >= m_maxFrames)
                m_window.RequestClose();
        }

    private:
        Engine::Render::Window &m_window;
        uint64_t m_maxFrames;
    };

    uint64_t ParseFramesArg(int argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--frames" && i + 1 < argc)
                return static_cast<uint64_t>(std::stoull(argv[i + 1]));
        }
        return 0;
    }

    std::vector<PointVertex> ToWhitePointVertices(const std::vector<Eigen::Vector3f> &points) {
        std::vector<PointVertex> vertices;
        vertices.reserve(points.size());
        for (const Eigen::Vector3f &p : points)
            vertices.push_back({{p.x(), p.y(), p.z()}, {255, 255, 255, 255}});
        return vertices;
    }

} // namespace

int main(int argc, char **argv) {
    try {
        const uint64_t framesLimit = ParseFramesArg(argc, argv);

        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {1024, 768, "TSDF Point Cloud Viewer"};
        Engine::Render::Application app(descriptor);

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D extent = app.GetSwapChain().Extent();
        const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        camera.SetPerspective(60.0f * 3.14159265f / 180.0f, aspect, 0.05f, 100.0f);
        camera.SetOrbit({0.0f, 0.0f, 0.0f}, 6.0f);

        // Build the plane fixture and integrate it into a DirectionalTSDF, mirroring the
        // integrate step of tsdf_slice_debug's "plane" scene. Task 1 only renders the raw
        // input samples (white); the TSDF itself is built now so Task 2/3 can add the
        // extracted-cloud and slice point sets without re-plumbing this setup.
        Engine::Spatial::DirectionalTSDF tsdf;
        tsdf.Build(app.GetContext());
        const tsdf_fixtures::PlaneFixture plane = tsdf_fixtures::MakePlaneFixture();
        tsdf.Integrate(plane.points, plane.normals, Eigen::Vector3f(0.0f, 0.0f, 5.0f),
                       Eigen::Vector3f::Zero());

        const std::string shaderDir = VIEWER_SHADER_DIR;
        Engine::Render::RenderGraph graph;

        auto pointCloudPassOwned = std::make_unique<PointCloudPass>(
                app.GetContext(), app.GetSwapChain().Format(), shaderDir);
        PointCloudPass *pointCloudPass = pointCloudPassOwned.get();
        graph.AddPass(std::move(pointCloudPassOwned));
        graph.AddPass(std::make_unique<FrameLimiterPass>(app.GetWindow(), framesLimit));

        // Window backend is fixed to GLFW (ApplicationDescriptor default, not overridden
        // above), so the base Window& is always actually a GlfwWindow -- safe to downcast to
        // reach Handle(), which ImGui's GLFW backend needs.
        auto &glfwWindow = static_cast<Engine::Render::GlfwWindow &>(app.GetWindow());
        auto imGuiPassOwned = std::make_unique<ImGuiPass>(
                app.GetContext(), glfwWindow.Handle(), app.GetSwapChain().Format(),
                app.GetSwapChain().ImageCount());
        ImGuiPass *imGuiPass = imGuiPassOwned.get();
        // Added last: ImGuiPass's RenderingScope loads (does not clear) the swapchain image,
        // so it must run after PointCloudPass within the same frame to draw the panel over
        // the points rather than wiping them out.
        graph.AddPass(std::move(imGuiPassOwned));

        pointCloudPass->SetPointSize(4.0f);
        pointCloudPass->SetPointSet(0, ToWhitePointVertices(plane.points));
        imGuiPass->SetPointCount(static_cast<uint32_t>(plane.points.size()));

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
                               1.0f, 20.0f));
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

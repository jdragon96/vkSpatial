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
#include "TSDF/Backends/DirectionalTSDF.h"

#include "imgui.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    // Shared interactive state for the TSDF viewer. The UI (built in the ImGuiPass::SetUi
    // lambda below) mutates it; the app's between-frames rebuild reads it and writes back the
    // stats fields. `dirty` requests a full TSDF rebuild (scene/quality change); the show*
    // flags drive cheap PointCloudPass::SetVisible toggles with no rebuild.
    struct ViewerState {
        int scene = 0;         // 0 = plane, 1 = interproximal
        int maxDirections = 1; // K in IntegrationQuality (1..2)
        bool viewAngle = false;
        bool showInput = true, showExtracted = true, showSliceP = true, showSliceN = true;
        int extractColor = 0; // 0 = by direction bitmask, 1 = flat green
        bool dirty = true;    // request rebuild (starts true so the first frame builds)

        // Stats, filled by the rebuild and displayed in the panel.
        size_t nInput = 0, nExtracted = 0;
        float extractedZMean = 0.0f; // mean |z| of extracted points
        float crossP = 0.0f;         // +Z-layer (dir 4) central-column zero-crossing z
        float crossN = 0.0f;         // -Z-layer (dir 5) central-column zero-crossing z
    };

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

    // Initial scene: --scene plane|interproximal (default plane). 0 = plane, 1 = interproximal.
    int ParseSceneArg(int argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--scene" && i + 1 < argc)
                return std::string(argv[i + 1]) == "interproximal" ? 1 : 0;
        }
        return 0;
    }

    // Debug overrides for the initial ViewerState so the quality/color rebuild paths can be
    // exercised headlessly (the UI can still change them interactively afterwards):
    //   --maxdir N   initial IntegrationQuality.maxDirections
    //   --viewangle  enable view-angle weighting
    int ParseIntArg(int argc, char **argv, const char *name, int fallback) {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == name && i + 1 < argc)
                return std::stoi(argv[i + 1]);
        return fallback;
    }

    bool ParseFlag(int argc, char **argv, const char *name) {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == name)
                return true;
        return false;
    }

    constexpr float kVoxelSize = 0.1f;
    constexpr float kTruncation = 0.3f;

    // Rebuilds the whole TSDF + all point sets from the current ViewerState. MUST be called
    // between frames (never inside a RenderPass::Execute): Integrate()/PointCloud() and the
    // slice readback all submit their own GPU work / read back from the device.
    //
    // DirectionalTSDF has no Reset/Clear and re-integrating into a used volume accumulates,
    // so each rebuild constructs a fresh volume for a clean field. Prints the resulting stats
    // to stdout (the numeric cross-check oracle, since the visuals aren't capturable).
    void Rebuild(ViewerState &state, Engine::Core::Context &ctx,
                 std::unique_ptr<TSDF::DirectionalTSDF> &tsdf, PointCloudPass &pass) {
        using namespace TSDF;

        tsdf = std::make_unique<DirectionalTSDF>();
        tsdf->Build(ctx, kVoxelSize, kTruncation);

        IntegrationQuality quality;
        quality.maxDirections = static_cast<uint32_t>(state.maxDirections < 1 ? 1 : state.maxDirections);
        quality.dirExponent = 4;
        quality.viewAngleWeight = state.viewAngle;
        tsdf->SetIntegrationQuality(quality);

        std::vector<Eigen::Vector3f> inputPoints;
        float sliceZHalf = 0.0f;
        if (state.scene == 1) {
            const tsdf_fixtures::InterproximalFixture fx = tsdf_fixtures::MakeInterproximalFixture();
            // Two separate integrate passes, one per camera (matches tsdf_slice_debug).
            tsdf->Integrate(fx.pB, fx.nB, Eigen::Vector3f(0, 0, 5), Eigen::Vector3f::Zero());
            tsdf->Integrate(fx.pA, fx.nA, Eigen::Vector3f(0, 0, -5), Eigen::Vector3f::Zero());
            inputPoints = fx.allPoints;
            sliceZHalf = 0.5f; // matches tsdf_slice_debug's interproximal slice half-extent
        } else {
            const tsdf_fixtures::PlaneFixture plane = tsdf_fixtures::MakePlaneFixture();
            tsdf->Integrate(plane.points, plane.normals, Eigen::Vector3f(0, 0, 5),
                            Eigen::Vector3f::Zero());
            inputPoints = plane.points;
            sliceZHalf = 2.0f * kTruncation; // matches tsdf_slice_debug's plane slice half-extent
        }

        // Set 0: INPUT (white).
        std::vector<PointVertex> inputVerts;
        inputVerts.reserve(inputPoints.size());
        for (const Eigen::Vector3f &p : inputPoints)
            inputVerts.push_back({{p.x(), p.y(), p.z()}, {255, 255, 255, 255}});
        pass.SetPointSet(0, inputVerts);

        // Set 1: EXTRACTED (by direction bitmask or flat green).
        std::vector<PointVertex> extractedVerts;
        const std::vector<ExtractedPoint> &cloud = tsdf->PointCloud();
        extractedVerts.reserve(cloud.size());
        float sumAbsZ = 0.0f;
        for (const ExtractedPoint &e : cloud) {
            const tsdf_fixtures::Rgb c = state.extractColor == 0
                                                 ? tsdf_fixtures::dirColor(e.dirMask)
                                                 : tsdf_fixtures::Rgb{40, 220, 40};
            extractedVerts.push_back(
                    {{e.position.x(), e.position.y(), e.position.z()}, {c.r, c.g, c.b, 255}});
            sumAbsZ += std::fabs(e.position.z());
        }
        pass.SetPointSet(1, extractedVerts);

        // Sets 2/3: SLICE +Z (dir 4) / SLICE -Z (dir 5) of the TSDF field at y=0.
        const tsdf_fixtures::SliceResult sliceP =
                tsdf_fixtures::BuildSlice(*tsdf, 4, kVoxelSize, 2.0f, sliceZHalf);
        const tsdf_fixtures::SliceResult sliceN =
                tsdf_fixtures::BuildSlice(*tsdf, 5, kVoxelSize, 2.0f, sliceZHalf);
        pass.SetPointSet(2, sliceP.points);
        pass.SetPointSet(3, sliceN.points);

        // Visibility from the show* flags.
        pass.SetVisible(0, state.showInput);
        pass.SetVisible(1, state.showExtracted);
        pass.SetVisible(2, state.showSliceP);
        pass.SetVisible(3, state.showSliceN);

        // Stats.
        state.nInput = inputPoints.size();
        state.nExtracted = cloud.size();
        state.extractedZMean = cloud.empty() ? 0.0f : sumAbsZ / static_cast<float>(cloud.size());
        state.crossP = sliceP.crossZ;
        state.crossN = sliceN.crossZ;
        state.dirty = false;

        std::cout << "[rebuild] scene=" << (state.scene == 1 ? "interproximal" : "plane")
                  << " maxDir=" << quality.maxDirections
                  << " viewAngle=" << (quality.viewAngleWeight ? 1 : 0)
                  << " extractColor=" << (state.extractColor == 0 ? "dir" : "green")
                  << " | nInput=" << state.nInput << " nExtracted=" << state.nExtracted
                  << " |z|mean=" << state.extractedZMean << " crossP(dir4)=" << state.crossP
                  << " crossN(dir5)=" << state.crossN << std::endl;
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

        // The TSDF volume + all point sets are (re)built by Rebuild() between frames whenever
        // state.dirty is set (see the render loop below). state.dirty starts true, so the
        // first loop iteration performs the initial build before any frame is in flight. The
        // volume is a unique_ptr because Rebuild constructs a fresh one per rebuild (no
        // Reset/Clear on DirectionalTSDF; re-integrating a used volume would accumulate).
        ViewerState state;
        state.scene = ParseSceneArg(argc, argv);
        state.maxDirections = std::clamp(ParseIntArg(argc, argv, "--maxdir", 1), 1, 2);
        state.viewAngle = ParseFlag(argc, argv, "--viewangle");
        std::unique_ptr<TSDF::DirectionalTSDF> tsdf;

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
        // Panel body: exactly what used to live in ImGuiPass.cpp::Execute (Task 1 refactor),
        // now supplied via the generic ImGuiPass::SetUi callback. `state` and `pointCloudPass`
        // are app-scope locals that outlive the render loop below.
        imGuiPass->SetUi([&state, pointCloudPass]() {
            ImGui::Begin("TSDF Viewer");

            ImGui::SeparatorText("Scene");
            if (ImGui::Combo("scene", &state.scene, "plane\0interproximal\0\0"))
                state.dirty = true;

            ImGui::SeparatorText("Integration quality");
            if (ImGui::SliderInt("maxDirections", &state.maxDirections, 1, 2))
                state.dirty = true;
            if (ImGui::Checkbox("view-angle weight", &state.viewAngle))
                state.dirty = true;
            if (ImGui::Combo("extract color", &state.extractColor, "direction\0green\0\0"))
                state.dirty = true;

            // Layer toggles are cheap: flip PointCloudPass visibility immediately, no rebuild.
            ImGui::SeparatorText("Layers");
            if (ImGui::Checkbox("input (white)", &state.showInput))
                pointCloudPass->SetVisible(0, state.showInput);
            if (ImGui::Checkbox("extracted", &state.showExtracted))
                pointCloudPass->SetVisible(1, state.showExtracted);
            if (ImGui::Checkbox("slice +Z (dir 4)", &state.showSliceP))
                pointCloudPass->SetVisible(2, state.showSliceP);
            if (ImGui::Checkbox("slice -Z (dir 5)", &state.showSliceN))
                pointCloudPass->SetVisible(3, state.showSliceN);

            ImGui::SeparatorText("Stats");
            ImGui::Text("input points:     %zu", state.nInput);
            ImGui::Text("extracted points: %zu", state.nExtracted);
            ImGui::Text("extracted |z| mean: %.5f mm", state.extractedZMean);
            ImGui::Text("+Z crossing (dir4): %.4f mm", state.crossP);
            ImGui::Text("-Z crossing (dir5): %.4f mm", state.crossN);

            ImGui::Spacing();
            if (ImGui::Button("Re-integrate"))
                state.dirty = true;

            ImGui::End();
        });

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

        // Manual render loop (replicates Engine::Render::Application::Run) so the TSDF
        // rebuild can run BETWEEN frames. Application::Run() exposes no per-frame hook, and
        // Rebuild() submits its own GPU compute + reads back from the device -- which must
        // never happen inside a RenderPass::Execute (mid graphics command buffer). The UI
        // only sets state.dirty; the actual rebuild happens here, before BeginFrame, guarded
        // by a device-idle wait so no graphics frame is in flight while vertex buffers are
        // re-uploaded (the Renderer is single-frame-in-flight). No engine change required.
        Engine::Core::Context &ctx = app.GetContext();
        try {
            while (!app.GetWindow().ShouldClose()) {
                app.GetWindow().PollEvents();

                const VkExtent2D size = app.GetWindow().FramebufferSize();
                if (size.width == 0 || size.height == 0)
                    continue;

                if (state.dirty) {
                    vkDeviceWaitIdle(ctx.device);
                    Rebuild(state, ctx, tsdf, *pointCloudPass);
                }

                if (!app.GetRenderer().BeginFrame(size.width, size.height))
                    continue;
                app.GetRenderer().Render(app.GetView());
                app.GetRenderer().EndFrame();
            }
        } catch (...) {
            vkDeviceWaitIdle(ctx.device);
            throw;
        }
        vkDeviceWaitIdle(ctx.device);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

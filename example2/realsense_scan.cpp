// RealSense scan pipeline — a depth camera driving the full Track/Map/Render pipeline, with the
// surface accumulating on screen as you sweep the sensor over an object.
//
//   depth frames -> BackprojectDepth -> ICP against the model -> TSDF integrate -> extracted surface
//   \___________________________________/  \_______________/     \____________/
//        ReconstructionThread                RegistrationThread    IntegrationThread
//
// This is the first source in the repo for which tracking is a real problem. The synthetic
// scan_out frames each cover the whole scene from every side, so there is no viewpoint to
// recover and "identity" is the only tracker that works on them. A depth camera sees one side of
// one thing at a time, which is exactly what point-to-plane ICP is for -- so "icp" is the default
// here, and the residual RMSE in the stats panel is a real measurement rather than a zero.
//
//   ./realsense_scan                          live camera
//   ./realsense_scan --record ./capture       live camera, saving raw depth as it goes
//   ./realsense_scan --replay ./capture       replay a recording, no hardware needed
//        [--tracker icp|icp-cpu|identity|global] [--voxel 0.01] [--truncation 0.03]
//        [--width 640] [--height 480] [--fps 30]
//
// Defaults are indoor scale. MapConfig ships with baseVoxel 0.5 m for the 190 m synthetic scenes;
// a D435 works between 0.3 and 5 m, where 0.5 m voxels would quantise a whole object into a
// handful of cells.

#include "ImGuiPass.h"
#include "PointCloudPass.h"

#include "Engine/Core/Context.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"
#include "Pipeline/Pipeline.h"
#include "Pipeline/Reconstruction/DepthCameraFrameSource.h"
#include "Pipeline/Reconstruction/DepthRecording.h"
#include "Pipeline/Reconstruction/RealSenseDepthProvider.h"
#include "Pipeline/Registration/Tracker.h"
#include "utilities/ArgParser.h"

#include "imgui.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ep = Pipeline;
using Eigen::Vector3f;

namespace {

    constexpr int kSetSurface = 0;
    constexpr int kSetNew = 1;

    enum class EColorMode { Normal, Age, Weight };

    // The world frame is the first camera's frame, which is the image convention: +y down, +z
    // forward. Rotating 180 degrees about x puts y up and the scene in front of a viewer looking
    // down -z. A rotation, not a mirror -- a single-axis negation would flip handedness and invert
    // every displayed normal.
    inline Vector3f SensorToView(const Vector3f &v) { return Vector3f(v.x(), -v.y(), -v.z()); }

    struct Color {
        uint8_t r, g, b;
    };

    Color Ramp(float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        const float x = t * 4.0f;
        const int band = std::min(3, int(x));
        const float f = x - float(band);
        static constexpr float kStops[5][3] = {
                {0.10f, 0.20f, 0.90f}, {0.10f, 0.85f, 0.90f}, {0.15f, 0.85f, 0.20f},
                {0.95f, 0.90f, 0.15f}, {0.95f, 0.20f, 0.15f}};
        auto mix = [&](int c) {
            return kStops[band][c] + f * (kStops[band + 1][c] - kStops[band][c]);
        };
        return {uint8_t(mix(0) * 255.0f), uint8_t(mix(1) * 255.0f), uint8_t(mix(2) * 255.0f)};
    }

    // A voxel's centre is not where the surface is: the surface is the zero crossing, which the
    // stored tsdf locates inside the voxel. Rendering centres instead would quantise the surface
    // to the voxel grid and hide exactly the sub-voxel accuracy the TSDF is accumulating.
    inline Vector3f SurfacePosition(const TSDFVoxel &voxel, float truncationDistance) {
        return voxel.center - voxel.normal * (voxel.tsdf * truncationDistance);
    }

    void BuildModelPoints(const ep::ModelSnapshot &snapshot, EColorMode mode, float minimumWeight,
                          float weightRampMax, std::vector<PointVertex> &surface,
                          std::vector<PointVertex> &fresh) {
        surface.clear();
        fresh.clear();
        surface.reserve(snapshot.entries.size());

        const float frameSpan = std::max(1.0f, float(snapshot.processedFrame));
        for (std::size_t i = 0; i < snapshot.entries.size(); ++i) {
            const TSDFVoxel &voxel = snapshot.entries[i];
            // Low-weight voxels have seen too little evidence to be surface yet. They are the ones
            // that become holes at extraction, so hiding them here shows the same surface a mesh
            // would give you.
            if (voxel.weight < minimumWeight) continue;

            const Vector3f position = SensorToView(SurfacePosition(voxel, snapshot.truncationDistance));
            Color color{200, 205, 215};
            if (mode == EColorMode::Normal) {
                const Vector3f n = SensorToView(voxel.normal) * 0.5f + Vector3f::Constant(0.5f);
                color = {uint8_t(std::clamp(n.x(), 0.0f, 1.0f) * 255.0f),
                         uint8_t(std::clamp(n.y(), 0.0f, 1.0f) * 255.0f),
                         uint8_t(std::clamp(n.z(), 0.0f, 1.0f) * 255.0f)};
            } else if (mode == EColorMode::Age) {
                color = Ramp(float(voxel.firstFrame) / frameSpan);
            } else {
                color = Ramp(voxel.weight / std::max(1e-3f, weightRampMax));
            }

            const PointVertex vertex{{position.x(), position.y(), position.z()},
                                     {color.r, color.g, color.b, 255}};
            surface.push_back(vertex);
            // isNew is parallel to entries: voxels first filled on this frame. Seeing them
            // separately is how you tell a sensor that is still adding surface from one that is
            // only re-observing what it already has.
            if (i < snapshot.isNew.size() && snapshot.isNew[i])
                fresh.push_back({{position.x(), position.y(), position.z()}, {255, 90, 60, 255}});
        }
    }

    std::unique_ptr<ep::IDepthProvider> OpenDevice(int width, int height, int fps) {
#ifdef VKBVH_HAS_REALSENSE
        return std::make_unique<ep::RealSenseDepthProvider>(width, height, fps);
#else
        (void) width;
        (void) height;
        (void) fps;
        throw std::runtime_error("realsense_scan: built without librealsense2. Install it and "
                                 "reconfigure, or scan a recording with --replay <dir>.");
#endif
    }

} // namespace

namespace {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Interrupt handling
    ///////////////////////////////////////////////////////////////////////////////////////////////

    // Ctrl-C is how a live viewer actually gets stopped, and a signal that kills the process
    // unwinds nothing: no destructor runs, so the camera is left streaming with its USB interface
    // claimed and the next run cannot open it. Catching the signal turns it into a normal exit
    // through the same shutdown path as closing the window.
    //
    // A handler may only touch a volatile sig_atomic_t, so it sets a flag and the render loop acts
    // on it. The default disposition is restored first, so a second Ctrl-C still kills a process
    // that has become wedged.
    volatile std::sig_atomic_t g_interrupted = 0;

    void HandleInterrupt(int signalNumber) {
        std::signal(signalNumber, SIG_DFL);
        g_interrupted = 1;
    }

    void InstallInterruptHandler() {
        std::signal(SIGINT, HandleInterrupt);
        std::signal(SIGTERM, HandleInterrupt);
    }

} // namespace

int main(int argc, char **argv) {
    InstallInterruptHandler();
    try {
        auto arg = util::BuildArgParser(argc, argv)
                           .Option("--replay")
                           .Option("--record")
                           .Option("--tracker", "icp")
                           .Option("--voxel", 0.01)
                           .Option("--truncation", 0.03)
                           .Option("--width", 640)
                           .Option("--height", 480)
                           .Option("--fps", 30);

        const std::string replayDirectory = arg.Value("--replay");
        const std::string recordDirectory = arg.Value("--record");
        const std::string trackerName = arg.Value("--tracker");
        const bool live = replayDirectory.empty();
        if (!live && !recordDirectory.empty())
            throw std::runtime_error("realsense_scan: --replay and --record are exclusive.");

        const int width = arg.ValueInt("--width");
        const int height = arg.ValueInt("--height");
        const int fps = arg.ValueInt("--fps");

        ep::TrackerRegistry registry = ep::TrackerRegistry::Default();

        ep::Pipeline::Config config;
        config.map.baseVoxel = arg.ValueFloat("--voxel");
        config.map.truncation = arg.ValueFloat("--truncation");
        config.map.submap = false; // one level until a plain scan is known good
        config.acquisition.type = ep::EAcquisitionType::DepthCamera;
        // A camera keeps producing whether or not the map keeps up, so live must drop to bound
        // latency. A recording waits, so replaying it losslessly costs only wall-clock.
        config.acquisition.realTime = live;

        // Built fresh on every stage build, including each Reconfigure. Capturing an already-open
        // device instead would hand the rebuilt pipeline a source the previous one has closed.
        config.acquisition.makeSource = [=]() -> std::unique_ptr<ep::IFrameSource> {
            std::unique_ptr<ep::IDepthProvider> device =
                    live ? OpenDevice(width, height, fps)
                         : std::make_unique<ep::RecordedDepthProvider>(replayDirectory);
            if (!recordDirectory.empty())
                device = std::make_unique<ep::DepthRecorder>(std::move(device), recordDirectory);
            return std::make_unique<ep::DepthCameraFrameSource>(std::move(device));
        };

        std::printf("source    : %s\n", live ? "live device" : replayDirectory.c_str());
        std::printf("tracker   : %s\n", trackerName.c_str());
        std::printf("map       : voxel %.4f m, truncation %.4f m\n", config.map.baseVoxel,
                    config.map.truncation);

        // Constructed before the window on purpose: the Pipeline constructor builds its stages,
        // which calls makeSource synchronously. A missing camera or an occupied recording
        // directory therefore reports as one CLI line, with no window ever appearing -- and the
        // device is opened exactly once, which a separate pre-flight probe would not manage.
        ep::Pipeline pipeline(config, registry.Create(trackerName));

        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {1400, 900, "RealSense Scan Pipeline"};
        Engine::Render::Application app(descriptor);
        Engine::Core::Context &context = app.GetContext();

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D extent = app.GetSwapChain().Extent();
        camera.SetPerspective(60.0f * 3.14159265f / 180.0f,
                              float(extent.width) / float(extent.height), 0.02f, 60.0f);
        camera.SetOrbit(vkMath::Vec3(0.0f, 0.0f, -1.0f), 1.8f);

        Engine::Render::RenderGraph graph;
        auto pointsOwned = std::make_unique<PointCloudPass>(context, app.GetSwapChain().Format(),
                                                            VOXDBG_SHADER_DIR);
        PointCloudPass *points = pointsOwned.get();
        graph.AddPass(std::move(pointsOwned));
        auto &glfwWindow = static_cast<Engine::Render::GlfwWindow &>(app.GetWindow());
        auto imguiOwned = std::make_unique<ImGuiPass>(context, glfwWindow.Handle(),
                                                      app.GetSwapChain().Format(),
                                                      app.GetSwapChain().ImageCount());
        ImGuiPass *imgui = imguiOwned.get();
        graph.AddPass(std::move(imguiOwned));
        points->SetPointSize(2.0f);

        struct State {
            bool showSurface = true, showNew = true;
            EColorMode colorMode = EColorMode::Normal;
            float pointSize = 2.0f;
            float minimumWeight = 1.0f;
            float weightRampMax = 20.0f;
            bool paused = false;
        } state;

        auto applyVisibility = [&] {
            points->SetVisible(kSetSurface, state.showSurface);
            points->SetVisible(kSetNew, state.showNew);
        };
        applyVisibility();

        std::vector<PointVertex> surfaceVertices, newVertices;
        std::shared_ptr<const ep::ModelSnapshot> snapshot;
        std::string workerError;

        imgui->SetUi([&] {
            ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(330.0f, 0.0f), ImGuiCond_Always);
            ImGui::Begin("Scan", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse);
            if (!workerError.empty()) ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "%s", workerError.c_str());
            if (ImGui::Button(state.paused ? "resume" : "pause")) {
                state.paused = !state.paused;
                pipeline.SetPaused(state.paused);
            }
            ImGui::SeparatorText("Layers");
            if (ImGui::Checkbox("surface", &state.showSurface)) applyVisibility();
            if (ImGui::Checkbox("new this frame", &state.showNew)) applyVisibility();

            ImGui::SeparatorText("Display");
            int mode = int(state.colorMode);
            if (ImGui::Combo("color", &mode, "normal\0age\0weight\0")) state.colorMode = EColorMode(mode);
            ImGui::PushItemWidth(ImGui::CalcItemWidth() / 3.0f);
            ImGui::InputFloat("point size", &state.pointSize, 0.0f, 0.0f, "%.4f");
            points->SetPointSize(std::clamp(state.pointSize, 1.0f, 16.0f));
            // The same threshold that predicts holes in an extracted mesh: below it a voxel has
            // not seen enough evidence to be surface.
            ImGui::InputFloat("min weight", &state.minimumWeight, 0.0f, 0.0f, "%.4f");
            if (state.colorMode == EColorMode::Weight)
                ImGui::InputFloat("weight ramp", &state.weightRampMax, 0.0f, 0.0f, "%.4f");
            ImGui::PopItemWidth();
            ImGui::End();

            // Stage stats live in their own panel on the right, one section per pipeline stage.
            const ep::PipelineStats stats = pipeline.GetStats();
            const float panelWidth = 330.0f;
            ImGui::SetNextWindowPos(ImVec2(float(app.GetWindow().FramebufferSize().width) /
                                                   ImGui::GetIO().DisplayFramebufferScale.x -
                                                   panelWidth - 10.0f,
                                           10.0f),
                                    ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(panelWidth, 0.0f), ImGuiCond_Always);
            ImGui::Begin("Pipeline", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse);

            ImGui::SeparatorText("Acquire");
            ImGui::Text("frames        %llu", (unsigned long long) stats.acquiredFrames);
            ImGui::Text("per frame     %.2f ms", stats.acquireMsAvg);
            ImGui::Text("queued        %zu", stats.captureDepth);

            ImGui::SeparatorText("Track");
            ImGui::Text("tracker       %s", trackerName.c_str());
            ImGui::Text("frames        %llu", (unsigned long long) stats.alignedFrames);
            ImGui::Text("per frame     %.2f ms", stats.alignMsAvg);
            // The number that says whether tracking is working. It is a real residual here, unlike
            // on the synthetic scans where identity is the only viable tracker.
            ImGui::Text("residual rmse %.5f m", stats.trackerRmseAvg);
            ImGui::Text("queued        %zu  dropped %zu", stats.trackDepth, stats.trackDropped);

            ImGui::SeparatorText("Integrate");
            ImGui::Text("frames        %llu", (unsigned long long) stats.integratedFrames);
            ImGui::Text("per frame     %.2f ms", stats.integrateMsAvg);

            ImGui::SeparatorText("Map");
            if (snapshot) {
                ImGui::Text("voxels        %zu", snapshot->entries.size());
                ImGui::Text("shown         %zu", surfaceVertices.size());
                ImGui::Text("new this fr.  %zu", newVertices.size());
                ImGui::Text("tiles         %u base / %u detail", snapshot->baseTiles,
                            snapshot->detailTiles);
                ImGui::Text("hash          %llu / %llu",
                            (unsigned long long) snapshot->map.filledCount,
                            (unsigned long long) snapshot->map.hashCapacity);
                ImGui::Text("insert fails  %llu",
                            (unsigned long long) snapshot->map.insertFailureCount);
                if (snapshot->windowLimitRefusals)
                    ImGui::TextColored(ImVec4(1, 0.7f, 0.2f, 1), "refused points %u",
                                       snapshot->windowLimitRefusals);
            } else {
                ImGui::TextDisabled("no model yet");
            }
            ImGui::Text("display       %.1f fps", double(ImGui::GetIO().Framerate));
            ImGui::End();
        });

        app.GetView().SetScene(&scene);
        app.GetView().SetCamera(&camera);
        app.GetView().SetRenderGraph(&graph);

        Engine::Render::MouseListenerGroup mouse(app.GetWindow().Mouse());
        mouse.Add(Engine::Render::MouseEventType::Drag, [&](Engine::Render::MouseEvent &e) {
            if (ImGui::GetIO().WantCaptureMouse) return;
            if (e.button != Engine::Render::MouseButton::Left) return;
            const VkExtent2D size = app.GetWindow().FramebufferSize();
            if (size.width == 0 || size.height == 0) return;
            if (!camera.IsTrackballDragging()) {
                camera.BeginTrackballDrag(e.x, e.y, int(size.width), int(size.height));
                return;
            }
            camera.DragTrackball(e.x, e.y, int(size.width), int(size.height));
            e.handled = true;
        });
        mouse.Add(Engine::Render::MouseEventType::ButtonUp, [&](Engine::Render::MouseEvent &e) {
            if (e.button == Engine::Render::MouseButton::Left) camera.EndTrackballDrag();
        });
        mouse.Add(Engine::Render::MouseEventType::Scroll, [&](Engine::Render::MouseEvent &e) {
            if (ImGui::GetIO().WantCaptureMouse) return;
            camera.SetDistance(std::clamp(
                    camera.GetDistance() * std::exp(float(-e.scrollY) * 0.08f), 0.05f, 40.0f));
            e.handled = true;
        });
        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape) app.GetWindow().RequestClose();
        });

        pipeline.Start();

        int uploadedFrame = -1;
        EColorMode uploadedColorMode = state.colorMode;
        float uploadedMinimumWeight = state.minimumWeight;
        try {
            while (!app.GetWindow().ShouldClose() && !g_interrupted) {
                app.GetWindow().PollEvents();
                const VkExtent2D size = app.GetWindow().FramebufferSize();
                if (size.width == 0 || size.height == 0) continue;

                // A worker stage that throws -- a camera unplugged mid-scan, a full hash -- must
                // surface in the window rather than deadlock the render loop waiting for frames
                // that will never come.
                if (workerError.empty()) {
                    try {
                        pipeline.CheckErrors();
                    } catch (const std::exception &e) {
                        workerError = e.what();
                    }
                }

                std::shared_ptr<const ep::ModelSnapshot> latest = pipeline.LatestModel();
                const bool modelAdvanced = latest && (!snapshot || latest->processedFrame != uploadedFrame);
                const bool displayChanged = uploadedColorMode != state.colorMode ||
                                            uploadedMinimumWeight != state.minimumWeight;
                if (latest && (modelAdvanced || displayChanged)) {
                    snapshot = latest;
                    BuildModelPoints(*snapshot, state.colorMode, state.minimumWeight,
                                     state.weightRampMax, surfaceVertices, newVertices);
                    // SetPointSet may reallocate a buffer the previous frame's command buffer is
                    // still reading; one idle covers both uploads.
                    vkDeviceWaitIdle(context.device);
                    points->SetPointSet(kSetSurface, surfaceVertices);
                    points->SetPointSet(kSetNew, newVertices);
                    uploadedFrame = snapshot->processedFrame;
                    uploadedColorMode = state.colorMode;
                    uploadedMinimumWeight = state.minimumWeight;
                }

                if (!app.GetRenderer().BeginFrame(size.width, size.height)) continue;
                app.GetRenderer().Render(app.GetView());
                app.GetRenderer().EndFrame();
            }
        } catch (...) {
            pipeline.Stop();
            vkDeviceWaitIdle(context.device);
            throw;
        }
        pipeline.Stop();
        vkDeviceWaitIdle(context.device);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

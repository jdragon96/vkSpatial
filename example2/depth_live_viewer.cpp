// Live depth-camera viewer — watch what the sensor actually sees, as an oriented point cloud.
//
// The whole point of this tool is that it renders the SAME thing the reconstruction pipeline
// consumes: every frame goes through BackprojectDepth, so what is on screen is exactly the
// `Frame` a tracker and a TSDF would receive. A dropped boundary, an inverted normal, or a
// filter threshold set too tight is visible here rather than three stages downstream.
//
// The source is an IDepthProvider, so the same viewer serves a live device and a recording:
//
//   ./depth_live_viewer                              live camera (640x480 @ 30, USB2-safe)
//   ./depth_live_viewer --record ./capture           live camera, saving raw depth as it runs
//   ./depth_live_viewer --replay ./capture           replay a recording, no hardware needed
//        [--width 640] [--height 480] [--fps 30] [--loop]
//
// The depth-discontinuity filter is editable while it runs. That is the only way to pick its
// two thresholds honestly: too loose welds foreground to background across object boundaries,
// too tight eats the edges of real surfaces, and both look fine in a summary statistic.

#include "ImGuiPass.h"
#include "PointCloudPass.h"

#include "Engine/Core/Context.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"
#include "Pipeline/Acquisition/DepthCameraFrameSource.h"
#include "Pipeline/Acquisition/DepthRecording.h"
#include "Pipeline/Acquisition/D435DepthProvider.h"
#include "utilities/ArgParser.h"

#include "imgui.h"

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Vector3f;
using Pipeline::BackprojectDepth;
using Pipeline::CameraIntrinsics;
using Pipeline::DepthFilterOptions;
using Pipeline::DepthFrame;
using Pipeline::IDepthProvider;

namespace {

    /// Point sets. Kept as named constants because visibility and colour are set from three
    /// different places (construction, the UI callback, the per-frame upload).
    constexpr int kSetSurface = 0;
    constexpr int kSetNormals = 1;
    constexpr int kSetFrustum = 2;

    enum class EColorMode { Normal, Depth, Flat };

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Display transform
    ///////////////////////////////////////////////////////////////////////////////////////////////

    // Depth-camera coordinates are the image convention: +x right, +y DOWN, +z forward into the
    // scene. A viewer with +y up would show every scene upside down and behind itself. Rotating
    // 180 degrees about x maps the sensor frame to the viewing frame: y becomes up, and the
    // scene lands in front of a camera looking down -z.
    //
    // This is a rotation, not a mirror -- determinant +1 -- so chirality survives and a normal
    // transforms by the same matrix as a position. Mirroring instead (say, negating y alone)
    // would flip the handedness and make every normal point the wrong way, which is precisely
    // the class of bug this viewer exists to catch.
    inline Vector3f SensorToView(const Vector3f &v) { return Vector3f(v.x(), -v.y(), -v.z()); }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Colour
    ///////////////////////////////////////////////////////////////////////////////////////////////

    struct Color {
        uint8_t r, g, b;
    };

    // Blue -> cyan -> green -> yellow -> red over t in [0,1]. A ramp with distinguishable
    // mid-tones; a plain grey ramp hides the centimetre-scale steps that matter here.
    Color DepthRamp(float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        const float x = t * 4.0f;
        const int band = std::min(3, int(x));
        const float f = x - float(band);
        static constexpr float kStops[5][3] = {
                {0.10f, 0.20f, 0.90f}, {0.10f, 0.85f, 0.90f}, {0.15f, 0.85f, 0.20f},
                {0.95f, 0.90f, 0.15f}, {0.95f, 0.20f, 0.15f}};
        auto mix = [&](int channel) {
            return kStops[band][channel] + f * (kStops[band + 1][channel] - kStops[band][channel]);
        };
        return {uint8_t(mix(0) * 255.0f), uint8_t(mix(1) * 255.0f), uint8_t(mix(2) * 255.0f)};
    }

    Color NormalColor(const Vector3f &viewNormal) {
        const Vector3f c = viewNormal * 0.5f + Vector3f::Constant(0.5f);
        return {uint8_t(std::clamp(c.x(), 0.0f, 1.0f) * 255.0f),
                uint8_t(std::clamp(c.y(), 0.0f, 1.0f) * 255.0f),
                uint8_t(std::clamp(c.z(), 0.0f, 1.0f) * 255.0f)};
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Per-frame geometry
    ///////////////////////////////////////////////////////////////////////////////////////////////

    struct FrameStats {
        std::size_t validPixelCount = 0; // depth > 0 in the raw image
        std::size_t pointCount = 0;      // survived back-projection AND the discontinuity filter
        float nearestDepth = 0.0f, farthestDepth = 0.0f;
        double grabMilliseconds = 0.0, backprojectMilliseconds = 0.0;
    };

    // The gap between validPixelCount and pointCount is the filter's cost, and it is the number
    // to watch while tuning: a filter that is rejecting most of a clean scene is mis-set.
    void MeasureRawDepth(const DepthFrame &frame, FrameStats &stats) {
        stats.validPixelCount = 0;
        stats.nearestDepth = 0.0f;
        stats.farthestDepth = 0.0f;
        for (const float z: frame.depth) {
            if (z <= 0.0f) continue;
            if (stats.validPixelCount == 0) stats.nearestDepth = stats.farthestDepth = z;
            stats.nearestDepth = std::min(stats.nearestDepth, z);
            stats.farthestDepth = std::max(stats.farthestDepth, z);
            ++stats.validPixelCount;
        }
    }

    void BuildSurfacePoints(const Pipeline::Frame &frame, EColorMode mode, float rampNear,
                            float rampFar, std::vector<PointVertex> &out) {
        out.clear();
        out.reserve(frame.pts.size());
        const float span = std::max(1e-4f, rampFar - rampNear);
        for (std::size_t i = 0; i < frame.pts.size(); ++i) {
            const Vector3f position = SensorToView(frame.pts[i]);
            Color color{200, 205, 215};
            if (mode == EColorMode::Normal) color = NormalColor(SensorToView(frame.nrm[i]));
            else if (mode == EColorMode::Depth)
                color = DepthRamp((frame.pts[i].z() - rampNear) / span);
            out.push_back({{position.x(), position.y(), position.z()},
                           {color.r, color.g, color.b, 255}});
        }
    }

    // Short line segments along each normal, drawn as points. Sampled, because one segment per
    // surface point would bury the surface itself under its own hair.
    void BuildNormalTufts(const Pipeline::Frame &frame, int stride, float length,
                          std::vector<PointVertex> &out) {
        out.clear();
        if (stride <= 0) return;
        constexpr int kSamplesPerTuft = 4;
        out.reserve(frame.pts.size() / std::size_t(stride) * kSamplesPerTuft);
        for (std::size_t i = 0; i < frame.pts.size(); i += std::size_t(stride)) {
            const Vector3f base = SensorToView(frame.pts[i]);
            const Vector3f direction = SensorToView(frame.nrm[i]);
            for (int s = 1; s <= kSamplesPerTuft; ++s) {
                const Vector3f p = base + direction * (length * float(s) / float(kSamplesPerTuft));
                out.push_back({{p.x(), p.y(), p.z()}, {255, 90, 60, 255}});
            }
        }
    }

    // The sensor's viewing frustum at a fixed range, so the scene has a reference for where the
    // camera is and which way it looks. Built from the intrinsics, so a wrong cx/cy is visible
    // as an off-centre frustum rather than as a silently skewed reconstruction.
    void BuildFrustum(const CameraIntrinsics &intrinsics, float range,
                      std::vector<PointVertex> &out) {
        out.clear();
        if (intrinsics.fx <= 0.0f || intrinsics.fy <= 0.0f) return;
        auto corner = [&](float u, float v) {
            return SensorToView(Vector3f((u - intrinsics.cx) / intrinsics.fx * range,
                                         (v - intrinsics.cy) / intrinsics.fy * range, range));
        };
        const Vector3f apex = SensorToView(Vector3f::Zero());
        const Vector3f corners[4] = {corner(0.0f, 0.0f), corner(float(intrinsics.width), 0.0f),
                                     corner(float(intrinsics.width), float(intrinsics.height)),
                                     corner(0.0f, float(intrinsics.height))};
        constexpr int kSamplesPerEdge = 48;
        auto edge = [&](const Vector3f &a, const Vector3f &b) {
            for (int s = 0; s <= kSamplesPerEdge; ++s) {
                const Vector3f p = a + (b - a) * (float(s) / float(kSamplesPerEdge));
                out.push_back({{p.x(), p.y(), p.z()}, {90, 220, 255, 255}});
            }
        };
        for (int c = 0; c < 4; ++c) {
            edge(apex, corners[c]);
            edge(corners[c], corners[(c + 1) % 4]);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Source selection
    ///////////////////////////////////////////////////////////////////////////////////////////////

    std::unique_ptr<IDepthProvider> OpenDevice(int width, int height, int fps) {
#ifdef VKBVH_HAS_REALSENSE
        return std::make_unique<Pipeline::D435DepthProvider>(
                Realsense::D435StreamOptions{width, height, fps, false, "high-accuracy"});
#else
        (void) width;
        (void) height;
        (void) fps;
        throw std::runtime_error(
                "depth_live_viewer: built without librealsense2, so there is no live device to "
                "open. Install it and reconfigure, or view a recording with --replay <dir>.");
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
                           .Option("--width", 640)  // 640x480 @ 30 is the highest depth mode both
                           .Option("--height", 480) // USB 2.1 and USB 3.x offer; 848x480 tops out
                           .Option("--fps", 30);    // at 10 Hz over USB 2.1.
        const bool loop = util::HasFlag(argc, argv, "--loop");
        const std::string replayDirectory = arg.Value("--replay");
        const std::string recordDirectory = arg.Value("--record");
        const bool live = replayDirectory.empty();

        if (!live && !recordDirectory.empty())
            throw std::runtime_error("depth_live_viewer: --replay and --record are exclusive; "
                                     "a recording is already on disk.");

        // The source is opened before the window: a missing camera should print one clear line,
        // not flash an empty window first.
        std::unique_ptr<IDepthProvider> source;
        if (live) {
            source = OpenDevice(arg.ValueInt("--width"), arg.ValueInt("--height"),
                                arg.ValueInt("--fps"));
            if (!recordDirectory.empty())
                source = std::make_unique<Pipeline::DepthRecorder>(std::move(source),
                                                                   recordDirectory);
        } else {
            source = std::make_unique<Pipeline::RecordedDepthProvider>(replayDirectory);
        }

        const CameraIntrinsics intrinsics = source->Intrinsics();
        const std::string sourceLabel =
                live ? (recordDirectory.empty() ? std::string("live device")
                                                : "live device -> " + recordDirectory)
                     : "replay " + replayDirectory;
        std::printf("source    : %s\n", sourceLabel.c_str());
        std::printf("intrinsics: %dx%d  fx %.2f fy %.2f  cx %.2f cy %.2f\n", intrinsics.width,
                    intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy);

        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {1280, 800, "Depth Live Viewer"};
        Engine::Render::Application app(descriptor);
        Engine::Core::Context &context = app.GetContext();

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D extent = app.GetSwapChain().Extent();
        camera.SetPerspective(60.0f * 3.14159265f / 180.0f,
                              float(extent.width) / float(extent.height), 0.02f, 60.0f);
        // The scene sits down -z after SensorToView, so orbit a target inside it rather than the
        // origin -- the origin is the sensor itself, at the very edge of the geometry.
        constexpr float kOrbitDepth = 1.5f;
        camera.SetOrbit(vkMath::Vec3(0.0f, 0.0f, -kOrbitDepth), 2.5f);

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
            bool streaming = true;
            bool showSurface = true, showNormals = false, showFrustum = true;
            EColorMode colorMode = EColorMode::Normal;
            float pointSize = 2.0f;
            float rampNear = 0.3f, rampFar = 5.0f;
            int normalStride = 400;
            float normalLength = 0.04f;
            DepthFilterOptions filter; // live-editable: see the header comment
            bool streamEnded = false;
            int frameIndex = 0;
        } state;

        FrameStats stats;
        std::vector<PointVertex> surfaceVertices, normalVertices, frustumVertices;

        BuildFrustum(intrinsics, 1.0f, frustumVertices);
        vkDeviceWaitIdle(context.device);
        points->SetPointSet(kSetFrustum, frustumVertices);

        // Every SetPointSet may reallocate the vertex buffer, which the previous frame's command
        // buffer can still be reading. One idle before the batch of uploads covers all of them.
        auto uploadFrame = [&](const Pipeline::Frame &frame) {
            BuildSurfacePoints(frame, state.colorMode, state.rampNear, state.rampFar,
                               surfaceVertices);
            BuildNormalTufts(frame, state.normalStride, state.normalLength, normalVertices);
            vkDeviceWaitIdle(context.device);
            points->SetPointSet(kSetSurface, surfaceVertices);
            points->SetPointSet(kSetNormals, normalVertices);
        };

        auto applyVisibility = [&]() {
            points->SetVisible(kSetSurface, state.showSurface);
            points->SetVisible(kSetNormals, state.showNormals);
            points->SetVisible(kSetFrustum, state.showFrustum);
        };
        applyVisibility();

        imgui->SetUi([&]() {
            ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(340.0f, 0.0f), ImGuiCond_Always);
            ImGui::Begin("Depth Camera", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse);

            ImGui::Text("%s", live ? "live device" : "replay");
            ImGui::SameLine();
            if (ImGui::Button(state.streaming ? "pause" : "resume"))
                state.streaming = !state.streaming;
            if (state.streamEnded) ImGui::TextColored(ImVec4(1, 0.7f, 0.2f, 1), "stream ended");

            ImGui::SeparatorText("Layers");
            if (ImGui::Checkbox("surface", &state.showSurface)) applyVisibility();
            if (ImGui::Checkbox("normals", &state.showNormals)) applyVisibility();
            if (ImGui::Checkbox("frustum", &state.showFrustum)) applyVisibility();

            ImGui::SeparatorText("Color");
            int mode = int(state.colorMode);
            if (ImGui::Combo("mode", &mode, "normal\0depth\0flat\0")) state.colorMode = EColorMode(mode);
            ImGui::PushItemWidth(ImGui::CalcItemWidth() / 3.0f);
            if (state.colorMode == EColorMode::Depth) {
                ImGui::InputFloat("ramp near", &state.rampNear, 0.0f, 0.0f, "%.4f");
                ImGui::InputFloat("ramp far", &state.rampFar, 0.0f, 0.0f, "%.4f");
            }
            if (ImGui::InputFloat("point size", &state.pointSize, 0.0f, 0.0f, "%.4f"))
                points->SetPointSize(std::clamp(state.pointSize, 1.0f, 16.0f));
            if (state.showNormals) {
                ImGui::InputInt("normal stride", &state.normalStride);
                ImGui::InputFloat("normal length", &state.normalLength, 0.0f, 0.0f, "%.4f");
                state.normalStride = std::max(1, state.normalStride);
            }
            ImGui::PopItemWidth();

            // Both thresholds reject a neighbour whose depth differs by more than
            // max(minimumDepthJump, relativeDepthJump * z); the point goes with the normal.
            ImGui::SeparatorText("Discontinuity filter");
            ImGui::PushItemWidth(ImGui::CalcItemWidth() / 3.0f);
            ImGui::InputFloat("relative", &state.filter.relativeDepthJump, 0.0f, 0.0f, "%.4f");
            ImGui::InputFloat("minimum (m)", &state.filter.minimumDepthJump, 0.0f, 0.0f, "%.4f");
            ImGui::PopItemWidth();
            const double kept = stats.validPixelCount
                                        ? 100.0 * double(stats.pointCount) / double(stats.validPixelCount)
                                        : 0.0;
            ImGui::Text("kept %.1f%% of valid pixels", kept);

            ImGui::SeparatorText("Frame");
            ImGui::Text("frame          %d", state.frameIndex);
            ImGui::Text("valid pixels   %zu / %d", stats.validPixelCount,
                        intrinsics.width * intrinsics.height);
            ImGui::Text("points         %zu", stats.pointCount);
            ImGui::Text("depth range    %.3f .. %.3f m", stats.nearestDepth, stats.farthestDepth);
            ImGui::Text("grab           %.2f ms", stats.grabMilliseconds);
            ImGui::Text("backproject    %.2f ms", stats.backprojectMilliseconds);
            ImGui::Text("display        %.1f fps", double(ImGui::GetIO().Framerate));
            ImGui::End();
        });

        app.GetView().SetScene(&scene);
        app.GetView().SetCamera(&camera);
        app.GetView().SetRenderGraph(&graph);

        // ImGui owns the cursor whenever it is over a panel, so a drag or a scroll meant for a
        // slider must not also spin the 3D camera behind it.
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
            if (e.keyCode == Engine::Render::KeyCode::Space) state.streaming = !state.streaming;
        });

        DepthFrame depthFrame;
        Pipeline::Frame lastFrame;
        DepthFilterOptions appliedFilter = state.filter;
        EColorMode appliedColorMode = state.colorMode;

        using Clock = std::chrono::steady_clock;
        auto elapsedMilliseconds = [](Clock::time_point start) {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        };

        try {
            while (!app.GetWindow().ShouldClose() && !g_interrupted) {
                app.GetWindow().PollEvents();
                const VkExtent2D size = app.GetWindow().FramebufferSize();
                if (size.width == 0 || size.height == 0) continue;

                bool needsUpload = false;
                if (state.streaming && !state.streamEnded) {
                    const Clock::time_point grabStart = Clock::now();
                    const bool got = source->Grab(depthFrame);
                    stats.grabMilliseconds = elapsedMilliseconds(grabStart);
                    if (!got) {
                        // A recording runs out; a device does not. Looping a recording keeps the
                        // window useful instead of freezing on the last frame.
                        if (!live && loop) {
                            source = std::make_unique<Pipeline::RecordedDepthProvider>(replayDirectory);
                            state.frameIndex = 0;
                            continue;
                        }
                        state.streamEnded = true;
                    } else {
                        MeasureRawDepth(depthFrame, stats);
                        const Clock::time_point projectStart = Clock::now();
                        lastFrame = BackprojectDepth(depthFrame, intrinsics, state.filter);
                        stats.backprojectMilliseconds = elapsedMilliseconds(projectStart);
                        stats.pointCount = lastFrame.pts.size();
                        ++state.frameIndex;
                        needsUpload = true;
                    }
                }

                // Paused, the filter and the colour mode must still take effect -- that is how
                // you tune them: freeze a frame, then watch the thresholds eat into it.
                if (!needsUpload && !lastFrame.pts.empty()) {
                    if (appliedFilter.relativeDepthJump != state.filter.relativeDepthJump ||
                        appliedFilter.minimumDepthJump != state.filter.minimumDepthJump) {
                        lastFrame = BackprojectDepth(depthFrame, intrinsics, state.filter);
                        stats.pointCount = lastFrame.pts.size();
                        needsUpload = true;
                    } else if (appliedColorMode != state.colorMode) {
                        needsUpload = true;
                    }
                }
                if (needsUpload) {
                    uploadFrame(lastFrame);
                    appliedFilter = state.filter;
                    appliedColorMode = state.colorMode;
                }

                if (!app.GetRenderer().BeginFrame(size.width, size.height)) continue;
                app.GetRenderer().Render(app.GetView());
                app.GetRenderer().EndFrame();
            }
        } catch (...) {
            vkDeviceWaitIdle(context.device);
            throw;
        }
        vkDeviceWaitIdle(context.device);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

// Live windowed viewer for the real chair scan: loads scanData/frame_*.ply, integrates it into
// a TSDF (Simple / Directional / Compact-Directional, switchable at runtime), extracts the
// surface, and renders both the raw input observations and the extracted surface as colored,
// trackball-orbitable point clouds with an ImGui control panel.
//
// This mirrors example2/tsdf_viewer.cpp's windowed-app shape exactly (Application + Camera +
// RenderGraph{PointCloudPass, ImGuiPass-last} + manual render loop with a between-frames
// dirty-triggered rebuild). The frame loader / per-frame camera model / bbox-voxel fit are
// ported verbatim from directional_tsdf_chair_benchmark.cpp (and mirrored again in
// tsdf_realdata_benchmark.cpp, which also shows the three TSDF variants' Build/Integrate/
// Extract call shapes side by side on this same scan). The one genuinely new piece here is
// fitting the camera to the chair's real-world (~827mm) scale -- tsdf_viewer's synthetic-scene
// near/far (0.05/100) would not span it.
#include "ImGuiPass.h"
#include "PointCloudPass.h"

#include "Engine/Core/Context.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"
#include "Engine/Spatial/CompactDirectionalTSDF.h"
#include "Engine/Spatial/DirectionalTSDF.h"
#include "Engine/Spatial/SimpleTSDF.h"

#include "imgui.h"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    // ---------------------------------------------------------------------------------------
    // Frame loading (ported verbatim from directional_tsdf_chair_benchmark.cpp): the PLY layout
    // is x,y,z,nx,ny,nz, already registered into a common mm frame. The PLY carries no camera
    // pose, so each frame's "camera" is estimated as centroid + standoff*meanNormal (normals
    // face the sensor).
    // ---------------------------------------------------------------------------------------
    struct Frame {
        std::vector<Eigen::Vector3f> points, normals;
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        Eigen::Vector3f meanNormal = Eigen::Vector3f::Zero();
    };

    bool loadFrame(const std::string &path, Frame &out) {
        std::ifstream in(path);
        if (!in) return false;
        std::string line;
        bool inData = false;
        while (std::getline(in, line)) {
            if (!inData) {
                if (line.rfind("end_header", 0) == 0) inData = true;
                continue;
            }
            std::istringstream ss(line);
            float x, y, z, nx, ny, nz;
            if (!(ss >> x >> y >> z >> nx >> ny >> nz)) continue;
            out.points.emplace_back(x, y, z);
            out.normals.emplace_back(nx, ny, nz);
            out.centroid += Eigen::Vector3f(x, y, z);
            out.meanNormal += Eigen::Vector3f(nx, ny, nz);
        }
        if (out.points.empty()) return false;
        out.centroid /= float(out.points.size());
        if (out.meanNormal.norm() > 1e-6f) out.meanNormal.normalize();
        else
            out.meanNormal = Eigen::Vector3f(0, 0, 1);
        return true;
    }

    // Per-frame camera estimate shared by all three TSDF methods (matches
    // directional_tsdf_chair_benchmark.cpp / tsdf_realdata_benchmark.cpp).
    Eigen::Vector3f camFor(const Frame &fr) {
        return fr.centroid + 1000.0f * fr.meanNormal;
    }

    // ---------------------------------------------------------------------------------------
    // CLI parsing (mirrors tsdf_viewer.cpp's small hand-rolled parsers).
    // ---------------------------------------------------------------------------------------
    uint64_t ParseFramesArg(int argc, char **argv) {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--frames" && i + 1 < argc)
                return static_cast<uint64_t>(std::stoull(argv[i + 1]));
        return 0;
    }

    int ParseIntArg(int argc, char **argv, const char *name, int fallback) {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == name && i + 1 < argc)
                return std::stoi(argv[i + 1]);
        return fallback;
    }

    std::string ParseStringArg(int argc, char **argv, const char *name, const std::string &fallback) {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == name && i + 1 < argc)
                return std::string(argv[i + 1]);
        return fallback;
    }

    // Watches the render graph's frame counter and requests the window close once `maxFrames`
    // frames have been rendered (0 = no limit) -- gives `--frames N` an exit-0 smoke signal.
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

    // ---------------------------------------------------------------------------------------
    // Shared interactive state. The ImGui panel mutates method/showInput/showExtracted/
    // colorMode; the between-frames rebuild (below) reads method/colorMode when state.dirty is
    // set, and writes the stats fields back for the panel to display.
    // ---------------------------------------------------------------------------------------
    struct ChairViewerState {
        int method = 2; // 0 = Simple, 1 = Directional, 2 = Compact
        bool showInput = true, showExtracted = true;
        int colorMode = 0; // 0 = normal, 1 = method, 2 = flat
        bool dirty = true; // starts true so the first frame builds

        // Stats, filled by Rebuild() and displayed in the panel.
        size_t nInput = 0, nExtracted = 0;
        double buildMs = 0.0;
        double memKB = 0.0;
    };

    const char *MethodName(int method) {
        switch (method) {
            case 0:
                return "Simple";
            case 1:
                return "Directional";
            default:
                return "Compact";
        }
    }

    uint8_t ToByte(float unit) {
        return static_cast<uint8_t>(std::clamp(unit, 0.0f, 1.0f) * 255.0f + 0.5f);
    }

    struct Rgb {
        uint8_t r, g, b;
    };

    // Flat per-method identity color (colorMode == 1).
    Rgb MethodColor(int method) {
        switch (method) {
            case 0:
                return {160, 160, 160}; // Simple: grey
            case 1:
                return {230, 140, 30}; // Directional: orange
            default:
                return {30, 180, 180}; // Compact: teal
        }
    }

    // Rebuilds the whole TSDF + both point sets from the current ChairViewerState. MUST be
    // called between frames (never inside a RenderPass::Execute): Build/Integrate/Extract all
    // submit their own GPU work / read back from the device. Each call constructs a fresh TSDF
    // instance (methods have no uniform Reset semantics, and switching methods needs a clean
    // volume anyway), integrates ALL loaded frames, extracts the surface, and uploads both the
    // (subsampled) input cloud and the colored extracted cloud to the PointCloudPass.
    void Rebuild(ChairViewerState &state, Engine::Core::Context &ctx, PointCloudPass &pass,
                 const std::vector<Frame> &frames, const Eigen::Vector3f &center,
                 const Eigen::Vector3f &bbMin, float voxelSize, float truncation) {
        using namespace Engine::Spatial;

        const auto t0 = std::chrono::steady_clock::now();

        OrientedPointCloud recon;
        double memKB = 0.0;

        if (state.method == 0) {
            // SimpleTSDF: normal-free baseline (w=1 per observation).
            SimpleTSDF tsdf;
            tsdf.Build(ctx, voxelSize, truncation, 1u << 22, 1u << 17);
            for (const Frame &fr: frames)
                tsdf.Integrate(fr.points, camFor(fr));
            recon = tsdf.ExtractPointCloud();
            memKB = double(tsdf.FilledCount()) * 16.0 / 1024.0;
        } else if (state.method == 1) {
            // DirectionalTSDF: streaming host/GPU cache, Unified backend (whole scan fits one
            // fixed window, so the first frame's missing set exceeds the Streaming staging cap).
            DirectionalTSDF tsdf;
            tsdf.Build(ctx, voxelSize, truncation, 1u << 18, 1u << 17, 1u << 21,
                       ResidencyMode::Unified);
            tsdf.SetIntegrationQuality({3, 4, true});
            for (const Frame &fr: frames)
                tsdf.Integrate(fr.points, fr.normals, camFor(fr), center);
            recon = tsdf.ExtractOrientedCloud();
            memKB = double(tsdf.HostStore().Size()) * 4096.0 / 1024.0;
        } else {
            // CompactDirectionalTSDF: per-(voxel,direction) flat hash, single 512^3 window
            // anchored below the scene's bbox min (margin = truncation + 1 voxel, matching
            // tsdf_realdata_benchmark.cpp's windowMinCorner derivation).
            CompactDirectionalTSDF tsdf;
            const Eigen::Vector3f windowMinCorner =
                    bbMin - Eigen::Vector3f::Constant(truncation + voxelSize);
            tsdf.Build(ctx, voxelSize, truncation, 1u << 22, 1u << 17, windowMinCorner);
            tsdf.SetIntegrationQuality({3, 4, true});
            for (const Frame &fr: frames)
                tsdf.Integrate(fr.points, fr.normals, camFor(fr));
            recon = tsdf.ExtractPointCloud(1u << 21, /*merge=*/true);
            memKB = double(tsdf.FilledCount()) * 16.0 / 1024.0;
        }

        if (recon.points.empty()) {
            std::cerr << "[rebuild] WARNING: " << MethodName(state.method)
                      << " extracted an EMPTY point cloud -- window/voxel fit may be wrong for "
                         "this scene.\n";
        }

        // Set 0: INPUT, subsampled (keep every 4th point) -- a full ~900k-point upload is heavy
        // for an interactive viewer. Dim grey so the extracted surface reads clearly on top.
        std::vector<PointVertex> inputVerts;
        size_t inputEstimate = 0;
        for (const Frame &fr: frames) inputEstimate += (fr.points.size() + 3) / 4;
        inputVerts.reserve(inputEstimate);
        for (const Frame &fr: frames)
            for (size_t i = 0; i < fr.points.size(); i += 4) {
                const Eigen::Vector3f &p = fr.points[i];
                inputVerts.push_back({{p.x(), p.y(), p.z()}, {130, 140, 150, 255}});
            }
        pass.SetPointSet(0, inputVerts);

        // Set 1: EXTRACTED, colored per state.colorMode.
        std::vector<PointVertex> extractedVerts;
        extractedVerts.reserve(recon.points.size());
        const Rgb methodColor = MethodColor(state.method);
        for (size_t i = 0; i < recon.points.size(); ++i) {
            const Eigen::Vector3f &p = recon.points[i];
            uint8_t r, g, b;
            if (state.colorMode == 0) {
                const Eigen::Vector3f n =
                        i < recon.normals.size() ? recon.normals[i] : Eigen::Vector3f::Zero();
                r = ToByte(n.x() * 0.5f + 0.5f);
                g = ToByte(n.y() * 0.5f + 0.5f);
                b = ToByte(n.z() * 0.5f + 0.5f);
            } else if (state.colorMode == 1) {
                r = methodColor.r;
                g = methodColor.g;
                b = methodColor.b;
            } else {
                r = g = b = 255;
            }
            extractedVerts.push_back({{p.x(), p.y(), p.z()}, {r, g, b, 255}});
        }
        pass.SetPointSet(1, extractedVerts);

        pass.SetVisible(0, state.showInput);
        pass.SetVisible(1, state.showExtracted);

        const auto t1 = std::chrono::steady_clock::now();

        state.nInput = inputVerts.size();
        state.nExtracted = recon.points.size();
        state.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        state.memKB = memKB;
        state.dirty = false;

        std::printf("[rebuild] method=%s nIn=%zu nExt=%zu mem=%.1fKB build=%.1fms\n",
                    MethodName(state.method), state.nInput, state.nExtracted, state.memKB,
                    state.buildMs);
    }

} // namespace

int main(int argc, char **argv) {
    try {
        const uint64_t framesLimit = ParseFramesArg(argc, argv);
        const std::string scandir = ParseStringArg(argc, argv, "--scandir", "scanData");
        const int maxFrames = ParseIntArg(argc, argv, "--maxframes", 12);

        // ---- load up to maxFrames chair frames ----
        std::vector<Frame> frames;
        for (int i = 0; i < maxFrames; ++i) {
            std::ostringstream name;
            name << scandir << "/frame_" << std::setw(4) << std::setfill('0') << i << ".ply";
            Frame fr;
            if (!loadFrame(name.str(), fr)) break;
            std::cout << "loaded " << name.str() << ": " << fr.points.size() << " points\n";
            frames.push_back(std::move(fr));
        }
        if (frames.empty())
            throw std::runtime_error("no frames found under '" + scandir + "' (expected frame_0000.ply ...)");

        // ---- scene bbox -> center/extent/voxelSize/truncation (all three methods share these) ----
        Eigen::Vector3f bbMin = frames[0].points[0], bbMax = frames[0].points[0];
        size_t totalIn = 0;
        for (const auto &fr: frames)
            for (const auto &p: fr.points) {
                bbMin = bbMin.cwiseMin(p);
                bbMax = bbMax.cwiseMax(p);
                ++totalIn;
            }
        const Eigen::Vector3f center = 0.5f * (bbMin + bbMax);
        const Eigen::Vector3f extent = bbMax - bbMin;
        const float maxExtent = extent.maxCoeff();
        const float radius = 0.5f * maxExtent;
        const float voxelSize = (maxExtent / 400.0f) * 1.15f;
        const float truncation = 3.0f * voxelSize;

        std::cout << "\nscene: " << frames.size() << " frames, " << totalIn << " points\n"
                  << "  bbox extent " << extent.x() << " x " << extent.y() << " x " << extent.z()
                  << " mm, center (" << center.x() << ", " << center.y() << ", " << center.z()
                  << ")\n"
                  << "  voxelSize " << voxelSize << " mm, truncation " << truncation << " mm\n\n";

        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {1280, 800, "Chair TSDF Viewer"};
        Engine::Render::Application app(descriptor);

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D extent2D = app.GetSwapChain().Extent();
        const float aspect = static_cast<float>(extent2D.width) / static_cast<float>(extent2D.height);
        // CAMERA FIT: the chair scan is ~827mm across -- tsdf_viewer's synthetic-scene near/far
        // (0.05/100) would clip almost the entire model. near=1mm is comfortably inside the
        // trackball's min orbit distance; far covers >12x the scene radius with margin.
        camera.SetPerspective(60.0f * 3.14159265f / 180.0f, aspect, 1.0f,
                              std::max(5000.0f, radius * 12.0f));
        camera.SetOrbit(vkMath::Vec3(center.x(), center.y(), center.z()), radius * 2.2f);

        ChairViewerState state;

        const std::string shaderDir = CHAIR_VIEWER_SHADER_DIR;
        Engine::Render::RenderGraph graph;

        auto pointCloudPassOwned = std::make_unique<PointCloudPass>(
                app.GetContext(), app.GetSwapChain().Format(), shaderDir);
        PointCloudPass *pointCloudPass = pointCloudPassOwned.get();
        graph.AddPass(std::move(pointCloudPassOwned));
        graph.AddPass(std::make_unique<FrameLimiterPass>(app.GetWindow(), framesLimit));

        // Window backend is fixed to GLFW (ApplicationDescriptor default), so the base Window&
        // is always actually a GlfwWindow -- safe to downcast to reach Handle() for ImGui's
        // GLFW backend.
        auto &glfwWindow = static_cast<Engine::Render::GlfwWindow &>(app.GetWindow());
        auto imGuiPassOwned = std::make_unique<ImGuiPass>(
                app.GetContext(), glfwWindow.Handle(), app.GetSwapChain().Format(),
                app.GetSwapChain().ImageCount());
        ImGuiPass *imGuiPass = imGuiPassOwned.get();
        // Added last: ImGuiPass's RenderingScope loads (does not clear) the swapchain image, so
        // it must run after PointCloudPass to draw the panel over the points.
        graph.AddPass(std::move(imGuiPassOwned));

        pointCloudPass->SetPointSize(3.0f);

        imGuiPass->SetUi([&state, pointCloudPass, voxelSize]() {
            ImGui::Begin("Chair TSDF Viewer");

            ImGui::SeparatorText("Method");
            if (ImGui::RadioButton("Simple", &state.method, 0)) state.dirty = true;
            ImGui::SameLine();
            if (ImGui::RadioButton("Directional", &state.method, 1)) state.dirty = true;
            ImGui::SameLine();
            if (ImGui::RadioButton("Compact", &state.method, 2)) state.dirty = true;

            ImGui::SeparatorText("Layers");
            if (ImGui::Checkbox("show input", &state.showInput))
                pointCloudPass->SetVisible(0, state.showInput);
            if (ImGui::Checkbox("show extracted", &state.showExtracted))
                pointCloudPass->SetVisible(1, state.showExtracted);

            ImGui::SeparatorText("Color mode");
            if (ImGui::Combo("extracted color", &state.colorMode, "normal\0method\0flat\0\0"))
                state.dirty = true;

            ImGui::SeparatorText("Stats");
            ImGui::Text("method:           %s", MethodName(state.method));
            ImGui::Text("input points:     %zu", state.nInput);
            ImGui::Text("extracted points: %zu", state.nExtracted);
            ImGui::Text("voxel size:       %.3f mm", voxelSize);
            ImGui::Text("memory:           %.1f KB", state.memKB);
            ImGui::Text("build time:       %.1f ms", state.buildMs);

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
                camera.BeginTrackballDrag(e.x, e.y, static_cast<int>(size.width),
                                          static_cast<int>(size.height));
                return;
            }

            camera.DragTrackball(e.x, e.y, static_cast<int>(size.width),
                                 static_cast<int>(size.height));
            e.handled = true;
        });
        trackball.Add(Engine::Render::MouseEventType::ButtonUp, [&](Engine::Render::MouseEvent &e) {
            if (e.button == Engine::Render::MouseButton::Left)
                camera.EndTrackballDrag();
        });
        trackball.Add(Engine::Render::MouseEventType::Scroll, [&](Engine::Render::MouseEvent &e) {
            camera.SetDistance(std::clamp(
                    camera.GetDistance() * std::exp(static_cast<float>(-e.scrollY) * 0.08f),
                    radius * 0.2f, radius * 20.0f));
            e.handled = true;
        });
        // Right-drag: pan (translate) the view in the current view plane.
        trackball.Add(Engine::Render::MouseEventType::Drag, [&](Engine::Render::MouseEvent &e) {
            if (e.button != Engine::Render::MouseButton::Right)
                return;
            const VkExtent2D size = app.GetWindow().FramebufferSize();
            if (size.width == 0 || size.height == 0)
                return;
            camera.Pan(e.deltaX, e.deltaY, static_cast<int>(size.width),
                       static_cast<int>(size.height));
            e.handled = true;
        });

        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape)
                app.GetWindow().RequestClose();
        });

        // Manual render loop (replicates Engine::Render::Application::Run) so Rebuild() can run
        // BETWEEN frames -- it submits its own GPU compute + reads back from the device, which
        // must never happen inside a RenderPass::Execute. The UI only sets state.dirty; the
        // rebuild happens here, guarded by a device-idle wait (single-frame-in-flight Renderer).
        Engine::Core::Context &ctx = app.GetContext();
        try {
            while (!app.GetWindow().ShouldClose()) {
                app.GetWindow().PollEvents();

                const VkExtent2D size = app.GetWindow().FramebufferSize();
                if (size.width == 0 || size.height == 0)
                    continue;

                if (state.dirty) {
                    vkDeviceWaitIdle(ctx.device);
                    Rebuild(
                            state,
                            ctx,
                            *pointCloudPass,
                            frames,
                            center,
                            bbMin,
                            voxelSize,
                            truncation);
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

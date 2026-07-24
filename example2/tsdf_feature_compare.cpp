// SimpleTSDF-vs-DirectionalTSDF feature-preservation comparison.
//
// Integrates the SAME analytic multi-view samples into BOTH TSDFs, extracts both point
// clouds, and reports per-region (flat / curved / edge) ground-truth error.
//
// The metric is the oracle: DirectionalTSDF (per-direction layers) should beat the averaged
// SimpleTSDF on EDGE error (it preserves the sharp corner / rim) while tying on flat/curved.
//
//   tsdf_feature_compare --shape cube|cylinder [--voxel <f>] [--dump]
//   tsdf_feature_compare --shape cube|cylinder [--voxel <f>] [--frames N]
//
// --dump computes once, prints the Task-2 metric table, and exits before any window is
// created (the headless correctness oracle). Without --dump this opens a windowed viewer
// (Task 3): an A/B toggle between the Simple/Directional extracted clouds (same camera),
// error/region color modes, and an ImGui panel with the same per-region metric table.
#include "shape_fixtures.h"

#include "ImGuiPass.h"
#include "PointCloudPass.h"

#include "Engine/Core/Context.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"
#include "Engine/Spatial/DirectionalTSDF.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"
#include "Engine/Spatial/SimpleTSDF.h"

#include "imgui.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using fixtures::Region;
using fixtures::Shape;

namespace {

    constexpr float kVoxelDefault = 0.1f;
    constexpr float kTruncation = 0.3f;

    struct RegionStat {
        double sum = 0.0;
        double maxErr = 0.0;
        std::size_t count = 0;
        void add(float e) {
            sum += double(e);
            maxErr = std::max(maxErr, double(e));
            ++count;
        }
        double mean() const { return count ? sum / double(count) : 0.0; }
    };

    // Region is {Flat, Curved, Edge}; keep the stats array's size tied to the enum so a future
    // Region addition fails to compile here instead of silently truncating/overflowing the table.
    constexpr int kNumRegions = 3;
    static_assert(kNumRegions == static_cast<int>(Region::Edge) + 1,
                  "RegionStat array size must track fixtures::Region's enumerator count");

    // err = NearestDistance(true surface), region = ClassifyRegion; accumulate per region.
    void accumulate(Shape s, float voxel, const std::vector<Eigen::Vector3f> &pts,
                    std::array<RegionStat, kNumRegions> &stats) {
        for (const auto &p : pts) {
            const float e = fixtures::NearestDistance(s, p);
            stats[static_cast<int>(fixtures::ClassifyRegion(s, p, voxel))].add(e);
        }
    }

    // Per-region normal-angle error (degrees): angle between each extracted normal and the
    // analytic surface normal at that point.
    void accumulateNormals(Shape s, float voxel, const std::vector<Eigen::Vector3f> &pts,
                           const std::vector<Eigen::Vector3f> &normals,
                           std::array<RegionStat, kNumRegions> &stats) {
        for (std::size_t i = 0; i < pts.size(); ++i) {
            const float ang = fixtures::NormalAngleDeg(normals[i], fixtures::NearestNormal(s, pts[i]));
            stats[static_cast<int>(fixtures::ClassifyRegion(s, pts[i], voxel))].add(ang);
        }
    }

    const char *regionName(int r) {
        switch (r) {
            case static_cast<int>(Region::Flat): return "flat";
            case static_cast<int>(Region::Curved): return "curved";
            case static_cast<int>(Region::Edge): return "edge";
            default: return "?";
        }
    }

    // Shared compute (Task 2 + Task 3 share this): integrates the SAME multi-view samples into
    // both TSDFs, extracts both clouds, and scores both against the analytic ground truth. Used
    // by both the headless --dump path and the windowed GUI's rebuild() so the two paths can
    // never drift apart -- there is exactly one place that computes the comparison.
    struct CompareResult {
        std::vector<fixtures::View> views;
        std::size_t nInput = 0, maxPerView = 0;
        std::vector<Eigen::Vector3f> simplePts;
        std::vector<Eigen::Vector3f> dirPts;             // legacy directional positions
        std::vector<Eigen::Vector3f> dirNormals;         // legacy directional normals
        std::vector<Eigen::Vector3f> dirPtsRefined;      // refined directional positions
        std::vector<Eigen::Vector3f> dirNormalsRefined;  // refined directional normals
        std::array<RegionStat, kNumRegions> simpleStats{};
        std::array<RegionStat, kNumRegions> dirStats{};            // legacy position error
        std::array<RegionStat, kNumRegions> dirStatsRefined{};     // refined position error
        std::array<RegionStat, kNumRegions> dirNormStats{};        // legacy normal-angle error
        std::array<RegionStat, kNumRegions> dirNormStatsRefined{}; // refined normal-angle error
        std::vector<Eigen::Vector3f> dirPtsStored;         // mode 2 positions
        std::vector<Eigen::Vector3f> dirNormalsStored;     // mode 2 normals
        std::array<RegionStat, kNumRegions> dirStatsStored{};     // stored position error
        std::array<RegionStat, kNumRegions> dirNormStatsStored{}; // stored normal-angle error
    };

    CompareResult RunCompare(Engine::Core::Context &ctx, Shape shape, float voxel,
                             float truncation) {
        CompareResult r;
        r.views = fixtures::SampleViews(shape, voxel);
        for (const auto &v : r.views) {
            r.nInput += v.points.size();
            r.maxPerView = std::max(r.maxPerView, v.points.size());
        }

        // ---- SimpleTSDF: single averaged field (rounds sharp features) ----
        Engine::Spatial::SimpleTSDF simple;
        simple.Build(ctx, voxel, truncation);
        for (const auto &v : r.views) simple.Integrate(v.points, v.camPos);
        const Engine::Spatial::OrientedPointCloud simpleCloud = simple.ExtractPointCloud();
        r.simplePts = simpleCloud.points;

        // ---- DirectionalTSDF, legacy extraction (axis-crossing average) ----
        Engine::Spatial::DirectionalTSDF dirLegacy;
        dirLegacy.Build(ctx, voxel, truncation);
        dirLegacy.SetIntegrationQuality({3, 4, true}); // maxDirections=3, dirExponent=4, viewAngleWeight
        dirLegacy.SetExtractMode(0);
        for (const auto &v : r.views)
            dirLegacy.Integrate(v.points, v.normals, v.camPos, Eigen::Vector3f::Zero());
        for (const auto &e : dirLegacy.PointCloud()) {
            r.dirPts.push_back(e.position);
            r.dirNormals.push_back(e.normal);
        }

        // ---- DirectionalTSDF, refined extraction (sub-voxel gradient projection) ----
        // Same synthetic input is deterministic at this sample budget, so the two integrated
        // volumes match and only the extraction differs. (No public re-extract API exists;
        // a second Build+Integrate is the clean way to get both clouds.)
        Engine::Spatial::DirectionalTSDF dirRefined;
        dirRefined.Build(ctx, voxel, truncation);
        dirRefined.SetIntegrationQuality({3, 4, true});
        dirRefined.SetExtractMode(1);
        for (const auto &v : r.views)
            dirRefined.Integrate(v.points, v.normals, v.camPos, Eigen::Vector3f::Zero());
        for (const auto &e : dirRefined.PointCloud()) {
            r.dirPtsRefined.push_back(e.position);
            r.dirNormalsRefined.push_back(e.normal);
        }

        // ---- DirectionalTSDF, stored-gradient extraction (mode 2) ----
        Engine::Spatial::DirectionalTSDF dirStored;
        dirStored.Build(ctx, voxel, truncation);
        dirStored.SetIntegrationQuality({3, 4, true});
        dirStored.SetExtractMode(2);
        for (const auto &v : r.views)
            dirStored.Integrate(v.points, v.normals, v.camPos, Eigen::Vector3f::Zero());
        for (const auto &e : dirStored.PointCloud()) {
            r.dirPtsStored.push_back(e.position);
            r.dirNormalsStored.push_back(e.normal);
        }

        accumulate(shape, voxel, r.simplePts, r.simpleStats);
        accumulate(shape, voxel, r.dirPts, r.dirStats);
        accumulate(shape, voxel, r.dirPtsRefined, r.dirStatsRefined);
        accumulateNormals(shape, voxel, r.dirPts, r.dirNormals, r.dirNormStats);
        accumulateNormals(shape, voxel, r.dirPtsRefined, r.dirNormalsRefined, r.dirNormStatsRefined);
        accumulate(shape, voxel, r.dirPtsStored, r.dirStatsStored);
        accumulateNormals(shape, voxel, r.dirPtsStored, r.dirNormalsStored, r.dirNormStatsStored);
        return r;
    }

    // The Task-2/3 verification table (--dump oracle): legacy vs refined position error, plus
    // the refined-vs-legacy normal-angle error.
    void PrintReport(const char *shapeName, float voxel, float truncation,
                      const CompareResult &r) {
        std::printf("=== tsdf_feature_compare  shape=%s  voxel=%.3f  truncation=%.3f ===\n",
                    shapeName, voxel, truncation);
        std::printf("views=%zu  nInput=%zu (max/view=%zu)  nSimple=%zu  nDir=%zu  nDirRefined=%zu  "
                    "nDirStored=%zu\n",
                    r.views.size(), r.nInput, r.maxPerView, r.simplePts.size(), r.dirPts.size(),
                    r.dirPtsRefined.size(), r.dirPtsStored.size());

        std::printf("[position error mm]\n");
        std::printf("%-7s | %11s %11s | %11s %11s | %11s %11s | %11s %11s\n", "region",
                    "Simple.mean", "Simple.max", "Dir.mean", "Dir.max", "FD.mean", "FD.max",
                    "Stored.mean", "Stored.max");
        std::printf("--------+-------------------------+-------------------------"
                    "+-------------------------+-------------------------\n");
        for (int reg = 0; reg < kNumRegions; ++reg) {
            if (r.simpleStats[reg].count == 0 && r.dirStats[reg].count == 0 &&
                r.dirStatsRefined[reg].count == 0 && r.dirStatsStored[reg].count == 0)
                continue;
            std::printf("%-7s | %9.4f %11.4f | %9.4f %11.4f | %9.4f %11.4f | %9.4f %11.4f\n",
                        regionName(reg), r.simpleStats[reg].mean(), r.simpleStats[reg].maxErr,
                        r.dirStats[reg].mean(), r.dirStats[reg].maxErr,
                        r.dirStatsRefined[reg].mean(), r.dirStatsRefined[reg].maxErr,
                        r.dirStatsStored[reg].mean(), r.dirStatsStored[reg].maxErr);
        }

        std::printf("[normal-angle error deg]  (edge = reference only, normal is discontinuous)\n");
        std::printf("%-7s | %11s %11s | %11s %11s | %11s %11s\n", "region", "Dir.mean", "Dir.max",
                    "FD.mean", "FD.max", "Stored.mean", "Stored.max");
        std::printf("--------+-------------------------+-------------------------"
                    "+-------------------------\n");
        for (int reg = 0; reg < kNumRegions; ++reg) {
            if (r.dirNormStats[reg].count == 0 && r.dirNormStatsRefined[reg].count == 0 &&
                r.dirNormStatsStored[reg].count == 0)
                continue;
            std::printf("%-7s | %9.4f %11.4f | %9.4f %11.4f | %9.4f %11.4f\n", regionName(reg),
                        r.dirNormStats[reg].mean(), r.dirNormStats[reg].maxErr,
                        r.dirNormStatsRefined[reg].mean(), r.dirNormStatsRefined[reg].maxErr,
                        r.dirNormStatsStored[reg].mean(), r.dirNormStatsStored[reg].maxErr);
        }
        std::printf("(position mm; angle deg; 1 world unit == 1 mm)\n");
    }

    // ---------------------------------------------------------------------------------------
    // Task 3: windowed A/B viewer state + rebuild + render-graph plumbing below.
    // ---------------------------------------------------------------------------------------

    // Shared interactive state for the feature-compare viewer. The UI (SetUi lambda in main)
    // mutates it; the between-frames rebuild() reads it and writes back the metrics. `dirty`
    // requests a full recompute (shape/voxel/colorMode change); `method`/`showInput` drive
    // cheap PointCloudPass::SetVisible toggles with no rebuild (the A/B toggle must be instant).
    struct CompareState {
        int shape = 0;     // 0 = cube, 1 = cylinder
        int method = 1;    // 0 = Simple, 1 = Directional
        int colorMode = 0; // 0 = error, 1 = region
        bool showInput = false;
        float voxel = kVoxelDefault;
        bool dirty = true; // starts true so the first loop iteration performs the initial build

        // Metrics for both methods, per region -- filled by rebuild(), shown in the panel.
        std::size_t nInput = 0, nSimple = 0, nDir = 0;
        std::array<RegionStat, kNumRegions> simpleStats{};
        std::array<RegionStat, kNumRegions> dirStats{};
    };

    // Rebuilds both TSDFs from the current CompareState and uploads 3 PointCloudPass sets:
    // 0 = INPUT (all view points, white), 1 = SIMPLE extracted, 2 = DIRECTIONAL extracted (both
    // colored per state.colorMode). MUST be called between frames (never inside a
    // RenderPass::Execute): RunCompare's Integrate()/PointCloud()/ExtractPointCloud() all submit
    // their own GPU work / read back from the device.
    void rebuild(CompareState &state, Engine::Core::Context &ctx, PointCloudPass &pass) {
        const Shape shape = (state.shape == 1) ? Shape::Cylinder : Shape::Cube;
        const char *shapeName = (state.shape == 1) ? "cylinder" : "cube";

        const CompareResult r = RunCompare(ctx, shape, state.voxel, kTruncation);

        state.nInput = r.nInput;
        state.nSimple = r.simplePts.size();
        state.nDir = r.dirPts.size();
        state.simpleStats = r.simpleStats;
        state.dirStats = r.dirStats;

        const float maxErr = 3.0f * state.voxel;
        auto colorFor = [&](const Eigen::Vector3f &p) -> fixtures::Rgb {
            if (state.colorMode == 1)
                return fixtures::regionColor(fixtures::ClassifyRegion(shape, p, state.voxel));
            return fixtures::errorColor(fixtures::NearestDistance(shape, p), maxErr);
        };

        // Set 0: INPUT (all view samples, white).
        std::vector<PointVertex> inputVerts;
        inputVerts.reserve(r.nInput);
        for (const auto &v : r.views)
            for (const auto &p : v.points)
                inputVerts.push_back({{p.x(), p.y(), p.z()}, {255, 255, 255, 255}});
        pass.SetPointSet(0, inputVerts);

        // Set 1: SIMPLE extracted.
        std::vector<PointVertex> simpleVerts;
        simpleVerts.reserve(r.simplePts.size());
        for (const auto &p : r.simplePts) {
            const fixtures::Rgb c = colorFor(p);
            simpleVerts.push_back({{p.x(), p.y(), p.z()}, {c.r, c.g, c.b, 255}});
        }
        pass.SetPointSet(1, simpleVerts);

        // Set 2: DIRECTIONAL extracted.
        std::vector<PointVertex> dirVerts;
        dirVerts.reserve(r.dirPts.size());
        for (const auto &p : r.dirPts) {
            const fixtures::Rgb c = colorFor(p);
            dirVerts.push_back({{p.x(), p.y(), p.z()}, {c.r, c.g, c.b, 255}});
        }
        pass.SetPointSet(2, dirVerts);

        pass.SetVisible(0, state.showInput);
        pass.SetVisible(1, state.method == 0);
        pass.SetVisible(2, state.method == 1);

        state.dirty = false;

        std::cout << "[rebuild] shape=" << shapeName << " voxel=" << state.voxel
                  << " method=" << (state.method == 0 ? "simple" : "directional")
                  << " color=" << (state.colorMode == 0 ? "error" : "region")
                  << " | nInput=" << state.nInput << " nSimple=" << state.nSimple
                  << " nDir=" << state.nDir;
        for (int reg = 0; reg < kNumRegions; ++reg) {
            if (state.simpleStats[reg].count == 0 && state.dirStats[reg].count == 0) continue;
            std::cout << " | " << regionName(reg) << " S=" << state.simpleStats[reg].mean() << "/"
                      << state.simpleStats[reg].maxErr << " D=" << state.dirStats[reg].mean()
                      << "/" << state.dirStats[reg].maxErr;
        }
        std::cout << std::endl;
    }

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

} // namespace

int main(int argc, char **argv) {
    Shape shapeArg = Shape::Cube;
    float voxelArg = kVoxelDefault;
    bool dump = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shape" && i + 1 < argc) {
            const std::string v = argv[++i];
            shapeArg = (v == "cylinder") ? Shape::Cylinder : Shape::Cube;
        } else if (a == "--voxel" && i + 1 < argc) {
            voxelArg = std::stof(argv[++i]);
        } else if (a == "--dump") {
            dump = true;
        }
    }
    const char *shapeName = (shapeArg == Shape::Cube) ? "cube" : "cylinder";

    // --dump: headless correctness oracle. Computes once, prints the metric table, and returns
    // before any window is created -- no Application/render deps touched on this path.
    if (dump) {
        Engine::Core::Context ctx;
        const CompareResult r = RunCompare(ctx, shapeArg, voxelArg, kTruncation);
        PrintReport(shapeName, voxelArg, kTruncation, r);
        return 0;
    }

    try {
        const uint64_t framesLimit = ParseFramesArg(argc, argv);

        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {1280, 800, "TSDF Feature Compare: Simple vs Directional"};
        Engine::Render::Application app(descriptor);

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D extent = app.GetSwapChain().Extent();
        const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        camera.SetPerspective(60.0f * 3.14159265f / 180.0f, aspect, 0.05f, 100.0f);
        camera.SetOrbit({0.0f, 0.0f, 0.0f}, 8.0f); // shapes sit within radius ~1.5..2.6

        // The comparison is (re)built by rebuild() between frames whenever state.dirty is set
        // (see the render loop below). state.dirty starts true, so the first loop iteration
        // performs the initial build before any frame is in flight.
        CompareState state;
        state.shape = (shapeArg == Shape::Cylinder) ? 1 : 0;
        state.voxel = voxelArg;

        const std::string shaderDir = FEATURE_SHADER_DIR;
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

        imGuiPass->SetUi([&state, pointCloudPass]() {
            ImGui::Begin("Feature Compare");

            ImGui::SeparatorText("Shape");
            if (ImGui::Combo("shape", &state.shape, "cube\0cylinder\0\0"))
                state.dirty = true;

            ImGui::SeparatorText("Method (A/B -- Tab also toggles)");
            bool methodChanged = false;
            methodChanged |= ImGui::RadioButton("Simple", &state.method, 0);
            ImGui::SameLine();
            methodChanged |= ImGui::RadioButton("Directional", &state.method, 1);
            if (methodChanged) {
                // Visibility flip only -- both extracted sets are already uploaded, so no
                // rebuild is needed for the A/B toggle.
                pointCloudPass->SetVisible(1, state.method == 0);
                pointCloudPass->SetVisible(2, state.method == 1);
            }

            ImGui::SeparatorText("Color mode");
            if (ImGui::Combo("color", &state.colorMode, "error\0region\0\0"))
                state.dirty = true; // recolor needs a rebuild (colors are baked into the vertices)

            if (ImGui::Checkbox("show input", &state.showInput))
                pointCloudPass->SetVisible(0, state.showInput);

            ImGui::SeparatorText("Per-region ground-truth error (mm)");
            ImGui::Text("%-7s %11s %11s %11s %11s", "region", "Simple.mean", "Simple.max",
                        "Dir.mean", "Dir.max");
            for (int reg = 0; reg < kNumRegions; ++reg) {
                if (state.simpleStats[reg].count == 0 && state.dirStats[reg].count == 0)
                    continue;
                ImGui::Text("%-7s %11.4f %11.4f %11.4f %11.4f", regionName(reg),
                            state.simpleStats[reg].mean(), state.simpleStats[reg].maxErr,
                            state.dirStats[reg].mean(), state.dirStats[reg].maxErr);
            }
            ImGui::Text("nInput=%zu  nSimple=%zu  nDir=%zu", state.nInput, state.nSimple,
                        state.nDir);

            ImGui::Spacing();
            if (ImGui::Button("Re-run"))
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
            if (e.keyCode == Engine::Render::KeyCode::Escape) {
                app.GetWindow().RequestClose();
            } else if (e.keyCode == Engine::Render::KeyCode::Tab) {
                // A/B toggle: flip method + visibility immediately, no rebuild.
                state.method = (state.method == 0) ? 1 : 0;
                pointCloudPass->SetVisible(1, state.method == 0);
                pointCloudPass->SetVisible(2, state.method == 1);
            }
        });

        // Manual render loop (mirrors tsdf_viewer.cpp) so the comparison can be rebuilt BETWEEN
        // frames. Application::Run() exposes no per-frame hook, and rebuild() submits its own
        // GPU compute + reads back from the device -- which must never happen inside a
        // RenderPass::Execute (mid graphics command buffer). The UI only sets state.dirty; the
        // actual rebuild happens here, before BeginFrame, guarded by a device-idle wait so no
        // graphics frame is in flight while vertex buffers are re-uploaded (the Renderer is
        // single-frame-in-flight). No engine change required.
        Engine::Core::Context &ctx = app.GetContext();
        try {
            while (!app.GetWindow().ShouldClose()) {
                app.GetWindow().PollEvents();

                const VkExtent2D size = app.GetWindow().FramebufferSize();
                if (size.width == 0 || size.height == 0)
                    continue;

                if (state.dirty) {
                    vkDeviceWaitIdle(ctx.device);
                    rebuild(state, ctx, *pointCloudPass);
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

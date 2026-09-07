// RealSense scan pipeline — a depth camera driving the full Track/Map/Render pipeline, with the
// surface accumulating on screen as you sweep the sensor over an object.
//
//   depth frames -> depth front end -> ICP against the model -> TSDF integrate -> accumulated surface
//   \_______________________________/   \___________________/    \_____________________________/
//          AcquisitionThread                RegistrationThread            IntegrationThread
//
// The depth front end is src/Realsense: confidence score, threshold, back-project, normals, voxel
// thinning and compaction all on the device, with only the survivors read back. It replaced a host
// one that back-projected per pixel, and a live camera left no choice -- measured over capture/
// (476 frames, --gpu-downsample 0.01) the host front end handed ICP 3.2M map entries and 240.6 ms
// per frame, which is 4 fps against a sensor producing 30. This one hands it 616k and 30.4 ms.
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
//        [--score-threshold 0.9]                            confidence a pixel must reach
//        [--gpu-downsample 0]                               device-side thinning; default --voxel, 0 = off
//        [--high-accuracy]                                  D435 High Accuracy visual preset (live only)
//        [--bootstrap-frames 5] [--bootstrap-fitness 0.70]  hold fusion until tracking locks on
//        [--min-fuse-fitness 0] [--max-fuse-rmse 0]         per-frame fusion quality gates (0 = off)
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
#include "Pipeline/Acquisition/AcquisitionThread.h"
#include "Pipeline/Pipeline.h"
#include "Pipeline/Registration/GpuIcpTracker.h"
#include "Pipeline/Registration/Tracker.h"
#include "Realsense/RealSenseD435.h"
#include "Realsense/RealSenseD435Recorder.h"
#include "Registration/RegistrationParam.h"
#include "utilities/ArgParser.h"

#include "imgui.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
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
    // Map-structure overlays. Same three boxes, colors and meanings as voxel_fill_debugger, so a
    // reading learned on the folder debugger transfers to a live scan unchanged.
    constexpr int kSetTileBox = 2;   // coarse tile windows the map has opened
    constexpr int kSetAllocBox = 3;  // the AABB the map has actually allocated
    constexpr int kSetSubmapBox = 4; // dense (detail) submap regions

    enum class EColorMode { Normal,
                            Age,
                            Weight };

    // The world frame is the first camera's frame, which is the image convention: +y down, +z
    // forward. Rotating 180 degrees about x puts y up and the scene in front of a viewer looking
    // down -z. A rotation, not a mirror -- a single-axis negation would flip handedness and invert
    // every displayed normal.
    inline Vector3f SensorToView(const Vector3f &v) { return Vector3f(v.x(), -v.y(), -v.z()); }

    struct Color {
        uint8_t r, g, b;
    };

    // A world-space AABB drawn as its 12 edges, each sampled densely enough to read as a line in
    // the point pipeline (PointCloudPass draws points, not line lists). Sampling follows the box's
    // longest side in voxels and is clamped so a huge tile window does not cost a million points.
    // Points are converted per-vertex with SensorToView: the transform negates two axes, so
    // converting the min/max corners instead would swap them and build the box inside out.
    std::vector<PointVertex> BoxEdgePoints(const Vector3f &minimumCorner, const Vector3f &maximumCorner,
                                           float voxel, Color color) {
        std::vector<PointVertex> vertices;
        const Vector3f corner[8] = {
                {minimumCorner.x(), minimumCorner.y(), minimumCorner.z()},
                {maximumCorner.x(), minimumCorner.y(), minimumCorner.z()},
                {maximumCorner.x(), maximumCorner.y(), minimumCorner.z()},
                {minimumCorner.x(), maximumCorner.y(), minimumCorner.z()},
                {minimumCorner.x(), minimumCorner.y(), maximumCorner.z()},
                {maximumCorner.x(), minimumCorner.y(), maximumCorner.z()},
                {maximumCorner.x(), maximumCorner.y(), maximumCorner.z()},
                {minimumCorner.x(), maximumCorner.y(), maximumCorner.z()}};
        static const int kEdge[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        const int samplesPerEdge =
                std::clamp(int((maximumCorner - minimumCorner).maxCoeff() / std::max(1e-6f, voxel)), 24, 512);
        vertices.reserve(std::size_t(12 * (samplesPerEdge + 1)));
        for (const auto &edge: kEdge)
            for (int i = 0; i <= samplesPerEdge; ++i) {
                const float t = float(i) / float(samplesPerEdge);
                const Vector3f world = corner[edge[0]] + t * (corner[edge[1]] - corner[edge[0]]);
                const Vector3f view = SensorToView(world);
                vertices.push_back({{view.x(), view.y(), view.z()}, {color.r, color.g, color.b, 255}});
            }
        return vertices;
    }

    // The three overlays for one snapshot. Kept together so the sets can never disagree about
    // which snapshot they describe.
    void BuildMapStructureBoxes(const ep::ModelSnapshot &snapshot,
                                std::vector<PointVertex> &tileBoxes,
                                std::vector<PointVertex> &allocBox,
                                std::vector<PointVertex> &submapBoxes) {
        tileBoxes.clear();
        allocBox.clear();
        submapBoxes.clear();
        const float voxel = snapshot.voxel > 0.0f ? snapshot.voxel : 0.01f;
        for (const auto &box: snapshot.baseCoreBoxes) {
            const std::vector<PointVertex> edges = BoxEdgePoints(box.first, box.second, voxel, {40, 220, 220});
            tileBoxes.insert(tileBoxes.end(), edges.begin(), edges.end());
        }
        if (snapshot.hasAlloc)
            allocBox = BoxEdgePoints(snapshot.allocMin, snapshot.allocMax, voxel, {255, 160, 40});
        for (const auto &box: snapshot.denseBlockBoxes) {
            const std::vector<PointVertex> edges = BoxEdgePoints(box.first, box.second, voxel, {230, 60, 230});
            submapBoxes.insert(submapBoxes.end(), edges.begin(), edges.end());
        }
    }

    Color Ramp(float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        const float x = t * 4.0f;
        const int band = std::min(3, int(x));
        const float f = x - float(band);
        static constexpr float kStops[5][3] = {
                {0.10f, 0.20f, 0.90f},
                {0.10f, 0.85f, 0.90f},
                {0.15f, 0.85f, 0.20f},
                {0.95f, 0.90f, 0.15f},
                {0.95f, 0.20f, 0.15f}};
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

    // No VKBVH_HAS_REALSENSE guard: RealSenseD435 builds either way, and without the SDK Open
    // throws with its own message when this is actually called.
    std::unique_ptr<ep::IDepthProvider> OpenDevice(const Realsense::D435StreamOptions &stream) {
        auto camera = std::make_unique<Realsense::RealSenseD435>();
        camera->Open(stream);
        return camera;
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
                           .Option("--voxel", 0.05)
                           .Option("--truncation", 0.05)
                           .Option("--width", 640)
                           .Option("--height", 480)
                           .Option("--fps", 30)
                           // Fusion gate. Unlike the library default (off), a live scan turns the
                           // bootstrap run ON: a hand-held sweep's first frames are the ones most
                           // likely to seed the map from a pose nothing has corroborated yet.
                           .Option("--bootstrap-frames", 5)
                           .Option("--bootstrap-fitness", 0.70)
                           .Option("--min-fuse-fitness", 0.7) // 0 = off
                           .Option("--max-fuse-rmse", 0.02)   // 0 = off
                           // Weight only the occluded side of the truncation band down, instead
                           // of both sides equally (Bylow / Voxblox eq. 5).
                           .Option("--behind-dropoff")
                           // Truncation band = N * sigma_z(z); 0 = the fixed band.
                           .Option("--band-sigma", 0.0)
                           // Device-side. Raises the matcher's own rejection thresholds, which is a
                           // judgement made on the raw stereo pair -- information no gate this side
                           // of the cable can see. Live only: a recording was made after the
                           // matcher already decided.
                           .Option("--high-accuracy")
                           // --- gpu front end ---------------------------------------------
                           // The confidence a pixel must reach to survive. Measured on capture/:
                           // 0.9 leaves zero depth-cliff pixels where 0.0 leaves 1.06%, and it is
                           // half of why the long stretched streaks disappear (radius 4 is the
                           // other half -- the two are a pair, see NormalEstimation.md).
                           .Option("--score-threshold", 0.9)
                           // Device-side voxel thinning, BEFORE the readback, so it cuts the
                           // transfer too. Negative = follow --voxel, which is the size below
                           // which the map cannot represent the difference anyway; 0 = off.
                           .Option("--gpu-downsample", -1.0);

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
        config.map.behindSurfaceDropoff = arg.Has("--behind-dropoff");
        config.map.bandSigmaMultiplier = arg.ValueFloat("--band-sigma");
        // A camera keeps producing whether or not the map keeps up, so live must drop to bound
        // latency. A recording waits, so replaying it losslessly costs only wall-clock.
        config.acquisition.realTime = live;
        config.fusion.bootstrapConsecutiveFrames = arg.ValueInt("--bootstrap-frames");
        config.fusion.bootstrapMinFitness = arg.ValueFloat("--bootstrap-fitness");
        config.fusion.minimumFusionFitness = arg.ValueFloat("--min-fuse-fitness");
        config.fusion.maximumFusionRmse = arg.ValueFloat("--max-fuse-rmse");

        const bool highAccuracyPreset = arg.Has("--high-accuracy");

        // The front end publishes what it discarded here. Owned by the caller because the front
        // end is built on the acquisition thread and never handed back.
        const auto gpuStats = std::make_shared<ep::GpuFrontEndStats>();

        // Device-side thinning, tied to the map: half the base voxel is the map's detail level, so
        // thinning to it removes points the map provably cannot tell apart while keeping every
        // point the detail level CAN represent. Derived rather than set, because the two drifting
        // apart is silent -- too coarse quietly deletes representable surface, too fine pays for a
        // pass that removes nothing.
        const auto detailVoxelFor = [](float baseVoxel) { return baseVoxel * 0.5f; };
        const float gpuDownsampleVoxel = arg.ValueFloat("--gpu-downsample");

        Realsense::D435StreamOptions stream;
        stream.width = width;
        stream.height = height;
        stream.fps = fps;
        stream.enableInfrared = false; // this tool scores depth only
        stream.visualPreset = highAccuracyPreset ? "high-accuracy" : "default";

        config.acquisition.stream = stream;
        config.acquisition.scoreThreshold = arg.ValueFloat("--score-threshold");
        // The flag now only says WHETHER to thin; the size follows the map voxel. A negative
        // value (the default) means on.
        config.acquisition.downSample.enabled = gpuDownsampleVoxel != 0.0f;
        config.acquisition.downSample.detailVoxelMeters = detailVoxelFor(config.map.baseVoxel);
        config.acquisition.gpuStats = gpuStats;
        // Already thinned on the device, before the readback. Thinning again on the host would only
        // pay for the same reduction twice.
        config.acquisition.downsampleVoxel = 0.0f;

        if (recordDirectory.empty()) {
            config.acquisition.source = live ? ep::EAcquisitionSource::Realsense
                                             : ep::EAcquisitionSource::RealsenseFile;
            config.acquisition.recordingDirectory = replayDirectory;
        } else {
            // Recording needs a recorder in the chain, and the enum describes a device, not a
            // chain. So this one case builds it. It costs nothing beyond the write: the recorder's
            // file format is the device's own Z16, so the frame reaches the kernels untouched.
            config.acquisition.makeProvider = [=]() -> std::unique_ptr<ep::IDepthProvider> {
                return std::make_unique<Realsense::RealSenseD435Recorder>(OpenDevice(stream),
                                                                          recordDirectory);
            };
        }

        std::printf("source    : %s\n", live ? "live device" : replayDirectory.c_str());
        std::printf("tracker   : %s\n", trackerName.c_str());
        std::printf("map       : voxel %.4f m, truncation %.4f m\n", config.map.baseVoxel,
                    config.map.truncation);

        // Constructed before the window on purpose: the Pipeline constructor builds its stages,
        // which opens the device synchronously. A missing camera or an occupied recording
        // directory therefore reports as one CLI line, with no window ever appearing -- and the
        // device is opened exactly once, which a separate pre-flight probe would not manage.
        // TrackerRegistry::Factory takes no arguments, so building a configured tracker is a
        // create-then-configure pair. Kept in one place so the option panel's rebuild and the
        // initial construction cannot drift.
        Registration::RegistrationParam trackerParam;
        trackerParam.maxCorrDist = 0.0f; // 0 = leave Track's own voxel-derived value alone
        trackerParam.minFitness = 0.0f;
        trackerParam.maxStepMeters = 0.0f; // 0 = the tracker's voxel-scaled default
        trackerParam.minInliers = 0;       // 0 = leave the solver default

        const auto makeTracker = [&registry](const std::string &name,
                                             const Registration::RegistrationParam &param)
                -> std::unique_ptr<ep::Tracker> {
            std::unique_ptr<ep::Tracker> made = registry.Create(name);
            if (!made) throw std::runtime_error("realsense_scan: unknown tracker '" + name + "'");
            made->Configure(param); // reaches icp, icp-cpu and icp+global alike
            return made;
        };

        std::unique_ptr<ep::Tracker> tracker = makeTracker(trackerName, trackerParam);
        std::printf("depth     : score >=%.2f, normals %s r%d, downsample %s%s\n",
                    config.acquisition.scoreThreshold,
                    config.acquisition.normal.estimator.c_str(),
                    config.acquisition.normal.planeFitRadius,
                    config.acquisition.downSample.enabled ? "on" : "off",
                    highAccuracyPreset ? ", high-accuracy preset" : "");
        std::printf("denoise   : behind-dropoff %s, band %.1f sigma\n",
                    arg.Has("--behind-dropoff") ? "on" : "off", arg.ValueFloat("--band-sigma"));

        ep::Pipeline pipeline(config, std::move(tracker));

        // Options are edited into `pending` and only reach the pipeline on Apply. Reconfigure
        // replays from frame 0 by design -- an accumulating map cannot be retro-changed -- and
        // that is also what makes a comparison meaningful: a map that is half one setting and half
        // another measures nothing.
        struct PendingOptions {
            ep::Pipeline::Config config;
            std::string trackerName;
            Registration::RegistrationParam trackerParam;
            bool dirty = false;
        };
        PendingOptions pending{config, trackerName, trackerParam, false};
        PendingOptions applied = pending;
        std::vector<std::string> trackerNames = registry.Names();


        if (!pipeline.VisualPresetRefusal().empty())
            std::printf("preset    : NOT applied -- %s\n", pipeline.VisualPresetRefusal().c_str());

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
            bool showTileBox = true, showAllocBox = true, showSubmapBox = true;
            EColorMode colorMode = EColorMode::Normal;
            float pointSize = 2.0f;
            float minimumWeight = 1.0f;
            float weightRampMax = 20.0f;
            bool paused = false;
        } state;

        // Reconfigure rebuilds every worker stage in place and replays from frame 0. The viewer's
        // own accumulators are reset with it, or the next frame's counters read as a continuation
        // of a run that no longer exists.
        // Shown in the Visualize panel; written by a worker rethrow and by a failed Apply.
        std::string workerError;

        // Runs on the RENDER thread, inside the ImGui callback. An escaping exception there takes
        // the process down -- which is what a failed rebuild used to do: Reconfigure destroys the
        // old stages before building the new ones (it must, the old one still holds the device), so
        // a device that will not reopen throws with every stage already gone.
        const auto applyPending = [&] {
            try {
                // Re-derived here, not just where it is displayed: the panel computes it before
                // the Map section runs, so a base-voxel edit would otherwise be applied with the
                // previous frame's detail voxel.
                pending.config.acquisition.downSample.detailVoxelMeters =
                        detailVoxelFor(pending.config.map.baseVoxel);
                pipeline.Reconfigure(pending.config,
                                     makeTracker(pending.trackerName, pending.trackerParam));
                applied = pending;
                applied.dirty = false;
                pending.dirty = false;
                pipeline.SetPaused(state.paused);
                workerError.clear();
            } catch (const std::exception &e) {
                // The pipeline is now unbuilt and stays that way; Pipeline's accessors tolerate it
                // so this panel keeps drawing. Say so instead of dying, and keep the edits so they
                // can be applied again once the device is free.
                workerError = std::string("apply failed: ") + e.what();
            }
        };

        auto applyVisibility = [&] {
            points->SetVisible(kSetSurface, state.showSurface);
            points->SetVisible(kSetNew, state.showNew);
            points->SetVisible(kSetTileBox, state.showTileBox);
            points->SetVisible(kSetAllocBox, state.showAllocBox);
            points->SetVisible(kSetSubmapBox, state.showSubmapBox);
        };
        applyVisibility();

        std::vector<PointVertex> surfaceVertices, newVertices;
        std::vector<PointVertex> tileBoxVertices, allocBoxVertices, submapBoxVertices;
        std::shared_ptr<const ep::ModelSnapshot> snapshot;

        // Three panels, matching the agreed layout:
        //   left top     visualisation options -- what is DRAWN, applied immediately
        //   left bottom  per-stage statistics  -- read-only, all stages at once
        //   right        per-stage OPTIONS in tabs -- what is COMPUTED, applied on Apply
        //
        // The split is by when a control takes effect, not by subject. Anything on the left is a
        // view change and lands on the next frame; anything on the right rebuilds the pipeline and
        // restarts from frame 0, which is why they must not sit in the same panel.
        imgui->SetUi([&] {
            const ep::PipelineStats stats = pipeline.GetStats();
            const float columnWidth = 320.0f;
            const float logicalHeight = float(app.GetWindow().FramebufferSize().height) /
                                        ImGui::GetIO().DisplayFramebufferScale.y;
            const float logicalWidth = float(app.GetWindow().FramebufferSize().width) /
                                       ImGui::GetIO().DisplayFramebufferScale.x;
            constexpr ImGuiWindowFlags kPanelFlags =
                    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;

            ///////////////////////////////////////////////////////////////////////////////////////
            // Left top -- visualisation
            ///////////////////////////////////////////////////////////////////////////////////////
            ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(columnWidth, 0.0f), ImGuiCond_Always);
            ImGui::Begin("Visualize", nullptr, kPanelFlags);
            if (!workerError.empty())
                ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "%s", workerError.c_str());
            if (ImGui::Button(state.paused ? "resume" : "pause")) {
                state.paused = !state.paused;
                pipeline.SetPaused(state.paused);
            }

            ImGui::SeparatorText("Layers");
            if (ImGui::Checkbox("surface", &state.showSurface)) applyVisibility();
            if (ImGui::Checkbox("new this frame", &state.showNew)) applyVisibility();
            if (ImGui::Checkbox("tile windows", &state.showTileBox)) applyVisibility();
            if (ImGui::Checkbox("allocated box", &state.showAllocBox)) applyVisibility();
            // Empty unless the map runs submaps, which realsense_scan leaves off; kept so turning
            // them on needs no viewer change.
            if (ImGui::Checkbox("submap regions", &state.showSubmapBox)) applyVisibility();
            ImGui::Text("boxes: %zu tile / %s alloc / %zu submap",
                        snapshot ? snapshot->baseCoreBoxes.size() : 0u,
                        (snapshot && snapshot->hasAlloc) ? "1" : "0",
                        snapshot ? snapshot->denseBlockBoxes.size() : 0u);

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
            const float visualizeBottom = ImGui::GetWindowPos().y + ImGui::GetWindowSize().y;
            ImGui::End();

            ///////////////////////////////////////////////////////////////////////////////////////
            // Left bottom -- statistics, every stage at once
            ///////////////////////////////////////////////////////////////////////////////////////
            ImGui::SetNextWindowPos(ImVec2(10.0f, visualizeBottom + 10.0f), ImGuiCond_Always);
            // Clamped: Visualize sizes itself to its content, so on a short window it can reach
            // past where Statistics would start, and a negative height is a broken window rather
            // than a small one.
            ImGui::SetNextWindowSize(ImVec2(columnWidth,
                                            std::max(120.0f, logicalHeight - visualizeBottom - 20.0f)),
                                     ImGuiCond_Always);
            ImGui::Begin("Statistics", nullptr, kPanelFlags);

            ImGui::SeparatorText("Acquire");
            ImGui::Text("frames        %llu", (unsigned long long) stats.acquiredFrames);
            ImGui::Text("per frame     %.2f ms", stats.acquireMsAvg);
            ImGui::Text("queued        %zu", stats.captureDepth);
            ImGui::Text("points kept   %u this fr. / %llu total",
                        gpuStats->lastFramePoints.load(),
                        (unsigned long long) gpuStats->emittedPoints.load());
            // Two different refusals, never summed: the border is the estimator's domain and scales
            // with the stencil, while "support" is the scene refusing the pixel.
            ImGui::Text("normal reject %llu border / %llu support",
                        (unsigned long long) gpuStats->normalOutOfDomain.load(),
                        (unsigned long long) gpuStats->normalNoSupport.load());
            // Both ceilings fail OPEN -- the point is dropped, the frame still looks fine -- so they
            // are loud when nonzero and invisible otherwise.
            {
                const unsigned long long insertFailures = gpuStats->downSampleInsertFailures.load();
                const unsigned long long outOfRange = gpuStats->downSampleOutOfRange.load();
                if (insertFailures || outOfRange)
                    ImGui::TextColored(ImVec4(1, 0.7f, 0.2f, 1),
                                       "downsample    %llu full / %llu out of range",
                                       insertFailures, outOfRange);
            }

            ImGui::SeparatorText("Register");
            ImGui::Text("tracker       %s", applied.trackerName.c_str());
            ImGui::Text("frames        %llu", (unsigned long long) stats.alignedFrames);
            ImGui::Text("per frame     %.2f ms", stats.alignMsAvg);
            // The number that says whether tracking is working. It is a real residual here, unlike
            // on the synthetic scans where identity is the only viable tracker.
            ImGui::Text("residual rmse %.5f m", stats.trackerRmseAvg);
            ImGui::Text("queued        %zu  dropped %zu", stats.trackDepth, stats.trackDropped);

            ImGui::SeparatorText("Integrate");
            ImGui::Text("frames        %llu", (unsigned long long) stats.integratedFrames);
            ImGui::Text("per frame     %.2f ms", stats.integrateMsAvg);
            ImGui::Text("skipped       %llu", (unsigned long long) stats.skippedFusions);
            // While the gate holds, the map deliberately stays at its seed frame. Without this the
            // screen is indistinguishable from a pipeline that has simply stopped working.
            if (!stats.fusionArmed)
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f), "BOOTSTRAP  holding %llu fr.",
                                   (unsigned long long) stats.bootstrapHeldFrames);
            if (stats.fusionRejectedByFitness > 0 || stats.fusionRejectedByRmse > 0)
                ImGui::Text("gate reject   %llu fitness / %llu rmse",
                            (unsigned long long) stats.fusionRejectedByFitness,
                            (unsigned long long) stats.fusionRejectedByRmse);

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

            ///////////////////////////////////////////////////////////////////////////////////////
            // Right -- per-stage options, one tab per stage
            ///////////////////////////////////////////////////////////////////////////////////////
            ImGui::SetNextWindowPos(ImVec2(logicalWidth - columnWidth - 10.0f, 10.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(columnWidth, logicalHeight - 20.0f), ImGuiCond_Always);
            ImGui::Begin("Options", nullptr, kPanelFlags);

            // Edited into `pending`; nothing reaches the pipeline until Apply. One Apply for the
            // whole list, because Reconfigure rebuilds the entire pipeline from one Config -- a
            // per-section Apply would imply the stages restart independently, which they do not.
            bool changed = false;

            // Every stage listed top to bottom, sections only. The list scrolls inside a child
            // sized to leave the Apply row its space; without the child it would simply grow past
            // the bottom of the window and Apply -- the one control that makes any of it take
            // effect -- would become unreachable.
            const float applyRowHeight = ImGui::GetFrameHeightWithSpacing() +
                                         ImGui::GetStyle().ItemSpacing.y * 2.0f;
            ImGui::BeginChild("options", ImVec2(0.0f, -applyRowHeight), false);

            ImGui::SeparatorText("Acquire");
            changed |= ImGui::SliderFloat("score >=", &pending.config.acquisition.scoreThreshold, 0.0f, 1.0f, "%.2f");
            changed |= ImGui::Checkbox("normals", &pending.config.acquisition.normal.enabled);
            if (pending.config.acquisition.normal.enabled)
                changed |= ImGui::SliderInt("plane-fit r", &pending.config.acquisition.normal.planeFitRadius, 1, 8);
            changed |= ImGui::Checkbox("downsample", &pending.config.acquisition.downSample.enabled);
            // Read-only on purpose: it is half the map's base voxel, kept in step with the Map
            // section below rather than editable here, so the two cannot be set to disagree.
            pending.config.acquisition.downSample.detailVoxelMeters =
                    detailVoxelFor(pending.config.map.baseVoxel);
            ImGui::BeginDisabled();
            ImGui::InputFloat("detail voxel", &pending.config.acquisition.downSample.detailVoxelMeters,
                              0.0f, 0.0f, "%.4f");
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("= base voxel / 2");

            ImGui::SeparatorText("Register");
            {
                int current = 0;
                for (std::size_t i = 0; i < trackerNames.size(); ++i)
                    if (trackerNames[i] == pending.trackerName) current = int(i);
                std::vector<const char *> labels;
                labels.reserve(trackerNames.size());
                for (const std::string &name: trackerNames) labels.push_back(name.c_str());
                if (ImGui::Combo("tracker", &current, labels.data(), int(labels.size()))) {
                    pending.trackerName = trackerNames[std::size_t(current)];
                    changed = true;
                }
            }
            // 0 on any of these means "leave the tracker's own default", which for maxCorrDist and
            // maxStepMeters is derived from the map voxel -- so a hard-coded value here is usually
            // worse than none. Widening maxCorrDist grows the GPU LocalGrid cell count cubically.
            changed |= ImGui::InputFloat("max corr dist", &pending.trackerParam.maxCorrDist, 0.0f, 0.0f, "%.4f");
            changed |= ImGui::SliderFloat("min fitness", &pending.trackerParam.minFitness, 0.0f, 1.0f, "%.2f");
            changed |= ImGui::InputFloat("max step [m]", &pending.trackerParam.maxStepMeters, 0.0f, 0.0f, "%.4f");
            changed |= ImGui::InputInt("min inliers", &pending.trackerParam.minInliers);

            ImGui::SeparatorText("Map");
            changed |= ImGui::InputFloat("base voxel", &pending.config.map.baseVoxel, 0.0f, 0.0f, "%.4f");
            changed |= ImGui::InputFloat("truncation", &pending.config.map.truncation, 0.0f, 0.0f, "%.4f");
            changed |= ImGui::Checkbox("submap", &pending.config.map.submap);
            changed |= ImGui::Checkbox("point-to-plane", &pending.config.map.pointToPlane);
            changed |= ImGui::SliderFloat("confidence", &pending.config.map.confidence, 0.0f, 1.0f, "%.2f");
            changed |= ImGui::Checkbox("hermite", &pending.config.map.hermite);

            ImGui::SeparatorText("Fusion gate");
            // The gates that decide whether a TRACKED frame is allowed into the map at all,
            // layered on top of ShouldFuse's per-cause policy. 0 disables each.
            changed |= ImGui::InputInt("bootstrap frames", &pending.config.fusion.bootstrapConsecutiveFrames);
            changed |= ImGui::SliderFloat("bootstrap fitness", &pending.config.fusion.bootstrapMinFitness, 0.0f, 1.0f, "%.2f");
            changed |= ImGui::SliderFloat("min fuse fitness", &pending.config.fusion.minimumFusionFitness, 0.0f, 1.0f, "%.2f");
            changed |= ImGui::InputFloat("max fuse rmse", &pending.config.fusion.maximumFusionRmse, 0.0f, 0.0f, "%.4f");

            ImGui::EndChild();

            pending.dirty = pending.dirty || changed;

            // Outside the child, so it stays put however far the list has been scrolled.
            ImGui::Separator();
            if (!pending.dirty) ImGui::BeginDisabled();
            if (ImGui::Button("Apply & restart")) {
                // On a live camera the accumulated map is discarded and cannot be recovered, so
                // that case asks first. A replay just plays again.
                if (live) ImGui::OpenPopup("confirm restart");
                else
                    applyPending();
            }
            ImGui::SameLine();
            if (ImGui::Button("Revert")) {
                pending = applied;
                pending.dirty = false;
            }
            if (!pending.dirty) ImGui::EndDisabled();
            ImGui::SameLine();
            if (pending.dirty) ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f), "unapplied");
            else
                ImGui::TextDisabled("applied");

            if (ImGui::BeginPopupModal("confirm restart", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::TextUnformatted("This restarts acquisition and DISCARDS the current map.");
                if (ImGui::Button("Restart")) {
                    applyPending();
                    ImGui::CloseCurrentPopup();
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
            }
            ImGui::End();
        });

        app.GetView().SetScene(&scene);
        app.GetView().SetCamera(&camera);
        app.GetView().SetRenderGraph(&graph);

        Engine::Render::MouseListenerGroup mouse(app.GetWindow().Mouse());
        mouse.Add(Engine::Render::MouseEventType::Drag, [&](Engine::Render::MouseEvent &e) {
            if (ImGui::GetIO().WantCaptureMouse) return;
            const VkExtent2D size = app.GetWindow().FramebufferSize();
            if (size.width == 0 || size.height == 0) return;

            // Right drag translates the orbit target in the view plane. Camera::Pan takes a
            // per-event delta rather than an absolute position, so unlike the trackball it needs
            // no begin/end -- there is no accumulated state to reset.
            if (e.button == Engine::Render::MouseButton::Right) {
                camera.Pan(e.deltaX, e.deltaY, int(size.width), int(size.height));
                e.handled = true;
                return;
            }
            if (e.button != Engine::Render::MouseButton::Left) return;
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
                    if (modelAdvanced) {
                        BuildMapStructureBoxes(*snapshot, tileBoxVertices, allocBoxVertices,
                                               submapBoxVertices);
                        points->SetPointSet(kSetTileBox, tileBoxVertices);
                        points->SetPointSet(kSetAllocBox, allocBoxVertices);
                        points->SetPointSet(kSetSubmapBox, submapBoxVertices);
                    }
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

#include "AsyncTsdfMapper.h"
#include "ImGuiPass.h"
#include "PointCloudPass.h"
#include "VoxelFillDebug.h"

#include "Engine/Core/Context.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"
#include "Engine/Spatial/AdvancedTSDF.h"
#include "Engine/Spatial/SubmapAdvancedTSDF.h"
#include "Engine/Spatial/TiledAdvancedTSDF.h"

#include "utilities/ArgParser.h"
#include "utilities/PointCloudIO.h"
#include "utilities/StageProfiler.h"

#include "imgui.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Eigen::Vector3f;
using Engine::Spatial::AdvancedEntry;
namespace fs = std::filesystem;

namespace {

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Reused verbatim from tsdf_folder_eval.cpp: nextPow2, estimateCamera. Arg parsing via the
    // fluent util::ArgParser (utilities/ArgParser.h); PLY point I/O from utilities/PointCloudIO.h
    // (util::LoadPly).
    ///////////////////////////////////////////////////////////////////////////////////////////

    uint32_t nextPow2(uint32_t v) {
        if (v <= 1) return 1;
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

    // Estimate the camera position for a captured cloud: the visible points face the sensor, so
    // the camera sits along the mean normal, outside the cloud.
    Vector3f estimateCamera(const std::vector<Vector3f> &pts, const std::vector<Vector3f> &nrm) {
        Vector3f centroid = Vector3f::Zero(), meanN = Vector3f::Zero();
        for (const auto &p: pts) centroid += p;
        for (const auto &n: nrm) meanN += n;
        centroid /= float(std::max<size_t>(1, pts.size()));
        Vector3f mn = pts[0], mx = pts[0];
        for (const auto &p: pts) {
            mn = mn.cwiseMin(p);
            mx = mx.cwiseMax(p);
        }
        const float diag = (mx - mn).norm();
        if (meanN.norm() > 1e-6f) meanN.normalize();
        else
            meanN = Vector3f(0, 0, 1);
        return centroid + std::max(1.0f, 3.0f * diag) * meanN;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Reused verbatim from object_scan_viewer.cpp: pushCloud, cameraMarker.
    ///////////////////////////////////////////////////////////////////////////////////////////

    void pushCloud(std::vector<PointVertex> &out, const std::vector<Vector3f> &pts,
                   uint8_t r, uint8_t g, uint8_t b) {
        for (const auto &p: pts) out.push_back({{p.x(), p.y(), p.z()}, {r, g, b, 255}});
    }

    // A small red camera marker: the eye plus a short segment toward the origin (view direction).
    std::vector<PointVertex> cameraMarker(const Vector3f &eye) {
        std::vector<PointVertex> v;
        for (int i = 0; i < 30; ++i) {
            const float t = float(i) / 30.0f * 0.15f;
            const Vector3f p = eye * (1.0f - t);
            v.push_back({{p.x(), p.y(), p.z()}, {255, 40, 40, 255}});
        }
        return v;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // New: bounding-box wireframes as point sets (PointCloudPass draws points only).
    ///////////////////////////////////////////////////////////////////////////////////////////

    // Sample points along the 12 edges of an AABB [mn,mx]; density ~ one point per voxel along the
    // longest edge (capped), so the box reads as a wireframe in the point renderer.
    std::vector<PointVertex> boxEdges(const Vector3f &mn, const Vector3f &mx, float voxel,
                                      uint8_t r, uint8_t g, uint8_t b) {
        std::vector<PointVertex> v;
        const Vector3f c[8] = {{mn.x(), mn.y(), mn.z()}, {mx.x(), mn.y(), mn.z()}, {mx.x(), mx.y(), mn.z()}, {mn.x(), mx.y(), mn.z()}, {mn.x(), mn.y(), mx.z()}, {mx.x(), mn.y(), mx.z()}, {mx.x(), mx.y(), mx.z()}, {mn.x(), mx.y(), mx.z()}};
        static const int E[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        const int per = std::clamp(int((mx - mn).maxCoeff() / std::max(1e-6f, voxel)), 24, 512);
        for (const auto &e: E)
            for (int i = 0; i <= per; ++i) {
                const float t = float(i) / float(per);
                const Vector3f p = c[e[0]] + t * (c[e[1]] - c[e[0]]);
                v.push_back({{p.x(), p.y(), p.z()}, {r, g, b, 255}});
            }
        return v;
    }

    // AABB of downloaded voxel entries, expanded by half a voxel to the true voxel extent.
    bool entriesAabb(const std::vector<AdvancedEntry> &e, float voxel, Vector3f &mn, Vector3f &mx) {
        if (e.empty()) return false;
        mn = Vector3f::Constant(1e30f);
        mx = Vector3f::Constant(-1e30f);
        for (const auto &en: e) {
            mn = mn.cwiseMin(en.center);
            mx = mx.cwiseMax(en.center);
        }
        const Vector3f h = Vector3f::Constant(0.5f * voxel);
        mn -= h;
        mx += h;
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // New: build the "occupied" (colored by mode; below-threshold dimmed grey or skipped) and
    // "new this frame" highlight point sets from downloaded AdvancedTSDF entries.
    ///////////////////////////////////////////////////////////////////////////////////////////

    void buildVoxelSets(const std::vector<AdvancedEntry> &entries, const std::vector<char> &isNew,
                        voxdbg::ColorMode mode, float trunc, float wMax, float wThresh, bool hideBelow,
                        const std::vector<int> &firstFrame, int nFrames,
                        std::vector<PointVertex> &occupied, std::vector<PointVertex> &newThis) {
        occupied.clear();
        newThis.clear();
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const AdvancedEntry &e = entries[i];
            const bool below = voxdbg::belowThreshold(e.weight, wThresh);
            if (below && hideBelow) continue;
            voxdbg::Rgba c;
            switch (mode) {
                case voxdbg::ColorMode::TsdfSign:
                    c = voxdbg::tsdfColor(e.tsdf, trunc);
                    break;
                case voxdbg::ColorMode::Weight:
                    c = voxdbg::weightColor(e.weight, wMax);
                    break;
                case voxdbg::ColorMode::FillFrame:
                    c = voxdbg::fillFrameColor(i < firstFrame.size() ? firstFrame[i] : -1, nFrames);
                    break;
                case voxdbg::ColorMode::Direction:
                    c = voxdbg::directionColor(uint8_t(e.direction));
                    break;
            }
            if (below) c = {80, 80, 80, 255}; // extraction drop-out, dimmed
            occupied.push_back({{e.center.x(), e.center.y(), e.center.z()}, {c[0], c[1], c[2], c[3]}});
            if (i < isNew.size() && isNew[i])
                newThis.push_back({{e.center.x(), e.center.y(), e.center.z()}, {255, 240, 40, 255}});
        }
    }

} // namespace

int main(int argc, char **argv) {
    try {
        util::ArgParser arg =
                util::BuildArgParser(argc, argv)
                        .Must("--dir", "usage: voxel_fill_debugger --dir <folder> [--voxel v] "
                                       "[--trunc t] [--no-p2p] [--conf L] [--hermite] [--wthresh w] "
                                       "[--tile-hash N] [--block V] [--detail-k K] [--dump]")
                        .Option("--voxel") // default is runtime-computed (extent / 200)
                        .Option("--trunc") // default is runtime-computed (voxel * 3)
                        .Option("--conf", 0.5)
                        .Option("--wthresh", 0.0)
                        .Option("--tile-hash", 1 << 20)
                        .Option("--block", 32)
                        .Option("--detail-k", 4.0);
        if (!arg) return 2;
        const std::string dir = arg.Value("--dir");
        if (!fs::is_directory(dir)) {
            std::cerr << "not a directory: " << dir << "\n";
            return 2;
        }

        std::vector<std::string> framePaths;
        for (const auto &e: fs::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            const std::string name = e.path().filename().string();
            if (name.rfind("ground_truth", 0) == 0) continue;
            if (e.path().extension() == ".ply" && name.rfind("frame_", 0) == 0)
                framePaths.push_back(e.path().string());
        }
        std::sort(framePaths.begin(), framePaths.end());
        if (framePaths.empty()) throw std::runtime_error("no frame_*.ply found in " + dir);

        // Frames use asyncmap::MapperFrame directly so they can be moved into the mapper (which is
        // shared read-only with the render thread for the input-cloud/camera layers).
        std::vector<asyncmap::MapperFrame> frames;
        Vector3f bbMin = Vector3f::Constant(1e30f), bbMax = Vector3f::Constant(-1e30f);
        size_t maxFramePts = 0;
        for (const auto &p: framePaths) {
            asyncmap::MapperFrame fr;
            if (!util::LoadPly(p, fr.pts, fr.nrm) || fr.nrm.size() != fr.pts.size()) {
                std::fprintf(stderr, "skip (no normals): %s\n", p.c_str());
                continue;
            }
            fr.cam = estimateCamera(fr.pts, fr.nrm);
            for (const auto &q: fr.pts) {
                bbMin = bbMin.cwiseMin(q);
                bbMax = bbMax.cwiseMax(q);
            }
            maxFramePts = std::max(maxFramePts, fr.pts.size());
            frames.push_back(std::move(fr));
        }
        if (frames.empty()) throw std::runtime_error("no usable frames (need per-point normals)");

        const bool p2p = !arg.Has("--no-p2p");
        const float conf = arg.ValueFloat("--conf");
        const bool hermite = arg.Has("--hermite");
        const Vector3f span = bbMax - bbMin;
        const float extent = span.norm();
        const float voxel = arg.ValueFloat("--voxel", extent / 200.0f);
        const float trunc = arg.ValueFloat("--trunc", voxel * 3.0f);
        const float wThreshArg = arg.ValueFloat("--wthresh");

        // SubmapAdvancedTSDF: base TiledAdvancedTSDF at `voxel` (all points) + a detail level at
        // voxel/2 in DENSE blocks only. Density is precomputed from ALL frames up front so the dense
        // set is fixed while scrubbing; the dense blocks are the "submap regions" (drawn magenta).
        // --block/--detail-k tune density; --tile-hash sizes both levels (detail = half voxel needs
        // a bigger hash). Sparse scenes -> no dense blocks -> behaves like a plain tiled map.
        const uint32_t tileHash =
                nextPow2(uint32_t(arg.ValueFloat("--tile-hash")));
        const uint32_t maxPts = nextPow2(uint32_t(std::max<std::size_t>(maxFramePts, 1u << 15)));
        const int blockVoxels = int(arg.ValueFloat("--block"));
        const float detailK = arg.ValueFloat("--detail-k");

        const int nFrames = int(frames.size());
        const float detailVoxel = voxel * 0.5f; // finest level -> key voxel so base+detail keys are unique

        std::printf("dir       : %s  (%d frames, extent %.4f)\n", dir.c_str(), nFrames, extent);
        std::printf("submap    : base %.4f + detail %.4f, block %d vox, detail-k %.1f, per-tile "
                    "hash %u\n",
                    voxel, detailVoxel, blockVoxels, detailK, tileHash);

        // Mapper config, shared by the headless --dump path and the live async worker.
        asyncmap::Config cfg;
        cfg.baseVoxel = voxel;
        cfg.truncation = trunc;
        cfg.blockVoxels = blockVoxels;
        cfg.detailK = detailK;
        cfg.tileHash = tileHash;
        cfg.maxPoints = maxPts;
        cfg.quality = {3, 4, true};
        cfg.pointToPlane = p2p;
        cfg.confidence = conf;
        cfg.hermite = hermite;

        // Headless per-frame stats: SYNCHRONOUS submap (baseline timing), no window/render deps.
        if (arg.Has("--dump") || arg.Has("--no-view")) {
            Engine::Core::Context ctx;
            Engine::Spatial::SubmapAdvancedTSDF submap;
            submap.Build(ctx, voxel, trunc, blockVoxels, detailK, tileHash, maxPts);
            submap.SetIntegrationQuality(cfg.quality);
            submap.SetPointToPlane(p2p);
            submap.SetConfidenceWeight(conf);
            submap.SetHermitePosition(hermite);
            for (const auto &fr: frames) submap.AddDensity(fr.pts); // density from all frames
            submap.FinalizeDensity();
            voxdbg::FillTracker tracker(detailVoxel);
            util::StageProfiler prof;
            for (int f = 0; f < nFrames; ++f) {
                {
                    util::ScopedStageTimer t(prof, "integrate");
                    submap.Integrate(frames[f].pts, frames[f].nrm, frames[f].cam);
                }
                std::vector<AdvancedEntry> entries;
                {
                    util::ScopedStageTimer t(prof, "download");
                    entries = submap.DownloadEntries();
                }
                std::vector<char> isNew;
                {
                    util::ScopedStageTimer t(prof, "tracker");
                    isNew = tracker.update(entries, f);
                }
                std::size_t below = 0;
                for (const auto &e: entries)
                    if (voxdbg::belowThreshold(e.weight, wThreshArg)) ++below;
                std::size_t nnew = 0;
                for (char c: isNew) nnew += (c != 0);
                Vector3f aMn, aMx, aSz = Vector3f::Zero();
                if (entriesAabb(entries, voxel, aMn, aMx)) aSz = aMx - aMn;
                std::printf("frame %3d: occupied %zu  new %zu  below-wthresh %zu  base/detail tiles "
                            "%u/%u  allocBox(%.2f,%.2f,%.2f)\n",
                            f, entries.size(), nnew, below, submap.BaseTileCount(),
                            submap.DetailTileCount(), aSz.x(), aSz.y(), aSz.z());
            }
            std::printf("\n%s", prof.Report("--dump per-frame TSDF pipeline").c_str());
            std::printf("[--dump] done.\n");
            return 0;
        }

        ///////////////////////////////////////////////////////////////////////////////////////
        // Live viewer — the TSDF runs on a background AsyncTsdfMapper (its own device); this thread
        // only renders the latest immutable snapshot, so the window stays smooth regardless of
        // integrate/download cost. Frames are shared read-only with the worker.
        ///////////////////////////////////////////////////////////////////////////////////////

        auto sharedFrames =
                std::make_shared<std::vector<asyncmap::MapperFrame>>(std::move(frames));
        asyncmap::AsyncTsdfMapper mapper;
        mapper.Start(cfg, sharedFrames);
        mapper.RequestFrame(0);

        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {1280, 800, "Voxel Fill Debugger"};
        Engine::Render::Application app(descriptor);
        Engine::Core::Context &appCtx = app.GetContext();

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D ext = app.GetSwapChain().Extent();
        const Vector3f center = 0.5f * (bbMin + bbMax);
        camera.SetPerspective(60.0f * 3.14159265f / 180.0f, float(ext.width) / float(ext.height),
                              std::max(1e-3f, extent * 1e-3f), std::max(10.0f, extent * 6.0f));
        camera.SetOrbit(vkMath::Vec3(center.x(), center.y(), center.z()), extent * 2.0f);

        const std::string shaderDir = VOXDBG_SHADER_DIR;
        Engine::Render::RenderGraph graph;
        auto pcOwned = std::make_unique<PointCloudPass>(appCtx, app.GetSwapChain().Format(), shaderDir);
        PointCloudPass *pc = pcOwned.get();
        graph.AddPass(std::move(pcOwned));
        auto &glfwWindow = static_cast<Engine::Render::GlfwWindow &>(app.GetWindow());
        auto imguiOwned = std::make_unique<ImGuiPass>(appCtx, glfwWindow.Handle(),
                                                      app.GetSwapChain().Format(),
                                                      app.GetSwapChain().ImageCount());
        ImGuiPass *imgui = imguiOwned.get();
        graph.AddPass(std::move(imguiOwned));
        pc->SetPointSize(3.0f);

        struct State {
            int frame = 0;
            bool playing = false;
            float fps = 12.0f;
            voxdbg::ColorMode mode = voxdbg::ColorMode::TsdfSign;
            float wThresh = 0.0f;
            bool hideBelow = false;
            bool showOccupied = true, showNew = true, showInput = true, showCamera = true;
            bool showWindowBox = true, showAllocBox = true, showSubmapBox = true;
            int lastRequested = -1; // last frame handed to the mapper
            float wMax = 1.0f;
            bool dirty = false;
        } state;
        state.wThresh = wThreshArg;

        util::StageProfiler prof;                        // render-thread stages only
        std::shared_ptr<const asyncmap::MapSnapshot> snap; // latest consumed snapshot

        // Rebuild the render point-sets from an immutable snapshot (worker output). SetPointSet
        // reallocates PointCloudPass buffers; the prior frame's command buffer may still be in
        // flight, so wait once here before touching them.
        auto refreshSets = [&](const asyncmap::MapSnapshot &s) {
            {
                util::ScopedStageTimer t(prof, "waitIdle");
                vkDeviceWaitIdle(appCtx.device);
            }
            std::vector<PointVertex> occ, nw;
            {
                util::ScopedStageTimer t(prof, "buildSets");
                buildVoxelSets(s.entries, s.isNew, state.mode, trunc, state.wMax, state.wThresh,
                               state.hideBelow, s.firstFrame, nFrames, occ, nw);
            }
            util::ScopedStageTimer t(prof, "upload"); // CPU box gen + all 7 SetPointSet uploads
            pc->SetPointSet(0, occ);
            pc->SetPointSet(1, nw);
            const int pf = std::clamp(s.processedFrame, 0, nFrames - 1);
            std::vector<PointVertex> in;
            pushCloud(in, (*sharedFrames)[pf].pts, 100, 110, 120);
            pc->SetPointSet(2, in);
            pc->SetPointSet(3, cameraMarker((*sharedFrames)[pf].cam));

            std::vector<PointVertex> tileBoxes;
            for (const auto &b: s.baseCoreBoxes) {
                const std::vector<PointVertex> e = boxEdges(b.first, b.second, voxel, 40, 220, 220);
                tileBoxes.insert(tileBoxes.end(), e.begin(), e.end());
            }
            pc->SetPointSet(4, tileBoxes);
            if (s.hasAlloc)
                pc->SetPointSet(5, boxEdges(s.allocMin, s.allocMax, voxel, 255, 160, 40));
            else
                pc->SetPointSet(5, {});

            std::vector<PointVertex> submapBoxes;
            for (const auto &b: s.denseBlockBoxes) {
                const std::vector<PointVertex> e = boxEdges(b.first, b.second, voxel, 230, 60, 230);
                submapBoxes.insert(submapBoxes.end(), e.begin(), e.end());
            }
            pc->SetPointSet(6, submapBoxes);
        };

        imgui->SetUi([&]() {
            ImGui::Begin("Voxel Fill Debug");
            ImGui::Text("frame %d / %d", state.frame + 1, nFrames);
            if (ImGui::SliderInt("frame", &state.frame, 0, std::max(0, nFrames - 1)))
                state.playing = false;
            ImGui::Checkbox("play", &state.playing);
            ImGui::SameLine();
            if (ImGui::Button("restart")) {
                state.frame = 0;
                state.playing = true;
            }
            ImGui::SliderFloat("fps", &state.fps, 1.0f, 30.0f, "%.0f");
            ImGui::SeparatorText("Color mode");
            int m = int(state.mode);
            bool cm = false;
            cm |= ImGui::RadioButton("tsdf", &m, 0);
            ImGui::SameLine();
            cm |= ImGui::RadioButton("weight", &m, 1);
            ImGui::SameLine();
            cm |= ImGui::RadioButton("fill-frame", &m, 2);
            ImGui::SameLine();
            cm |= ImGui::RadioButton("direction", &m, 3);
            if (cm) {
                state.mode = voxdbg::ColorMode(m);
                state.dirty = true;
            }
            if (ImGui::SliderFloat("weight thresh", &state.wThresh, 0.0f, state.wMax, "%.3f"))
                state.dirty = true;
            if (ImGui::Checkbox("hide below thresh", &state.hideBelow)) state.dirty = true;
            ImGui::SeparatorText("Layers");
            if (ImGui::Checkbox("occupied", &state.showOccupied)) pc->SetVisible(0, state.showOccupied);
            if (ImGui::Checkbox("new this frame", &state.showNew)) pc->SetVisible(1, state.showNew);
            if (ImGui::Checkbox("input", &state.showInput)) pc->SetVisible(2, state.showInput);
            if (ImGui::Checkbox("camera", &state.showCamera)) pc->SetVisible(3, state.showCamera);
            if (ImGui::Checkbox("base tile windows", &state.showWindowBox))
                pc->SetVisible(4, state.showWindowBox);
            if (ImGui::Checkbox("allocated box", &state.showAllocBox))
                pc->SetVisible(5, state.showAllocBox);
            if (ImGui::Checkbox("submap regions (detail)", &state.showSubmapBox))
                pc->SetVisible(6, state.showSubmapBox);
            ImGui::SeparatorText("Stats");
            const int behind = mapper.RequestedFrame() - mapper.ProcessedFrame();
            ImGui::Text("worker: processed %d / requested %d  (%d behind)", mapper.ProcessedFrame(),
                        mapper.RequestedFrame(), behind > 0 ? behind : 0);
            if (snap) {
                ImGui::Text("occupied voxels: %zu", snap->entries.size());
                std::size_t below = 0;
                for (const auto &e: snap->entries)
                    if (voxdbg::belowThreshold(e.weight, state.wThresh)) ++below;
                ImGui::Text("below thresh: %zu", below);
                ImGui::Text("base/detail tiles: %u / %u, dense blocks: %u", snap->baseTiles,
                            snap->detailTiles, snap->denseBlocks);
                if (snap->hasAlloc) {
                    const Vector3f sz = snap->allocMax - snap->allocMin;
                    ImGui::Text("allocated box: %.2f x %.2f x %.2f", sz.x(), sz.y(), sz.z());
                }
                ImGui::SeparatorText("Worker stage times (ms, latest frame)");
                ImGui::Text("integrate %8.2f", snap->integrateMs);
                ImGui::Text("download  %8.2f", snap->downloadMs);
                ImGui::Text("tracker   %8.2f", snap->trackerMs);
            } else {
                ImGui::Text("(waiting for first snapshot...)");
            }
            ImGui::SeparatorText("Render stage times (avg / last ms)");
            for (const char *s: {"waitIdle", "buildSets", "upload", "render"})
                ImGui::Text("%-10s %8.2f / %8.2f", s, prof.AvgMs(s), prof.LastMs(s));
            ImGui::End();
        });

        app.GetView().SetScene(&scene);
        app.GetView().SetCamera(&camera);
        app.GetView().SetRenderGraph(&graph);

        Engine::Render::MouseListenerGroup mouse(app.GetWindow().Mouse());
        mouse.Add(Engine::Render::MouseEventType::Drag, [&](Engine::Render::MouseEvent &e) {
            if (e.button != Engine::Render::MouseButton::Left) return;
            const VkExtent2D s = app.GetWindow().FramebufferSize();
            if (s.width == 0 || s.height == 0) return;
            if (!camera.IsTrackballDragging()) {
                camera.BeginTrackballDrag(e.x, e.y, int(s.width), int(s.height));
                return;
            }
            camera.DragTrackball(e.x, e.y, int(s.width), int(s.height));
            e.handled = true;
        });
        mouse.Add(Engine::Render::MouseEventType::ButtonUp, [&](Engine::Render::MouseEvent &e) {
            if (e.button == Engine::Render::MouseButton::Left) camera.EndTrackballDrag();
        });
        mouse.Add(Engine::Render::MouseEventType::Scroll, [&](Engine::Render::MouseEvent &e) {
            camera.SetDistance(std::clamp(camera.GetDistance() * std::exp(float(-e.scrollY) * 0.08f),
                                          extent * 0.2f, extent * 20.0f));
            e.handled = true;
        });
        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape) app.GetWindow().RequestClose();
        });

        double lastAdvance = 0.0;
        try {
            while (!app.GetWindow().ShouldClose()) {
                app.GetWindow().PollEvents();
                const VkExtent2D s = app.GetWindow().FramebufferSize();
                if (s.width == 0 || s.height == 0) continue;

                // Play advances only once the worker has caught up to the current target AND the fps
                // interval elapsed, so every frame's fill is shown (paced by worker throughput).
                const double now = ImGui::GetTime();
                if (state.playing && nFrames > 0 && mapper.ProcessedFrame() >= state.frame &&
                    (now - lastAdvance) >= 1.0 / double(std::max(1.0f, state.fps))) {
                    state.frame = (state.frame + 1) % nFrames; // loop
                    lastAdvance = now;
                }
                if (state.frame != state.lastRequested) {
                    mapper.RequestFrame(state.frame); // coalesced; non-blocking
                    state.lastRequested = state.frame;
                }

                // Consume the latest worker snapshot (never blocks on integrate/download).
                std::shared_ptr<const asyncmap::MapSnapshot> latest = mapper.Latest();
                bool needRefresh = state.dirty;
                if (latest && latest.get() != snap.get()) {
                    snap = latest;
                    state.wMax = 1.0f;
                    for (const auto &e: snap->entries) state.wMax = std::max(state.wMax, e.weight);
                    state.wThresh = std::min(state.wThresh, state.wMax);
                    needRefresh = true;
                }
                if (snap && needRefresh) {
                    refreshSets(*snap);
                    state.dirty = false;
                }

                util::ScopedStageTimer t(prof, "render");
                if (!app.GetRenderer().BeginFrame(s.width, s.height)) continue;
                app.GetRenderer().Render(app.GetView());
                app.GetRenderer().EndFrame();
            }
        } catch (...) {
            vkDeviceWaitIdle(appCtx.device);
            mapper.Stop();
            throw;
        }
        vkDeviceWaitIdle(appCtx.device);
        mapper.Stop();
        std::printf("\n%s", prof.Report("render-thread pipeline").c_str());
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

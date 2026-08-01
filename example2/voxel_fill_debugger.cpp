#include "ImGuiPass.h"
#include "PointCloudPass.h"
#include "VoxelFillDebug.h"

#include "Engine/Core/Context.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"
#include "Engine/Spatial/AdvancedTSDF.h"

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
    // Reused verbatim from tsdf_folder_eval.cpp: arg helpers, nextPow2, readPly, estimateCamera.
    ///////////////////////////////////////////////////////////////////////////////////////////

    std::string strArg(int argc, char **argv, const char *k, const std::string &d) {
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == k) return argv[i + 1];
        return d;
    }
    float floatArg(int argc, char **argv, const char *k, float d) {
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == k) return float(std::atof(argv[i + 1]));
        return d;
    }
    bool flag(int argc, char **argv, const char *k) {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == k) return true;
        return false;
    }

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

    // Tolerant ASCII-PLY reader: x y z always; nx ny nz if the header declares them.
    bool readPly(const std::string &path, std::vector<Vector3f> &pts, std::vector<Vector3f> &nrm) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        size_t count = 0;
        bool ascii = false, hasN = false;
        std::vector<std::string> props;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string tok;
            ss >> tok;
            if (tok == "format") {
                std::string fmt;
                ss >> fmt;
                ascii = (fmt == "ascii");
            } else if (tok == "element") {
                std::string e;
                ss >> e;
                if (e == "vertex") ss >> count;
            } else if (tok == "property") {
                std::string t, name;
                ss >> t >> name;
                props.push_back(name);
            } else if (tok == "end_header")
                break;
        }
        if (!ascii) return false;
        hasN = std::find(props.begin(), props.end(), "nx") != props.end();
        const size_t stride = props.size();
        pts.reserve(pts.size() + count);
        for (size_t i = 0; i < count && std::getline(f, line); ++i) {
            std::istringstream ss(line);
            std::vector<float> vals(stride, 0.0f);
            for (size_t j = 0; j < stride; ++j) ss >> vals[j];
            pts.emplace_back(vals[0], vals[1], vals[2]);
            if (hasN && stride >= 6) nrm.emplace_back(vals[3], vals[4], vals[5]);
        }
        return !pts.empty();
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
        const Vector3f c[8] = {{mn.x(), mn.y(), mn.z()}, {mx.x(), mn.y(), mn.z()},
                               {mx.x(), mx.y(), mn.z()}, {mn.x(), mx.y(), mn.z()},
                               {mn.x(), mn.y(), mx.z()}, {mx.x(), mn.y(), mx.z()},
                               {mx.x(), mx.y(), mx.z()}, {mn.x(), mx.y(), mx.z()}};
        static const int E[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                                     {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
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
                        const voxdbg::FillTracker &tracker, float voxel, int nFrames,
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
                    c = voxdbg::fillFrameColor(tracker.firstFrame(voxdbg::keyOf(e, voxel)), nFrames);
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
        const std::string dir = strArg(argc, argv, "--dir", "");
        if (dir.empty() || !fs::is_directory(dir)) {
            std::cerr << "usage: voxel_fill_debugger --dir <folder> [--voxel v] [--trunc t] "
                         "[--no-p2p] [--conf L] [--hermite] [--wthresh w] [--dump]\n";
            return 2;
        }

        // Collect frame_*.ply (sorted); ground_truth.ply (object_scan_viewer output) is excluded.
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

        // Read frames + accumulate a world bbox for auto voxel sizing (verbatim pattern from
        // tsdf_folder_eval.cpp).
        struct Frame {
            std::vector<Vector3f> pts, nrm;
            Vector3f cam;
        };
        std::vector<Frame> frames;
        Vector3f bbMin = Vector3f::Constant(1e30f), bbMax = Vector3f::Constant(-1e30f);
        size_t maxFramePts = 0;
        for (const auto &p: framePaths) {
            Frame fr;
            if (!readPly(p, fr.pts, fr.nrm) || fr.nrm.size() != fr.pts.size()) {
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

        const bool p2p = !flag(argc, argv, "--no-p2p");
        const float conf = floatArg(argc, argv, "--conf", 0.5f);
        const bool hermite = flag(argc, argv, "--hermite");
        const Vector3f span = bbMax - bbMin;
        const float extent = span.norm();
        const float voxel = floatArg(argc, argv, "--voxel", extent / 200.0f);
        const float trunc = floatArg(argc, argv, "--trunc", voxel * 3.0f);
        const float wThreshArg = floatArg(argc, argv, "--wthresh", 0.0f);

        // AdvancedTSDF is a SINGLE movable 512^3-voxel window (no tiling in this debugger — see
        // tsdf_folder_eval/TiledAdvancedTSDF for that). Size and place the window to the data;
        // refuse voxel sizes that would need more than one window.
        const float maxSpan = span.maxCoeff();
        const int margin = int(std::ceil(trunc / voxel)) + 2; // truncation ghost band
        const long axisVox = long(std::ceil(maxSpan / voxel)) + 2L * margin;
        if (axisVox > 512) {
            const float minVoxel = maxSpan / float(512 - 2 * margin);
            std::fprintf(stderr,
                         "error: voxel %.4f too fine — object spans %ld voxels/axis but a single "
                         "AdvancedTSDF window is 512^3.\n  use --voxel >= %.4f (single-window "
                         "debugger; tiling is out of scope).\n",
                         voxel, axisVox, minVoxel);
            return 3;
        }
        const Vector3f windowMinCorner = bbMin - float(margin) * Vector3f::Constant(voxel);
        const double surf = 2.0 * double(span.x() * span.y() + span.y() * span.z() +
                                         span.z() * span.x());
        const double shell = 2.0 * double(trunc) / double(voxel);
        const double estEntries = (surf / (double(voxel) * double(voxel))) * shell * 1.5;
        const uint32_t hashCap =
                std::max(1u << 20, nextPow2(uint32_t(std::min(estEntries * 2.0, double(1u << 24)))));
        const uint32_t maxPts = nextPow2(uint32_t(std::max<std::size_t>(maxFramePts, 1u << 15)));

        Engine::Core::Context ctx;
        Engine::Spatial::AdvancedTSDF tsdf;
        tsdf.Build(ctx, voxel, trunc, hashCap, maxPts, windowMinCorner);
        tsdf.SetIntegrationQuality({3, 4, true});
        tsdf.SetPointToPlane(p2p);
        tsdf.SetConfidenceWeight(conf);
        tsdf.SetHermitePosition(hermite);

        // World-space bounds of the movable 512^3 window the TSDF hashes into (constant after
        // Build). Voxels outside this box are dropped at integration, so this box = the reachable
        // region; the "allocated" box (from downloaded entries) grows inside it per frame.
        const int kWindow = 512;
        const Eigen::Vector3i originVoxel = tsdf.OriginVoxel();
        const Vector3f winMin = originVoxel.cast<float>() * voxel;
        const Vector3f winMax = (originVoxel + Eigen::Vector3i::Constant(kWindow)).cast<float>() * voxel;

        const int nFrames = int(frames.size());
        voxdbg::FillTracker tracker(voxel);

        std::printf("dir       : %s  (%d frames, extent %.4f)\n", dir.c_str(), nFrames, extent);
        std::printf("advanced  : voxel %.4f, trunc %.4f, hashCap %u, window %ld vox/axis\n", voxel,
                    trunc, hashCap, axisVox);

        // Headless per-frame stats: no window, no render deps touched.
        if (flag(argc, argv, "--dump") || flag(argc, argv, "--no-view")) {
            const Vector3f winSize = winMax - winMin;
            std::printf("window box: min(%.2f,%.2f,%.2f) max(%.2f,%.2f,%.2f) size(%.2f,%.2f,%.2f)\n",
                        winMin.x(), winMin.y(), winMin.z(), winMax.x(), winMax.y(), winMax.z(),
                        winSize.x(), winSize.y(), winSize.z());
            for (int f = 0; f < nFrames; ++f) {
                tsdf.Integrate(frames[f].pts, frames[f].nrm, frames[f].cam);
                const auto entries = tsdf.DownloadEntries();
                const auto isNew = tracker.update(entries, f);
                std::size_t below = 0;
                for (const auto &e: entries)
                    if (voxdbg::belowThreshold(e.weight, wThreshArg)) ++below;
                std::size_t nnew = 0;
                for (char c: isNew) nnew += (c != 0);
                Vector3f aMn, aMx, aSz = Vector3f::Zero();
                if (entriesAabb(entries, voxel, aMn, aMx)) aSz = aMx - aMn;
                std::printf("frame %3d: occupied %zu  new %zu  below-wthresh %zu  "
                            "allocBox(%.2f,%.2f,%.2f)\n",
                            f, entries.size(), nnew, below, aSz.x(), aSz.y(), aSz.z());
            }
            std::printf("[--dump] done.\n");
            return 0;
        }

        ///////////////////////////////////////////////////////////////////////////////////////
        // Live viewer — scaffold reused from object_scan_viewer.cpp; scrubbing/state is new.
        ///////////////////////////////////////////////////////////////////////////////////////

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
            bool showWindowBox = true, showAllocBox = true;
            int shown = -1;
            float wMax = 1.0f;
            bool dirty = false;
        } state;
        state.wThresh = wThreshArg;

        std::vector<AdvancedEntry> curEntries;
        std::vector<char> curNew;
        auto rebuildTo = [&](int target) {
            target = std::clamp(target, 0, std::max(0, nFrames - 1));
            if (nFrames == 0) return;
            vkDeviceWaitIdle(appCtx.device);
            if (target < state.shown) {
                tsdf.Reset();
                tracker.reset();
                state.shown = -1;
            }
            for (int f = state.shown + 1; f <= target; ++f) {
                tsdf.Integrate(frames[f].pts, frames[f].nrm, frames[f].cam);
                curEntries = tsdf.DownloadEntries();
                curNew = tracker.update(curEntries, f);
            }
            state.shown = target;
            state.wMax = 1.0f;
            for (const auto &e: curEntries) state.wMax = std::max(state.wMax, e.weight);
            state.wThresh = std::min(state.wThresh, state.wMax);
        };
        auto refreshSets = [&]() {
            // SetPointSet() below reallocates PointCloudPass's vertex buffers (destroying the
            // old ones); the previous frame's command buffer may still be in flight, so wait
            // here too (not just in rebuildTo) — this path also runs standalone from the
            // dirty-only branch (color-mode/threshold change with no frame change).
            vkDeviceWaitIdle(appCtx.device);
            std::vector<PointVertex> occ, nw;
            buildVoxelSets(curEntries, curNew, state.mode, trunc, state.wMax, state.wThresh,
                           state.hideBelow, tracker, voxel, nFrames, occ, nw);
            pc->SetPointSet(0, occ);
            pc->SetPointSet(1, nw);
            std::vector<PointVertex> in;
            pushCloud(in, frames[std::clamp(state.shown, 0, nFrames - 1)].pts, 100, 110, 120);
            pc->SetPointSet(2, in);
            pc->SetPointSet(3, cameraMarker(frames[std::clamp(state.shown, 0, nFrames - 1)].cam));
            // 512^3 window box (cyan, constant) + currently-allocated voxel box (orange, grows).
            pc->SetPointSet(4, boxEdges(winMin, winMax, voxel, 40, 220, 220));
            Vector3f aMn, aMx;
            if (entriesAabb(curEntries, voxel, aMn, aMx))
                pc->SetPointSet(5, boxEdges(aMn, aMx, voxel, 255, 160, 40));
            else
                pc->SetPointSet(5, {});
        };
        rebuildTo(0);
        refreshSets();

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
            if (ImGui::Checkbox("window box (512^3)", &state.showWindowBox))
                pc->SetVisible(4, state.showWindowBox);
            if (ImGui::Checkbox("allocated box", &state.showAllocBox))
                pc->SetVisible(5, state.showAllocBox);
            ImGui::SeparatorText("Stats");
            ImGui::Text("occupied voxels: %zu", curEntries.size());
            std::size_t below = 0;
            for (const auto &e: curEntries)
                if (voxdbg::belowThreshold(e.weight, state.wThresh)) ++below;
            ImGui::Text("below thresh: %zu", below);
            ImGui::Text("window box: %.1f (=512 x %.3f)", float(kWindow) * voxel, voxel);
            Vector3f aMn, aMx;
            if (entriesAabb(curEntries, voxel, aMn, aMx)) {
                const Vector3f sz = aMx - aMn;
                ImGui::Text("allocated box: %.2f x %.2f x %.2f", sz.x(), sz.y(), sz.z());
            }
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

                const double now = ImGui::GetTime();
                if (state.playing && nFrames > 0 &&
                    (now - lastAdvance) >= 1.0 / double(std::max(1.0f, state.fps))) {
                    state.frame = (state.frame + 1) % nFrames; // loop
                    lastAdvance = now;
                }
                if (state.frame != state.shown) {
                    rebuildTo(state.frame);
                    refreshSets();
                    state.dirty = false;
                } else if (state.dirty) {
                    refreshSets();
                    state.dirty = false;
                }

                if (!app.GetRenderer().BeginFrame(s.width, s.height)) continue;
                app.GetRenderer().Render(app.GetView());
                app.GetRenderer().EndFrame();
            }
        } catch (...) {
            vkDeviceWaitIdle(appCtx.device);
            throw;
        }
        vkDeviceWaitIdle(appCtx.device);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

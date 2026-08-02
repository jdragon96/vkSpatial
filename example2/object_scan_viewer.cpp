// Object scan viewer — record a synthetic multi-view scan of a mesh and WATCH the camera orbit.
// No TSDF here: this only captures + saves; reconstruction/eval is a separate tool
// (tsdf_folder_eval).
//
//   1. Load a mesh (PLY or OBJ) and TRANSLATE its bbox centre to the origin.
//   2. Orbit a trackball camera around the origin in a SPIRAL (elevation sweep + azimuth turns).
//   3. From each view, capture the visible surface as an oriented point cloud (WORLD coords).
//   4. Save each frame as <out>/frame_%04d.ply (+ ground_truth.ply + manifest).
//   5. Live window: watch the scan camera (red) orbit the object; the this-frame capture is
//      highlighted. --no-view records + saves headless (no display).
//
//   ./object_scan_viewer --mesh bunny.obj --out scans/bunny [--frames 90] [--turns 3]
//        [--sample 120000] [--width 320] [--height 240] [--fov 55] [--no-view]

#include "ImGuiPass.h"
#include "PointCloudPass.h"

#include "Engine/Core/Context.h"
#include "Engine/Eval/ScanDataset.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"
#include "utilities/ArgParser.h"
#include "utilities/ObjectScanner.h"
#include "utilities/PlyMesh.h"

#include "imgui.h"

#include <Eigen/Core>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Engine::Eval::CaptureParams;
using Engine::Eval::ScanFrame;
using Engine::Eval::TrackballParams;
using Eigen::Vector3f;
using util::TriMesh;

namespace {

    // Arg parsing via the fluent util::ArgParser (utilities/ArgParser.h).

    bool endsWith(const std::string &s, const std::string &suf) {
        if (s.size() < suf.size()) return false;
        std::string a = s.substr(s.size() - suf.size());
        for (char &c : a) c = char(std::tolower((unsigned char) c));
        return a == suf;
    }

    // Minimal OBJ loader (v / f with v, v/vt, v/vt/vn, v//vn; fan-triangulated).
    bool LoadObjMesh(const std::string &path, TriMesh &mesh) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string tok;
            ss >> tok;
            if (tok == "v") {
                float x, y, z;
                ss >> x >> y >> z;
                mesh.vertices.emplace_back(x, y, z);
            } else if (tok == "f") {
                std::vector<int> idx;
                std::string vtok;
                while (ss >> vtok) {
                    int vi = std::atoi(vtok.c_str());
                    if (vi < 0) vi = int(mesh.vertices.size()) + 1 + vi;
                    if (vi >= 1) idx.push_back(vi - 1);
                }
                for (size_t k = 1; k + 1 < idx.size(); ++k)
                    mesh.faces.emplace_back(idx[0], idx[k], idx[k + 1]);
            }
        }
        return !mesh.vertices.empty();
    }

    // Translate the mesh so its bbox centre is the origin; return the extent diagonal.
    float CenterMesh(TriMesh &mesh) {
        Vector3f mn = mesh.vertices[0], mx = mesh.vertices[0];
        for (const auto &v : mesh.vertices) { mn = mn.cwiseMin(v); mx = mx.cwiseMax(v); }
        const Vector3f c = 0.5f * (mn + mx);
        for (auto &v : mesh.vertices) v -= c;
        return (mx - mn).norm();
    }

    // Area-weighted surface sampling → dense oriented cloud (robust to low-poly meshes).
    void SampleSurface(const TriMesh &mesh, const std::vector<Vector3f> &vn, int target,
                       std::vector<Vector3f> &outP, std::vector<Vector3f> &outN) {
        std::vector<float> cum;
        cum.reserve(mesh.faces.size());
        float total = 0.0f;
        for (const auto &fc : mesh.faces) {
            const Vector3f &a = mesh.vertices[fc[0]], &b = mesh.vertices[fc[1]], &c = mesh.vertices[fc[2]];
            total += 0.5f * (b - a).cross(c - a).norm();
            cum.push_back(total);
        }
        if (total <= 0.0f) return;
        std::mt19937 rng(1234567u);
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        outP.reserve(target);
        outN.reserve(target);
        for (int s = 0; s < target; ++s) {
            const float r = U(rng) * total;
            int fi = int(std::lower_bound(cum.begin(), cum.end(), r) - cum.begin());
            fi = std::min(fi, int(mesh.faces.size()) - 1);
            const auto &fc = mesh.faces[fi];
            float u = U(rng), v = U(rng);
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            const float w = 1.0f - u - v;
            const Vector3f &A = mesh.vertices[fc[0]], &B = mesh.vertices[fc[1]], &C = mesh.vertices[fc[2]];
            Vector3f p = w * A + u * B + v * C;
            Vector3f n = w * vn[fc[0]] + u * vn[fc[1]] + v * vn[fc[2]];
            if (n.norm() < 1e-8f) n = (B - A).cross(C - A);
            n.normalize();
            outP.push_back(p);
            outN.push_back(n);
        }
    }

    void pushCloud(std::vector<PointVertex> &out, const std::vector<Vector3f> &pts,
                   uint8_t r, uint8_t g, uint8_t b) {
        for (const auto &p : pts) out.push_back({{p.x(), p.y(), p.z()}, {r, g, b, 255}});
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

} // namespace

int main(int argc, char **argv) {
    try {
        util::ArgParser arg =
                util::BuildArgParser(argc, argv)
                        .Must("--mesh", "usage: object_scan_viewer --mesh <file.ply|.obj> --out "
                                        "<dir> [--frames N] [--turns T] [--sample N] [--no-view]")
                        .Option("--out", "scan_out")
                        .Option("--frames", 90)
                        .Option("--turns", 3.0)
                        .Option("--elev-start", 75.0)
                        .Option("--elev-end", -30.0)
                        .Option("--sample", 120000)
                        .Option("--width", 320)
                        .Option("--height", 240)
                        .Option("--fov", 55.0);
        if (!arg) return 2;
        const std::string meshPath = arg.Value("--mesh");
        const std::string outDir = arg.Value("--out");
        const int frames = arg.ValueInt("--frames");
        const float turns = arg.ValueFloat("--turns");
        const float elevStart = arg.ValueFloat("--elev-start");
        const float elevEnd = arg.ValueFloat("--elev-end");
        const int sample = arg.ValueInt("--sample");
        const int capW = arg.ValueInt("--width");
        const int capH = arg.ValueInt("--height");
        const float fov = arg.ValueFloat("--fov");
        const bool noView = arg.Has("--no-view");

        // 1. Load mesh (+ vertex normals) and centre it at the origin.
        TriMesh mesh;
        if (endsWith(meshPath, ".obj")) {
            if (!LoadObjMesh(meshPath, mesh)) throw std::runtime_error("cannot load OBJ: " + meshPath);
        } else {
            util::LoadPlyMesh(meshPath, mesh);
        }
        if (mesh.vertices.empty() || mesh.faces.empty())
            throw std::runtime_error("mesh has no triangles: " + meshPath);
        const std::vector<Vector3f> vnormals = util::ComputeVertexNormals(mesh);
        const float extent = CenterMesh(mesh);

        std::vector<Vector3f> objP, objN;
        SampleSurface(mesh, vnormals, sample, objP, objN);
        const float radius = extent * 1.4f;

        // 2. Spiral trackball trajectory around the origin.
        TrackballParams tp;
        tp.orbitCenter = Vector3f::Zero();
        tp.radius = radius;
        tp.numFrames = frames;
        tp.startElevationDeg = elevStart;
        tp.endElevationDeg = elevEnd;
        tp.azimuthTurns = turns;

        CaptureParams cp;
        cp.width = capW;
        cp.height = capH;
        cp.fovYDeg = fov;

        // 3. Capture per view + 4. save world-space frame PLYs.
        util::ObjectScanner scanner;
        scanner.RecordFromPoints(objP, objN, tp, cp);
        const size_t written = scanner.WriteDataset(outDir, "mesh:" + meshPath);
        const auto &sframes = scanner.Frames();

        std::size_t total = 0;
        for (const auto &f : sframes) total += f.points.size();
        std::printf("mesh      : %s (%zu verts, %zu tris; extent %.4f) → centred, %zu surface pts\n",
                    meshPath.c_str(), mesh.vertices.size(), mesh.faces.size(), extent, objP.size());
        std::printf("trajectory: %d frames, elev %.0f→%.0f, %.1f turns, radius %.4f\n", frames,
                    elevStart, elevEnd, turns, radius);
        std::printf("saved     : %zu frames → %s/  (frame_%%04d.ply, ground_truth.ply, manifest.txt)\n",
                    written, outDir.c_str());
        std::printf("points/frame: total %zu, avg %.0f\n", total,
                    written ? double(total) / double(written) : 0.0);

        if (noView) {
            std::printf("[--no-view] scan complete (no window).\n");
            return 0;
        }

        // 5. Live viewer: watch the scan camera orbit the object.
        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {1280, 800, "Object Scan Viewer"};
        Engine::Render::Application app(descriptor);
        Engine::Core::Context &ctx = app.GetContext();

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D ext = app.GetSwapChain().Extent();
        camera.SetPerspective(60.0f * 3.14159265f / 180.0f, float(ext.width) / float(ext.height),
                              std::max(1e-3f, extent * 1e-3f), std::max(10.0f, radius * 12.0f));
        camera.SetOrbit(vkMath::Vec3(0.0f, 0.0f, 0.0f), radius * 2.0f);

        const std::string shaderDir = SCAN_VIEWER_SHADER_DIR;
        Engine::Render::RenderGraph graph;
        auto pcOwned = std::make_unique<PointCloudPass>(ctx, app.GetSwapChain().Format(), shaderDir);
        PointCloudPass *pc = pcOwned.get();
        graph.AddPass(std::move(pcOwned));
        auto &glfwWindow = static_cast<Engine::Render::GlfwWindow &>(app.GetWindow());
        auto imguiOwned = std::make_unique<ImGuiPass>(ctx, glfwWindow.Handle(),
                                                      app.GetSwapChain().Format(),
                                                      app.GetSwapChain().ImageCount());
        ImGuiPass *imgui = imguiOwned.get();
        graph.AddPass(std::move(imguiOwned));
        pc->SetPointSize(2.5f);

        // Set 0: the full object (dim grey reference) — static so you see the object.
        {
            std::vector<PointVertex> ov;
            pushCloud(ov, objP, 120, 128, 140);
            pc->SetPointSet(0, ov);
        }

        struct State {
            int frame = 0;
            bool playing = true;
            float fps = 12.0f;
            bool showObject = true, showCapture = true, showCamera = true;
        } state;
        const int nFrames = int(sframes.size());

        auto showFrame = [&](int fi) {
            fi = std::clamp(fi, 0, std::max(0, nFrames - 1));
            vkDeviceWaitIdle(ctx.device);
            if (nFrames == 0) return;
            const ScanFrame &f = sframes[size_t(fi)];
            std::vector<PointVertex> cur;
            pushCloud(cur, f.points, 255, 220, 40); // this-frame capture (yellow)
            pc->SetPointSet(1, cur);
            pc->SetPointSet(2, cameraMarker(f.cameraPos)); // scan camera (red), orbiting
        };
        showFrame(0);

        imgui->SetUi([&]() {
            ImGui::Begin("Object Scan");
            ImGui::Text("scan camera frame %d / %d", state.frame + 1, nFrames);
            if (ImGui::SliderInt("frame", &state.frame, 0, std::max(0, nFrames - 1)))
                state.playing = false;
            ImGui::Checkbox("play", &state.playing);
            ImGui::SameLine();
            if (ImGui::Button("restart")) { state.frame = 0; state.playing = true; }
            ImGui::SliderFloat("fps", &state.fps, 1.0f, 60.0f, "%.0f");
            ImGui::SeparatorText("Layers");
            if (ImGui::Checkbox("object", &state.showObject)) pc->SetVisible(0, state.showObject);
            if (ImGui::Checkbox("this-frame capture", &state.showCapture)) pc->SetVisible(1, state.showCapture);
            if (ImGui::Checkbox("scan camera", &state.showCamera)) pc->SetVisible(2, state.showCamera);
            ImGui::SeparatorText("Info");
            ImGui::Text("captured this frame: %zu pts",
                        nFrames ? sframes[size_t(std::clamp(state.frame, 0, nFrames - 1))].points.size()
                                : size_t(0));
            ImGui::Text("saved to: %s/", outDir.c_str());
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
                                          radius * 0.2f, radius * 20.0f));
            e.handled = true;
        });
        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape) app.GetWindow().RequestClose();
        });

        double lastAdvance = 0.0;
        int shown = -1;
        try {
            while (!app.GetWindow().ShouldClose()) {
                app.GetWindow().PollEvents();
                const VkExtent2D s = app.GetWindow().FramebufferSize();
                if (s.width == 0 || s.height == 0) continue;

                const double now = ImGui::GetTime();
                if (state.playing && nFrames > 0 &&
                    (now - lastAdvance) >= 1.0 / double(std::max(1.0f, state.fps))) {
                    state.frame = (state.frame + 1) % nFrames; // loop the orbit video
                    lastAdvance = now;
                }
                if (state.frame != shown) { showFrame(state.frame); shown = state.frame; }

                if (!app.GetRenderer().BeginFrame(s.width, s.height)) continue;
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

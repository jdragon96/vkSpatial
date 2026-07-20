// Synthetic scan-dataset generator for TSDF / DirectionalTSDF validation (Integrate + Extract).
//
// Thin CLI over util::ObjectScanner: records a trackball-down multi-view scan of an analytic
// object OR a 3D point-cloud file, and writes a per-frame PLY sequence + ground truth +
// manifest. The recorded frames are DirectionalTSDF::Integrate-ready; a consumer replays them:
//     scanner.Replay([&](size_t, const auto& f){
//         tsdf.Integrate(f.points, f.normals, f.cameraPos, f.aabbCenterHint); });
//
//   ./scan_dataset_gen [--object sphere|box|torus] [--file scan.ply] [--out <dir>]
//                      [--frames N] [--radius R] [--size S] [--width W] [--height H]
//                      [--fov F] [--elev-start deg] [--elev-end deg] [--turns T] [--gt N]
//
//   ./scan_dataset_gen --object torus --out data/torus --turns 2.0     # analytic, self-occlusion
//   ./scan_dataset_gen --file data/torus/ground_truth.ply --out data/torus_rescan  # from a 3D file

#include "Engine/Eval/SyntheticSurface.h"
#include "utilities/ObjectScanner.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using namespace Engine::Eval;

namespace {

    struct Config {
        std::string object = "sphere";
        std::string file; // if set, record from this PLY point cloud instead of an analytic object
        std::string out = "scan_dataset";
        int frames = 60;
        float radius = 3.0f;
        float size = 1.0f;
        int width = 160, height = 120;
        float fovYDeg = 55.0f;
        float elevStart = 80.0f, elevEnd = -20.0f, turns = 1.5f;
        int gtCount = 20000;
        int splat = -1; // occluder splat radius (px); <0 = path default (mesh 0, point cloud 1)
    };

    const char *argValue(int argc, char **argv, const char *key) {
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == key) return argv[i + 1];
        return nullptr;
    }

    Config parseArgs(int argc, char **argv) {
        Config c;
        if (const char *v = argValue(argc, argv, "--object")) c.object = v;
        if (const char *v = argValue(argc, argv, "--file")) c.file = v;
        if (const char *v = argValue(argc, argv, "--out")) c.out = v;
        if (const char *v = argValue(argc, argv, "--frames")) c.frames = std::atoi(v);
        if (const char *v = argValue(argc, argv, "--radius")) c.radius = float(std::atof(v));
        if (const char *v = argValue(argc, argv, "--size")) c.size = float(std::atof(v));
        if (const char *v = argValue(argc, argv, "--width")) c.width = std::atoi(v);
        if (const char *v = argValue(argc, argv, "--height")) c.height = std::atoi(v);
        if (const char *v = argValue(argc, argv, "--fov")) c.fovYDeg = float(std::atof(v));
        if (const char *v = argValue(argc, argv, "--elev-start")) c.elevStart = float(std::atof(v));
        if (const char *v = argValue(argc, argv, "--elev-end")) c.elevEnd = float(std::atof(v));
        if (const char *v = argValue(argc, argv, "--turns")) c.turns = float(std::atof(v));
        if (const char *v = argValue(argc, argv, "--gt")) c.gtCount = std::atoi(v);
        if (const char *v = argValue(argc, argv, "--splat")) c.splat = std::atoi(v);
        return c;
    }

    std::unique_ptr<Surface> makeSurface(const Config &c, std::string &descOut) {
        const Eigen::Vector3f center = Eigen::Vector3f::Zero();
        if (c.object == "box") {
            descOut = "box half=" + std::to_string(c.size);
            return std::make_unique<BoxSurface>(center, Eigen::Vector3f::Constant(c.size));
        }
        if (c.object == "torus") {
            descOut = "torus R=" + std::to_string(c.size) + " r=" + std::to_string(c.size * 0.4f);
            return std::make_unique<TorusSurface>(center, c.size, c.size * 0.4f);
        }
        descOut = "sphere r=" + std::to_string(c.size);
        return std::make_unique<SphereSurface>(center, c.size);
    }

} // namespace

int main(int argc, char **argv) {
    const Config c = parseArgs(argc, argv);

    TrackballParams tp;
    tp.orbitCenter = Eigen::Vector3f::Zero();
    tp.radius = c.radius;
    tp.numFrames = c.frames;
    tp.startElevationDeg = c.elevStart;
    tp.endElevationDeg = c.elevEnd;
    tp.azimuthTurns = c.turns;

    CaptureParams cp;
    cp.width = c.width;
    cp.height = c.height;
    cp.fovYDeg = c.fovYDeg;
    if (c.splat >= 0) cp.splatRadius = c.splat;

    util::ObjectScanner scanner;
    std::string desc;
    if (!c.file.empty()) {
        desc = "file:" + c.file;
        scanner.RecordFromFile(c.file, tp, cp); // depth-buffer occlusion capture
    } else {
        std::unique_ptr<Surface> surface = makeSurface(c, desc);
        scanner.Record(*surface, tp, cp, uint32_t(c.gtCount)); // exact ray-marched capture
    }

    const size_t written = scanner.WriteDataset(c.out, desc);

    // Summary via the Replay callback — the same entry point a TSDF consumer would use.
    size_t total = 0, mn = SIZE_MAX, mx = 0;
    scanner.Replay([&](size_t, const util::ObjectScanner::Frame &f) {
        total += f.points.size();
        mn = std::min(mn, f.points.size());
        mx = std::max(mx, f.points.size());
    });
    if (written == 0) mn = 0;

    // For file/mesh input the camera is auto-fit to the geometry bbox — report the actual fit.
    float shownRadius = tp.radius;
    Eigen::Vector3f shownCenter = tp.orbitCenter;
    const bool autoFit = !c.file.empty() && !scanner.FramePoses().empty();
    if (autoFit) {
        shownCenter = scanner.FramePoses()[0].target;
        shownRadius = (scanner.FramePoses()[0].eye - shownCenter).norm();
    }

    std::printf("object    : %s\n", desc.c_str());
    std::printf("trajectory: %d poses, elevation %.0f°→%.0f°, %.1f turns, radius %.3f%s\n",
                tp.numFrames, tp.startElevationDeg, tp.endElevationDeg, tp.azimuthTurns, shownRadius,
                autoFit ? " (auto-fit)" : "");
    if (autoFit)
        std::printf("fit center: %.3f %.3f %.3f\n", shownCenter.x(), shownCenter.y(), shownCenter.z());
    std::printf("camera    : %dx%d, fovY %.0f°\n", cp.width, cp.height, cp.fovYDeg);
    std::printf("frames    : %zu (non-empty)\n", written);
    std::printf("points/frame: total %zu, min %zu, max %zu, avg %.0f\n",
                total, mn, mx, written ? double(total) / double(written) : 0.0);
    std::printf("ground truth: %zu points\n", scanner.GroundTruth().size());
    std::printf("output dir : %s/  (frame_%%04d.ply, ground_truth.ply, manifest.txt)\n", c.out.c_str());
    return 0;
}

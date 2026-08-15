// TSDF pipeline debugger — replays every frame_*.ply in a folder through the TSDF class and
// reports, per frame, what the pipeline actually did: how the splitter divided the points, how
// many windows exist, how full their hashes are, and how much of the map sits below a weight
// threshold (a hole predictor -- those voxels have seen too little evidence to extract cleanly).
//
// Headless. The live viewer in voxel_fill_debugger goes through Pipeline, whose MapConfig
// still builds a SubmapAdvancedTSDF directly; porting that is separate work.

#include "Engine/Core/Context.h"
#include "TSDF/TSDF.h"

#include "utilities/ArgParser.h"
#include "utilities/PointCloudIO.h"
#include "utilities/StageProfiler.h"

#include <Eigen/Core>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Eigen::Vector3f;

namespace {

    uint32_t NextPowerOfTwo(uint32_t v) {
        if (v <= 1) return 1;
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

    struct Frame {
        std::vector<Vector3f> points, normals;
        Vector3f camera;
    };

    // The scans carry no pose, so place the camera off the mean normal at a few scene diagonals --
    // far enough that the view direction is effectively constant over the frame. Same estimate
    // tsdf_folder_eval uses, so the two tools describe the same reconstruction.
    Vector3f EstimateCamera(const std::vector<Vector3f> &points, const std::vector<Vector3f> &normals) {
        Vector3f centroid = Vector3f::Zero(), meanNormal = Vector3f::Zero();
        for (const Vector3f &p: points) centroid += p;
        for (const Vector3f &n: normals) meanNormal += n;
        centroid /= float(std::max<std::size_t>(1, points.size()));

        Vector3f minimum = points[0], maximum = points[0];
        for (const Vector3f &p: points) {
            minimum = minimum.cwiseMin(p);
            maximum = maximum.cwiseMax(p);
        }
        const float diagonal = (maximum - minimum).norm();
        if (meanNormal.norm() > 1e-6f) meanNormal.normalize();
        else meanNormal = Vector3f(0.0f, 0.0f, 1.0f);
        return centroid + std::max(1.0f, 3.0f * diagonal) * meanNormal;
    }

    std::vector<std::string> CollectFramePaths(const std::string &dir) {
        std::vector<std::string> paths;
        for (const auto &entry: fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            const std::string name = entry.path().filename().string();
            if (name.rfind("ground_truth", 0) == 0) continue;
            if (entry.path().extension() == ".ply" && name.rfind("frame_", 0) == 0)
                paths.push_back(entry.path().string());
        }
        std::sort(paths.begin(), paths.end());
        return paths;
    }

    // AABB of the filled voxels, expanded by half a voxel to the true extent.
    bool VoxelBounds(const std::vector<TSDFVoxel> &voxels, float voxel, Vector3f &size) {
        if (voxels.empty()) return false;
        Vector3f minimum = Vector3f::Constant(1e30f);
        Vector3f maximum = Vector3f::Constant(-1e30f);
        for (const TSDFVoxel &v: voxels) {
            minimum = minimum.cwiseMin(v.center);
            maximum = maximum.cwiseMax(v.center);
        }
        size = (maximum - minimum) + Vector3f::Constant(voxel);
        return true;
    }

} // namespace

int main(int argc, char **argv) {
    try {
        util::ArgParser arg =
                util::BuildArgParser(argc, argv)
                        .Must("--dir", "usage: tsdf_pipeline_debugger --dir <folder> [--voxel v] "
                                       "[--trunc t] [--splitter dense|none] [--hash linear|bucketed] "
                                       "[--no-submap] [--wthresh w] [--hash-cap N] [--window-vox V] "
                                       "[--max-window N] [--no-p2p] [--conf L] [--hermite]")
                        .Option("--voxel") // default: extent / 200
                        .Option("--trunc") // default: voxel * 2
                        .Option("--splitter", "dense")
                        .Option("--hash", "linear")
                        .Option("--wthresh", 0.0)
                        .Option("--hash-cap", 1 << 19)
                        .Option("--window-vox", 512)
                        .Option("--max-window", 0)
                        .Option("--conf", 0.5);
        if (!arg) return 2;

        const std::string dir = arg.Value("--dir");
        const std::vector<std::string> framePaths = CollectFramePaths(dir);
        if (framePaths.empty()) throw std::runtime_error("no frame_*.ply found in " + dir);

        std::vector<Frame> frames;
        Vector3f boundsMinimum = Vector3f::Constant(1e30f), boundsMaximum = Vector3f::Constant(-1e30f);
        std::size_t maxFramePoints = 0;
        for (const std::string &path: framePaths) {
            Frame frame;
            if (!util::LoadPly(path, frame.points, frame.normals) ||
                frame.normals.size() != frame.points.size()) {
                std::fprintf(stderr, "skip (no normals): %s\n", path.c_str());
                continue;
            }
            frame.camera = EstimateCamera(frame.points, frame.normals);
            for (const Vector3f &p: frame.points) {
                boundsMinimum = boundsMinimum.cwiseMin(p);
                boundsMaximum = boundsMaximum.cwiseMax(p);
            }
            maxFramePoints = std::max(maxFramePoints, frame.points.size());
            frames.push_back(std::move(frame));
        }
        if (frames.empty()) throw std::runtime_error("no usable frames (need per-point normals)");
        const float extent = (boundsMaximum - boundsMinimum).norm();

        const float voxel = arg.ValueFloat("--voxel", extent / 200.0f);
        const float truncation = arg.ValueFloat("--trunc", voxel * 2.0f);
        const float weightThreshold = arg.ValueFloat("--wthresh");
        const uint32_t maxPointPerFrame =
                NextPowerOfTwo(uint32_t(std::max<std::size_t>(maxFramePoints, 1u << 15)));

        TSDFConfiguration config;
        config.backend = "advanced";
        config.splitter = arg.Value("--splitter");
        config.useSubmap = !arg.Has("--no-submap");
        config.windowVoxels = int(arg.ValueFloat("--window-vox"));
        config.maxResidentWindow = int(arg.ValueFloat("--max-window"));
        config.backendConfig.voxelSize = voxel;
        config.backendConfig.truncation = truncation;
        config.backendConfig.hashCapacity = NextPowerOfTwo(uint32_t(arg.ValueFloat("--hash-cap")));
        config.backendConfig.maxPointPerFrame = maxPointPerFrame;
        config.backendConfig.hash = arg.Value("--hash");
        config.backendConfig.pointToPlane = !arg.Has("--no-p2p");
        config.backendConfig.confidenceWeight = arg.ValueFloat("--conf");
        config.backendConfig.hermitePosition = arg.Has("--hermite");
        config.backendConfig.maxDirections = 3;
        config.backendConfig.directionExponent = 4;
        config.backendConfig.viewAngleWeight = true;
        config.splitterConfig.baseResolution = voxel;
        config.splitterConfig.maxPointPerFrame = int(maxPointPerFrame);

        const std::string resolvedHash = ResolveHashName(config.backendConfig.hash);
        if (resolvedHash != config.backendConfig.hash)
            std::fprintf(stderr, "warning: --hash %s is unknown; falling back to %s\n",
                         config.backendConfig.hash.c_str(), resolvedHash.c_str());

        Engine::Core::Context context;
        TSDF tsdf;
        tsdf.Build(context, config);

        std::printf("dir       : %s  (%d frames, extent %.4f)\n", dir.c_str(), int(frames.size()),
                    extent);
        std::printf("map       : base %.4f + detail %.4f, window %d vox (%.2f m), hash %s %u/window\n",
                    voxel, voxel * 0.5f, config.windowVoxels,
                    voxel * float(config.windowVoxels), resolvedHash.c_str(),
                    config.backendConfig.hashCapacity);
        std::printf("splitter  : %s, submap %s, p2p %s, confidence %.2f, hermite %s\n\n",
                    config.splitter.c_str(), config.useSubmap ? "on" : "off",
                    config.backendConfig.pointToPlane ? "on" : "off",
                    config.backendConfig.confidenceWeight,
                    config.backendConfig.hermitePosition ? "on" : "off");

        util::StageProfiler profiler;
        std::vector<TSDFVoxel> voxels;
        int overIntegrate = 0, overDownload = 0;

        for (int f = 0; f < int(frames.size()); ++f) {
            {
                util::ScopedStageTimer timer(profiler, "integrate");
                tsdf.Integrate(frames[f].points, frames[f].normals, frames[f].camera);
            }
            {
                util::ScopedStageTimer timer(profiler, "download");
                tsdf.Download(voxels);
            }
            if (f > 0) { // frame 0 pays shader compilation and first-touch window creation
                if (profiler.LastMs("integrate") > 30.0) ++overIntegrate;
                if (profiler.LastMs("download") > 30.0) ++overDownload;
            }

            std::size_t below = 0, fresh = 0;
            for (const TSDFVoxel &v: voxels) {
                if (v.weight < weightThreshold) ++below;
                if (v.firstFrame == f) ++fresh;
            }

            const DataSplitter::DividePointOutput &divided = tsdf.LastDivision();
            const TSDFBackendStats stats = tsdf.Stats();
            Vector3f boxSize = Vector3f::Zero();
            VoxelBounds(voxels, voxel, boxSize);

            std::printf("frame %3d: int %6.2f dl %6.2f %s occupied %8zu new %7zu below %7zu | "
                        "base/detail %7zu/%-7zu dense %4u | windows %3zu load %.3f | "
                        "box(%.2f,%.2f,%.2f)\n",
                        f, profiler.LastMs("integrate"), profiler.LastMs("download"),
                        (f > 0 && profiler.LastMs("integrate") > 30.0) ? "OVER" : "    ",
                        voxels.size(), fresh, below,
                        divided.baseIndex.size(), divided.detailIndex.size(),
                        divided.denseBlockCount,
                        tsdf.WindowCount(), stats.LoadFactor(),
                        boxSize.x(), boxSize.y(), boxSize.z());

            if (divided.blockInsertFailureCount > 0 || divided.cellInsertFailureCount > 0)
                std::printf("           splitter lost statistics: %u point(s) past the block table, "
                            "%u cell claim(s) past their probe budget\n",
                            divided.blockInsertFailureCount, divided.cellInsertFailureCount);
        }

        const TSDFBackendStats stats = tsdf.Stats();
        std::printf("\n%s", profiler.Report("per-frame TSDF pipeline").c_str());
        std::printf("[budget] frames past #0 over 30ms:  integrate=%d  download=%d  (of %d)\n",
                    overIntegrate, overDownload, int(frames.size()) - 1);
        std::printf("[map]    windows %u, occupied %llu / %llu slots (load %.3f), %.1f MB, "
                    "grows %llu\n",
                    stats.tableCount, (unsigned long long) stats.filledCount,
                    (unsigned long long) stats.hashCapacity, stats.LoadFactor(),
                    double(stats.deviceMemoryBytes) / (1024.0 * 1024.0),
                    (unsigned long long) stats.growCount);

        if (stats.insertFailureCount > 0)
            std::printf("WARNING: %llu observations were dropped -- this hash's load factor limit "
                        "is too high for this scene\n",
                        (unsigned long long) stats.insertFailureCount);
        if (tsdf.WindowLimitRefusalCount() > 0)
            std::printf("WARNING: %u point(s) were refused because --max-window %d was reached -- "
                        "that geometry is missing from the map\n",
                        tsdf.WindowLimitRefusalCount(), config.maxResidentWindow);

        const Engine::Core::OrientedPointCloud surface = tsdf.Extract();
        std::printf("[extract] %zu oriented points\n", surface.points.size());
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}

// Headless ICP-quality diagnostic: runs the REAL reconstruction Pipeline (integrate + track) over a
// folder of frame_*.ply frames with a chosen tracker, then reports reconstruction + tracker stats.
//
// Purpose: scan_out is pre-registered (frames already in world frame), so `identity` is the correct
// tracker and its reconstruction is the ground-truth reference. This tool measures how far the `icp`
// tracker's reconstruction DRIFTS from the `identity` reference (nearest-neighbour RMSE, both
// directions) plus each tracker's average residual RMSE / align time -- a headless stand-in for the
// GUI viewer's live quality, so icp-vs-identity (and clean-map-vs-WIP-map) can be A/B'd with numbers.
//
// Usage: icp_quality_diag --dir <folder> [--voxel v] [--trackers identity,icp[,icp-cpu]]

#include "Engine/Pipeline/Pipeline.h"
#include "Engine/Pipeline/Registration/Tracker.h"       // TrackerRegistry
#include "Engine/Pipeline/Reconstruction/FrameLoader.h" // LoadFrames / ComputeBounds
#include "Engine/Eval/RmseMetrics.h"                     // NearestNeighbourRMSE
#include "utilities/ArgParser.h"

#include <Eigen/Core>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace ep = Engine::Pipeline;
namespace fs = std::filesystem;

namespace {

    std::vector<std::string> collectFramePaths(const std::string &dir) {
        std::vector<std::string> paths;
        for (const auto &e: fs::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            const std::string name = e.path().filename().string();
            if (name.rfind("ground_truth", 0) == 0) continue;
            if (e.path().extension() == ".ply" && name.rfind("frame_", 0) == 0)
                paths.push_back(e.path().string());
        }
        std::sort(paths.begin(), paths.end());
        return paths;
    }

    struct RunResult {
        std::vector<Eigen::Vector3f> reconPoints; // occupied-voxel centres of the final model
        int processedFrames = -1;
        double alignMsAvg = 0.0;
        double trackerRmseAvg = 0.0;
        std::size_t entries = 0;
    };

    // Drive the pipeline to completion over all frames, then snapshot the final model + stats.
    RunResult runTracker(const std::vector<std::string> &framePaths, float voxel,
                         const std::string &trackerName) {
        ep::Pipeline::Config config;
        config.map.baseVoxel = voxel;
        config.acquisition.type = ep::EAcquisitionType::File;
        config.acquisition.framePaths = framePaths;
        config.acquisition.intervalMs = 0.0; // as fast as the stages allow
        config.acquisition.loop = false;

        ep::TrackerRegistry registry = ep::TrackerRegistry::Default();
        ep::Pipeline pipe(config, registry.Create(trackerName));
        pipe.Start();
        pipe.SetPaused(false);

        const int lastFrame = int(framePaths.size()) - 1;
        int seen = -1, stableAtEnd = 0;
        const auto start = std::chrono::steady_clock::now();
        while (true) {
            pipe.CheckErrors(); // rethrow any worker-stage exception
            const int processed = pipe.ProcessedFrame();
            if (processed >= lastFrame) {
                if (++stableAtEnd > 15) break; // all frames integrated + settled
            } else {
                stableAtEnd = 0;
            }
            seen = processed;
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(180)) {
                std::printf("  [warn] tracker '%s' timed out at frame %d/%d\n", trackerName.c_str(), seen,
                            lastFrame);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        RunResult r;
        const ep::PipelineStats stats = pipe.GetStats();
        r.processedFrames = stats.processedFrame;
        r.alignMsAvg = stats.alignMsAvg;
        r.trackerRmseAvg = stats.trackerRmseAvg;
        if (const std::shared_ptr<const ep::ModelSnapshot> model = pipe.LatestModel()) {
            r.entries = model->entries.size();
            r.reconPoints.reserve(model->entries.size());
            for (const Engine::Spatial::AdvancedEntry &e: model->entries) r.reconPoints.push_back(e.center);
        }
        pipe.Stop();
        return r;
    }

} // namespace

int main(int argc, char **argv) {
    try {
        util::ArgParser arg =
                util::BuildArgParser(argc, argv)
                        .Must("--dir",
                              "usage: icp_quality_diag --dir <folder> [--voxel v] [--trackers identity,icp]")
                        .Option("--voxel")                  // default runtime-computed (extent / 200)
                        .Option("--trackers", "identity,icp");
        if (!arg) return 2; // a Must was unsatisfied (usage already printed)

        const std::string dir = arg.Value("--dir");
        const std::vector<std::string> framePaths = collectFramePaths(dir);
        if (framePaths.empty()) {
            std::printf("no frame_*.ply found in %s\n", dir.c_str());
            return 1;
        }
        const std::vector<ep::Frame> frames = ep::LoadFrames(framePaths);
        const ep::FrameBounds bounds = ep::ComputeBounds(frames);
        const float voxel = arg.ValueFloat("--voxel", bounds.Extent() / 200.0f);

        // Split "identity,icp" into names.
        std::vector<std::string> trackers;
        {
            const std::string s = arg.Value("--trackers");
            std::size_t i = 0;
            while (i < s.size()) {
                std::size_t j = s.find(',', i);
                if (j == std::string::npos) j = s.size();
                if (j > i) trackers.push_back(s.substr(i, j - i));
                i = j + 1;
            }
        }

        std::printf("dir      : %s  (%zu frames, extent %.4f)\n", dir.c_str(), framePaths.size(),
                    bounds.Extent());
        std::printf("voxel    : %.4f\n\n", voxel);
        std::printf("%-10s | %8s | %10s | %13s | %9s\n", "tracker", "frames", "align ms", "trackerRmse",
                    "entries");
        std::printf("-----------|----------|------------|---------------|----------\n");

        std::vector<RunResult> results;
        for (const std::string &t: trackers) {
            RunResult r = runTracker(framePaths, voxel, t);
            std::printf("%-10s | %8d | %10.2f | %13.6f | %9zu\n", t.c_str(), r.processedFrames, r.alignMsAvg,
                        r.trackerRmseAvg, r.entries);
            results.push_back(std::move(r));
        }

        // If an `identity` run exists, score every other tracker's reconstruction against it (identity
        // == the pre-registered ground-truth reference). Higher RMSE => that tracker drifted the surface.
        int identityIndex = -1;
        for (std::size_t i = 0; i < trackers.size(); ++i)
            if (trackers[i] == "identity") identityIndex = int(i);

        if (identityIndex >= 0 && !results[identityIndex].reconPoints.empty()) {
            const std::vector<Eigen::Vector3f> &ref = results[identityIndex].reconPoints;
            std::printf("\nReconstruction drift vs identity (nearest-neighbour RMSE, world units):\n");
            for (std::size_t i = 0; i < trackers.size(); ++i) {
                if (int(i) == identityIndex || results[i].reconPoints.empty()) continue;
                const float aToRef = Engine::Eval::NearestNeighbourRMSE(results[i].reconPoints, ref);
                const float refToA = Engine::Eval::NearestNeighbourRMSE(ref, results[i].reconPoints);
                std::printf("  %-10s : recon->identity %.5f   identity->recon %.5f\n", trackers[i].c_str(),
                            aToRef, refToA);
            }
            std::printf("\n(~0 = that tracker reproduced the identity reconstruction; large = it drifted.)\n");
        }
        return 0;
    } catch (const std::exception &e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
}

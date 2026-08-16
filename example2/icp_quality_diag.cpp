// Headless ICP-quality diagnostic: runs the REAL reconstruction Pipeline (integrate + track) over a
// folder of frame_*.ply frames with a chosen tracker, then reports reconstruction + tracker stats.
//
// Purpose: scan_out is pre-registered (frames already in world frame), so `identity` is the correct
// tracker and its reconstruction is the ground-truth reference. This tool measures how far the `icp`
// tracker's reconstruction DRIFTS from the `identity` reference (nearest-neighbour RMSE, both
// directions) plus each tracker's average residual RMSE / align time -- a headless stand-in for the
// GUI viewer's live quality, so icp-vs-identity (and clean-map-vs-WIP-map) can be A/B'd with numbers.
//
// Usage: icp_quality_diag --dir <folder of frame_*.ply> [--voxel v] [--trackers identity,icp]
//        icp_quality_diag --replay <depth recording>   [--voxel v] [--trackers identity,icp]
//
// --replay drives the same comparison from a raw depth recording, so a real camera capture can be
// scored the same way. Note what "identity" means there: a hand-held camera's frames are NOT
// pre-registered, so identity is no longer ground truth -- it is the null hypothesis, the
// reconstruction you get by pretending the sensor never moved. Drift away from it is the signal
// that ICP found motion, not that ICP is wrong.

#include "Pipeline/Pipeline.h"
#include "Pipeline/Registration/Tracker.h"       // TrackerRegistry
#include "Pipeline/Reconstruction/DepthCameraFrameSource.h"
#include "Pipeline/Reconstruction/DepthRecording.h"
#include "Pipeline/Reconstruction/FrameLoader.h" // LoadFrames / ComputeBounds
#include "Engine/Eval/RmseMetrics.h"                     // NearestNeighbourRMSE
#include "utilities/ArgParser.h"

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <memory>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace ep = Pipeline;
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
        // Per-stage counters. The headline "frames" is the last INTEGRATED frame index, which on
        // its own cannot say whether a short run means the source ended, the tracker stalled, or
        // the lossy trackedFrames channel threw the rest away.
        std::uint64_t acquired = 0, aligned = 0, integrated = 0;
        std::size_t dropped = 0;
    };

    // Drive the pipeline to completion over all frames, then snapshot the final model + stats.
    RunResult runTracker(const ep::AcquisitionConfig &acquisition, int lastFrame, float voxel,
                         float truncation, const std::string &trackerName) {
        // acquisition already carries downsampleVoxel; Pipeline only fills it when it is 0.
        ep::Pipeline::Config config;
        config.map.baseVoxel = voxel;
        if (truncation > 0.0f) config.map.truncation = truncation;
        config.acquisition = acquisition;

        ep::TrackerRegistry registry = ep::TrackerRegistry::Default();
        ep::Pipeline pipe(config, registry.Create(trackerName));
        pipe.Start();
        pipe.SetPaused(false);

        // Two completion tests, because reaching the last frame is not guaranteed: trackedFrames
        // is a lossy channel (capacity 4, drops when full), so whenever integration is slower than
        // tracking the final frame index is simply never processed. Waiting only on that burns the
        // whole timeout on a run that finished long ago.
        static constexpr int kStallTicks = 100; // 100 * 20 ms = 2 s of no new integration
        int seen = -1, stableAtEnd = 0, stalledTicks = 0;
        std::uint64_t lastIntegrated = 0;
        const auto start = std::chrono::steady_clock::now();
        while (true) {
            pipe.CheckErrors(); // rethrow any worker-stage exception
            const ep::PipelineStats live = pipe.GetStats();
            const int processed = pipe.ProcessedFrame();
            if (lastFrame >= 0 && processed >= lastFrame) {
                if (++stableAtEnd > 15) break; // all frames integrated + settled
            } else {
                stableAtEnd = 0;
            }
            if (live.integratedFrames == lastIntegrated) {
                if (live.integratedFrames > 0 && ++stalledTicks > kStallTicks) break;
            } else {
                stalledTicks = 0;
                lastIntegrated = live.integratedFrames;
            }
            if (processed != seen && processed % 25 == 0)
                std::printf("  [%s] frame %d/%d  integrated %llu  dropped %zu\n",
                            trackerName.c_str(), processed, lastFrame,
                            (unsigned long long) live.integratedFrames, live.trackDropped);
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
        r.acquired = stats.acquiredFrames;
        r.aligned = stats.alignedFrames;
        r.integrated = stats.integratedFrames;
        r.dropped = stats.trackDropped;
        if (const std::shared_ptr<const ep::ModelSnapshot> model = pipe.LatestModel()) {
            r.entries = model->entries.size();
            r.reconPoints.reserve(model->entries.size());
            for (const TSDFVoxel &e: model->entries) r.reconPoints.push_back(e.center);
        }
        pipe.Stop();
        return r;
    }

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        util::ArgParser arg =
                util::BuildArgParser(argc, argv)
                        .Option("--dir")
                        .Option("--replay")
                        .Option("--voxel")     // default runtime-computed (extent / 200)
                        .Option("--truncation")// default: MapConfig's, or 3 voxels for --replay
                        .Option("--downsample") // acquisition-stage voxel; default = the map's finest
                        .Option("--trackers", "identity,icp");

        const std::string dir = arg.Value("--dir");
        const std::string replayDirectory = arg.Value("--replay");
        if (dir.empty() == replayDirectory.empty()) {
            std::printf("usage: icp_quality_diag --dir <folder of frame_*.ply> [--voxel v] "
                        "[--trackers identity,icp]\n"
                        "       icp_quality_diag --replay <depth recording> [--voxel v] "
                        "[--truncation t] [--trackers identity,icp]\n");
            return 2;
        }

        ep::AcquisitionConfig acquisition;
        int lastFrame = -1;
        float voxel = 0.0f;
        float truncation = arg.ValueFloat("--truncation", 0.0f);
        std::string label;

        if (!dir.empty()) {
            const std::vector<std::string> framePaths = collectFramePaths(dir);
            if (framePaths.empty()) {
                std::printf("no frame_*.ply found in %s\n", dir.c_str());
                return 1;
            }
            const ep::FrameBounds bounds = ep::ComputeBounds(ep::LoadFrames(framePaths));
            voxel = arg.ValueFloat("--voxel", bounds.Extent() / 200.0f);
            acquisition.type = ep::EAcquisitionType::File;
            acquisition.framePaths = framePaths;
            acquisition.intervalMs = 0.0; // as fast as the stages allow
            acquisition.loop = false;
            lastFrame = int(framePaths.size()) - 1;
            char buf[512];
            std::snprintf(buf, sizeof buf, "%s  (%zu frames, extent %.4f)", dir.c_str(),
                          framePaths.size(), bounds.Extent());
            label = buf;
        } else {
            // Opened once here for its frame count and intrinsics; each run builds its own.
            const ep::RecordedDepthProvider probe(replayDirectory);
            const ep::CameraIntrinsics k = probe.Intrinsics();
            lastFrame = probe.FrameCount() - 1;
            voxel = arg.ValueFloat("--voxel", 0.01f); // indoor scale, not the 190 m synthetic default
            if (truncation <= 0.0f) truncation = 3.0f * voxel;

            acquisition.type = ep::EAcquisitionType::DepthCamera;
            acquisition.makeSource = [replayDirectory]() -> std::unique_ptr<ep::IFrameSource> {
                return std::make_unique<ep::DepthCameraFrameSource>(
                        std::make_unique<ep::RecordedDepthProvider>(replayDirectory));
            };
            char buf[512];
            std::snprintf(buf, sizeof buf, "%s  (%d depth frames, %dx%d, fx %.2f)",
                          replayDirectory.c_str(), probe.FrameCount(), k.width, k.height, k.fx);
            label = buf;
        }

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

        // The map and the TRACKER want different things from the input. A point finer than a map
        // voxel is invisible to the map, but ICP still uses it to find correspondences, so tying
        // the two together is an assumption worth being able to break.
        const float downsample = arg.ValueFloat("--downsample", 0.0f);
        if (downsample != 0.0f) acquisition.downsampleVoxel = downsample;

        std::printf("source   : %s\n", label.c_str());
        std::printf("voxel    : %.4f   truncation : %.4f   downsample : %s\n\n", voxel,
                    truncation > 0.0f ? truncation : 1.5f,
                    downsample == 0.0f ? "(map's finest)"
                                       : (downsample < 0.0f ? "off" : std::to_string(downsample).c_str()));
        std::printf("%-10s | %6s | %8s | %11s | %9s | %6s %6s %6s %6s\n", "tracker", "frames",
                    "align ms", "trackerRmse", "entries", "acq", "align", "integ", "drop");
        std::printf("-----------|--------|----------|-------------|-----------|"
                    "-------------------------------\n");

        std::vector<RunResult> results;
        for (const std::string &t: trackers) {
            RunResult r = runTracker(acquisition, lastFrame, voxel, truncation, t);
            std::printf("%-10s | %6d | %8.2f | %11.6f | %9zu | %6llu %6llu %6llu %6zu\n", t.c_str(),
                        r.processedFrames, r.alignMsAvg, r.trackerRmseAvg, r.entries,
                        (unsigned long long) r.acquired, (unsigned long long) r.aligned,
                        (unsigned long long) r.integrated, r.dropped);
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

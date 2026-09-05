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
//        icp_quality_diag --replay <depth recording>   [--voxel v] [--trackers identity,icp] [--no-p2p]
//
// --replay drives the same comparison from a raw depth recording, so a real camera capture can be
// scored the same way. Note what "identity" means there: a hand-held camera's frames are NOT
// pre-registered, so identity is no longer ground truth -- it is the null hypothesis, the
// reconstruction you get by pretending the sensor never moved. Drift away from it is the signal
// that ICP found motion, not that ICP is wrong.

#include "Pipeline/Pipeline.h"
#include "Pipeline/Registration/GpuIcpTracker.h"
#include "Pipeline/Registration/Tracker.h"       // TrackerRegistry
#include "Pipeline/Acquisition/DepthCameraFrameSource.h"
#include "Pipeline/Realsense/GpuDepthFrameSource.h"
#include "Pipeline/Acquisition/DepthRecording.h"
#include "Pipeline/Acquisition/FrameLoader.h" // LoadFrames / ComputeBounds
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

    float g_minFitness = 0.0f; // set from --min-fitness before any run
    // TSDF integration knobs, so a map setting can be A/B'd on a real recording. Valid to compare
    // only because a recording now runs lock-stepped (FrameHandshake): before that, changing the map
    // changed which map version the tracker saw, so the trajectory diverged and the comparison
    // measured thread scheduling instead of the setting.
    bool g_pointToPlane = true;
    // Range-adaptive truncation band multiplier (0 = off, the shipped default).
    float g_bandSigmaMultiplier = 0.0f;
    // Occluded-side-only weight profile (off = the symmetric one, the shipped default).
    bool g_behindSurfaceDropoff = false;
    bool g_submap = true;
    // Depth prefilter window for the --replay front end (0 = off, the shipped default).
    int g_prefilterWindow = 0;
    // Confidence gates for the --replay front end, all off by default. Globals for the same reason
    // the prefilter window is one: makeSource is rebuilt per tracker run and captures by value.
    ep::DepthFilterOptions g_depthGates;
    std::string g_frontEnd = "cpu";
    float g_gpuScoreThreshold = 0.9f;
    float g_gpuDownsampleVoxel = 0.0f;

    // Shared so the counters survive every makeSource rebuild and can be reported after the runs.
    std::shared_ptr<ep::DepthFilterStats> g_depthGateStats;


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

    // Fusion gate settings from the CLI; off unless the caller asks for them.
    ep::FusionGateConfig g_fusion;

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
        double stepAvg = 0.0, stepMax = 0.0, turnMax = 0.0, pathLength = 0.0;
        std::uint64_t rejected = 0, noModel = 0, noLocal = 0, fewInliers = 0, lowOverlap = 0;
        std::uint64_t implausibleMotion = 0;
        std::uint64_t skippedFusions = 0;
        std::uint64_t bootstrapHeld = 0, rejectedByFitness = 0, rejectedByRmse = 0;
        bool fusionArmed = true;
    };

    // Drive the pipeline to completion over all frames, then snapshot the final model + stats.
    RunResult runTracker(const ep::AcquisitionConfig &acquisition, int lastFrame, float voxel,
                         float truncation, const std::string &trackerName) {
        // acquisition already carries downsampleVoxel; Pipeline only fills it when it is 0.
        ep::Pipeline::Config config;
        config.map.baseVoxel = voxel;
        config.map.pointToPlane = g_pointToPlane;
        config.map.bandSigmaMultiplier = g_bandSigmaMultiplier;
        config.map.behindSurfaceDropoff = g_behindSurfaceDropoff;
        config.map.submap = g_submap;
        if (truncation > 0.0f) config.map.truncation = truncation;
        config.acquisition = acquisition;
        config.fusion = g_fusion;

        ep::TrackerRegistry registry = ep::TrackerRegistry::Default();
        std::unique_ptr<ep::Tracker> tracker = registry.Create(trackerName);
        if (g_minFitness > 0.0f)
            if (auto *gpu = dynamic_cast<ep::GpuIcpTracker *>(tracker.get())) gpu->SetMinFitness(g_minFitness);
        ep::Pipeline pipe(config, std::move(tracker));
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
            // Generous, and it has to be: a recording runs LOCK-STEPPED (FrameHandshake), so the
            // registration and integration stages no longer overlap and a replay takes roughly the
            // sum of their per-frame costs. At 477 frames x ~0.35 s the old 180 s wall cut runs off
            // near frame 350, which reads exactly like a diverged tracker -- two runs that differed
            // only in where the timeout landed looked like nondeterminism.
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(1800)) {
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
        r.rejected = stats.trackRejected;
        r.noModel = stats.rejectedNoModel;
        r.noLocal = stats.rejectedNoLocalTarget;
        r.fewInliers = stats.rejectedTooFewInliers;
        r.lowOverlap = stats.rejectedLowOverlap;
        r.implausibleMotion = stats.rejectedImplausibleMotion;
        r.skippedFusions = stats.skippedFusions;
        r.bootstrapHeld = stats.bootstrapHeldFrames;
        r.rejectedByFitness = stats.fusionRejectedByFitness;
        r.rejectedByRmse = stats.fusionRejectedByRmse;
        r.fusionArmed = stats.fusionArmed;
        r.stepAvg = stats.poseDeltaMetersAvg;
        r.stepMax = stats.poseDeltaMetersMax;
        r.turnMax = stats.poseDeltaDegreesMax;
        r.pathLength = stats.trajectoryLengthMeters;
        if (const std::shared_ptr<const ep::ModelSnapshot> model = pipe.LatestModel()) {
            r.entries = model->entries.size();
            // Observable ceilings: a detail level that cannot be inserted disappears silently, so
            // an entry count that moved the wrong way is diagnosable only from here.
            std::printf("  [%s] base tiles %u  detail tiles %u  dense blocks %u  "
                        "hash insert failures %llu  window refusals %u\n",
                        trackerName.c_str(), model->baseTiles, model->detailTiles, model->denseBlocks,
                        (unsigned long long) model->map.insertFailureCount,
                        model->windowLimitRefusals);
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
                        .Option("--min-fitness", 0.0)
                        .Option("--prefilter", 0)
                        // Confidence gates, 0 = off for each: --near/--far in metres,
                        // --min-neighbours out of 8, --max-incidence in degrees. They discard
                        // measurements, so the per-cause counts are printed with the run.
                        .Option("--near", 0.0)
                        .Option("--far", 0.0)
                        .Option("--min-neighbours", 0)
                        .Option("--max-incidence", 0.0)
                        // Range-adaptive truncation band: band = N * sigma_z(z), floored at 2
                        // voxels and capped at --truncation. 0 = off (the fixed band).
                        .Option("--band-sigma", 0.0)
                        // Weight only the occluded side of the band down (Bylow / Voxblox eq. 5)
                        // instead of both sides equally.
                        .Option("--behind-dropoff")
                        // Test the point against all eight neighbours for a depth step, not only
                        // the two the normal is differenced from.
                        .Option("--symmetric-guard")
                        // Base-only map: no density classifier, no detail level, no detail hash.
                        .Option("--no-submap")
                        .Option("--trackers", "identity,icp")
                        .Option("--bootstrap-frames", 0)      // 0 = off
                        .Option("--bootstrap-fitness", 0.70)
                        .Option("--min-fuse-fitness", 0.0)    // 0 = off
                        .Option("--max-fuse-rmse", 0.0)  // 0 = off
                        // Which depth front end builds the frames. "cpu" is BackprojectDepth on
                        // the host; "gpu" is Realsense::RealSensePipeline on the device. Everything
                        // downstream is identical either way, so the table separates the two.
                        .Option("--frontend", "cpu")
                        .Option("--gpu-threshold", 0.9)   // confidence bar, gpu front end only
                        .Option("--gpu-downsample", 0.0); // detail voxel [m]; 0 = off

        g_frontEnd = arg.Value("--frontend");
        if (g_frontEnd != "cpu" && g_frontEnd != "gpu")
            throw std::runtime_error("--frontend must be cpu or gpu, not '" + g_frontEnd + "'");
        g_gpuScoreThreshold = float(arg.ValueFloat("--gpu-threshold"));
        g_gpuDownsampleVoxel = float(arg.ValueFloat("--gpu-downsample"));

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
            // Two front ends over the SAME recording. The CPU one back-projects and estimates
            // normals per pixel on the host; the GPU one hands the frame to
            // Realsense::RealSensePipeline and reads back only the survivors. Selecting here rather
            // than in two tools is the point -- everything downstream is then identical, so a
            // difference in the table below is the front end and nothing else.
            const bool useGpuFrontEnd = g_frontEnd == "gpu";
            acquisition.makeSource = [replayDirectory, useGpuFrontEnd, k]()
                    -> std::unique_ptr<ep::IFrameSource> {
                if (useGpuFrontEnd) {
                    Realsense::ValidationScoreOptions score;
                    score.focalLengthPixels = k.fx;
                    score.depthScale = 0.001f;
                    // The recording predates this module and carries no baseline, so the D435
                    // datasheet value stands in. Every comparison here is relative.
                    score.baselineMeters = 0.05f;
                    Realsense::DownSampleOptions downSample;
                    downSample.enabled = g_gpuDownsampleVoxel > 0.0f;
                    downSample.detailVoxelMeters = g_gpuDownsampleVoxel;
                    return std::make_unique<ep::GpuDepthFrameSource>(
                            std::make_unique<ep::RecordedDepthProvider>(replayDirectory), score,
                            Realsense::NormalEstimationOptions{}, downSample, g_gpuScoreThreshold);
                }
                ep::DepthFilterOptions filter = g_depthGates;
                filter.prefilterWindow = g_prefilterWindow;
                return std::make_unique<ep::DepthCameraFrameSource>(
                        std::make_unique<ep::RecordedDepthProvider>(replayDirectory), filter,
                        g_depthGateStats);
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
        // Scoring is offline: every configuration must process the SAME frames, or a slower
        // setting silently processes fewer and the comparison measures the drop rate instead of
        // the tracker.
        g_fusion.bootstrapConsecutiveFrames = arg.ValueInt("--bootstrap-frames");
        g_fusion.bootstrapMinFitness = arg.ValueFloat("--bootstrap-fitness");
        g_fusion.minimumFusionFitness = arg.ValueFloat("--min-fuse-fitness");
        g_fusion.maximumFusionRmse = arg.ValueFloat("--max-fuse-rmse");

        acquisition.realTime = false;

        g_minFitness = arg.ValueFloat("--min-fitness", 0.0f);
        g_pointToPlane = !arg.Has("--no-p2p");
        g_bandSigmaMultiplier = arg.ValueFloat("--band-sigma");
        g_behindSurfaceDropoff = arg.Has("--behind-dropoff");
        g_submap = !arg.Has("--no-submap");
        g_depthGates.symmetricDepthJumpGuard = arg.Has("--symmetric-guard");
        g_prefilterWindow = arg.ValueInt("--prefilter", 0);
        g_depthGates.minimumDepthMeters = arg.ValueFloat("--near");
        g_depthGates.maximumDepthMeters = arg.ValueFloat("--far");
        g_depthGates.minimumValidNeighbours = arg.ValueInt("--min-neighbours");
        g_depthGates.maximumIncidenceDegrees = arg.ValueFloat("--max-incidence");
        g_depthGateStats = std::make_shared<ep::DepthFilterStats>();
        const float downsample = arg.ValueFloat("--downsample", 0.0f);
        if (downsample != 0.0f) acquisition.downsampleVoxel = downsample;

        std::printf("source   : %s\n", label.c_str());
        std::printf("map      : point-to-plane %s   depth prefilter %d\n",
                    g_pointToPlane ? "on" : "off", g_prefilterWindow);
        std::printf("band     : adaptive %.1f sigma_z (0 = fixed truncation), behind-dropoff %s\n",
                    g_bandSigmaMultiplier, g_behindSurfaceDropoff ? "on" : "off");
        std::printf("guard    : symmetric depth-jump %s\n",
                    g_depthGates.symmetricDepthJumpGuard ? "on" : "off");
        std::printf("gates    : range %.2f-%.2f m   neighbours >=%d   incidence <=%.0f deg\n",
                    g_depthGates.minimumDepthMeters, g_depthGates.maximumDepthMeters,
                    g_depthGates.minimumValidNeighbours, g_depthGates.maximumIncidenceDegrees);
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

        // A hand-held sensor at 30 fps moves well under 0.05 m and a few degrees per frame. These
        // are what separate a tracker that spread the frames because the camera really moved from
        // one that diverged -- the residual rmse cannot, because it only scores the correspondences
        // the tracker itself picked.
        std::printf("\nPer-frame motion the tracker claims (30 fps hand-held: step < ~0.05 m):\n");
        std::printf("%-10s | %10s | %10s | %10s | %12s | %8s\n", "tracker", "step avg", "step max",
                    "turn max", "path length", "rejected");
        std::printf("-----------|------------|------------|------------|--------------|---------\n");
        for (std::size_t i = 0; i < trackers.size(); ++i)
            std::printf("%-10s | %10.4f | %10.4f | %9.2f d | %12.3f | %8llu\n", trackers[i].c_str(),
                        results[i].stepAvg, results[i].stepMax, results[i].turnMax,
                        results[i].pathLength, (unsigned long long) results[i].rejected);

        // `skipped` is the subset NOT fused. no-model / no-local-map frames ARE still fused: there is
        // no local map for their wrong pose to corrupt, and refusing them would stop the map ever
        // bootstrapping or ever growing into new territory (see ShouldFuse in Pipeline/Types.h).
        std::printf("\nWhy tracks were rejected (too-few-inliers / low-overlap frames are NOT fused; "
                    "no-model / no-local-map ones are):\n");
        std::printf("%-10s | %9s | %13s | %14s | %11s | %12s | %8s\n", "tracker", "no model",
                    "no local map", "too few inliers", "low overlap", "implausible", "skipped");
        std::printf("-----------|-----------|---------------|----------------|-------------|"
                    "--------------|---------\n");
        for (std::size_t i = 0; i < trackers.size(); ++i)
            std::printf("%-10s | %9llu | %13llu | %14llu | %11llu | %12llu | %8llu\n",
                        trackers[i].c_str(),
                        (unsigned long long) results[i].noModel,
                        (unsigned long long) results[i].noLocal,
                        (unsigned long long) results[i].fewInliers,
                        (unsigned long long) results[i].lowOverlap,
                        (unsigned long long) results[i].implausibleMotion,
                        (unsigned long long) results[i].skippedFusions);

        // The FusionGate's share of those skipped fusions. Printed only when the gate is on, so a
        // default run's report stays exactly as it was.
        const bool gateOn = g_fusion.bootstrapConsecutiveFrames > 0 || g_fusion.minimumFusionFitness > 0.0f ||
                            g_fusion.maximumFusionRmse > 0.0f;
        if (gateOn) {
            std::printf("\nFusion gate (bootstrap %d frames @ fitness %.2f; steady minFitness %.2f, maxRmse %.4f):\n",
                        g_fusion.bootstrapConsecutiveFrames, g_fusion.bootstrapMinFitness,
                        g_fusion.minimumFusionFitness, g_fusion.maximumFusionRmse);
            std::printf("%-10s | %15s | %18s | %15s | %6s\n", "tracker", "bootstrap held", "rejected by fitness",
                        "rejected by rmse", "armed");
            std::printf("-----------|-----------------|--------------------|-----------------|-------\n");
            for (std::size_t i = 0; i < trackers.size(); ++i)
                std::printf("%-10s | %15llu | %18llu | %15llu | %6s\n",
                            trackers[i].c_str(),
                            (unsigned long long) results[i].bootstrapHeld,
                            (unsigned long long) results[i].rejectedByFitness,
                            (unsigned long long) results[i].rejectedByRmse,
                            results[i].fusionArmed ? "yes" : "NO");
        }

        // Per cause, never summed: rejectedByRange counts PIXELS (the gate runs before
        // back-projection) while the other two count candidate POINTS, and a gate set too tight
        // shows up only as a thinner reconstruction unless it is reported. Totals span every
        // tracker run, since one shared counter outlives each makeSource rebuild.
        if (g_depthGateStats && !replayDirectory.empty()) {
            std::printf("\nDepth gates over all runs: %llu points kept, rejected %llu range (px) / "
                        "%llu support / %llu incidence\n",
                        (unsigned long long) g_depthGateStats->emittedPoints.load(),
                        (unsigned long long) g_depthGateStats->rejectedByRange.load(),
                        (unsigned long long) g_depthGateStats->rejectedByNeighbourSupport.load(),
                        (unsigned long long) g_depthGateStats->rejectedByIncidence.load());
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

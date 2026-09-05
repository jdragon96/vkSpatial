#include "VoxelFillDebug.h"          // voxdbg::belowThreshold
#include "VoxelFillRenderStrategy.h" // VoxelFillRenderStrategy

#include "Engine/Core/Context.h"
#include "Pipeline/Pipeline.h"                   // Pipeline / Config / MapConfig / Frame
#include "Pipeline/Acquisition/FrameLoader.h" // LoadFrames / ComputeBounds
#include "Pipeline/Registration/Tracker.h"       // TrackerRegistry / Tracker
#include "Pipeline/Render/RenderThread.h"        // RenderThread
#include "Mesh/ExtractorRegistry.h"
#include "Mesh/MeshConnectivity.h"
#include "Mesh/VoxelField.h"
#include "TSDF/TSDF.h"                           // headless --dump map

#include "utilities/ArgParser.h"
#include "utilities/StageProfiler.h"

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ep = Pipeline;
using Eigen::Vector3f;

namespace {

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

    // AABB of downloaded voxel entries, expanded by half a voxel to the true voxel extent.
    bool entriesAabb(const std::vector<TSDFVoxel> &e, float voxel, Vector3f &mn, Vector3f &mx) {
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

    // Apply the per-stage integration toggles onto a MapConfig. Shared by the initial config build and
    // every live toggle-rebuild, so the two paths can never drift. `confWeight` is the confidence weight
    // used when confidence is enabled (0 disables the confidence term).
    void applyOpts(ep::MapConfig &m, const VoxelFillRenderStrategy::Opts &o, float confWeight) {
        m.submap = o.submap;
        m.pointToPlane = o.pointToPlane;
        m.confidence = o.confidence ? confWeight : 0.0f;
        m.hermite = o.hermite;
        m.downsample = o.downsample;
    }

    // The numbers a TSDF change is judged by. Memory and load factor say whether the map fits;
    // probes say whether the hash is still cheap at that load; drops and refusals say whether
    // anything was lost getting there.
    void reportMetrics(const TSDF &tsdf, float voxel, std::size_t occupied) {
        const TSDFBackendStats stats = tsdf.Stats();
        const double megabyte = 1024.0 * 1024.0;

        std::printf("\n== TSDF metrics ==\n");
        std::printf("  memory      %8.1f MB total   %7.3f MB/window   %6.1f B/occupied voxel\n",
                    double(stats.deviceMemoryBytes) / megabyte,
                    stats.tableCount ? double(stats.deviceMemoryBytes) / megabyte / stats.tableCount : 0.0,
                    occupied ? double(stats.deviceMemoryBytes) / double(occupied) : 0.0);
        std::printf("  occupancy   %8llu / %llu slots   load %.3f   grows %llu\n",
                    (unsigned long long) stats.filledCount,
                    (unsigned long long) stats.hashCapacity, stats.LoadFactor(),
                    (unsigned long long) stats.growCount);
        std::printf("  windows     %8u total   base %zu   detail %zu\n",
                    stats.tableCount, tsdf.BaseWindowCount(), tsdf.DetailWindowCount());

        if (stats.probeQueryCount > 0) {
            // Knuth's successful-search estimate for linear probing, for comparison.
            const double alpha = stats.LoadFactor();
            const double predicted = alpha < 1.0 ? 0.5 * (1.0 + 1.0 / (1.0 - alpha)) : 0.0;
            std::printf("  probes      %8.2f slots/lookup   worst %u   (linear-probe theory at this "
                        "load: %.2f)\n",
                        stats.AverageProbes(), stats.probeSlotMax, predicted);
            std::printf("              %8llu lookups over %llu slots examined\n",
                        (unsigned long long) stats.probeQueryCount,
                        (unsigned long long) stats.probeSlotTotal);
        } else {
            std::printf("  probes      (pass --probe-stats to measure; costs three atomics per "
                        "lookup)\n");
        }

        const float detailVoxel = voxel * 0.5f;
        std::printf("  voxel       base %.4f m   detail %.4f m\n", voxel, detailVoxel);

        if (stats.insertFailureCount > 0)
            std::printf("  WARNING     %llu observation(s) dropped -- the load factor limit is too "
                        "high for this scene\n",
                        (unsigned long long) stats.insertFailureCount);
        if (tsdf.WindowLimitRefusalCount() > 0)
            std::printf("  WARNING     %u point(s) refused by the window ceiling -- that geometry is "
                        "missing\n",
                        tsdf.WindowLimitRefusalCount());
    }

    int runDump(const ep::MapConfig &cfg, const std::vector<ep::Frame> &frames, float wThresh,
                bool probeStats) {
        Engine::Core::Context ctx;
        TSDF tsdf;
        TSDFConfiguration config;
        config.backend = "advanced";
        config.splitter = cfg.submap ? "dense" : "none";
        config.useSubmap = cfg.submap;
        config.backendConfig.voxelSize = cfg.baseVoxel;
        config.backendConfig.truncation = cfg.truncation;
        config.backendConfig.hashCapacity = cfg.tileHash;
        config.backendConfig.maxPointPerFrame = cfg.maxPoints;
        config.backendConfig.pointToPlane = cfg.pointToPlane;
        config.backendConfig.confidenceWeight = cfg.confidence;
        config.backendConfig.hermitePosition = cfg.hermite;
        config.backendConfig.maxDirections = cfg.maxDirections;
        config.backendConfig.directionExponent = cfg.directionExponent;
        config.backendConfig.viewAngleWeight = cfg.viewAngleWeight;
        config.splitterConfig.baseResolution = cfg.baseVoxel;
        config.splitterConfig.blockVoxels = cfg.blockVoxels;
        config.splitterConfig.maxPointPerFrame = int(cfg.maxPoints);
        config.backendConfig.probeStats = probeStats;
        tsdf.Build(ctx, config);

        const int nFrames = int(frames.size());
        util::StageProfiler prof;
        std::vector<TSDFVoxel> entries; // hoisted: reused each frame (warm buffer)
        int overInt = 0, overDl = 0;    // frames past the first that breached the 30 ms budget
        for (int f = 0; f < nFrames; ++f) {
            {
                util::ScopedStageTimer t(prof, "integrate");
                tsdf.Integrate(frames[f].pts, frames[f].nrm, frames[f].cam);
            }
            {
                util::ScopedStageTimer t(prof, "download");
                tsdf.Download(entries);
            }
            if (f > 0) { // the first frame is allowed to be slow (window creation + first-touch)
                if (prof.LastMs("integrate") > 30.0) ++overInt;
                if (prof.LastMs("download") > 30.0) ++overDl;
            }
            std::size_t below = 0, nnew = 0;
            for (const auto &e: entries) {
                if (voxdbg::belowThreshold(e.weight, wThresh)) ++below;
                if (e.firstFrame == f) ++nnew; // GPU-stamped first-seen -> "new this frame"
            }
            Vector3f aMn, aMx, aSz = Vector3f::Zero();
            if (entriesAabb(entries, cfg.baseVoxel, aMn, aMx)) aSz = aMx - aMn;
            std::printf("frame %3d: int %6.2f dl %5.2f %s occupied %zu  new %zu  below %zu  "
                        "base/detail windows %zu/%zu  allocBox(%.2f,%.2f,%.2f)\n",
                        f, prof.LastMs("integrate"), prof.LastMs("download"),
                        (f > 0 && prof.LastMs("integrate") > 30.0) ? "OVER" : "    ", entries.size(),
                        nnew, below, tsdf.BaseWindowCount(), tsdf.DetailWindowCount(), aSz.x(),
                        aSz.y(), aSz.z());
        }
        std::printf("\n%s", prof.Report("--dump per-frame TSDF pipeline").c_str());
        std::printf("[budget] frames past #0 over 30ms:  integrate=%d  download=%d  (of %d)\n", overInt,
                    overDl, nFrames - 1);
        reportMetrics(tsdf, cfg.baseVoxel, entries.size());
        // Same extraction the viewer runs, so "the mesh does not render" can be told apart from
        // "the extractor produced nothing".
        const Mesh::VoxelField field = Mesh::FromVoxels(entries, cfg.baseVoxel);
        Mesh::ExtractorRegistry registry = Mesh::ExtractorRegistry::Default();
        if (std::unique_ptr<Mesh::IsoSurfaceExtractor> extractor = registry.Create("mc")) {
            Mesh::ExtractParams params;
            params.isoLevel = 0.0f;
            const Mesh::SurfaceMesh mesh = extractor->Extract(field, params);
            const Mesh::ConnectivityReport report = Mesh::AnalyzeConnectivity(mesh);
            std::printf("\n== mesh (mc) ==\n");
            std::printf("  vertices %zu   triangles %zu\n", mesh.vertices.size(),
                        mesh.triangles.size());
            std::printf("  boundary edges (holes) %zu   non-manifold %zu   bowtie %zu\n",
                        report.boundaryEdges.size(), report.nonManifoldEdges.size(),
                        report.nonManifoldVertices.size());
        } else {
            std::printf("\n== mesh == registry.Create(\"mc\") returned null\n");
        }

        std::printf("\n[--dump] done.\n");
        return 0;
    }

    int runViewer(std::shared_ptr<const std::vector<ep::Frame>> frames,
                  const ep::MapConfig &baseMap,
                  const std::vector<std::string> &framePaths,
                  const ep::FrameBounds &bounds,
                  float wThresh,
                  const std::string &trackerName,
                  double intervalMs,
                  bool loop,
                  const VoxelFillRenderStrategy::Opts &initOpts) {
        ep::TrackerRegistry registry = ep::TrackerRegistry::Default();
        const float confValue = baseMap.confidence > 0.0f ? baseMap.confidence : 0.5f;

        auto makeConfig = [&](const VoxelFillRenderStrategy::Opts &o, const ep::MapConfig &base) {
            ep::MapConfig m = base;
            applyOpts(m, o, confValue);
            ep::Pipeline::Config config;
            config.map = m;
            config.acquisition.source = ep::EAcquisitionSource::PlyFolder;
            config.acquisition.framePaths = framePaths;
            config.acquisition.intervalMs = intervalMs;
            config.acquisition.loop = loop;
            return config;
        };

        ep::Pipeline pipe(makeConfig(initOpts, baseMap), registry.Create(trackerName));
        pipe.SetPaused(true);
        pipe.Start();

        VoxelFillRenderStrategy::Params params;
        params.frames = frames;
        params.voxel = baseMap.baseVoxel;
        params.trunc = baseMap.truncation;
        params.nFrames = int(frames->size());
        params.center = bounds.Center();
        params.extent = bounds.Extent();
        params.wThresh = wThresh;
        params.trackerName = trackerName;
        params.map = baseMap;
        params.intervalMs = intervalMs;
        params.loop = loop;
        params.opts = initOpts;
        params.onRebuild = [&](const VoxelFillRenderStrategy::Opts &o, const ep::MapConfig &m) {
            pipe.Reconfigure(makeConfig(o, m), registry.Create(trackerName));
            pipe.SetPaused(false); // resume playing so the effect of the toggle is visible
        };
        VoxelFillRenderStrategy strategy(std::move(params));

        ep::RenderThread rt({1280, 800, "Voxel Fill Debugger"});
        rt.Run(pipe, strategy); // blocks on the main thread until the window closes; stops the pipe
        return 0;
    }

    // Collect sorted frame_*.ply paths from `dir` (skipping ground_truth_*). Directory listing only --
    // the clouds themselves are read by the Reconstruction loader (ep::LoadFrames), not here.
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

} // namespace

int main(int argc, char **argv) {
    try {
        util::ArgParser arg =
                util::BuildArgParser(argc, argv)
                        .Must("--dir", "usage: voxel_fill_debugger --dir <folder> [--voxel v] "
                                       "[--trunc t] [--submap] [--downsample] [--no-p2p] [--conf L] "
                                       "[--hermite] [--wthresh w] [--tile-hash N] [--block V] "
                                       "[--detail-k K] [--detail-trunc-vox R] [--max-points N] "
                                       "[--tracker a] [--interval ms] [--loop] [--dump] [--probe-stats]")
                        .Option("--voxel") // default runtime-computed (extent / 200)
                        .Option("--trunc") // default runtime-computed (voxel * 2, real-time band)
                        .Option("--conf", 0.5)
                        .Option("--wthresh", 0.0)
                        .Option("--tile-hash", 1 << 19)
                        .Option("--block", 32)
                        .Option("--detail-k", 4.0)
                        .Option("--detail-trunc-vox", 3.0) // detail band radius in detail voxels
                        .Option("--max-points")            // default = largest loaded frame
                        .Option("--tracker", "identity")
                        .Option("--interval", 33.0); // ~30 fps pacing; 0 = as fast as consumed
        if (!arg) return 2;

        // ---- Reconstruction reads the clouds (shared loader), main only lists + configures ----
        const std::string dir = arg.Value("--dir");
        const std::vector<std::string> framePaths = collectFramePaths(dir);
        if (framePaths.empty()) throw std::runtime_error("no frame_*.ply found in " + dir);
        auto frames = std::make_shared<std::vector<ep::Frame>>(ep::LoadFrames(framePaths));
        if (frames->empty()) throw std::runtime_error("no usable frames (need per-point normals)");
        const ep::FrameBounds bounds = ep::ComputeBounds(*frames);

        // ---- Config: derive voxel/trunc/max-points from the data + flags, fill the MapConfig ----
        const float voxel = arg.ValueFloat("--voxel", bounds.Extent() / 200.0f);
        // Default band = 2 voxels: the GPU integrate splat cost scales ~(trunc/voxel)^3, and a 2-voxel
        // band keeps every occupied surface voxel (measured ~equal to a 3-voxel band) while cutting
        // per-frame integrate ~30% -- the real-time-friendly default. Pass --trunc for a wider band
        // (better far-field fill / noisier-scan fusion) at higher cost.
        const float trunc = arg.ValueFloat("--trunc", voxel * 2.0f);
        const float wThresh = arg.ValueFloat("--wthresh");
        const uint32_t maxPts =
                arg.Has("--max-points")
                        ? nextPow2(uint32_t(arg.ValueFloat("--max-points")))
                        : nextPow2(uint32_t(std::max<std::size_t>(bounds.maxFramePoints, 1u << 15)));

        VoxelFillRenderStrategy::Opts opts;
        opts.submap = arg.Has("--submap");         // off by default; opt in with --submap
        opts.downsample = arg.Has("--downsample"); // off by default (no-op on sparse scans)
        opts.pointToPlane = !arg.Has("--no-p2p");
        opts.confidence = arg.ValueFloat("--conf") > 0.0f;
        opts.hermite = arg.Has("--hermite");

        ep::MapConfig cfg;
        cfg.baseVoxel = voxel;
        cfg.truncation = trunc;
        cfg.blockVoxels = int(arg.ValueFloat("--block"));
        cfg.detailK = arg.ValueFloat("--detail-k");
        cfg.detailTruncVoxels = arg.ValueFloat("--detail-trunc-vox");
        cfg.tileHash = nextPow2(uint32_t(arg.ValueFloat("--tile-hash")));
        cfg.maxPoints = maxPts;
        cfg.probeStats = arg.Has("--probe-stats");
        applyOpts(cfg, opts, arg.ValueFloat("--conf"));

        std::printf("dir       : %s  (%d frames, extent %.4f)\n", dir.c_str(), int(frames->size()),
                    bounds.Extent());
        std::printf("map       : base %.4f + detail %.4f (band %.0f detail-vox), block %d vox, "
                    "detail-k %.1f, per-tile hash %u, submap %s\n",
                    voxel, voxel * 0.5f, cfg.detailTruncVoxels, cfg.blockVoxels, cfg.detailK,
                    cfg.tileHash, cfg.submap ? "on" : "off");
        std::printf("options   : p2p %s, confidence %.2f, hermite %s, downsample %s\n",
                    cfg.pointToPlane ? "on" : "off", cfg.confidence, cfg.hermite ? "on" : "off",
                    cfg.downsample ? "on" : "off");

        // ---- Run: headless benchmark or the live viewer ----
        if (arg.Has("--dump") || arg.Has("--no-view")) return runDump(cfg, *frames, wThresh, arg.Has("--probe-stats"));
        return runViewer(frames, cfg, framePaths, bounds, wThresh, arg.Value("--tracker"),
                         arg.ValueFloat("--interval"), arg.Has("--loop"), opts);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}

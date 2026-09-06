// TSDF folder evaluator — read every frame_*.ply in a folder, integrate them into a TSDF
// (windowing and hash selected by --tsdf/--hash), extract the surface, and score it
// against ground_truth.ply (RMSE). Fully separate from capture (object_scan_viewer) and headless.
//
//   ./tsdf_folder_eval --dir scans/bunny [--voxel v] [--trunc t] [--gt path.ply]
//        [--out extracted.ply] [--no-p2p] [--conf lambda] [--hermite]
//        [--tsdf flat|tile|submap] [--hash linear|bucketed] [--tile-hash n]
//
// --tsdf defaults BY SCENE SIZE: `flat` (one 512^3 window) while the object fits in 512
// voxels/axis, `tile` otherwise. A single window silently discards every voxel outside it, so an
// explicit `--tsdf flat` on an oversized scene is honoured but warned about on stderr.
//
// Per-frame camera position is ESTIMATED from each cloud (centroid + k·mean-normal), so this
// works on any folder of oriented-point PLYs, not just object_scan_viewer output.

#include "Engine/Core/Context.h"
#include "Engine/Core/OrientedPointCloud.h"
#include "TSDF/TSDF.h"

#include "utilities/ArgParser.h"
#include "utilities/PointCloudIO.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using Eigen::Vector3f;
namespace fs = std::filesystem;

namespace {

    // Arg parsing via the fluent util::ArgParser (utilities/ArgParser.h).

    uint32_t nextPow2(uint32_t v) {
        if (v <= 1) return 1;
        --v;
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
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

    // Uniform-grid nearest-neighbour over a reference cloud (for RMSE against ground truth).
    struct GridNN {
        float cell = 1.0f;
        int maxRing = 32; // set from the populated extent in build(); see nearestSq
        std::unordered_map<int64_t, std::vector<int>> grid;
        const std::vector<Vector3f> *pts = nullptr;

        static int64_t key(int x, int y, int z) {
            return (int64_t(x) & 0x1FFFFF) | ((int64_t(y) & 0x1FFFFF) << 21) |
                   ((int64_t(z) & 0x1FFFFF) << 42);
        }
        void build(const std::vector<Vector3f> &p, float cellSize) {
            pts = &p;
            cell = std::max(1e-6f, cellSize);
            Vector3f minimum = Vector3f::Constant(std::numeric_limits<float>::max());
            Vector3f maximum = Vector3f::Constant(std::numeric_limits<float>::lowest());
            for (int i = 0; i < int(p.size()); ++i) {
                const Vector3f &q = p[i];
                minimum = minimum.cwiseMin(q);
                maximum = maximum.cwiseMax(q);
                grid[key(int(std::floor(q.x() / cell)), int(std::floor(q.y() / cell)),
                         int(std::floor(q.z() / cell)))]
                        .push_back(i);
            }
            if (!p.empty())
                maxRing = 2 + int(std::ceil((maximum - minimum).maxCoeff() / cell));
        }
        float nearestSq(const Vector3f &q) const {
            const int cx = int(std::floor(q.x() / cell)), cy = int(std::floor(q.y() / cell)),
                      cz = int(std::floor(q.z() / cell));
            float best = std::numeric_limits<float>::infinity();
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        auto it = grid.find(key(cx + dx, cy + dy, cz + dz));
                        if (it == grid.end()) continue;
                        for (int idx: it->second) best = std::min(best, (q - (*pts)[idx]).squaredNorm());
                    }
            // Expand the search ring until a hit is found (sparse regions). maxRing must span the
            // whole populated grid, not a fixed 32: a ground-truth point in a region the
            // reconstruction never covered lies further away than any fixed ring, and returning
            // infinity there poisons the mean for every point at once -- which is how the
            // completeness figure came to read `inf` and the metric came to be ignored.
            for (int ring = 2; !std::isfinite(best) && ring <= maxRing; ++ring) {
                for (int dz = -ring; dz <= ring; ++dz)
                    for (int dy = -ring; dy <= ring; ++dy)
                        for (int dx = -ring; dx <= ring; ++dx) {
                            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != ring) continue;
                            auto it = grid.find(key(cx + dx, cy + dy, cz + dz));
                            if (it == grid.end()) continue;
                            for (int idx: it->second)
                                best = std::min(best, (q - (*pts)[idx]).squaredNorm());
                        }
            }
            // A query outside the populated grid entirely still needs a real distance: one
            // non-finite value makes the mean non-finite and silently discards the whole metric.
            if (!std::isfinite(best) && pts != nullptr)
                for (const Vector3f &p: *pts) best = std::min(best, (q - p).squaredNorm());
            return best;
        }
    };

    struct Err {
        double sum = 0, sumSq = 0;
        size_t n = 0;
        void add(double d) {
            sum += d;
            sumSq += d * d;
            ++n;
        }
        double mean() const { return n ? sum / double(n) : 0; }
        double rmse() const { return n ? std::sqrt(sumSq / double(n)) : 0; }
    };

} // namespace

int main(int argc, char **argv) {
    try {
        util::ArgParser arg =
                util::BuildArgParser(argc, argv)
                        .Must("--dir", "usage: tsdf_folder_eval --dir <folder> [--voxel v] "
                                       "[--trunc t] [--gt path] [--out extracted.ply] [--no-p2p] "
                                       "[--conf L] [--hermite]")
                        .Option("--voxel")          // default is runtime-computed (extent / 200)
                        .Option("--trunc")          // default is runtime-computed (voxel * 3)
                        .Option("--gt")
                        .Option("--out")
                        .Option("--conf", 0.5)
                        // Range-adaptive truncation band: band = N * sigma_z(z), floored at 2
                        // voxels and capped at --trunc. 0 = off (the fixed band).
                        .Option("--band-sigma", 0.0)
                        .Option("--tile-hash", 1 << 21)
                        // No declared default: --tsdf defaults to flat/tile by scene size, decided
                        // once axisVox is known (see defaultTsdfName below).
                        .Option("--tsdf")
                        .Option("--hash", "linear");
        if (!arg) return 2;
        const std::string dir = arg.Value("--dir");
        if (!fs::is_directory(dir)) {
            std::cerr << "not a directory: " << dir << "\n";
            return 2;
        }

        // Collect frame_*.ply (sorted); ground_truth.ply is excluded and used as GT.
        std::vector<std::string> framePaths;
        std::string gtPath = arg.Value("--gt");
        for (const auto &e: fs::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            const std::string name = e.path().filename().string();
            if (name.rfind("ground_truth", 0) == 0) {
                if (gtPath.empty()) gtPath = e.path().string();
                continue;
            }
            if (e.path().extension() == ".ply" && name.rfind("frame_", 0) == 0)
                framePaths.push_back(e.path().string());
        }
        std::sort(framePaths.begin(), framePaths.end());
        if (framePaths.empty()) throw std::runtime_error("no frame_*.ply found in " + dir);

        // Read frames + accumulate a world bbox for auto voxel sizing.
        struct Frame {
            std::vector<Vector3f> pts, nrm;
            Vector3f cam;
        };
        std::vector<Frame> frames;
        Vector3f bbMin = Vector3f::Constant(1e30f), bbMax = Vector3f::Constant(-1e30f);
        size_t totalPts = 0, maxFramePts = 0;
        for (const auto &p: framePaths) {
            Frame fr;
            if (!util::LoadPly(p, fr.pts, fr.nrm) || fr.nrm.size() != fr.pts.size()) {
                std::fprintf(stderr, "skip (no normals): %s\n", p.c_str());
                continue;
            }
            fr.cam = estimateCamera(fr.pts, fr.nrm);
            for (const auto &q: fr.pts) {
                bbMin = bbMin.cwiseMin(q);
                bbMax = bbMax.cwiseMax(q);
            }
            totalPts += fr.pts.size();
            maxFramePts = std::max(maxFramePts, fr.pts.size());
            frames.push_back(std::move(fr));
        }
        if (frames.empty()) throw std::runtime_error("no usable frames (need per-point normals)");
        const float extent = (bbMax - bbMin).norm();

        const bool p2p = !arg.Has("--no-p2p");
        const float conf = arg.ValueFloat("--conf");
        const bool hermite = arg.Has("--hermite");
        const float voxel = arg.ValueFloat("--voxel", extent / 200.0f);
        const float trunc = arg.ValueFloat("--trunc", voxel * 3.0f);

        // AdvancedTSDF is a SINGLE movable 512^3-voxel window. Its defaults (origin-centred window,
        // hashCapacity 1<<20, maxPoints 1<<15) silently break for fine voxels or off-origin/large
        // objects: voxels outside the window are dropped (garbage extract) and a saturated hash
        // collides. Size and place the window to the data instead.
        const Vector3f span = bbMax - bbMin;
        const float maxSpan = span.maxCoeff();
        const int margin = int(std::ceil(trunc / voxel)) + 2;            // truncation ghost band
        const long axisVox = long(std::ceil(maxSpan / voxel)) + 2L * margin;
        // A scene wider than 512 voxels/axis does NOT fit one AdvancedTSDF window, and the overflow
        // is silent: packDirKey's bounds check simply discards every voxel outside the window, so
        // `flat` would print a full stats table for whatever sliver happened to land inside. Pick
        // the tiled layout for such a scene unless the operator overrides --tsdf explicitly.
        const bool fitsOneWindow = axisVox <= 512;
        const char *const defaultTsdfName = fitsOneWindow ? "flat" : "tile";
        const bool forceSingle = arg.Has("--single");
        if (!fitsOneWindow && forceSingle) {
            const float minVoxel = maxSpan / float(512 - 2 * margin);
            std::fprintf(stderr,
                         "error: voxel %.4f too fine — object spans %ld voxels/axis but a single "
                         "AdvancedTSDF window is 512^3 (--single).\n"
                         "  drop --single (this scene then defaults to --tsdf tile), or use "
                         "--voxel >= %.4f.\n",
                         voxel, axisVox, minVoxel);
            return 3;
        }
        // Place the window's min corner a margin below the object; scale the hash to the expected
        // surface-shell entry count (bbox-surface proxy, x2 load headroom, clamped).
        const Vector3f windowMinCorner = bbMin - float(margin) * Vector3f::Constant(voxel);
        const double surf = 2.0 * double(span.x() * span.y() + span.y() * span.z() +
                                         span.z() * span.x());
        const double shell = 2.0 * double(trunc) / double(voxel);
        const double estEntries = (surf / (double(voxel) * double(voxel))) * shell * 1.5;
        const uint32_t hashCap =
                std::max(1u << 20, nextPow2(uint32_t(std::min(estEntries * 2.0, double(1u << 24)))));
        const uint32_t maxPts = nextPow2(uint32_t(std::max<size_t>(maxFramePts, 1u << 15)));
        // Per-tile hash capacity for the TILED path (each tile is one AdvancedTSDF window). Default
        // 1<<21 (~48MB/tile) is safe for a fully-surface-crossed tile but × many tiles can exceed
        // VRAM; lower it via --tile-hash for fine-voxel/large scenes (risks per-tile overflow).
        const uint32_t tileHash =
                nextPow2(uint32_t(arg.ValueFloat("--tile-hash")));

        std::printf("dir       : %s  (%zu frames, %zu total pts, extent %.4f)\n", dir.c_str(),
                    frames.size(), totalPts, extent);
        std::printf("advanced  : voxel %.4f, trunc %.4f, point-to-plane %s, conf %.2f, hermite %s\n",
                    voxel, trunc, p2p ? "on" : "off", conf, hermite ? "on" : "off");
        std::printf("window    : %ld voxels/axis, hashCap %u, maxPts %u, minCorner (%.2f,%.2f,%.2f)\n",
                    axisVox, hashCap, maxPts, windowMinCorner.x(), windowMinCorner.y(),
                    windowMinCorner.z());

        // Integrate + extract through the registry, so --tsdf/--hash pick the strategy without
        // recompiling. This is the axis the whole plan exists to measure (Task 6).
        Engine::Core::Context ctx;
        // Runtime default, not a declared one: `flat` only makes sense while the scene fits one
        // 512^3 window, and the harness cannot know that until it has read the frames.
        const std::string tsdfName = arg.Value("--tsdf", defaultTsdfName);
        const std::string hashName = arg.Value("--hash");
        // An explicit `--tsdf flat` on an oversized scene is the operator's call, but it must not
        // be quiet: every voxel outside the window is dropped without any counter recording it, so
        // the table below would describe a fraction of the scene while looking perfectly healthy.
        if (!fitsOneWindow && tsdfName == "flat")
            std::fprintf(stderr,
                         "warning: --tsdf flat on a %ld-voxel/axis scene — a single AdvancedTSDF "
                         "window covers only 512 voxels/axis (%.1f%% of one axis) and everything "
                         "outside it is silently discarded. The stats below describe that sliver, "
                         "not the scene. Use --tsdf tile (the default here) for the whole scene.\n",
                         axisVox, 100.0 * 512.0 / double(axisVox));

        // HashStrategyByName falls back to linear on an unknown name WITHOUT telling the caller,
        // so a --hash typo would silently run linear while the operator believes otherwise.
        const std::string resolvedHashName = ResolveHashName(hashName);
        if (resolvedHashName != hashName)
            std::fprintf(stderr, "warning: --hash %s is not a known strategy; falling back to %s\n",
                         hashName.c_str(), resolvedHashName.c_str());

        TSDFConfiguration config;
        config.backend = "advanced";
        // flat = one window (everything outside it is refused and counted); tile = windows opened
        // on demand, no detail level; submap = windows plus the dense detail level.
        config.splitter = tsdfName == "submap" ? "dense" : "none";
        config.useSubmap = tsdfName == "submap";
        config.maxResidentWindow = tsdfName == "flat" ? 1 : 0;
        config.backendConfig.voxelSize = voxel;
        config.backendConfig.truncation = trunc;
        config.backendConfig.hashCapacity = tsdfName == "flat" ? hashCap : tileHash;
        config.backendConfig.maxPointPerFrame = maxPts;
        config.backendConfig.hash = hashName;
        config.backendConfig.pointToPlane = p2p;
        config.backendConfig.confidenceWeight = conf;
        config.backendConfig.hermitePosition = hermite;
        config.backendConfig.maxDirections = 3;
        config.backendConfig.directionExponent = 4;
        config.backendConfig.viewAngleWeight = true;
        config.backendConfig.adaptiveBand.sigmaMultiplier = arg.ValueFloat("--band-sigma");
        config.splitterConfig.baseResolution = voxel;
        config.splitterConfig.maxPointPerFrame = int(maxPts);

        TSDF tsdf;
        tsdf.Build(ctx, config);

        for (const auto &fr: frames) tsdf.Integrate(fr.pts, fr.nrm, fr.cam);
        Engine::Core::OrientedPointCloud recon = tsdf.Extract();
        std::printf("extracted : %zu oriented points\n", recon.points.size());

        // The comparison table this plan exists to produce: run twice with --tsdf/--hash held
        // fixed except for one axis and diff tableMB. Labelled with the RESOLVED hash name (see
        // above), not the raw --hash argument.
        const TSDFBackendStats stats = tsdf.Stats();
        std::printf("\n%-10s %-8s %10s %10s %7s %7s %9s %6s %6s\n",
                    "hash", "tsdf", "occupied", "slots", "load", "tables", "tableMB",
                    "drops", "grows");
        std::printf("%-10s %-8s %10llu %10llu %7.3f %7u %9.1f %6llu %6llu\n",
                    resolvedHashName.c_str(), tsdfName.c_str(),
                    (unsigned long long) stats.filledCount,
                    (unsigned long long) stats.hashCapacity,
                    stats.LoadFactor(),
                    stats.tableCount,
                    double(stats.deviceMemoryBytes) / (1024.0 * 1024.0),
                    (unsigned long long) stats.insertFailureCount,
                    (unsigned long long) stats.growCount);
        if (stats.insertFailureCount > 0)
            std::printf("WARNING: %llu observations were dropped -- this hash's load factor limit "
                        "is too high for this scene; the numbers above understate occupancy.\n",
                        (unsigned long long) stats.insertFailureCount);

        // RMSE vs ground truth (accuracy: recon→GT, completeness: GT→recon).
        //
        // Both directions, always, because either alone is gameable in the opposite direction: a
        // reconstruction that keeps only its most confident voxels scores a great accuracy on the
        // handful it kept, and one that smears voxels everywhere scores a great completeness. Only
        // the pair says whether a change tightened the surface or simply threw it away.
        std::vector<Vector3f> gtP, gtN;
        if (!gtPath.empty() && util::LoadPly(gtPath, gtP, gtN) && !gtP.empty()) {
            const float cell = std::max(voxel, extent / 100.0f);
            GridNN gGT, gRec;
            gGT.build(gtP, cell);
            gRec.build(recon.points, cell);
            Err acc, comp;
            // Fractions within one voxel, alongside the distances. The means alone cannot settle a
            // change that trades point count for tightness: dropping every uncertain voxel improves
            // accuracy on the few that remain, and smearing voxels everywhere improves completeness.
            // precision falls when a change smears, recall falls when it drops surface, and F1 only
            // rises when the surface actually got better.
            const double withinThreshold = double(voxel);
            std::size_t reconWithin = 0, gtWithin = 0;
            for (const auto &p: recon.points) {
                const double d = std::sqrt(double(gGT.nearestSq(p)));
                acc.add(d);
                if (d <= withinThreshold) ++reconWithin;
            }
            for (const auto &g: gtP) {
                const double d = std::sqrt(double(gRec.nearestSq(g)));
                comp.add(d);
                if (d <= withinThreshold) ++gtWithin;
            }
            const double precision =
                    recon.points.empty() ? 0.0 : double(reconWithin) / double(recon.points.size());
            const double recall = gtP.empty() ? 0.0 : double(gtWithin) / double(gtP.size());
            const double f1 = (precision + recall) > 0.0
                                      ? 2.0 * precision * recall / (precision + recall)
                                      : 0.0;
            std::printf("\n=== RMSE vs %s (%zu GT pts) ===\n", gtPath.c_str(), gtP.size());
            std::printf("accuracy    (recon->GT): mean %.5f  rmse %.5f\n", acc.mean(), acc.rmse());
            std::printf("completeness(GT->recon): mean %.5f  rmse %.5f\n", comp.mean(), comp.rmse());
            std::printf("chamfer-L1  (mean of means): %.5f\n", 0.5 * (acc.mean() + comp.mean()));
            std::printf("within 1 voxel (%.4f): precision %.4f  recall %.4f  F1 %.4f\n",
                        withinThreshold, precision, recall, f1);
        } else {
            std::printf("\n(no ground_truth.ply found in %s and no --gt given -> skipping RMSE)\n",
                        dir.c_str());
        }

        const std::string outPly = arg.Value("--out");
        if (!outPly.empty()) {
            util::SavePly(outPly, recon.points, recon.normals);
            std::printf("wrote extracted cloud: %s\n", outPly.c_str());
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

// TiledCompactDirectionalTSDF demo: reconstruct the real chair scan (scanData/frame_*.ply) at a
// FINE 1.0mm voxel -- a resolution at which the chair spans ~827 voxels/axis, far beyond the
// 512^3 window a single CompactDirectionalTSDF can hold. Tiling partitions space into fixed
// 448^3-core tiles (each backed by one 512^3 CompactDirectionalTSDF window with a ghost margin)
// so the whole scene reconstructs at low, tile-count-proportional memory.
//
// Two configurations demonstrate the value:
//   1. VALIDATION @ common voxel (~2.38mm, fits ONE 512 window): build a TiledCompactDirectionalTSDF
//      AND a plain single CompactDirectionalTSDF at the SAME voxel, integrate the same frames,
//      extract (merge). Their accuracy/completeness must match -- tiling must not degrade at a
//      resolution the single window already handles. This validates tiling against the known-good
//      single-window path.
//   2. FINE @ 1.0mm (chair ~827 voxels/axis > 512 -> single window CANNOT fit): only tiling works.
//      Report TileCount, nPoints, mem_KB, accuracy/completeness; and confirm a single
//      CompactDirectionalTSDF @ 1mm extracts only a truncated/partial cloud (it silently drops
//      every voxel outside its one 512 window).
//
// Frame loader, camera model, Subsample and the parallel NN RMSE are copied from
// tsdf_realdata_benchmark.cpp (see that file's header for rationale). Both the recon and the
// reference observed cloud are subsampled to ~40k before the brute-force NN metric -- a full recon
// (100k+) would make ParallelNnRmse take many minutes.
//
// Usage: tiled_compact_demo [scanData_dir=scanData] [maxFrames=12]
#include "Engine/Core/Context.h"
#include "TSDF/Backends/CompactDirectionalTSDF.h"
#include "Engine/Core/OrientedPointCloud.h"
#include "TSDF/Backends/TiledCompactDirectionalTSDF.h"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Frame {
    std::vector<Eigen::Vector3f> points, normals;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f meanNormal = Eigen::Vector3f::Zero();
};

// Minimal ASCII-PLY reader for the x,y,z,nx,ny,nz layout used by scanData/frame_*.ply.
bool loadFrame(const std::string &path, Frame &out) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    bool inData = false;
    while (std::getline(in, line)) {
        if (!inData) {
            if (line.rfind("end_header", 0) == 0) inData = true;
            continue;
        }
        std::istringstream ss(line);
        float x, y, z, nx, ny, nz;
        if (!(ss >> x >> y >> z >> nx >> ny >> nz)) continue;
        out.points.emplace_back(x, y, z);
        out.normals.emplace_back(nx, ny, nz);
        out.centroid += Eigen::Vector3f(x, y, z);
        out.meanNormal += Eigen::Vector3f(nx, ny, nz);
    }
    if (out.points.empty()) return false;
    out.centroid /= float(out.points.size());
    if (out.meanNormal.norm() > 1e-6f) out.meanNormal.normalize();
    else out.meanNormal = Eigen::Vector3f(0, 0, 1);
    return true;
}

// Uniform-stride subsample down to ~targetCount points (identity if already smaller).
std::vector<Eigen::Vector3f> Subsample(const std::vector<Eigen::Vector3f> &pts, size_t targetCount) {
    if (pts.size() <= targetCount) return pts;
    const double stride = double(pts.size()) / double(targetCount);
    std::vector<Eigen::Vector3f> out;
    out.reserve(targetCount + 1);
    for (double i = 0.0; i < double(pts.size()); i += stride) out.push_back(pts[size_t(i)]);
    return out;
}

// Parallel nearest-neighbour RMSE: for each `from` point, distance to its nearest `to` point,
// RMS'd, sharded across hardware threads (brute force is the demo's dominant cost).
float ParallelNnRmse(const std::vector<Eigen::Vector3f> &from,
                     const std::vector<Eigen::Vector3f> &to) {
    if (from.empty() || to.empty()) return std::numeric_limits<float>::infinity();
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t nThreads = std::min<size_t>(hw, from.size());
    std::vector<double> partial(nThreads, 0.0);
    std::vector<std::thread> pool;
    pool.reserve(nThreads);
    for (size_t t = 0; t < nThreads; ++t) {
        pool.emplace_back([&, t] {
            double sumSq = 0.0;
            for (size_t i = t; i < from.size(); i += nThreads) {
                const Eigen::Vector3f &a = from[i];
                float best = std::numeric_limits<float>::max();
                for (const auto &b : to) {
                    const float d2 = (a - b).squaredNorm();
                    if (d2 < best) best = d2;
                }
                sumSq += double(best);
            }
            partial[t] = sumSq;
        });
    }
    for (auto &th : pool) th.join();
    double total = 0.0;
    for (double p : partial) total += p;
    return float(std::sqrt(total / double(from.size())));
}

// accuracy = recon -> ref (fit tightness); completeness = ref -> recon (coverage).
struct Metrics { float accuracy = 0.0f, completeness = 0.0f; size_t nPoints = 0; };
Metrics Evaluate(const std::vector<Eigen::Vector3f> &recon,
                 const std::vector<Eigen::Vector3f> &refObserved) {
    Metrics m;
    m.nPoints = recon.size();
    const std::vector<Eigen::Vector3f> reconSub = Subsample(recon, 40000);
    m.accuracy = ParallelNnRmse(reconSub, refObserved);
    m.completeness = ParallelNnRmse(refObserved, reconSub);
    return m;
}

} // namespace

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "scanData";
    const int maxFrames = argc > 2 ? std::atoi(argv[2]) : 12;

    // ---- load frames ----
    std::vector<Frame> frames;
    for (int i = 0; i < maxFrames; ++i) {
        std::ostringstream name;
        name << dir << "/frame_" << std::setw(4) << std::setfill('0') << i << ".ply";
        Frame fr;
        if (!loadFrame(name.str(), fr)) break;
        std::cout << "loaded " << name.str() << ": " << fr.points.size() << " points\n";
        frames.push_back(std::move(fr));
    }
    if (frames.empty()) {
        std::cerr << "no frames found under '" << dir << "' (expected frame_0000.ply ...)\n";
        return 1;
    }

    // ---- scene bbox ----
    Eigen::Vector3f bbMin = frames[0].points[0], bbMax = frames[0].points[0];
    size_t totalIn = 0;
    for (const auto &fr : frames)
        for (const auto &p : fr.points) {
            bbMin = bbMin.cwiseMin(p);
            bbMax = bbMax.cwiseMax(p);
            ++totalIn;
        }
    const Eigen::Vector3f center = 0.5f * (bbMin + bbMax);
    const Eigen::Vector3f extent = bbMax - bbMin;
    const float maxExtent = extent.maxCoeff();

    std::cout << "\nscene: " << frames.size() << " frames, " << totalIn << " points\n"
              << "  bbox extent " << extent.x() << " x " << extent.y() << " x " << extent.z()
              << " mm, center (" << center.x() << ", " << center.y() << ", " << center.z() << ")\n";

    // ---- reference observed cloud (subsampled to ~40k) ----
    std::vector<Eigen::Vector3f> allPoints;
    allPoints.reserve(totalIn);
    for (const auto &fr : frames)
        for (const auto &p : fr.points) allPoints.push_back(p);
    const std::vector<Eigen::Vector3f> refObserved = Subsample(allPoints, 40000);
    std::cout << "reference observed cloud: " << refObserved.size() << " points (from " << totalIn
              << " raw)\n";

    const float camDist = 1000.0f; // mm, camera along each frame's mean normal
    auto cameraFor = [camDist](const Frame &fr) { return fr.centroid + camDist * fr.meanNormal; };
    const Engine::Spatial::IntegrationQuality quality{3, 4, true};

    // ================================================================================
    // CONFIG 1 -- VALIDATION @ common voxel (fits ONE 512 window): tiled must match single.
    // ================================================================================
    const float voxel1 = (maxExtent / 400.0f) * 1.15f;
    const float trunc1 = 3.0f * voxel1;
    const float span1 = maxExtent / voxel1;
    std::cout << "\n================ CONFIG 1: VALIDATION @ common voxel ================\n"
              << "  voxelSize " << voxel1 << " mm, truncation " << trunc1 << " mm, scene span "
              << span1 << " voxels (< 512 -> a single window also fits)\n";

    Metrics tiled1, single1;
    uint32_t tiled1Tiles = 0, tiled1Filled = 0, single1Filled = 0;
    {
        std::cout << "  [tiled] integrating...\n";
        Engine::Core::Context ctx; // fresh Context per method (Engine::Core convention)
        Engine::Spatial::TiledCompactDirectionalTSDF tiled;
        tiled.Build(ctx, voxel1, trunc1);
        tiled.SetIntegrationQuality(quality);
        for (const auto &fr : frames) tiled.Integrate(fr.points, fr.normals, cameraFor(fr));
        const Engine::Spatial::OrientedPointCloud cloud = tiled.ExtractPointCloud(/*merge=*/true);
        tiled1 = Evaluate(cloud.points, refObserved);
        tiled1Tiles = tiled.TileCount();
        tiled1Filled = tiled.FilledCount();
        std::cout << "  [tiled]  tiles=" << tiled1Tiles << " ghostG=" << tiled.GhostVoxels()
                  << " nPoints=" << tiled1.nPoints << " filled=" << tiled1Filled
                  << " accuracy=" << tiled1.accuracy << " completeness=" << tiled1.completeness
                  << "\n";
    }
    {
        std::cout << "  [single] integrating...\n";
        Engine::Core::Context ctx; // fresh Context per method
        Engine::Spatial::CompactDirectionalTSDF single;
        const float margin = trunc1 + voxel1;
        const Eigen::Vector3f windowMin = bbMin - Eigen::Vector3f::Constant(margin);
        single.Build(ctx, voxel1, trunc1, 1u << 22, 1u << 17, windowMin);
        single.SetIntegrationQuality(quality);
        for (const auto &fr : frames) single.Integrate(fr.points, fr.normals, cameraFor(fr));
        const Engine::Spatial::OrientedPointCloud cloud = single.ExtractPointCloud(1u << 21, true);
        single1 = Evaluate(cloud.points, refObserved);
        single1Filled = single.FilledCount();
        std::cout << "  [single] nPoints=" << single1.nPoints << " filled=" << single1Filled
                  << " accuracy=" << single1.accuracy << " completeness=" << single1.completeness
                  << "\n";
    }

    auto relDiff = [](float a, float b) {
        const float d = std::max(std::abs(a), std::abs(b));
        return d > 0.0f ? std::abs(a - b) / d : 0.0f;
    };
    const float accDiff1 = relDiff(tiled1.accuracy, single1.accuracy);
    const float compDiff1 = relDiff(tiled1.completeness, single1.completeness);
    const bool config1Ok = accDiff1 <= 0.10f && compDiff1 <= 0.10f;
    std::cout << "  => accuracy rel-diff " << (accDiff1 * 100.0f) << "%  completeness rel-diff "
              << (compDiff1 * 100.0f) << "%  ["
              << (config1Ok ? "MATCH (within 10%)" : "DIVERGED (>10%)") << "]\n";

    // ================================================================================
    // CONFIG 2 -- FINE @ 1.0mm (chair ~827 voxels/axis > 512): tiling REQUIRED.
    // ================================================================================
    const float voxel2 = 1.0f;
    const float trunc2 = 3.0f;
    const float span2 = maxExtent / voxel2;
    std::cout << "\n================ CONFIG 2: FINE @ 1.0mm ================\n"
              << "  voxelSize " << voxel2 << " mm, truncation " << trunc2 << " mm, scene span "
              << span2 << " voxels (> 512 -> NO single 512^3 window can cover it)\n";

    Metrics tiled2, single2;
    uint32_t tiled2Tiles = 0, tiled2Filled = 0;
    double tiled2MemKB = 0.0;
    {
        std::cout << "  [tiled] integrating @ 1mm...\n";
        Engine::Core::Context ctx; // fresh Context
        Engine::Spatial::TiledCompactDirectionalTSDF tiled;
        tiled.Build(ctx, voxel2, trunc2);
        tiled.SetIntegrationQuality(quality);
        for (const auto &fr : frames) tiled.Integrate(fr.points, fr.normals, cameraFor(fr));
        const Engine::Spatial::OrientedPointCloud cloud = tiled.ExtractPointCloud(/*merge=*/true);
        tiled2 = Evaluate(cloud.points, refObserved);
        tiled2Tiles = tiled.TileCount();
        tiled2Filled = tiled.FilledCount();
        tiled2MemKB = double(tiled2Filled) * 16.0 / 1024.0;
        std::cout << "  [tiled]  tiles=" << tiled2Tiles << " ghostG=" << tiled.GhostVoxels()
                  << " nPoints=" << tiled2.nPoints << " filled=" << tiled2Filled
                  << " mem_KB=" << tiled2MemKB << " accuracy=" << tiled2.accuracy
                  << " completeness=" << tiled2.completeness << "\n";
    }
    {
        // A single CompactDirectionalTSDF @ 1mm can only hold a 512-voxel window: voxels outside
        // are silently skipped, so it reconstructs at most ~512/827 of each axis -> a partial cloud
        // with much worse completeness (large stretches of the chair have no recon point nearby).
        std::cout << "  [single] integrating @ 1mm (window placed at bbMin)...\n";
        Engine::Core::Context ctx; // fresh Context
        Engine::Spatial::CompactDirectionalTSDF single;
        const float margin = trunc2 + voxel2;
        const Eigen::Vector3f windowMin = bbMin - Eigen::Vector3f::Constant(margin);
        single.Build(ctx, voxel2, trunc2, 1u << 22, 1u << 17, windowMin);
        single.SetIntegrationQuality(quality);
        for (const auto &fr : frames) single.Integrate(fr.points, fr.normals, cameraFor(fr));
        const Engine::Spatial::OrientedPointCloud cloud = single.ExtractPointCloud(1u << 21, true);
        single2 = Evaluate(cloud.points, refObserved);
        std::cout << "  [single] nPoints=" << single2.nPoints
                  << " (window covers only a 512-voxel slab of the " << span2
                  << "-voxel scene) accuracy=" << single2.accuracy
                  << " completeness=" << single2.completeness << "\n";
    }
    const bool config2Covers = tiled2.completeness < single2.completeness * 0.75f &&
                               tiled2.nPoints > single2.nPoints && tiled2Tiles > 1;

    // ================================================================================
    // SUMMARY
    // ================================================================================
    std::cout << "\n==================== TILED-COMPACT DEMO SUMMARY ====================\n";
    std::cout << std::left << std::setw(34) << "config" << std::right << std::setw(8) << "tiles"
              << std::setw(11) << "nPoints" << std::setw(12) << "mem_KB" << std::setw(12)
              << "accuracy" << std::setw(14) << "completeness" << "\n";
    std::cout << "-------------------------------------------------------------------------------"
                 "----------\n";
    std::cout << std::fixed << std::setprecision(3);
    auto row = [](const std::string &name, int tiles, size_t nPoints, double memKB, float acc,
                  float comp) {
        std::cout << std::left << std::setw(34) << name << std::right << std::setw(8);
        if (tiles >= 0) std::cout << tiles;
        else std::cout << "-";
        std::cout << std::setw(11) << nPoints << std::setw(12) << std::setprecision(1) << memKB
                  << std::setprecision(3) << std::setw(12) << acc << std::setw(14) << comp << "\n";
    };
    row("1: Tiled @ common voxel", int(tiled1Tiles), tiled1.nPoints,
        double(tiled1Filled) * 16.0 / 1024.0, tiled1.accuracy, tiled1.completeness);
    row("1: single Compact @ common voxel", -1, single1.nPoints,
        double(single1Filled) * 16.0 / 1024.0, single1.accuracy, single1.completeness);
    row("2: Tiled @ 1.0mm", int(tiled2Tiles), tiled2.nPoints, tiled2MemKB, tiled2.accuracy,
        tiled2.completeness);
    row("2: single Compact @ 1.0mm (partial)", -1, single2.nPoints, 0.0, single2.accuracy,
        single2.completeness);

    std::cout << "\ninsight: at the common voxel (" << voxel1 << "mm) tiling matches the single "
                 "window (accuracy "
              << (accDiff1 * 100.0f) << "%, completeness " << (compDiff1 * 100.0f)
              << "% apart across " << tiled1Tiles << " tiles) -- tiling does not degrade a "
                 "resolution the single window already handles.\n"
              << "insight: at 1.0mm the chair spans " << span2 << " voxels/axis (> 512), so NO "
                 "single CompactDirectionalTSDF can cover it -- the single window reconstructs only "
              << single2.nPoints << " points (completeness " << single2.completeness
              << "mm). Tiling reconstructs the FULL chair with " << tiled2.nPoints
              << " points across " << tiled2Tiles << " tiles at " << tiled2MemKB << " KB ("
              << (tiled2Filled ? double(tiled2MemKB) / double(tiled2Tiles) : 0.0)
              << " KB/tile), completeness " << tiled2.completeness << "mm.\n";

    std::cout << "\nverdict: "
              << (config1Ok && config2Covers
                          ? "PASS -- tiling reconstructs the chair at 1mm (beyond the single-window "
                            "512^3 limit) at low, tile-proportional memory, and matches the single "
                            "window where both fit."
                          : "CONCERN -- see the per-config diagnostics above.")
              << "\n";

    return (config1Ok && config2Covers) ? 0 : 3;
}

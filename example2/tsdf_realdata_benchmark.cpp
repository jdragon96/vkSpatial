// Real-data (chair scan) end-to-end validation: SimpleTSDF vs DirectionalTSDF vs
// CompactDirectionalTSDF, integrating the SAME registered multi-frame scan
// (scanData/frame_*.ply -- x,y,z,nx,ny,nz, common mm frame) at the SAME voxelSize/truncation,
// so the comparison isolates the algorithmic/storage difference rather than any input or
// resolution discrepancy. Goes beyond the synthetic cube/cylinder fixtures used by
// tsdf_benchmark.cpp -- there is no analytic ground truth here, so accuracy/completeness are
// measured as nearest-neighbour RMSE against a subsampled OBSERVED point cloud (the union of
// all raw scan frames), brute-force (Engine::Eval::NearestNeighbourRMSE), same as
// directional_tsdf_eval.cpp uses on synthetic data.
//
// Usage: tsdf_realdata_benchmark [scanData_dir=scanData] [maxFrames=24]
//
// Frame loading, the bbox -> center/voxelSize fit, and the per-frame camera model are ported
// verbatim from directional_tsdf_chair_benchmark.cpp (see that file for the rationale: the PLY
// carries no pose, so the camera is placed along each frame's mean normal at a fixed standoff).
//
// Each of the three TSDF variants gets its OWN fresh Engine::Core::Context (not one shared
// context across methods) -- Engine::Core is known to degrade under a shared-Context,
// repeated-build regime at N >= ~1000 points (docs/KNOWN_ISSUES; see also
// test/test_spatialIndex.cpp's "fresh Context per backend" fix), and every method here
// integrates real frames with ~10^4-10^5 points, so a shared context is not a safe pattern.
//
// mem_KB accounting mirrors tsdf_benchmark.cpp:
//   Simple / Compact : FilledCount() * sizeof(entry=16B) / 1024      (occupied entries only)
//   Directional       : resident (block,direction) groups * 4096B / 1024 (each group = 512
//                       voxels x 8B GpuTsdfVoxel -- the actual GPU pool residency cost, the
//                       block-granularity figure to beat) AND a per-voxel-projected figure
//                       (occupied_dir_voxels * 8B / 1024, only weight>0 voxels). Both come from
//                       enumerating the ENTIRE resident local window (50^3 groups x 6 directions)
//                       via DebugDownloadIndexGrid + DebugDownloadGroupVoxels -- for the Unified
//                       backend these are persistently-mapped UMA reads (no GPU submit/wait per
//                       call), so the full-window enumeration is cheap. (tsdf_benchmark.cpp reads
//                       block mem from HostStore().Size(); that is 0 for the Unified backend used
//                       here -- see the note at the Directional block -- so the resident-group
//                       count from the same index-grid sweep is used instead. It equals the last
//                       frame's residentCount.)
#include "Engine/Core/Context.h"
#include "Engine/Eval/RmseMetrics.h"
#include "Engine/Spatial/CompactDirectionalTSDF.h"
#include "Engine/Spatial/DirectionalTSDF.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"
#include "Engine/Spatial/SimpleTSDF.h"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
// Ported verbatim from directional_tsdf_chair_benchmark.cpp.
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

// Uniform-stride subsample down to ~targetCount points (identity if already smaller). Used to
// keep the brute-force NearestNeighbourRMSE tractable over the ~1.8M raw observed points.
std::vector<Eigen::Vector3f> Subsample(const std::vector<Eigen::Vector3f> &pts,
                                       size_t targetCount) {
    if (pts.size() <= targetCount) return pts;
    const double stride = double(pts.size()) / double(targetCount);
    std::vector<Eigen::Vector3f> out;
    out.reserve(targetCount + 1);
    for (double i = 0.0; i < double(pts.size()); i += stride) out.push_back(pts[size_t(i)]);
    return out;
}

// Parallel nearest-neighbour RMSE: returns EXACTLY the same quantity as
// Engine::Eval::NearestNeighbourRMSE(from, to) -- for each `from` point, the distance to its
// nearest `to` point, RMS'd -- but shards the O(|from|*|to|) brute force across hardware
// threads. That brute force (tens of thousands of points on each side, 6 calls per run) is the
// benchmark's dominant cost; the single-threaded Engine::Eval version makes a 24-frame run take
// tens of minutes. Verified equal to the header version on the sanity run before switching.
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

// Small ASCII-PLY export helper (positions only) for visual inspection.
void ExportPly(const std::string &path, const std::vector<Eigen::Vector3f> &pts) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "  (could not open " << path << " for writing, skipping export)\n";
        return;
    }
    out << "ply\nformat ascii 1.0\nelement vertex " << pts.size()
        << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto &p : pts) out << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
}

struct MethodResult {
    std::string name;
    double buildMs = 0.0;
    size_t nPoints = 0;
    double memKB = 0.0;
    float accuracyRmse = 0.0f;     // recon -> refObserved (fit tightness)
    float completenessRmse = 0.0f; // refObserved -> recon (coverage)
};

void PrintTable(const std::vector<MethodResult> &rows) {
    std::cout << "\n==================== REAL-DATA BENCHMARK SUMMARY ====================\n";
    std::cout << std::left << std::setw(24) << "method" << std::right << std::setw(10)
              << "nPoints" << std::setw(12) << "mem_KB" << std::setw(16) << "accuracy_rmse"
              << std::setw(20) << "completeness_rmse" << std::setw(12) << "build_ms" << "\n";
    std::cout << "-----------------------------------------------------------------------------"
                 "-------------------\n";
    for (const auto &r : rows) {
        std::cout << std::left << std::setw(24) << r.name << std::right << std::fixed
                  << std::setw(10) << r.nPoints << std::setprecision(2) << std::setw(12)
                  << r.memKB << std::setprecision(3) << std::setw(16) << r.accuracyRmse
                  << std::setw(20) << r.completenessRmse << std::setprecision(1) << std::setw(12)
                  << r.buildMs << "\n";
    }
}

} // namespace

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "scanData";
    const int maxFrames = argc > 2 ? std::atoi(argv[2]) : 24;

    // ---- load frames (verbatim reader/camera model from directional_tsdf_chair_benchmark) ----
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

    // ---- scene bbox -> center/extent/voxelSize/truncation, COMMON to all three methods ----
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
    // Window side = 50 groups * 8 voxels * voxelSize = 400 * voxelSize (DirectionalTSDF's local
    // window); leave ~15% margin. By construction maxExtent/voxelSize == 400/1.15 =~ 347.8
    // regardless of the scene's absolute size, comfortably under both DirectionalTSDF's 400-voxel
    // window and CompactDirectionalTSDF's 512-voxel window.
    const float voxelSize = (maxExtent / 400.0f) * 1.15f;
    const float truncation = 3.0f * voxelSize;
    const float spanVoxels = maxExtent / voxelSize;

    std::cout << "\nscene: " << frames.size() << " frames, " << totalIn << " points\n"
              << "  bbox extent " << extent.x() << " x " << extent.y() << " x " << extent.z()
              << " mm, center (" << center.x() << ", " << center.y() << ", " << center.z()
              << ")\n"
              << "  voxelSize " << voxelSize << " mm, truncation " << truncation << " mm\n"
              << "  scene span in voxels (maxExtent/voxelSize) = " << spanVoxels
              << " (must be < 512 for CompactDirectionalTSDF's 512^3 window)\n\n";

    if (spanVoxels >= 512.0f) {
        std::cerr << "STOP: scene span " << spanVoxels
                  << " voxels exceeds CompactDirectionalTSDF's 512^3 window at the common voxel "
                     "size -- aborting rather than silently dropping a method.\n";
        return 2;
    }

    // ---- reference observed cloud: concat all frames, uniform-stride subsample to ~40k ----
    std::vector<Eigen::Vector3f> allPoints;
    allPoints.reserve(totalIn);
    for (const auto &fr : frames)
        for (const auto &p : fr.points) allPoints.push_back(p);
    const std::vector<Eigen::Vector3f> refObserved = Subsample(allPoints, 40000);
    std::cout << "reference observed cloud: " << refObserved.size() << " points (subsampled from "
              << totalIn << " raw scan points)\n\n";

    const float camDist = 1000.0f; // mm, camera placed along each frame's mean normal
    auto cameraFor = [camDist](const Frame &fr) { return fr.centroid + camDist * fr.meanNormal; };

    std::vector<MethodResult> results;
    double dirMemBlockKB = 0.0, dirMemPerVoxelKB = 0.0;
    size_t dirOccupiedGroups = 0, dirOccupiedDirVoxels = 0;

    // ============================== SimpleTSDF ==============================
    // Honest normal-free single-field baseline: no normals, w=1 per observation.
    {
        std::cout << "=== SimpleTSDF (normal-free baseline) ===\n";
        Engine::Core::Context ctx; // fresh Context per method -- see file header note
        Engine::Spatial::SimpleTSDF tsdf;
        tsdf.Build(ctx, voxelSize, truncation, 1u << 22, 1u << 17);

        const auto t0 = std::chrono::steady_clock::now();
        for (size_t f = 0; f < frames.size(); ++f) {
            const Frame &fr = frames[f];
            tsdf.Integrate(fr.points, cameraFor(fr));
            std::cout << "  [simple] frame " << f << ": in=" << fr.points.size() << "\n";
        }
        const Engine::Spatial::OrientedPointCloud cloud = tsdf.ExtractPointCloud(2000000);
        const auto t1 = std::chrono::steady_clock::now();

        MethodResult r;
        r.name = "SimpleTSDF";
        r.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        r.nPoints = cloud.points.size();
        r.memKB = double(tsdf.FilledCount()) * 16.0 / 1024.0;
        const std::vector<Eigen::Vector3f> reconSub = Subsample(cloud.points, 40000);
        r.accuracyRmse = ParallelNnRmse(reconSub, refObserved);
        r.completenessRmse = ParallelNnRmse(refObserved, reconSub);
        std::cout << "  nPoints=" << r.nPoints << " mem_KB=" << r.memKB
                  << " build_ms=" << r.buildMs << " accuracy_rmse=" << r.accuracyRmse
                  << " completeness_rmse=" << r.completenessRmse << "\n\n";
        ExportPly("recon_simple.ply", cloud.points);
        results.push_back(r);
    }

    // ============================== DirectionalTSDF ==============================
    {
        std::cout << "=== DirectionalTSDF ===\n";
        Engine::Core::Context ctx; // fresh Context per method
        Engine::Spatial::DirectionalTSDF tsdf;
        const uint32_t poolCapacity = 1u << 18, maxPoints = 1u << 17, maxCandidates = 1u << 21;
        // Unified (UMA) backend: the whole scan lives in one fixed window, so the first frame's
        // missing set covers most of the model at once -- exceeds the Streaming staging cap
        // (chunked upload is an unimplemented Phase-5 TODO). Same choice as
        // directional_tsdf_chair_benchmark.cpp's runScan.
        tsdf.Build(ctx, voxelSize, truncation, poolCapacity, maxPoints, maxCandidates,
                   Engine::Spatial::ResidencyMode::Unified);
        tsdf.SetIntegrationQuality({3, 4, true});

        uint32_t lastResident = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t f = 0; f < frames.size(); ++f) {
            const Frame &fr = frames[f];
            tsdf.Integrate(fr.points, fr.normals, cameraFor(fr), center);
            const auto st = tsdf.LastFrameStats();
            lastResident = st.residentCount;
            std::cout << "  [directional] frame " << f << ": in=" << fr.points.size()
                      << " resident=" << st.residentCount << " cloud=" << tsdf.PointCloud().size()
                      << "\n";
        }
        const auto t1 = std::chrono::steady_clock::now();

        MethodResult r;
        r.name = "DirectionalTSDF";
        r.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const std::vector<Engine::Spatial::ExtractedPoint> &cloud = tsdf.PointCloud();
        r.nPoints = cloud.size();
        std::vector<Eigen::Vector3f> reconPts;
        reconPts.reserve(cloud.size());
        for (const auto &p : cloud) reconPts.push_back(p.position);

        const std::vector<Eigen::Vector3f> reconSub = Subsample(reconPts, 40000);
        r.accuracyRmse = ParallelNnRmse(reconSub, refObserved);
        r.completenessRmse = ParallelNnRmse(refObserved, reconSub);

        // Per-voxel-projected storage: enumerate the ENTIRE resident local window (50^3 groups
        // x 6 directions) via the index grid, download each occupied group's live voxel weights
        // (cheap: Unified backend's debug reads are persistently-mapped UMA memory, no GPU
        // submit/wait), and count weight>0 voxels. Mirrors tsdf_benchmark.cpp's
        // AnalyzeDirectionalStorage, adapted to loop the actual local window (LocalBase() +
        // [0,kLocalGroupGrid)) instead of a fixture-centered bounding box -- correct regardless
        // of the scene's absolute world position.
        {
            using namespace Engine::Spatial;
            const std::vector<uint32_t> indexGrid = tsdf.DebugDownloadIndexGrid();
            const Eigen::Vector3i localBase = tsdf.LocalBase();
            for (uint32_t lz = 0; lz < kLocalGroupGrid; ++lz)
                for (uint32_t ly = 0; ly < kLocalGroupGrid; ++ly)
                    for (uint32_t lx = 0; lx < kLocalGroupGrid; ++lx)
                        for (uint32_t d = 0; d < kNumDirections; ++d) {
                            const uint32_t idx = IndexGridOffset(lx, ly, lz, d);
                            if (indexGrid[idx] == kInvalidPoolIndex) continue;
                            ++dirOccupiedGroups;
                            const DirectionalGroupKey key{
                                    localBase.x() + int32_t(lx), localBase.y() + int32_t(ly),
                                    localBase.z() + int32_t(lz), uint8_t(d)};
                            const DirectionalHostStore::Group group =
                                    tsdf.DebugDownloadGroupVoxels(key);
                            for (uint32_t v = 0; v < kVoxelsPerGroup; ++v)
                                if (group[v].weight > 0.0f) ++dirOccupiedDirVoxels;
                        }
            dirMemPerVoxelKB = double(dirOccupiedDirVoxels) * 8.0 / 1024.0;
        }

        // Block memory = resident (block,direction) groups x 512 voxels x 8B = groups x 4096B.
        // The enumerated occupied_groups equals the last frame's residentCount (cross-checked
        // below). NOTE: HostStore().Size() -- the figure tsdf_benchmark.cpp uses for the
        // Streaming backend -- is 0 here because the Unified (UMA) backend keeps only a view
        // (m_storeView) and never runs the dirty write-back that Puts groups into the host store
        // (the window is fixed on `center` every frame, so nothing is ever evicted). The real
        // residency cost is the GPU pool, so block memory is counted from the resident group
        // count, not HostStore().
        const size_t hostStoreSize = tsdf.HostStore().Size();
        dirMemBlockKB = double(dirOccupiedGroups) * 4096.0 / 1024.0;
        r.memKB = dirMemBlockKB;

        std::cout << "  nPoints=" << r.nPoints << " mem_block_KB=" << dirMemBlockKB
                  << " mem_pervoxel_projected_KB=" << dirMemPerVoxelKB
                  << " occupied_groups=" << dirOccupiedGroups
                  << " (residentCount=" << lastResident << ", HostStore().Size()=" << hostStoreSize
                  << " [0 for Unified backend, see note])"
                  << " occupied_dir_voxels=" << dirOccupiedDirVoxels
                  << " build_ms=" << r.buildMs << " accuracy_rmse=" << r.accuracyRmse
                  << " completeness_rmse=" << r.completenessRmse << "\n\n";
        tsdf.ExportPointCloud("recon_directional.ply");
        results.push_back(r);
    }

    // ============================== CompactDirectionalTSDF ==============================
    {
        std::cout << "=== CompactDirectionalTSDF ===\n";
        Engine::Core::Context ctx; // fresh Context per method
        Engine::Spatial::CompactDirectionalTSDF tsdf;
        // margin: slack below bbMin so the negative side of the truncation band (and the
        // windowMinCorner's floor-to-voxel rounding) never falls outside the 512^3 window.
        // truncation + 1 voxel is comfortably enough given the ~164-voxel slack between
        // spanVoxels (~348) and the window size (512).
        const float margin = truncation + voxelSize;
        const Eigen::Vector3f windowMinCorner = bbMin - Eigen::Vector3f::Constant(margin);
        tsdf.Build(ctx, voxelSize, truncation, 1u << 22, 1u << 17, windowMinCorner);
        tsdf.SetIntegrationQuality({3, 4, true});

        const auto t0 = std::chrono::steady_clock::now();
        for (size_t f = 0; f < frames.size(); ++f) {
            const Frame &fr = frames[f];
            tsdf.Integrate(fr.points, fr.normals, cameraFor(fr));
            std::cout << "  [compact] frame " << f << ": in=" << fr.points.size() << "\n";
        }
        const Engine::Spatial::OrientedPointCloud cloud = tsdf.ExtractPointCloud(1u << 21);
        const auto t1 = std::chrono::steady_clock::now();

        MethodResult r;
        r.name = "CompactDirectionalTSDF";
        r.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        r.nPoints = cloud.points.size();
        r.memKB = double(tsdf.FilledCount()) * 16.0 / 1024.0;
        if (cloud.points.empty()) {
            std::cerr << "  WARNING: CompactDirectionalTSDF extracted an EMPTY point cloud -- "
                         "the scene may not fit its window at the common voxel size (see the "
                         "spanVoxels check above).\n";
        }
        const std::vector<Eigen::Vector3f> reconSub = Subsample(cloud.points, 40000);
        r.accuracyRmse = ParallelNnRmse(reconSub, refObserved);
        r.completenessRmse = ParallelNnRmse(refObserved, reconSub);
        std::cout << "  nPoints=" << r.nPoints << " mem_KB=" << r.memKB
                  << " build_ms=" << r.buildMs << " accuracy_rmse=" << r.accuracyRmse
                  << " completeness_rmse=" << r.completenessRmse << "\n\n";
        ExportPly("recon_compact.ply", cloud.points);
        results.push_back(r);

        // ---- CompactDirectionalTSDF (merged) ----
        // SAME build/integrate (tsdf's hash table above is untouched by extraction) -- just a
        // second extract dispatch with merge=true, clustering the raw candidates on the CPU
        // (ported from DirectionalTSDF::mergeCandidates) to dedup redundant candidates into
        // averaged points while preserving sharp corners. Memory is unchanged (merge is
        // extraction-only; FilledCount() reflects the SAME storage), so memKB is copied from
        // the raw row rather than re-measured.
        std::cout << "=== CompactDirectionalTSDF (merged) ===\n";
        const auto tm0 = std::chrono::steady_clock::now();
        const Engine::Spatial::OrientedPointCloud mergedCloud =
                tsdf.ExtractPointCloud(1u << 21, /*merge=*/true);
        const auto tm1 = std::chrono::steady_clock::now();

        MethodResult rm;
        rm.name = "CompactDirectionalTSDF(merged)";
        rm.buildMs = r.buildMs + std::chrono::duration<double, std::milli>(tm1 - tm0).count();
        rm.nPoints = mergedCloud.points.size();
        rm.memKB = r.memKB;
        // CRITICAL: subsample the merged recon to ~40k before the brute-force NN, same as the
        // raw row above -- a full (un-subsampled) merged recon would make ParallelNnRmse take
        // many minutes (O(recon*ref) brute force).
        const std::vector<Eigen::Vector3f> mergedSub = Subsample(mergedCloud.points, 40000);
        rm.accuracyRmse = ParallelNnRmse(mergedSub, refObserved);
        rm.completenessRmse = ParallelNnRmse(refObserved, mergedSub);
        std::cout << "  nPoints=" << rm.nPoints << " mem_KB=" << rm.memKB
                  << " build_ms=" << rm.buildMs << " accuracy_rmse=" << rm.accuracyRmse
                  << " completeness_rmse=" << rm.completenessRmse << "\n\n";
        ExportPly("recon_compact_merged.ply", mergedCloud.points);
        results.push_back(rm);
    }

    // ============================== summary ==============================
    PrintTable(results);

    std::cout << "\n[directional-storage] mem_block=" << dirMemBlockKB
              << " KB  occupied_dir_voxels=" << dirOccupiedDirVoxels
              << "  mem_pervoxel_projected=" << dirMemPerVoxelKB << " KB\n";

    if (results.size() == 4) {
        const MethodResult &simple = results[0];
        const MethodResult &dirc = results[1];
        const MethodResult &compact = results[2];
        const MethodResult &compactMerged = results[3];
        const double dirVsCompactX =
                compact.memKB > 0.0 ? dirc.memKB / compact.memKB : 0.0;
        const double compactVsSimpleX =
                simple.memKB > 0.0 ? compact.memKB / simple.memKB : 0.0;
        const double mergedPctOfRaw =
                compact.nPoints > 0
                        ? 100.0 * double(compactMerged.nPoints) / double(compact.nPoints)
                        : 0.0;
        std::cout << "\ninsight: Compact accuracy_rmse=" << compact.accuracyRmse
                  << " (Directional=" << dirc.accuracyRmse << ", Simple=" << simple.accuracyRmse
                  << ") | Compact completeness_rmse=" << compact.completenessRmse
                  << " (Directional=" << dirc.completenessRmse
                  << ") | Compact mem_KB=" << compact.memKB
                  << " vs Directional mem_block_KB=" << dirc.memKB << " (" << dirVsCompactX
                  << "x less) vs Simple mem_KB=" << simple.memKB << " (" << compactVsSimpleX
                  << "x of Simple)\n"
                  << "insight (merge/dedup): Compact(merged) nPoints=" << compactMerged.nPoints
                  << " vs raw=" << compact.nPoints << " (" << mergedPctOfRaw
                  << "% of raw, Directional=" << dirc.nPoints
                  << ") | accuracy_rmse merged=" << compactMerged.accuracyRmse
                  << " (raw=" << compact.accuracyRmse
                  << ") | completeness_rmse merged=" << compactMerged.completenessRmse
                  << " (raw=" << compact.completenessRmse << ", Directional=" << dirc.completenessRmse
                  << ")\n"
                  << "wrote recon_simple.ply, recon_directional.ply, recon_compact.ply, "
                     "recon_compact_merged.ply\n";
    }

    return 0;
}

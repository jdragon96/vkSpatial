// Headless SimpleTSDF-vs-DirectionalTSDF benchmark: accuracy + memory + speed on analytic
// ground-truth fixtures (example2/shape_fixtures.h).
//
// Both methods integrate the SAME per-shape multi-view samples (fixtures::SampleViews), so
// the comparison isolates the algorithmic difference (single averaged field + Marching-Cubes
// extraction vs 6-direction layers + per-direction point extraction) rather than any input
// discrepancy.
//
//   tsdf_benchmark [--voxel <f>] [--shape cube|cylinder|both] [--maxdir <k>]
//
// Reports, per shape x method (Simple(fine), Simple(coarse), Directional):
//   build_ms  - host wall-clock time to a ready extracted cloud (integrate loop + extract;
//               DirectionalTSDF runs extraction inside Integrate, so its build_ms is just the
//               integrate-loop time). Every GPU dispatch here is synchronous (SubmitOneShot
//               does vkQueueWaitIdle), so host wall-clock ~= GPU time for both methods.
//   nPoints   - extracted point count.
//   mem_KB    - occupied surface storage: Simple = FilledCount()*16B; Directional =
//               HostStore().Size()*4096B (each occupied (block,direction) group = 512 voxels
//               x 8B GpuTsdfVoxel).
//   acc_mean/acc_rmse - per-point ground-truth error (fixtures::NearestDistance) in mm,
//               overall mean and RMSE.
//   edge/flat/curved  - per-region (fixtures::ClassifyRegion) mean error in mm.
//
// ---------------------------------------------------------------------------------------------
// REAL 2-level variance-adaptive voxel grid (MrHash paper's core mechanism, Table-7 ablation
// form: fine + coarse only, no further levels):
//
//   1. Integrate a fine SimpleTSDF (voxel = --voxel, variance tracked) AND a coarse SimpleTSDF
//      (voxel = 2x fine) from the *same* input views, so the two differ only in bucket size.
//   2. Fine voxels (SimpleTSDF::DownloadVoxels) are grouped into coarse cells (floor(center /
//      coarseVoxel)); each cell's mean per-voxel variance drives a keep-fine-or-coarsen
//      decision at a threshold sigma.
//   3. The adaptive point cloud = fine-cloud points in high-variance cells U coarse-cloud
//      points in low-variance cells (a cell with no fine voxels is treated as coarse).
//   4. Memory = sum over occupied cells of (high-var ? fineVoxelsInCell*16B : 1*16B) -- an 8:1
//      coarsening in every cell that gets to coarsen.
//   5. Accuracy is scored against the same analytic ground truth as the other rows.
//
// HONEST-MEASUREMENT NOTE: this selects, per coarse cell, between two independently-extracted
// single-resolution Marching-Cubes meshes -- it does NOT implement multi-resolution Marching
// Cubes with transitional/dual voxels (the paper's actual boundary handling). Minor seams or
// double-coverage can occur where a fine cell borders a coarsened one. That is an accepted
// approximation for measuring the accuracy/memory *tradeoff*; it is not a claim of watertight
// multi-resolution reconstruction. Numbers are reported as measured, whatever they turn out to
// be -- see the printed [caveat] line and the per-shape insight summary.
//
// A second, unrelated honest surprise turned up at the default --voxel 0.1 (coarse = 0.2): the
// CUBE's Simple(coarse) row extracts ZERO points. Root-caused with a scratch diagnostic
// (occupied-voxel sign + 2x2x2-cluster dump, not checked in): fixtures::SampleViews picks its
// flat-face sample grid pitch purely from the per-view point budget (independent of voxel
// size), and for the cube fixture that pitch lands at EXACTLY 0.2 -- i.e. exactly the coarse
// voxel size. That aliasing leaves every other coarse-voxel column empty along each face, so
// voxel_tsdf_mc.comp's full-2x2x2-occupied-corners requirement is never satisfied anywhere
// (confirmed: 0 full clusters out of 1538 occupied coarse cells), and Marching Cubes emits no
// triangles at all. The cylinder does NOT hit this (curved theta/z + cap sampling gives denser
// tangential overlap: 335 full clusters), so its Simple(coarse) row is a normal, if inaccurate,
// baseline. This is a genuine fixture/voxel-size interaction, not a selection-logic bug: the
// per-cell variance grouping (which drives coarsen%/mem_KB in the sigma sweep) is computed
// straight from fine.DownloadVoxels() and is unaffected -- only the *coarse replacement
// geometry* is unavailable for the cube. Consequently the cube's "coarsened" sigma-sweep cells
// are holes (points deleted, nothing substituted) rather than genuine coarse surface; this is
// called out explicitly wherever it changes the reading (PrintAdaptiveInsight, "n/a" instead of
// a misleading 0.00000 for zero-sample buckets). Cylinder is the shape where the intended
// tradeoff is actually exercised end-to-end.
//
// Also prints DirectionalTSDF's internal integrate/extract/merge breakdown (LastFrameStats,
// summed over the view loop), a small MrHash-style single-resolution projection off
// SimpleTSDF::DownloadVoxels (legacy bonus section, kept for continuity with
// variance_adaptive_demo.cpp), and the new real 2-level sigma sweep described above.
//
// ---------------------------------------------------------------------------------------------
// [directional-storage] / Adaptive-Directional hybrid (Task: "can we get DirectionalTSDF
// accuracy at less memory?"):
//
//   Part 1 (AnalyzeDirectionalStorage): the existing Directional row's mem_KB
//   (dir.HostStore().Size()*4096B) counts whole 512-voxel groups -- mostly empty near a thin
//   surface band. This enumerates the ACTUAL occupied (weight>0) directional voxels and
//   projects what a per-voxel store (8B/voxel, same accuracy) would cost, plus a
//   directions-per-occupied-voxel histogram (pruning potential: most flat voxels only ever
//   get written from one dominant direction).
//
//   HONEST-MEASUREMENT NOTE: this reads the live GPU active pool (DebugDownloadIndexGrid +
//   DebugDownloadGroupVoxels), NOT DirectionalHostStore's values. Every view here integrates
//   with the same aabbCenterHint (Zero()), so DirectionalTSDF's local window never moves
//   across a shape's whole view loop; directional_tsdf_classify.comp's "inside window" test
//   therefore reclassifies every previously-active slot as "reusable" every frame, so the
//   dirty write-back path (the only place HostStore() is Put with real data) never fires.
//   HostStore() still holds one zero-filled Group per touched key (inserted once via
//   GetOrCreate the first time that key becomes resident) -- so HostStore().Size() remains a
//   correct OCCUPANCY count (matches mem_KB above) but HostStore().Get(key) is all-zero for
//   this run. The GPU pool holds the real values, hence the debug-pool reads below.
//
//   Part 2 (RunAdaptiveDirectional): reuses the sigma-sweep's per-coarse-cell variance
//   classification (median sigma) to build a hybrid surface = Directional points in
//   high-variance ("edge") cells U Simple(fine) points in low-variance ("flat") cells, with
//   memory = flat Simple voxels*16B + edge occupied-directional-voxels*8B (per-voxel
//   projected, restricted to edge cells, from the same Part-1 pass).
#include "shape_fixtures.h"

#include "Engine/Core/Context.h"
#include "Engine/Spatial/AdaptiveVoxelGrid.h"
#include "Engine/Spatial/CompactDirectionalTSDF.h"
#include "Engine/Spatial/DirectionalTSDF.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"
#include "Engine/Spatial/SimpleTSDF.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>

using fixtures::Region;
using fixtures::Shape;

namespace {

    constexpr float kTruncation = 0.3f;
    constexpr int kNumRegions = 3; // Flat, Curved, Edge

    const char *ShapeName(Shape s) { return s == Shape::Cube ? "cube" : "cylinder"; }

    // Running mean + RMSE over per-point ground-truth errors.
    struct ErrStats {
        double sum = 0.0, sumSq = 0.0;
        std::size_t count = 0;
        void add(double e) {
            sum += e;
            sumSq += e * e;
            ++count;
        }
        double mean() const { return count ? sum / double(count) : 0.0; }
        double rmse() const { return count ? std::sqrt(sumSq / double(count)) : 0.0; }
    };

    // Formats an ErrStats-derived value, printing "n/a" (not a misleading "0.00000") when the
    // bucket has zero samples -- either because the shape structurally has no points in that
    // region (e.g. Region::Curved never occurs for a cube) or because a row's extraction
    // produced zero points outright (see the cube Simple(coarse) finding below).
    std::string FmtErr(double v, bool valid, int width) {
        char buf[32];
        if (valid) std::snprintf(buf, sizeof buf, "%*.5f", width, v);
        else std::snprintf(buf, sizeof buf, "%*s", width, "n/a");
        return buf;
    }

    struct Row {
        std::string shape;
        std::string method;
        double buildMs = 0.0;
        std::size_t nPoints = 0;
        double memKB = 0.0;
        ErrStats overall;
        std::array<ErrStats, kNumRegions> perRegion; // Flat, Curved, Edge
    };

    void ScoreAgainstGT(Shape shape, float voxel, const std::vector<Eigen::Vector3f> &pts,
                        Row &row) {
        for (const auto &p : pts) {
            const double e = double(fixtures::NearestDistance(shape, p));
            row.overall.add(e);
            row.perRegion[static_cast<int>(fixtures::ClassifyRegion(shape, p, voxel))].add(e);
        }
    }

    void PrintHeader() {
        std::printf("%-9s %-20s %10s %9s %10s %10s %10s %9s %9s %9s\n", "shape", "method",
                    "build_ms", "nPoints", "mem_KB", "acc_mean", "acc_rmse", "edge", "flat",
                    "curved");
        std::printf("---------------------------------------------------------------------"
                    "--------------------------------------\n");
    }

    void PrintRow(const Row &r) {
        const auto &e = r.perRegion[static_cast<int>(Region::Edge)];
        const auto &f = r.perRegion[static_cast<int>(Region::Flat)];
        const auto &c = r.perRegion[static_cast<int>(Region::Curved)];
        std::printf("%-9s %-20s %10.3f %9zu %10.2f %s %s %s %s %s\n", r.shape.c_str(),
                    r.method.c_str(), r.buildMs, r.nPoints, r.memKB,
                    FmtErr(r.overall.mean(), r.overall.count > 0, 10).c_str(),
                    FmtErr(r.overall.rmse(), r.overall.count > 0, 10).c_str(),
                    FmtErr(e.mean(), e.count > 0, 9).c_str(),
                    FmtErr(f.mean(), f.count > 0, 9).c_str(),
                    FmtErr(c.mean(), c.count > 0, 9).c_str());
    }

    // ---- Legacy bonus: single-resolution MrHash-style variance-adaptive projection off
    // SimpleTSDF voxels. Mirrors example2/variance_adaptive_demo.cpp's premise check, condensed
    // to one threshold. Kept alongside the real 2-level sweep below (RunVarianceAdaptiveSweep)
    // for continuity -- this one never actually builds a coarse grid, it only *estimates* what
    // an 8:1 per-voxel coarsening would save.
    void PrintVarianceAdaptiveSection(Shape shape, float voxel,
                                      const std::vector<Engine::Spatial::VoxelStat> &voxels) {
        if (voxels.empty()) {
            std::printf("  [variance-adaptive-projection] %s: no occupied voxels, skipping\n",
                        ShapeName(shape));
            return;
        }
        std::array<double, kNumRegions> regionVarSum{};
        std::array<std::size_t, kNumRegions> regionCount{};
        double maxVar = 0.0;
        for (const auto &v : voxels) {
            const int r = static_cast<int>(fixtures::ClassifyRegion(shape, v.center, voxel));
            regionVarSum[r] += double(v.variance);
            ++regionCount[r];
            maxVar = std::max(maxVar, double(v.variance));
        }
        auto regionMean = [&](int r) {
            return regionCount[r] ? regionVarSum[r] / double(regionCount[r]) : 0.0;
        };

        // sigma = 5% of the max observed variance (same order as variance_adaptive_demo's sweep).
        const double sigma = 0.05 * maxVar;
        std::size_t coarsenable = 0;
        for (const auto &v : voxels)
            if (double(v.variance) <= sigma) ++coarsenable;
        const double coarsenPct =
                100.0 * double(coarsenable) / double(voxels.size());

        const double curMemKB = double(voxels.size()) * 16.0 / 1024.0;
        // MrHash-style coarsening merges 8 fine voxels (2x2x2) into 1 coarse voxel of the same
        // 16B entry size -> keeps 1/8 of the storage for every coarsen-able voxel.
        const double projectedKB =
                (double(voxels.size() - coarsenable) * 16.0 +
                 double(coarsenable) * 16.0 / 8.0) / 1024.0;

        std::printf("  [variance-adaptive-projection] %s: occupied=%zu  edge/flat/curved mean var = "
                    "%.3g / %.3g / %.3g\n",
                    ShapeName(shape), voxels.size(),
                    regionMean(static_cast<int>(Region::Edge)),
                    regionMean(static_cast<int>(Region::Flat)),
                    regionMean(static_cast<int>(Region::Curved)));
        std::printf("  [variance-adaptive-projection] %s: sigma=%.3g (5%% of max var) -> coarsen-able = "
                    "%zu/%zu (%.1f%%); mem %.2f KB -> projected %.2f KB (8:1 coarsen, -%.1f%%)\n",
                    ShapeName(shape), sigma, coarsenable, voxels.size(), coarsenPct, curMemKB,
                    projectedKB, 100.0 * (curMemKB - projectedKB) / curMemKB);
    }

    // ---- Real 2-level variance-adaptive voxel grid (Table-7 ablation: fine + coarse) --------

    // Coarse-cell index for a world-space point: floor(p / coarseVoxel), component-wise. Used
    // both to bucket fine voxels into coarse cells and to classify extracted cloud points.
    using CellKey = std::array<int, 3>;

    CellKey CellOf(const Eigen::Vector3f &p, float vc) {
        return {int(std::floor(double(p.x()) / double(vc))),
                int(std::floor(double(p.y()) / double(vc))),
                int(std::floor(double(p.z()) / double(vc)))};
    }

    struct CellAgg {
        double varSum = 0.0;
        std::size_t fineCount = 0; // number of fine voxels this coarse cell aggregates
        double meanVar() const { return fineCount ? varSum / double(fineCount) : 0.0; }
    };

    // Groups fine voxels into coarse cells (single source of truth, shared by the sigma
    // sweep and the Adaptive-Directional hybrid so both agree on the same cell statistics).
    std::map<CellKey, CellAgg> BuildCellStats(
            const std::vector<Engine::Spatial::VoxelStat> &fineVoxels, float vc) {
        std::map<CellKey, CellAgg> cellStats;
        for (const auto &v : fineVoxels) {
            CellAgg &agg = cellStats[CellOf(v.center, vc)];
            agg.varSum += double(v.variance);
            ++agg.fineCount;
        }
        return cellStats;
    }

    // p-th percentile (0..1) of the per-cell mean-variance distribution. Empty -> 0.
    double PercentileOfCellVars(const std::map<CellKey, CellAgg> &cellStats, double p) {
        if (cellStats.empty()) return 0.0;
        std::vector<double> cellVars;
        cellVars.reserve(cellStats.size());
        for (const auto &kv : cellStats) cellVars.push_back(kv.second.meanVar());
        std::sort(cellVars.begin(), cellVars.end());
        const std::size_t idx =
                std::min(cellVars.size() - 1, std::size_t(p * double(cellVars.size() - 1)));
        return cellVars[idx];
    }

    // Shared high-variance ("edge") cell test at a single (median) sigma -- used by the
    // Adaptive-Directional hybrid (Part 2) to select Directional-vs-Simple(fine) per cell.
    // The sigma sweep below (RunVarianceAdaptiveSweep) builds its own per-percentile sigmas
    // from the same BuildCellStats/PercentileOfCellVars but is otherwise independent.
    struct CellClassifier {
        std::map<CellKey, CellAgg> cellStats;
        float vc = 0.0f;
        double sigma = 0.0;

        bool IsHighVar(const Eigen::Vector3f &p) const {
            const auto it = cellStats.find(CellOf(p, vc));
            // A cell with no fine voxels has nothing to keep fine -> treat as coarse/flat.
            return it != cellStats.end() && it->second.meanVar() > sigma;
        }
    };

    struct SigmaRow {
        std::string shape;
        double sigma = 0.0;
        double coarsenPct = 0.0;
        double memKB = 0.0;
        std::size_t nPoints = 0; // not printed by PrintSigmaRow; used by Compact-Dir combine row
        ErrStats overall;
        std::array<ErrStats, kNumRegions> perRegion;
    };

    void PrintSigmaHeader() {
        std::printf("%-9s %12s %9s %10s %10s %10s %9s %9s %9s\n", "shape", "sigma", "coarsen%",
                    "mem_KB", "acc_mean", "acc_rmse", "edge", "flat", "curved");
        std::printf("---------------------------------------------------------------------"
                    "---------------------------------\n");
    }

    void PrintSigmaRow(const SigmaRow &r) {
        const auto &e = r.perRegion[static_cast<int>(Region::Edge)];
        const auto &f = r.perRegion[static_cast<int>(Region::Flat)];
        const auto &c = r.perRegion[static_cast<int>(Region::Curved)];
        std::printf("%-9s %12.4g %8.1f%% %10.2f %s %s %s %s %s\n", r.shape.c_str(), r.sigma,
                    r.coarsenPct, r.memKB,
                    FmtErr(r.overall.mean(), r.overall.count > 0, 10).c_str(),
                    FmtErr(r.overall.rmse(), r.overall.count > 0, 10).c_str(),
                    FmtErr(e.mean(), e.count > 0, 9).c_str(),
                    FmtErr(f.mean(), f.count > 0, 9).c_str(),
                    FmtErr(c.mean(), c.count > 0, 9).c_str());
    }

    // Groups fine voxels into coarse cells, sweeps sigma at 4 percentiles of the observed
    // per-cell mean-variance distribution (adaptive to whatever range this shape actually
    // produces -- a fixed fraction-of-max can clump if the distribution is skewed, which would
    // make every row look identical and hide a mis-wired selection), and for each sigma builds
    // the adaptive point cloud (fine points in high-var cells U coarse points in low-var/absent
    // cells), scoring it against the analytic ground truth.
    std::vector<SigmaRow> RunVarianceAdaptiveSweep(
            Shape shape, float classifyVoxel, float vc,
            const std::vector<Engine::Spatial::VoxelStat> &fineVoxels,
            const Engine::Spatial::OrientedPointCloud &fineCloud,
            const Engine::Spatial::OrientedPointCloud &coarseCloud) {
        std::vector<SigmaRow> out;

        std::map<CellKey, CellAgg> cellStats = BuildCellStats(fineVoxels, vc);

        if (cellStats.empty()) {
            std::printf("  [variance-adaptive-2level] %s: no fine voxels, skipping sigma sweep\n",
                        ShapeName(shape));
            return out;
        }

        const std::array<double, 4> percentiles = {0.25, 0.50, 0.75, 0.90};

        for (double p : percentiles) {
            const double sigma = PercentileOfCellVars(cellStats, p);

            SigmaRow row;
            row.shape = ShapeName(shape);
            row.sigma = sigma;

            std::size_t coarsenCells = 0;
            double memBytes = 0.0;
            for (const auto &kv : cellStats) {
                if (kv.second.meanVar() > sigma) {
                    memBytes += double(kv.second.fineCount) * 16.0; // keep fine, no coarsening
                } else {
                    memBytes += 16.0; // coarsened: 8 fine voxels -> 1 coarse voxel
                    ++coarsenCells;
                }
            }
            row.memKB = memBytes / 1024.0;
            row.coarsenPct = 100.0 * double(coarsenCells) / double(cellStats.size());

            auto isHighVar = [&](const Eigen::Vector3f &p) {
                const auto it = cellStats.find(CellOf(p, vc));
                // A cell with no fine voxels has nothing to keep fine -> treat as coarse.
                return it != cellStats.end() && it->second.meanVar() > sigma;
            };

            std::vector<Eigen::Vector3f> adaptivePts;
            adaptivePts.reserve(fineCloud.points.size() + coarseCloud.points.size());
            for (const auto &p : fineCloud.points)
                if (isHighVar(p)) adaptivePts.push_back(p);
            for (const auto &p : coarseCloud.points)
                if (!isHighVar(p)) adaptivePts.push_back(p);
            row.nPoints = adaptivePts.size();

            for (const auto &p : adaptivePts) {
                const double e = double(fixtures::NearestDistance(shape, p));
                row.overall.add(e);
                row.perRegion[static_cast<int>(fixtures::ClassifyRegion(shape, p, classifyVoxel))]
                        .add(e);
            }

            out.push_back(row);
        }

        return out;
    }

    // ---- SimpleTSDF build/extract/score plumbing (shared by fine + coarse rows) -------------

    struct BuiltSimple {
        Engine::Spatial::SimpleTSDF tsdf;
        Engine::Spatial::OrientedPointCloud cloud;
        double buildMs = 0.0;
    };

    BuiltSimple BuildSimpleTSDF(Engine::Core::Context &ctx, float voxelSize,
                                const std::vector<fixtures::View> &views) {
        BuiltSimple out;
        out.tsdf.Build(ctx, voxelSize, kTruncation);

        const auto t0 = std::chrono::steady_clock::now();
        for (const auto &v : views) out.tsdf.Integrate(v.points, v.camPos);
        out.cloud = out.tsdf.ExtractPointCloud();
        const auto t1 = std::chrono::steady_clock::now();

        out.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return out;
    }

    // Same as BuildSimpleTSDF, but integrates with the normal view-angle-weighted overload
    // (w = max(0, dot(normal, -rayDir)) per observation) -- isolates the effect of view-angle
    // weighting alone (no extra memory: same TSDFEntry/hash-table layout as Simple(fine)).
    BuiltSimple BuildSimpleTSDFWeighted(Engine::Core::Context &ctx, float voxelSize,
                                        const std::vector<fixtures::View> &views) {
        BuiltSimple out;
        out.tsdf.Build(ctx, voxelSize, kTruncation);

        const auto t0 = std::chrono::steady_clock::now();
        for (const auto &v : views) out.tsdf.Integrate(v.points, v.normals, v.camPos);
        out.cloud = out.tsdf.ExtractPointCloud();
        const auto t1 = std::chrono::steady_clock::now();

        out.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return out;
    }

    Row ScoreSimpleRow(Shape shape, const std::string &method, float classifyVoxel,
                       const BuiltSimple &built) {
        Row row;
        row.shape = ShapeName(shape);
        row.method = method;
        row.buildMs = built.buildMs;
        row.nPoints = built.cloud.points.size();
        row.memKB = double(built.tsdf.FilledCount()) * 16.0 / 1024.0;
        ScoreAgainstGT(shape, classifyVoxel, built.cloud.points, row);
        return row;
    }

    // ---- AdaptiveVoxelGrid (MrHash paper core): fine SimpleTSDF + CPU variance merge (2^3
    // blocks -> coarse when block mean sigma^2 < percentile threshold) + multi-resolution
    // Marching Cubes. Accuracy = mesh-vertex distance to GT (same fixtures::NearestDistance
    // metric as every other row). memKB = the mixed-grid representation (FineCount+CoarseCount)
    // *16B -- the adaptive benefit. NOTE: the current impl keeps the fine level GPU-resident
    // (the reduction is in the extracted/stored representation, not GPU VRAM).
    Row RunAdaptiveVoxelGrid(Engine::Core::Context &ctx, Shape shape, float voxel,
                             const std::vector<fixtures::View> &views, float percentile) {
        Engine::Spatial::AdaptiveVoxelGrid avg;
        avg.Build(ctx, voxel, kTruncation);
        avg.SetVariancePercentile(percentile);

        const auto t0 = std::chrono::steady_clock::now();
        for (const auto &v : views) avg.Integrate(v.points, v.camPos);
        const Engine::Spatial::AdaptiveMesh mesh = avg.ExtractMesh();
        const auto t1 = std::chrono::steady_clock::now();

        Row row;
        row.shape = ShapeName(shape);
        char m[48];
        std::snprintf(m, sizeof(m), "AdaptiveVoxelGrid(p%.2f)", percentile);
        row.method = m;
        row.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        row.nPoints = mesh.vertices.size();
        row.memKB = double(avg.FineCount() + avg.CoarseCount()) * 16.0 / 1024.0;
        ScoreAgainstGT(shape, voxel, mesh.vertices, row);
        return row;
    }

    // ---- Part 1: Directional per-voxel storage-waste measurement --------------------------

    struct DirStorageStats {
        std::size_t occupiedGroups = 0;      // (block,direction) groups currently resident
        std::size_t occupiedDirVoxels = 0;   // total (voxel,direction) instances with weight>0
        std::size_t occupiedDirVoxelsEdge = 0; // subset of the above in high-variance ("edge") cells
        std::size_t occupiedPositions = 0;   // distinct (block,local-voxel) positions with >=1 dir occupied
        std::size_t hist1 = 0, hist2 = 0, hist3plus = 0; // direction-count histogram over occupiedPositions
        double avgDirsPerVoxel = 0.0;
        double memBlockKB = 0.0;             // current: HostStore().Size()*4096B (whole-group granularity)
        double memPerVoxelProjectedKB = 0.0; // occupiedDirVoxels*8B (per-voxel granularity, same accuracy)
        double wasteFactor = 0.0;            // memBlockKB / memPerVoxelProjectedKB
    };

    // Enumerates DirectionalTSDF's currently-resident GPU active-pool groups over a generous
    // block bounding box around the shape, downloading each occupied group's live voxel data
    // (DebugDownloadGroupVoxels) to count weight>0 voxels -- see the file-header
    // HONEST-MEASUREMENT NOTE for why this reads the GPU pool rather than dir.HostStore()'s
    // values (HostStore()'s values are stale zeros for this benchmark's fixed-window usage
    // pattern; only its Size() -- an occupancy count -- is meaningful here).
    //
    // `isEdgeCell`, if set, additionally tallies occupiedDirVoxelsEdge -- the subset of
    // occupied voxels whose world-space center falls in a high-variance cell (Part 2's
    // Adaptive-Directional hybrid memory term) -- in the SAME pass, avoiding a second
    // GPU-download sweep.
    DirStorageStats AnalyzeDirectionalStorage(
            Engine::Spatial::DirectionalTSDF &dir, float voxel,
            const std::function<bool(const Eigen::Vector3f &)> &isEdgeCell) {
        using namespace Engine::Spatial;
        DirStorageStats out;
        out.memBlockKB = double(dir.HostStore().Size()) * 4096.0 / 1024.0;

        const std::vector<uint32_t> indexGrid = dir.DebugDownloadIndexGrid();
        const Eigen::Vector3i localBase = dir.LocalBase();

        auto slotFor = [&](const DirectionalGroupKey &key) -> uint32_t {
            const int lx = key.gx - localBase.x();
            const int ly = key.gy - localBase.y();
            const int lz = key.gz - localBase.z();
            if (lx < 0 || ly < 0 || lz < 0 || lx >= int(kLocalGroupGrid) ||
                ly >= int(kLocalGroupGrid) || lz >= int(kLocalGroupGrid))
                return kInvalidPoolIndex;
            return indexGrid[IndexGridOffset(uint32_t(lx), uint32_t(ly), uint32_t(lz),
                                             key.direction)];
        };

        // Generous bounding box: fixture half-extent (1.5 for both shapes; see
        // shape_fixtures.h) + the truncation band, converted to block units (kGroupDim
        // voxels/block) with a safety margin -- covers the shape regardless of which fixture.
        const float halfExtent =
                std::max({fixtures::kCubeHalf, fixtures::kCylRadius, fixtures::kCylHalfZ});
        const float maxCoord = halfExtent + kTruncation + 2.0f * voxel;
        const int voxelBound = int(std::ceil(double(maxCoord) / double(voxel))) + 2;
        const int blockBound = int(std::ceil(double(voxelBound) / double(kGroupDim))) + 1;

        for (int gz = -blockBound; gz <= blockBound; ++gz)
            for (int gy = -blockBound; gy <= blockBound; ++gy)
                for (int gx = -blockBound; gx <= blockBound; ++gx) {
                    std::array<uint8_t, kVoxelsPerGroup> dirCount{}; // per-local-voxel dir count
                    bool anyOccupied = false;
                    for (uint8_t d = 0; d < uint8_t(kNumDirections); ++d) {
                        const DirectionalGroupKey key{gx, gy, gz, d};
                        const uint32_t slot = slotFor(key);
                        if (slot == kInvalidPoolIndex) continue;
                        ++out.occupiedGroups;
                        const DirectionalHostStore::Group group = dir.DebugDownloadGroupVoxels(key);
                        for (uint32_t v = 0; v < kVoxelsPerGroup; ++v) {
                            if (group[v].weight <= 0.0f) continue;
                            ++out.occupiedDirVoxels;
                            ++dirCount[v];
                            anyOccupied = true;
                            if (isEdgeCell) {
                                const uint32_t lx = v % kGroupDim;
                                const uint32_t ly = (v / kGroupDim) % kGroupDim;
                                const uint32_t lz = v / (kGroupDim * kGroupDim);
                                const int gvx = gx * int(kGroupDim) + int(lx);
                                const int gvy = gy * int(kGroupDim) + int(ly);
                                const int gvz = gz * int(kGroupDim) + int(lz);
                                const Eigen::Vector3f center((float(gvx) + 0.5f) * voxel,
                                                             (float(gvy) + 0.5f) * voxel,
                                                             (float(gvz) + 0.5f) * voxel);
                                if (isEdgeCell(center)) ++out.occupiedDirVoxelsEdge;
                            }
                        }
                    }
                    if (!anyOccupied) continue;
                    for (uint32_t v = 0; v < kVoxelsPerGroup; ++v) {
                        const uint8_t c = dirCount[v];
                        if (c == 0) continue;
                        ++out.occupiedPositions;
                        if (c == 1) ++out.hist1;
                        else if (c == 2) ++out.hist2;
                        else ++out.hist3plus;
                    }
                }

        out.memPerVoxelProjectedKB = double(out.occupiedDirVoxels) * 8.0 / 1024.0;
        out.wasteFactor = out.memPerVoxelProjectedKB > 0.0
                                  ? out.memBlockKB / out.memPerVoxelProjectedKB
                                  : 0.0;
        out.avgDirsPerVoxel = out.occupiedPositions > 0
                                      ? double(out.occupiedDirVoxels) / double(out.occupiedPositions)
                                      : 0.0;
        return out;
    }

    struct DirectionalResult {
        Row row;
        std::vector<Eigen::Vector3f> points; // extracted cloud positions (for Part 2 reuse)
        DirStorageStats storage;
    };

    DirectionalResult RunDirectional(Engine::Core::Context &ctx, Shape shape, float voxel,
                                     uint32_t maxDir, const std::vector<fixtures::View> &views,
                                     const CellClassifier &classifier) {
        DirectionalResult out;
        Row &row = out.row;
        row.shape = ShapeName(shape);
        row.method = "Directional";

        Engine::Spatial::DirectionalTSDF dir;
        dir.Build(ctx, voxel, kTruncation);
        dir.SetIntegrationQuality({maxDir, 4, true});

        double sumIntegrateMs = 0.0, sumExtractMs = 0.0, sumMergeMs = 0.0;
        const auto t0 = std::chrono::steady_clock::now();
        for (const auto &v : views) {
            dir.Integrate(v.points, v.normals, v.camPos, Eigen::Vector3f::Zero());
            const auto st = dir.LastFrameStats();
            sumIntegrateMs += st.integrateMs;
            sumExtractMs += st.extractMs;
            sumMergeMs += st.mergeMs;
        }
        const auto t1 = std::chrono::steady_clock::now();

        row.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const auto &cloud = dir.PointCloud();
        row.nPoints = cloud.size();
        row.memKB = double(dir.HostStore().Size()) * 4096.0 / 1024.0;

        out.points.reserve(cloud.size());
        for (const auto &e : cloud) out.points.push_back(e.position);
        ScoreAgainstGT(shape, voxel, out.points, row);

        std::printf("  [directional-internal] %s: sum integrate=%.3fms extract=%.3fms "
                    "merge=%.3fms (over %zu frames)\n",
                    ShapeName(shape), sumIntegrateMs, sumExtractMs, sumMergeMs, views.size());

        // Part 1 storage-waste enumeration must run before `dir` goes out of scope --
        // the debug pool downloads need the live backend.
        auto isEdgeCell = [&classifier](const Eigen::Vector3f &p) { return classifier.IsHighVar(p); };
        out.storage = AnalyzeDirectionalStorage(dir, voxel, isEdgeCell);

        return out;
    }

    // ---- Compact-Directional: DirectionalTSDF logic in a per-voxel flat (voxel,dir) hash -----
    //
    // Same dominant-direction integrate + per-direction extract as the Directional row, but
    // stored one 16-byte entry per occupied (voxel,direction) key (like SimpleTSDF) instead of
    // DirectionalTSDF's 8^3=512-voxel blocks. This makes the "Directional accuracy is a
    // block-granularity storage cost, not fundamental" claim a REAL measurement: mem_KB here is
    // FilledCount()*16B (occupied entries only), compared directly against the Directional row's
    // block-granular mem_KB at the SAME per-direction accuracy.
    struct CompactDirectionalResult {
        Row raw;    // Compact-Directional: RAW per-(voxel,direction) candidates.
        Row merged; // Compact-Directional(merged): raw candidates clustered on the CPU
                    // (CompactDirectionalTSDF::ExtractPointCloud(..., merge=true)) -- ported
                    // from DirectionalTSDF::mergeCandidates, dedups redundant candidates into
                    // averaged points while preserving sharp corners.
    };

    CompactDirectionalResult RunCompactDirectional(Engine::Core::Context &ctx, Shape shape,
                                                   float voxel, uint32_t maxDir,
                                                   const std::vector<fixtures::View> &views) {
        Engine::Spatial::CompactDirectionalTSDF cd;
        cd.Build(ctx, voxel, kTruncation);
        cd.SetIntegrationQuality({maxDir, 4, true});

        const auto t0 = std::chrono::steady_clock::now();
        for (const auto &v : views) cd.Integrate(v.points, v.normals, v.camPos);
        const Engine::Spatial::OrientedPointCloud cloud = cd.ExtractPointCloud();
        const auto t1 = std::chrono::steady_clock::now();

        Row row;
        row.shape = ShapeName(shape);
        row.method = "Compact-Directional";
        row.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        row.nPoints = cloud.points.size();
        row.memKB = double(cd.FilledCount()) * 16.0 / 1024.0; // occupied entries * sizeof(DirEntry)
        ScoreAgainstGT(shape, voxel, cloud.points, row);

        // Merged extraction: SAME build/integrate (cd's hash table above is untouched by
        // extraction), just a second extract dispatch with merge=true -- no re-integrate cost.
        // Memory is unchanged (merge is extraction-only; FilledCount() reflects the SAME
        // storage), so memKB is copied from the raw row rather than re-measured.
        const auto t2 = std::chrono::steady_clock::now();
        const Engine::Spatial::OrientedPointCloud mergedCloud =
                cd.ExtractPointCloud(1u << 19, /*merge=*/true);
        const auto t3 = std::chrono::steady_clock::now();

        Row mergedRow;
        mergedRow.shape = ShapeName(shape);
        mergedRow.method = "Compact-Directional(merged)";
        mergedRow.buildMs =
                row.buildMs + std::chrono::duration<double, std::milli>(t3 - t2).count();
        mergedRow.nPoints = mergedCloud.points.size();
        mergedRow.memKB = row.memKB;
        ScoreAgainstGT(shape, voxel, mergedCloud.points, mergedRow);

        return {row, mergedRow};
    }

    // ---- Compact-Dir(var-adaptive): the two orthogonal memory axes COMBINED --------------
    //
    // Axis 1 (Compact-Directional): DirectionalTSDF's per-direction accuracy stored one
    // 16-byte entry per occupied (voxel,direction) key, not whole 512-voxel blocks.
    // Axis 2 (variance-adaptive resolution): coarsen (2x voxel) low-variance flat cells,
    // keep fine resolution only where variance is high (edges/curvature).
    //
    // This applies Axis 2's cell classification -- reusing the SAME cellStats the
    // Simple-only sigma sweep (RunVarianceAdaptiveSweep) uses, i.e. built from a fine
    // SimpleTSDF::DownloadVoxels() grouped into coarse (2*voxel) cells -- to a fine and a
    // coarse CompactDirectionalTSDF instead of a fine/coarse SimpleTSDF. The two Compact-
    // Directional structures supply their OWN accuracy (Directional-level) and their OWN
    // per-entry memory count (CompactEntry, 16B each); only the fine/coarse SELECTION test
    // (which cells stay fine) is shared with the Simple-only sweep.
    //
    // Same honesty caveat as RunVarianceAdaptiveSweep: this selects, per coarse cell,
    // between two independently-extracted point clouds (no transitional-voxel MC), so minor
    // seams at fine<->coarse cell boundaries are possible; and the cube's Simple(coarse)
    // all-coarse-MC-fails aliasing (see file header) does NOT apply here (Compact-Directional
    // extraction is per-(voxel,direction) hash-probe crossings, not Marching-Cubes corner
    // occupancy), but the cube's per-cell variance may still be near-uniform, making most
    // cells classify the same way (little to coarsen) -- reported as measured either way.
    struct CompactAdaptiveEval {
        double memKB = 0.0;
        std::size_t nPoints = 0;
        double coarsenPct = 0.0;
        ErrStats overall;
        std::array<ErrStats, kNumRegions> perRegion;
    };

    CompactAdaptiveEval EvalCompactAdaptiveSigma(
            Shape shape, float classifyVoxel, float vc, double sigma,
            const std::map<CellKey, CellAgg> &cellStats,
            const Engine::Spatial::OrientedPointCloud &fineCloud,
            const Engine::Spatial::OrientedPointCloud &coarseCloud,
            const std::vector<Engine::Spatial::CompactEntry> &fineEntries,
            const std::vector<Engine::Spatial::CompactEntry> &coarseEntries) {
        CompactAdaptiveEval ev;

        auto isHighVar = [&](const Eigen::Vector3f &p) {
            const auto it = cellStats.find(CellOf(p, vc));
            // A cell with no fine voxels has nothing to keep fine -> treat as coarse/flat.
            return it != cellStats.end() && it->second.meanVar() > sigma;
        };

        // Memory: fine-Compact entries kept (high-var cell) + coarse-Compact entries kept
        // (low-var cell), each entry = 16B (sizeof(DirEntry)).
        std::size_t fineMemCount = 0, coarseMemCount = 0;
        for (const auto &e : fineEntries)
            if (isHighVar(e.center)) ++fineMemCount;
        for (const auto &e : coarseEntries)
            if (!isHighVar(e.center)) ++coarseMemCount;
        ev.memKB = double(fineMemCount + coarseMemCount) * 16.0 / 1024.0;

        std::size_t coarsenCells = 0;
        for (const auto &kv : cellStats)
            if (!(kv.second.meanVar() > sigma)) ++coarsenCells;
        ev.coarsenPct =
                cellStats.empty() ? 0.0 : 100.0 * double(coarsenCells) / double(cellStats.size());

        // Adaptive surface = fine-Compact cloud points in high-var cells U coarse-Compact
        // cloud points in low-var cells, then merge/dedup to weld coincident points at the
        // fine<->coarse cell boundaries (within-voxel double-coverage). Cross-boundary
        // duplicates in DIFFERENT voxels are a residual point-pipeline characteristic -- the
        // paper's transitional-voxel seam fix is a MESH technique that does not apply here.
        std::vector<Eigen::Vector3f> pts, nrms;
        pts.reserve(fineCloud.points.size() + coarseCloud.points.size());
        nrms.reserve(pts.capacity());
        for (std::size_t i = 0; i < fineCloud.points.size(); ++i)
            if (isHighVar(fineCloud.points[i])) {
                pts.push_back(fineCloud.points[i]);
                nrms.push_back(fineCloud.normals[i]);
            }
        for (std::size_t i = 0; i < coarseCloud.points.size(); ++i)
            if (!isHighVar(coarseCloud.points[i])) {
                pts.push_back(coarseCloud.points[i]);
                nrms.push_back(coarseCloud.normals[i]);
            }
        const Engine::Spatial::OrientedPointCloud adaptive =
                Engine::Spatial::CompactDirectionalTSDF::MergeCandidates(pts, nrms, classifyVoxel);
        ev.nPoints = adaptive.points.size();

        for (const auto &p : adaptive.points) {
            const double e = double(fixtures::NearestDistance(shape, p));
            ev.overall.add(e);
            ev.perRegion[static_cast<int>(fixtures::ClassifyRegion(shape, p, classifyVoxel))]
                    .add(e);
        }
        return ev;
    }

    struct CompactAdaptiveResult {
        Row row; // representative (median-sigma) row for the main table
        std::vector<SigmaRow> sweep;
    };

    // Builds a fine (voxel) and coarse (2*voxel) CompactDirectionalTSDF from the same views,
    // then sweeps the SAME cell-variance percentiles as RunVarianceAdaptiveSweep to show the
    // combine's memory/accuracy tradeoff. `cellStats` must be pre-built via
    // BuildCellStats(fineSimpleVoxels, 2*voxel) (shared with the Simple-only sweep / hybrid
    // classifier so all rows agree on the same per-cell variance).
    CompactAdaptiveResult RunCompactVarianceAdaptive(
            Engine::Core::Context &ctx, Shape shape, float voxel, uint32_t maxDir,
            const std::vector<fixtures::View> &views,
            const std::map<CellKey, CellAgg> &cellStats) {
        CompactAdaptiveResult out;
        const float vc = 2.0f * voxel;

        Engine::Spatial::CompactDirectionalTSDF cdFine;
        Engine::Spatial::CompactDirectionalTSDF cdCoarse;
        cdFine.Build(ctx, voxel, kTruncation);
        cdFine.SetIntegrationQuality({maxDir, 4, true});
        cdCoarse.Build(ctx, vc, kTruncation);
        cdCoarse.SetIntegrationQuality({maxDir, 4, true});

        const auto t0 = std::chrono::steady_clock::now();
        for (const auto &v : views) {
            cdFine.Integrate(v.points, v.normals, v.camPos);
            cdCoarse.Integrate(v.points, v.normals, v.camPos);
        }
        const Engine::Spatial::OrientedPointCloud fineCloud = cdFine.ExtractPointCloud();
        const Engine::Spatial::OrientedPointCloud coarseCloud = cdCoarse.ExtractPointCloud();
        const auto t1 = std::chrono::steady_clock::now();
        const double buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        const std::vector<Engine::Spatial::CompactEntry> fineEntries = cdFine.DownloadEntries();
        const std::vector<Engine::Spatial::CompactEntry> coarseEntries = cdCoarse.DownloadEntries();

        const std::array<double, 4> percentiles = {0.25, 0.50, 0.75, 0.90};
        for (double p : percentiles) {
            const double sigma = PercentileOfCellVars(cellStats, p);
            const CompactAdaptiveEval ev = EvalCompactAdaptiveSigma(
                    shape, voxel, vc, sigma, cellStats, fineCloud, coarseCloud, fineEntries,
                    coarseEntries);

            SigmaRow sr;
            sr.shape = ShapeName(shape);
            sr.sigma = sigma;
            sr.coarsenPct = ev.coarsenPct;
            sr.memKB = ev.memKB;
            sr.nPoints = ev.nPoints;
            sr.overall = ev.overall;
            sr.perRegion = ev.perRegion;
            out.sweep.push_back(sr);
        }

        // Representative row for the main table: the median-sigma (0.50 percentile) sweep
        // entry, same convention as the main() loop's `midIdx` for the Simple-only sweep.
        Row row;
        row.shape = ShapeName(shape);
        row.method = "Compact-Dir(var-adaptive)";
        row.buildMs = buildMs;
        if (!out.sweep.empty()) {
            const std::size_t midIdx = std::min<std::size_t>(1, out.sweep.size() - 1);
            const SigmaRow &mid = out.sweep[midIdx];
            row.nPoints = mid.nPoints;
            row.memKB = mid.memKB;
            row.overall = mid.overall;
            row.perRegion = mid.perRegion;
        }
        out.row = row;
        return out;
    }

    void PrintDirectionalStorage(Shape shape, const DirStorageStats &s) {
        std::printf("  [directional-storage] %s: mem_block=%.2f KB  occupied_dir_voxels=%zu  "
                    "mem_pervoxel_projected=%.2f KB  waste_factor=%.1fx\n",
                    ShapeName(shape), s.memBlockKB, s.occupiedDirVoxels, s.memPerVoxelProjectedKB,
                    s.wasteFactor);
        const double pct1 = s.occupiedPositions ? 100.0 * double(s.hist1) / double(s.occupiedPositions) : 0.0;
        const double pct2 = s.occupiedPositions ? 100.0 * double(s.hist2) / double(s.occupiedPositions) : 0.0;
        const double pct3 = s.occupiedPositions ? 100.0 * double(s.hist3plus) / double(s.occupiedPositions) : 0.0;
        std::printf("  [directional-storage] %s: direction histogram over %zu occupied "
                    "positions -- 1 dir: %zu (%.1f%%)  2 dir: %zu (%.1f%%)  3+ dir: %zu "
                    "(%.1f%%)  avg=%.2f directions/occupied-voxel\n",
                    ShapeName(shape), s.occupiedPositions, s.hist1, pct1, s.hist2, pct2,
                    s.hist3plus, pct3, s.avgDirsPerVoxel);
    }

    // ---- Part 2: Adaptive-Directional hybrid (edge-only Directional, flat Simple(fine)) ----

    // outDirFrac: fraction of the classifier's occupied coarse CELLS flagged high-variance
    // ("edge") -- i.e. how much of the surface this hybrid actually routes through
    // Directional. HONEST-MEASUREMENT NOTE: the classifier's sigma is the MEDIAN (0.50
    // percentile) of the per-cell variance distribution (see main()), so by construction
    // ~50% of occupied cells are flagged high-variance regardless of geometric edge-ness --
    // this is a considerably more generous split than "true edges only", so some
    // ClassifyRegion-labeled Flat points end up sourced from Directional (pulling the
    // hybrid's measured flat_mm below plain Simple(fine)'s). Reported as measured; see
    // outDirFrac / the printed cell-split line for the actual split size.
    Row RunAdaptiveDirectional(Shape shape, float voxel, const Row &fineRow, const Row &dirRow,
                               const BuiltSimple &fine,
                               const std::vector<Engine::Spatial::VoxelStat> &fineVoxels,
                               const std::vector<Eigen::Vector3f> &dirPoints,
                               const CellClassifier &classifier, const DirStorageStats &dirStorage,
                               double &outMemFlatKB, double &outMemEdgeKB, double &outDirCellFrac) {
        Row row;
        row.shape = ShapeName(shape);
        row.method = "Adaptive-Dir";
        row.buildMs = fineRow.buildMs + dirRow.buildMs; // cost to build both source structures

        std::vector<Eigen::Vector3f> pts;
        pts.reserve(dirPoints.size() + fine.cloud.points.size());
        for (const auto &p : dirPoints)
            if (classifier.IsHighVar(p)) pts.push_back(p); // edge cells -> Directional
        for (const auto &p : fine.cloud.points)
            if (!classifier.IsHighVar(p)) pts.push_back(p); // flat cells -> Simple(fine)
        row.nPoints = pts.size();

        std::size_t flatFineVoxels = 0;
        std::size_t highVarCells = 0;
        for (const auto &kv : classifier.cellStats)
            if (kv.second.meanVar() > classifier.sigma) ++highVarCells;
        outDirCellFrac = classifier.cellStats.empty()
                                 ? 0.0
                                 : double(highVarCells) / double(classifier.cellStats.size());
        for (const auto &v : fineVoxels)
            if (!classifier.IsHighVar(v.center)) ++flatFineVoxels;
        outMemFlatKB = double(flatFineVoxels) * 16.0 / 1024.0;
        outMemEdgeKB = double(dirStorage.occupiedDirVoxelsEdge) * 8.0 / 1024.0;
        row.memKB = outMemFlatKB + outMemEdgeKB;

        ScoreAgainstGT(shape, voxel, pts, row);
        return row;
    }

    // Short prose comparing Simple(fine) vs Directional for one shape's row pair.
    void PrintInsight(const Row &simple, const Row &dir) {
        const double simpleEdge = simple.perRegion[static_cast<int>(Region::Edge)].mean();
        const double dirEdge = dir.perRegion[static_cast<int>(Region::Edge)].mean();
        const char *edgeWinner = dirEdge < simpleEdge ? "Directional" : "Simple(fine)";
        const char *memWinner = dir.memKB < simple.memKB ? "Directional" : "Simple(fine)";
        const char *speedWinner = dir.buildMs < simple.buildMs ? "Directional" : "Simple(fine)";
        const char *rmseWinner =
                dir.overall.rmse() < simple.overall.rmse() ? "Directional" : "Simple(fine)";

        std::printf("insight (%s): edge error winner=%s (Simple(fine)=%.5f Dir=%.5f mm) | "
                    "overall RMSE winner=%s (Simple(fine)=%.5f Dir=%.5f mm) | "
                    "memory winner=%s (Simple(fine)=%.2f Dir=%.2f KB) | "
                    "speed winner=%s (Simple(fine)=%.3f Dir=%.3f ms)\n",
                    simple.shape.c_str(), edgeWinner, simpleEdge, dirEdge, rmseWinner,
                    simple.overall.rmse(), dir.overall.rmse(), memWinner, simple.memKB,
                    dir.memKB, speedWinner, simple.buildMs, dir.buildMs);
    }

    // Short prose comparing the 2-level variance-adaptive sweep's representative (median-sigma)
    // row against the Simple(fine) and Simple(coarse) baselines for one shape.
    //
    // HONEST-MEASUREMENT NOTE (see also the file-header comment): for the cube fixture at the
    // default --voxel 0.1 (coarse = 0.2), Simple(coarse) extracts ZERO points -- verified via a
    // scratch diagnostic: the flat-face sample grid fixtures::SampleViews settles on (pitch
    // exactly 0.2, budget-driven, independent of voxel size) aliases exactly with the coarse
    // voxel size, so every other coarse-voxel column along each face is empty and Marching
    // Cubes' full-2x2x2-corner requirement is never satisfied anywhere. This is a genuine
    // property of this fixture/voxel-size combination, not a selection bug: the per-cell
    // variance grouping (cellStats, built straight off fine.DownloadVoxels()) is unaffected and
    // still produces a sane coarsen%/mem_KB spread (see the sigma-sweep table) -- only the
    // *coarse replacement geometry* is unavailable for the cube, so cube's "coarsened" cells in
    // the sweep are pure holes (points removed, nothing substituted), not genuine coarse
    // surface. Reported as measured; guarded below rather than hidden.
    void PrintAdaptiveInsight(const Row &fine, const Row &coarse, const SigmaRow &mid) {
        const double fineEdge = fine.perRegion[static_cast<int>(Region::Edge)].mean();
        const double midEdge = mid.perRegion[static_cast<int>(Region::Edge)].mean();

        if (coarse.overall.count == 0) {
            std::printf("insight (%s, variance-adaptive @ sigma=%.3g median, coarsen=%.1f%% of "
                        "cells): Simple(coarse) extracted 0 points here (all-coarse MC failed -- "
                        "grid/voxel aliasing, see comment above) so coarsened cells in the "
                        "adaptive sweep are holes, not real coarse geometry. fine vs adaptive "
                        "only: mem_KB fine=%.2f adaptive=%.2f | edge_mm fine=%.5f adaptive=%.5f "
                        "| RMSE_mm fine=%.5f adaptive=%.5f\n",
                        fine.shape.c_str(), mid.sigma, mid.coarsenPct, fine.memKB, mid.memKB,
                        fineEdge, midEdge, fine.overall.rmse(), mid.overall.rmse());
            return;
        }

        const double coarseEdge = coarse.perRegion[static_cast<int>(Region::Edge)].mean();
        std::printf("insight (%s, variance-adaptive @ sigma=%.3g median, coarsen=%.1f%% of "
                    "cells): mem_KB fine=%.2f coarse=%.2f adaptive=%.2f | "
                    "edge_mm fine=%.5f coarse=%.5f adaptive=%.5f | "
                    "RMSE_mm fine=%.5f coarse=%.5f adaptive=%.5f\n",
                    fine.shape.c_str(), mid.sigma, mid.coarsenPct, fine.memKB, coarse.memKB,
                    mid.memKB, fineEdge, coarseEdge, midEdge, fine.overall.rmse(),
                    coarse.overall.rmse(), mid.overall.rmse());
    }

    // Final insight (a)+(b): per-voxel projected Directional memory + waste factor, and the
    // directions-per-voxel pruning potential.
    void PrintStorageInsight(const Row &dirRow, const DirStorageStats &s) {
        std::printf("insight (%s, directional storage): mem_block=%.2f KB vs "
                    "mem_pervoxel_projected=%.2f KB -> waste factor %.1fx (Directional "
                    "accuracy is achievable at ~%.2f KB -- the %.0fx is block-granularity "
                    "waste, not fundamental) | avg %.2f directions/occupied-voxel (pruning "
                    "potential: %zu/%zu occupied positions are single-direction)\n",
                    dirRow.shape.c_str(), s.memBlockKB, s.memPerVoxelProjectedKB, s.wasteFactor,
                    s.memPerVoxelProjectedKB, s.wasteFactor, s.avgDirsPerVoxel, s.hist1,
                    s.occupiedPositions);
    }

    // Final insight (c): the Adaptive-Directional hybrid's partial result. NOTE: dirCellFrac
    // is the fraction of occupied coarse cells routed through Directional (median-sigma
    // split, ~50% by construction -- see RunAdaptiveDirectional's header comment) -- a much
    // more generous split than "edges only", so treat edge_mm/flat_mm as measured, not as a
    // clean edge-vs-flat isolation.
    void PrintHybridInsight(const Row &hybrid, const Row &dirRow, const Row &fineRow,
                            double memFlatKB, double memEdgeKB, double dirCellFrac) {
        const auto &he = hybrid.perRegion[static_cast<int>(Region::Edge)];
        const auto &hf = hybrid.perRegion[static_cast<int>(Region::Flat)];
        const auto &de = dirRow.perRegion[static_cast<int>(Region::Edge)];
        const auto &ff = fineRow.perRegion[static_cast<int>(Region::Flat)];
        std::printf("insight (%s, adaptive-directional hybrid, %.0f%% of occupied cells routed "
                    "through Directional): edge_mm=%s (Directional=%s) | flat_mm=%s "
                    "(Simple(fine)=%s) | RMSE_mm=%.5f (Directional=%.5f Simple(fine)=%.5f) | "
                    "mem_KB=%.2f (flat Simple=%.2f + edge Directional/voxel=%.2f) vs "
                    "whole-shape Directional=%.2f Simple(fine)=%.2f\n",
                    hybrid.shape.c_str(), 100.0 * dirCellFrac,
                    FmtErr(he.mean(), he.count > 0, 0).c_str(),
                    FmtErr(de.mean(), de.count > 0, 0).c_str(),
                    FmtErr(hf.mean(), hf.count > 0, 0).c_str(),
                    FmtErr(ff.mean(), ff.count > 0, 0).c_str(), hybrid.overall.rmse(),
                    dirRow.overall.rmse(), fineRow.overall.rmse(), hybrid.memKB, memFlatKB,
                    memEdgeKB, dirRow.memKB, fineRow.memKB);
    }

    // Headline insight: the two orthogonal memory axes combined. Compares the
    // Compact-Dir(var-adaptive) representative (median-sigma) row against Compact-Directional
    // (fine, Axis 1 alone), Simple(fine) (the memory bar to beat), and Directional (the
    // accuracy bar to match) -- states plainly whether "Directional accuracy at BELOW Simple
    // memory" was reached.
    void PrintCompactAdaptiveInsight(const Row &combo, const Row &compactFine,
                                     const Row &simpleFine, const Row &dirRow) {
        const bool belowSimpleMem = combo.memKB < simpleFine.memKB;
        const bool belowCompactFineMem = combo.memKB < compactFine.memKB;
        const bool nearDirAccuracy =
                combo.overall.rmse() <
                simpleFine.overall.rmse() -
                        0.5 * (simpleFine.overall.rmse() - dirRow.overall.rmse());
        std::printf("insight (%s, COMBINE: Compact-Dir x variance-adaptive): mem_KB "
                    "combo=%.2f  Compact-Directional(fine)=%.2f  Simple(fine)=%.2f  "
                    "Directional=%.2f | RMSE_mm combo=%.5f  Compact-Directional(fine)=%.5f  "
                    "Simple(fine)=%.5f  Directional=%.5f | combo is %s Simple(fine) memory, "
                    "%s Compact-Directional(fine) memory, RMSE %s Directional than Simple(fine)"
                    " -- headline (\"Directional accuracy at BELOW Simple memory\"): %s\n",
                    combo.shape.c_str(), combo.memKB, compactFine.memKB, simpleFine.memKB,
                    dirRow.memKB, combo.overall.rmse(), compactFine.overall.rmse(),
                    simpleFine.overall.rmse(), dirRow.overall.rmse(),
                    belowSimpleMem ? "BELOW" : "ABOVE",
                    belowCompactFineMem ? "below" : "above",
                    nearDirAccuracy ? "closer to" : "closer to Simple(fine), not",
                    (belowSimpleMem && nearDirAccuracy) ? "REACHED" : "NOT reached");
    }

    // Compares Compact-Directional's RAW candidate row against its merged/deduped row (and
    // Directional's point count as the target this is closing the gap toward).
    void PrintMergeInsight(const Row &raw, const Row &merged, const Row &dirRow) {
        const double pctOfRaw =
                raw.nPoints > 0 ? 100.0 * double(merged.nPoints) / double(raw.nPoints) : 0.0;
        const bool towardDirectional =
                dirRow.nPoints > 0 &&
                (merged.nPoints > dirRow.nPoints
                         ? merged.nPoints - dirRow.nPoints < raw.nPoints - dirRow.nPoints
                         : true);
        std::printf("insight (%s, Compact-Dir merge/dedup): nPoints raw=%zu merged=%zu "
                    "(%.1f%% of raw, Directional=%zu) | RMSE_mm raw=%.5f merged=%.5f | "
                    "edge_mm raw=%.5f merged=%.5f | merged nPoints moved %s Directional's count\n",
                    raw.shape.c_str(), raw.nPoints, merged.nPoints, pctOfRaw, dirRow.nPoints,
                    raw.overall.rmse(), merged.overall.rmse(),
                    raw.perRegion[static_cast<int>(Region::Edge)].mean(),
                    merged.perRegion[static_cast<int>(Region::Edge)].mean(),
                    towardDirectional ? "toward" : "away from");
    }

} // namespace

int main(int argc, char **argv) {
    float voxel = 0.1f;
    std::string shapeArg = "both";
    uint32_t maxDir = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--voxel" && i + 1 < argc) {
            voxel = std::stof(argv[++i]);
        } else if (a == "--shape" && i + 1 < argc) {
            shapeArg = argv[++i];
        } else if (a == "--maxdir" && i + 1 < argc) {
            maxDir = static_cast<uint32_t>(std::stoul(argv[++i]));
        }
    }

    std::vector<Shape> shapes;
    if (shapeArg == "cube") shapes = {Shape::Cube};
    else if (shapeArg == "cylinder") shapes = {Shape::Cylinder};
    else shapes = {Shape::Cube, Shape::Cylinder};

    std::printf("=== tsdf_benchmark  voxel=%.3f  truncation=%.3f  maxdir=%u  shapes=%s ===\n\n",
                voxel, kTruncation, maxDir, shapeArg.c_str());

    Engine::Core::Context ctx;

    std::vector<Row> rows;
    std::vector<SigmaRow> sigmaRows;
    std::vector<SigmaRow> compactSigmaRows;
    // Per-shape bundle for the trailing insight summaries.
    struct ShapeInsight {
        Row fine, coarse, dir, hybrid, compact, compactMerged, compactAdaptive;
        std::vector<SigmaRow> sweep;
        DirStorageStats storage;
        double memFlatKB = 0.0, memEdgeKB = 0.0, dirCellFrac = 0.0;
    };
    std::vector<ShapeInsight> insights;

    for (Shape shape : shapes) {
        const std::vector<fixtures::View> views = fixtures::SampleViews(shape, voxel);
        std::size_t nInput = 0;
        for (const auto &v : views) nInput += v.points.size();
        std::printf("-- %s: %zu views, %zu input points --\n", ShapeName(shape), views.size(),
                    nInput);

        // Fine SimpleTSDF (variance tracked) -- the existing Simple row, now labeled "fine" to
        // pair with the new "coarse" baseline below.
        BuiltSimple fine = BuildSimpleTSDF(ctx, voxel, views);
        Row fineRow = ScoreSimpleRow(shape, "Simple(fine)", voxel, fine);
        const std::vector<Engine::Spatial::VoxelStat> fineVoxels = fine.tsdf.DownloadVoxels();
        PrintVarianceAdaptiveSection(shape, voxel, fineVoxels);

        // Same fine voxel size, but integrated with normal view-angle weighting
        // (w = max(0, dot(normal, -rayDir))) -- isolates whether view-angle weighting alone
        // (no extra memory) recovers DirectionalTSDF's flat-region accuracy advantage.
        BuiltSimple weighted = BuildSimpleTSDFWeighted(ctx, voxel, views);
        Row weightedRow = ScoreSimpleRow(shape, "Simple(weighted)", voxel, weighted);

        // All-coarse baseline: SimpleTSDF at 2x the voxel size, integrating the SAME (fine-
        // sampled) views -- the "coarsen everywhere" lower bound.
        const float vc = 2.0f * voxel;
        BuiltSimple coarse = BuildSimpleTSDF(ctx, vc, views);
        Row coarseRow = ScoreSimpleRow(shape, "Simple(coarse)", voxel, coarse);

        // Real 2-level variance-adaptive sweep: fine voxels grouped into coarse cells, keep-
        // fine-or-coarsen decided per cell by mean variance vs sigma.
        std::vector<SigmaRow> sweep = RunVarianceAdaptiveSweep(shape, voxel, vc, fineVoxels,
                                                                fine.cloud, coarse.cloud);

        // Shared median-sigma cell classifier (Part 2's edge/flat split), built from the
        // SAME cell statistics the sigma sweep uses, at the same median (0.50) percentile as
        // its "median" row (sweep[1], see PrintAdaptiveInsight's midIdx below).
        CellClassifier classifier;
        classifier.cellStats = BuildCellStats(fineVoxels, vc);
        classifier.vc = vc;
        classifier.sigma = PercentileOfCellVars(classifier.cellStats, 0.50);

        DirectionalResult dirResult = RunDirectional(ctx, shape, voxel, maxDir, views, classifier);
        const Row &dirRow = dirResult.row;
        PrintDirectionalStorage(shape, dirResult.storage);

        // Compact-Directional: DirectionalTSDF accuracy from a per-voxel flat (voxel,dir) hash
        // at ~SimpleTSDF per-entry memory (measured, right after the Directional row). Also
        // measures the merged/deduped extraction variant (same build/integrate, clustered
        // extraction) right alongside it.
        CompactDirectionalResult compactResult =
                RunCompactDirectional(ctx, shape, voxel, maxDir, views);
        Row compactRow = compactResult.raw;
        Row compactMergedRow = compactResult.merged;

        // COMBINE: Compact-Directional's per-voxel storage (Axis 1) x variance-adaptive
        // coarsening of low-variance flat cells (Axis 2) -- reuses the SAME cellStats the
        // Simple-only sweep/hybrid classifier above use (built from fine SimpleTSDF variance),
        // applied to a fine+coarse CompactDirectionalTSDF pair instead of SimpleTSDF.
        CompactAdaptiveResult compactAdaptive =
                RunCompactVarianceAdaptive(ctx, shape, voxel, maxDir, views, classifier.cellStats);

        // Part 2: Adaptive-Directional hybrid -- Directional points in edge cells U
        // Simple(fine) points in flat cells, memory-costed the same way.
        double memFlatKB = 0.0, memEdgeKB = 0.0, dirCellFrac = 0.0;
        Row hybridRow = RunAdaptiveDirectional(shape, voxel, fineRow, dirRow, fine, fineVoxels,
                                               dirResult.points, classifier, dirResult.storage,
                                               memFlatKB, memEdgeKB, dirCellFrac);

        rows.push_back(fineRow);
        rows.push_back(weightedRow);
        rows.push_back(coarseRow);
        rows.push_back(dirRow);
        rows.push_back(compactRow);
        rows.push_back(compactMergedRow);
        rows.push_back(compactAdaptive.row);
        rows.push_back(hybridRow);
        rows.push_back(RunAdaptiveVoxelGrid(ctx, shape, voxel, views, 0.50f));
        rows.push_back(RunAdaptiveVoxelGrid(ctx, shape, voxel, views, 0.90f));
        sigmaRows.insert(sigmaRows.end(), sweep.begin(), sweep.end());
        compactSigmaRows.insert(compactSigmaRows.end(), compactAdaptive.sweep.begin(),
                                compactAdaptive.sweep.end());
        insights.push_back({fineRow, coarseRow, dirRow, hybridRow, compactRow, compactMergedRow,
                            compactAdaptive.row, sweep, dirResult.storage, memFlatKB, memEdgeKB,
                            dirCellFrac});

        std::printf("\n");
    }

    PrintHeader();
    for (const auto &r : rows) PrintRow(r);
    std::printf("\n");

    std::printf("-- variance-adaptive sigma sweep (real 2-level fine+coarse selection) --\n");
    PrintSigmaHeader();
    for (const auto &r : sigmaRows) PrintSigmaRow(r);
    std::printf("\n");

    std::printf("[caveat] the sigma-sweep rows above pick, per coarse cell, between two "
                "independently-extracted single-resolution MC meshes -- not true multi-"
                "resolution MC with transitional voxels -- so minor seams/double-coverage at "
                "fine<->coarse cell boundaries are possible. Numbers are reported as measured.\n\n");

    std::printf("-- Compact-Dir(var-adaptive) sigma sweep (COMBINE: per-voxel directional "
                "storage x variance-adaptive coarsening) --\n");
    PrintSigmaHeader();
    for (const auto &r : compactSigmaRows) PrintSigmaRow(r);
    std::printf("\n");

    std::printf("[caveat] same per-cell-selection-of-two-independently-extracted-clouds caveat "
                "as the sigma sweep above applies here too, now with Compact-Directional as both "
                "the fine and coarse source (see RunCompactVarianceAdaptive's header comment).\n\n");

    for (const auto &ins : insights) {
        PrintInsight(ins.fine, ins.dir);
        if (!ins.sweep.empty()) {
            const std::size_t midIdx = std::min<std::size_t>(1, ins.sweep.size() - 1);
            PrintAdaptiveInsight(ins.fine, ins.coarse, ins.sweep[midIdx]);
        }
        PrintStorageInsight(ins.dir, ins.storage);
        PrintHybridInsight(ins.hybrid, ins.dir, ins.fine, ins.memFlatKB, ins.memEdgeKB,
                          ins.dirCellFrac);
        PrintCompactAdaptiveInsight(ins.compactAdaptive, ins.compact, ins.fine, ins.dir);
        PrintMergeInsight(ins.compact, ins.compactMerged, ins.dir);
    }

    return 0;
}

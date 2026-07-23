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
#include "shape_fixtures.h"

#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalTSDF.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"
#include "Engine/Spatial/SimpleTSDF.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
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
        std::printf("%-9s %-15s %10s %9s %10s %10s %10s %9s %9s %9s\n", "shape", "method",
                    "build_ms", "nPoints", "mem_KB", "acc_mean", "acc_rmse", "edge", "flat",
                    "curved");
        std::printf("---------------------------------------------------------------------"
                    "---------------------------------\n");
    }

    void PrintRow(const Row &r) {
        const auto &e = r.perRegion[static_cast<int>(Region::Edge)];
        const auto &f = r.perRegion[static_cast<int>(Region::Flat)];
        const auto &c = r.perRegion[static_cast<int>(Region::Curved)];
        std::printf("%-9s %-15s %10.3f %9zu %10.2f %s %s %s %s %s\n", r.shape.c_str(),
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

    struct SigmaRow {
        std::string shape;
        double sigma = 0.0;
        double coarsenPct = 0.0;
        double memKB = 0.0;
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

        std::map<CellKey, CellAgg> cellStats;
        for (const auto &v : fineVoxels) {
            CellAgg &agg = cellStats[CellOf(v.center, vc)];
            agg.varSum += double(v.variance);
            ++agg.fineCount;
        }

        if (cellStats.empty()) {
            std::printf("  [variance-adaptive-2level] %s: no fine voxels, skipping sigma sweep\n",
                        ShapeName(shape));
            return out;
        }

        std::vector<double> cellVars;
        cellVars.reserve(cellStats.size());
        for (const auto &kv : cellStats) cellVars.push_back(kv.second.meanVar());
        std::sort(cellVars.begin(), cellVars.end());

        auto percentile = [&](double p) {
            const std::size_t idx =
                    std::min(cellVars.size() - 1, std::size_t(p * double(cellVars.size() - 1)));
            return cellVars[idx];
        };

        const std::array<double, 4> percentiles = {0.25, 0.50, 0.75, 0.90};

        for (double p : percentiles) {
            const double sigma = percentile(p);

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

    Row RunDirectional(Engine::Core::Context &ctx, Shape shape, float voxel, uint32_t maxDir,
                       const std::vector<fixtures::View> &views) {
        Row row;
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

        std::vector<Eigen::Vector3f> pts;
        pts.reserve(cloud.size());
        for (const auto &e : cloud) pts.push_back(e.position);
        ScoreAgainstGT(shape, voxel, pts, row);

        std::printf("  [directional-internal] %s: sum integrate=%.3fms extract=%.3fms "
                    "merge=%.3fms (over %zu frames)\n",
                    ShapeName(shape), sumIntegrateMs, sumExtractMs, sumMergeMs, views.size());

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
    // Per-shape bundle for the trailing insight summaries.
    struct ShapeInsight {
        Row fine, coarse, dir;
        std::vector<SigmaRow> sweep;
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

        Row dirRow = RunDirectional(ctx, shape, voxel, maxDir, views);

        rows.push_back(fineRow);
        rows.push_back(weightedRow);
        rows.push_back(coarseRow);
        rows.push_back(dirRow);
        sigmaRows.insert(sigmaRows.end(), sweep.begin(), sweep.end());
        insights.push_back({fineRow, coarseRow, dirRow, sweep});

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

    for (const auto &ins : insights) {
        PrintInsight(ins.fine, ins.dir);
        if (!ins.sweep.empty()) {
            const std::size_t midIdx = std::min<std::size_t>(1, ins.sweep.size() - 1);
            PrintAdaptiveInsight(ins.fine, ins.coarse, ins.sweep[midIdx]);
        }
    }

    return 0;
}

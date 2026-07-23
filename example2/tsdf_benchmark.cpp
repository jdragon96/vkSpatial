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
// Reports, per shape x method:
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
// Also prints DirectionalTSDF's internal integrate/extract/merge breakdown (LastFrameStats,
// summed over the view loop) as an extra insight line, and a small MrHash-style
// variance-adaptive-resolution estimate off SimpleTSDF::DownloadVoxels (bonus section).
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
        std::printf("%-9s %-12s %10s %9s %10s %10s %10s %9s %9s %9s\n", "shape", "method",
                    "build_ms", "nPoints", "mem_KB", "acc_mean", "acc_rmse", "edge", "flat",
                    "curved");
        std::printf("---------------------------------------------------------------------"
                    "-----------------------------\n");
    }

    void PrintRow(const Row &r) {
        std::printf("%-9s %-12s %10.3f %9zu %10.2f %10.5f %10.5f %9.5f %9.5f %9.5f\n",
                    r.shape.c_str(), r.method.c_str(), r.buildMs, r.nPoints, r.memKB,
                    r.overall.mean(), r.overall.rmse(),
                    r.perRegion[static_cast<int>(Region::Edge)].mean(),
                    r.perRegion[static_cast<int>(Region::Flat)].mean(),
                    r.perRegion[static_cast<int>(Region::Curved)].mean());
    }

    // ---- Bonus: MrHash-style variance-adaptive-resolution estimate off SimpleTSDF voxels ----
    // Mirrors example2/variance_adaptive_demo.cpp's premise check, condensed to one threshold.
    void PrintVarianceAdaptiveSection(Shape shape, float voxel,
                                      const std::vector<Engine::Spatial::VoxelStat> &voxels) {
        if (voxels.empty()) {
            std::printf("  [variance-adaptive] %s: no occupied voxels, skipping\n",
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

        std::printf("  [variance-adaptive] %s: occupied=%zu  edge/flat/curved mean var = "
                    "%.3g / %.3g / %.3g\n",
                    ShapeName(shape), voxels.size(),
                    regionMean(static_cast<int>(Region::Edge)),
                    regionMean(static_cast<int>(Region::Flat)),
                    regionMean(static_cast<int>(Region::Curved)));
        std::printf("  [variance-adaptive] %s: sigma=%.3g (5%% of max var) -> coarsen-able = "
                    "%zu/%zu (%.1f%%); mem %.2f KB -> projected %.2f KB (8:1 coarsen, -%.1f%%)\n",
                    ShapeName(shape), sigma, coarsenable, voxels.size(), coarsenPct, curMemKB,
                    projectedKB, 100.0 * (curMemKB - projectedKB) / curMemKB);
    }

    Row RunSimple(Engine::Core::Context &ctx, Shape shape, float voxel,
                 const std::vector<fixtures::View> &views) {
        Row row;
        row.shape = ShapeName(shape);
        row.method = "Simple";

        Engine::Spatial::SimpleTSDF simple;
        simple.Build(ctx, voxel, kTruncation);

        const auto t0 = std::chrono::steady_clock::now();
        for (const auto &v : views) simple.Integrate(v.points, v.camPos);
        const Engine::Spatial::OrientedPointCloud cloud = simple.ExtractPointCloud();
        const auto t1 = std::chrono::steady_clock::now();

        row.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        row.nPoints = cloud.points.size();
        row.memKB = double(simple.FilledCount()) * 16.0 / 1024.0;
        ScoreAgainstGT(shape, voxel, cloud.points, row);

        // Bonus section, computed off the same already-integrated instance (no extra work).
        const std::vector<Engine::Spatial::VoxelStat> voxels = simple.DownloadVoxels();
        PrintVarianceAdaptiveSection(shape, voxel, voxels);

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

    // Short prose comparing the two methods for one shape's row pair.
    void PrintInsight(const Row &simple, const Row &dir) {
        const double simpleEdge = simple.perRegion[static_cast<int>(Region::Edge)].mean();
        const double dirEdge = dir.perRegion[static_cast<int>(Region::Edge)].mean();
        const char *edgeWinner = dirEdge < simpleEdge ? "Directional" : "Simple";
        const char *memWinner = dir.memKB < simple.memKB ? "Directional" : "Simple";
        const char *speedWinner = dir.buildMs < simple.buildMs ? "Directional" : "Simple";
        const char *rmseWinner = dir.overall.rmse() < simple.overall.rmse() ? "Directional" : "Simple";

        std::printf("insight (%s): edge error winner=%s (Simple=%.5f Dir=%.5f mm) | "
                    "overall RMSE winner=%s (Simple=%.5f Dir=%.5f mm) | "
                    "memory winner=%s (Simple=%.2f Dir=%.2f KB) | "
                    "speed winner=%s (Simple=%.3f Dir=%.3f ms)\n",
                    simple.shape.c_str(), edgeWinner, simpleEdge, dirEdge, rmseWinner,
                    simple.overall.rmse(), dir.overall.rmse(), memWinner, simple.memKB,
                    dir.memKB, speedWinner, simple.buildMs, dir.buildMs);
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
    std::vector<std::pair<Row, Row>> pairs; // (Simple, Directional) per shape, for the insight summary

    for (Shape shape : shapes) {
        const std::vector<fixtures::View> views = fixtures::SampleViews(shape, voxel);
        std::size_t nInput = 0;
        for (const auto &v : views) nInput += v.points.size();
        std::printf("-- %s: %zu views, %zu input points --\n", ShapeName(shape), views.size(),
                    nInput);

        Row simpleRow = RunSimple(ctx, shape, voxel, views);
        Row dirRow = RunDirectional(ctx, shape, voxel, maxDir, views);

        rows.push_back(simpleRow);
        rows.push_back(dirRow);
        pairs.emplace_back(simpleRow, dirRow);
        std::printf("\n");
    }

    PrintHeader();
    for (const auto &r : rows) PrintRow(r);
    std::printf("\n");

    for (const auto &p : pairs) PrintInsight(p.first, p.second);

    return 0;
}

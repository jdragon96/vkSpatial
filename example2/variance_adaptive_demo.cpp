// Variance-Adaptive TSDF validation demo (MrHash foundation, walking skeleton).
//
// This is a render-free correctness/validation experiment for the paper's core premise:
//
//   De Rebotti et al., "Resolution Where It Counts: Hash-based GPU-Accelerated 3D
//   Reconstruction via Variance-Adaptive Voxel Grids", ACM TOG 2025 (see
//   docs/MRHASH_VS_DIRECTIONAL_TSDF.md for the full write-up).
//
// The paper drives per-voxel resolution purely from online TSDF variance sigma^2: keep fine
// resolution where variance is high (sharp edges/corners -- multiple observations disagree on
// the SDF because a single averaged field can't represent a crease), and coarsen where
// variance is low (flat/curved regions where repeated observations agree closely).
//
// This demo integrates the SAME analytic multi-view samples used by tsdf_feature_compare
// (example2/shape_fixtures.h) into a single SimpleTSDF, downloads every occupied voxel's
// online variance (SimpleTSDF::DownloadVoxels, backed by the new sumD2 accumulator), and:
//
//   1. Classifies every voxel by region (flat / curved / edge) and reports MEAN variance
//      per region -- the premise check. Edge variance should be clearly higher.
//   2. Sweeps a handful of variance thresholds sigma and reports what fraction of voxels
//      would be "coarsen-able" (variance <= sigma) at each -- the memory-saving payoff a
//      variance-adaptive grid would realize here. NO multi-resolution storage is implemented
//      (follow-on work); this only quantifies the potential.
//
// Usage: variance_adaptive_demo --shape cube|cylinder [--voxel <f>] [--dump]
#include "shape_fixtures.h"

#include "Engine/Core/Context.h"
#include "Engine/Spatial/SimpleTSDF.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using fixtures::Region;
using fixtures::Shape;

namespace {

    constexpr float kVoxelDefault = 0.1f;
    constexpr float kTruncation = 0.3f;

    struct RegionStat {
        double sum = 0.0;
        double maxVar = 0.0;
        std::size_t count = 0;
        void add(float v) {
            sum += double(v);
            maxVar = std::max(maxVar, double(v));
            ++count;
        }
        double mean() const { return count ? sum / double(count) : 0.0; }
    };

    // Region is {Flat, Curved, Edge}; keep the stats array's size tied to the enum so a future
    // Region addition fails to compile here instead of silently truncating/overflowing the table.
    constexpr int kNumRegions = 3;
    static_assert(kNumRegions == static_cast<int>(Region::Edge) + 1,
                  "RegionStat array size must track fixtures::Region's enumerator count");

    const char *regionName(int r) {
        switch (r) {
            case static_cast<int>(Region::Flat): return "flat";
            case static_cast<int>(Region::Curved): return "curved";
            case static_cast<int>(Region::Edge): return "edge";
            default: return "?";
        }
    }

} // namespace

int main(int argc, char **argv) {
    Shape shapeArg = Shape::Cube;
    float voxelArg = kVoxelDefault;
    bool dump = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shape" && i + 1 < argc) {
            const std::string v = argv[++i];
            shapeArg = (v == "cylinder") ? Shape::Cylinder : Shape::Cube;
        } else if (a == "--voxel" && i + 1 < argc) {
            voxelArg = std::stof(argv[++i]);
        } else if (a == "--dump") {
            dump = true;
        }
    }
    const char *shapeName = (shapeArg == Shape::Cube) ? "cube" : "cylinder";

    Engine::Core::Context ctx;

    Engine::Spatial::SimpleTSDF simple;
    simple.Build(ctx, voxelArg, kTruncation);

    const std::vector<fixtures::View> views = fixtures::SampleViews(shapeArg, voxelArg);
    for (const auto &v : views) simple.Integrate(v.points, v.camPos);

    const std::vector<Engine::Spatial::VoxelStat> voxels = simple.DownloadVoxels();

    // ---- Premise check: per-region mean variance ----
    std::array<RegionStat, kNumRegions> stats{};
    std::vector<float> allVar;
    allVar.reserve(voxels.size());
    for (const auto &vs : voxels) {
        const Region r = fixtures::ClassifyRegion(shapeArg, vs.center, voxelArg);
        stats[static_cast<int>(r)].add(vs.variance);
        allVar.push_back(vs.variance);
    }

    float maxVar = 0.0f;
    for (float v : allVar) maxVar = std::max(maxVar, v);

    // ---- Memory-saving estimate: variance threshold sweep ----
    // Thresholds spanning the observed range (fractions of the max observed variance).
    std::vector<float> thresholds;
    if (maxVar > 0.0f) {
        for (float frac : {0.01f, 0.02f, 0.05f, 0.1f, 0.2f, 0.4f})
            thresholds.push_back(frac * maxVar);
    } else {
        thresholds.push_back(0.0f);
    }

    printf("=== variance_adaptive_demo: shape=%s voxel=%.4f truncation=%.2f ===\n",
           shapeName, voxelArg, kTruncation);
    printf("occupied voxels (weight-gated) = %zu\n", voxels.size());

    if (dump) {
        printf("\n-- Per-region variance (premise: edge should be >> flat/curved) --\n");
        printf("%-8s %10s %14s %14s\n", "region", "count", "mean_var", "max_var");
        for (int r = 0; r < kNumRegions; ++r) {
            printf("%-8s %10zu %14.6g %14.6g\n", regionName(r), stats[r].count,
                   stats[r].mean(), stats[r].maxVar);
        }

        printf("\n-- Memory-saving estimate: variance threshold sweep --\n");
        printf("(keep-fine = variance > sigma;  coarsen-able = variance <= sigma)\n");
        printf("%-14s %12s %12s %14s\n", "sigma", "keep_fine", "coarsenable", "coarsen_pct");
        for (float th : thresholds) {
            std::size_t keep = 0;
            for (float v : allVar)
                if (v > th) ++keep;
            const std::size_t coarsen = allVar.size() - keep;
            const double pct = allVar.empty() ? 0.0 : 100.0 * double(coarsen) / double(allVar.size());
            printf("%-14.6g %12zu %12zu %13.1f%%\n", th, keep, coarsen, pct);
        }
    }

    const double edgeMean = stats[static_cast<int>(Region::Edge)].mean();
    const double flatMean = stats[static_cast<int>(Region::Flat)].mean();
    const double curvedMean = stats[static_cast<int>(Region::Curved)].mean();
    printf("\nPremise check: edge mean variance = %.6g | flat mean variance = %.6g | "
           "curved mean variance = %.6g\n",
           edgeMean, flatMean, curvedMean);
    if (flatMean > 0.0)
        printf("edge / flat ratio = %.2fx\n", edgeMean / flatMean);
    if (curvedMean > 0.0)
        printf("edge / curved ratio = %.2fx\n", edgeMean / curvedMean);

    return 0;
}

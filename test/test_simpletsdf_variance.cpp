// Regression test for online per-voxel TSDF variance (MrHash foundation).
//
// SimpleTSDF now accumulates sumD2 = sum(sdf^2) alongside the existing sumDW/sumW, letting
// SimpleTSDF::DownloadVoxels recover a per-voxel variance sigma^2 = E[d^2] - E[d]^2. The
// paper's premise (De Rebotti et al., ACM TOG 2025 -- see docs/MRHASH_VS_DIRECTIONAL_TSDF.md)
// is that this variance is HIGH at sharp edges/corners (multiple views disagree on the SDF
// because a single averaged field can't represent a crease) and LOW on flat regions.
//
// Reuses example2/shape_fixtures.h (render-free analytic cube/cylinder fixtures shared with
// tsdf_feature_compare and example2/variance_adaptive_demo) instead of duplicating a sampler.
#include <gtest/gtest.h>

#include "shape_fixtures.h"

#include "Engine/Core/Context.h"
#include "Engine/Spatial/SimpleTSDF.h"

#include <memory>
#include <stdexcept>
#include <vector>

using fixtures::Region;
using fixtures::Shape;

namespace {
    // Builds a Context once; skips the test if Vulkan is unavailable (mirrors
    // test_spatialIndex.cpp's CtxHolder).
    struct CtxHolder {
        std::unique_ptr<Engine::Core::Context> ctx;
        bool ok = false;
        CtxHolder() {
            try {
                ctx = std::make_unique<Engine::Core::Context>();
                ok = true;
            } catch (const std::exception &) {
                ok = false;
            }
        }
    };
} // namespace

TEST(SimpleTSDFVariance, EdgeVarianceExceedsFlatVariance) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    // Voxel size matters here: shape_fixtures.h's SampleViews uses only 8 widely-spaced
    // (cube-corner) viewpoints, so even a perfectly flat face is fused from rays up to ~55
    // degrees apart. That view-angle diversity, combined with a coarse voxel, injects
    // quantization noise into the "flat" bucket's variance (sdf == dot(p - voxelCenter,
    // rayDir), which varies with rayDir whenever the fixed offset (p - voxelCenter) is a
    // non-trivial fraction of a voxel). At the example2/variance_adaptive_demo default
    // (voxel=0.1) this shrinks the cube's edge/flat margin to ~1.8x -- present but weak.
    // Halving the voxel to 0.05 (still coarser than the demo's threshold sweep granularity,
    // and matching the paper's actual operating regime: start at FINE resolution, merge where
    // variance is low) shrinks that quantization noise faster than the genuine cross-face
    // disagreement at edges, giving a robust ~15x margin. See
    // .superpowers/sdd/variance-report.md for the full voxel-size sweep.
    constexpr float voxel = 0.05f;
    constexpr float truncation = 0.3f;

    Engine::Spatial::SimpleTSDF simple;
    simple.Build(*h.ctx, voxel, truncation);

    const std::vector<fixtures::View> views = fixtures::SampleViews(Shape::Cube, voxel);
    ASSERT_FALSE(views.empty());
    for (const auto &v : views) simple.Integrate(v.points, v.camPos);

    const std::vector<Engine::Spatial::VoxelStat> voxels = simple.DownloadVoxels();
    ASSERT_FALSE(voxels.empty());

    double edgeSum = 0.0, flatSum = 0.0;
    std::size_t edgeCount = 0, flatCount = 0;
    for (const auto &vs : voxels) {
        const Region r = fixtures::ClassifyRegion(Shape::Cube, vs.center, voxel);
        if (r == Region::Edge) {
            edgeSum += double(vs.variance);
            ++edgeCount;
        } else if (r == Region::Flat) {
            flatSum += double(vs.variance);
            ++flatCount;
        }
    }
    ASSERT_GT(edgeCount, 0u) << "no edge-region voxels observed -- fixture/classification issue";
    ASSERT_GT(flatCount, 0u) << "no flat-region voxels observed -- fixture/classification issue";

    const double edgeMean = edgeSum / double(edgeCount);
    const double flatMean = flatSum / double(flatCount);

    // The paper's premise: sharp-feature (edge) variance should clearly dominate flat-region
    // variance. 2x is a conservative margin given the demo's --dump output shows a much larger
    // separation in practice.
    EXPECT_GT(edgeMean, 2.0 * flatMean)
            << "edgeMean=" << edgeMean << " flatMean=" << flatMean;
}

// Task 1: AdaptiveVoxelGrid scaffold — composes SimpleTSDF for fine-resolution GPU
// integration. FineCount() must passthrough to SimpleTSDF::FilledCount() until Task 2
// implements the CPU variance merge (mixed-resolution grid).
//
// Task 2: CPU variance merge -> mixed-resolution grid. buildMixed() buckets fine voxels
// into 2x2x2 coarse blocks and coarsens flat/well-observed blocks; FineCount()/CoarseCount()
// now read the merged result (lazily built, cached until the next Set*/Integrate/Reset).
//
// Task 3: single-resolution CPU Marching Cubes over fine (level==0) voxels only, using MC
// tables transcribed verbatim from src/shader/voxel_common.glsl (MarchingCubesTables.h).
// CpuMcMatchesGpuMcAllFine validates the CPU MC against SimpleTSDF's GPU MC
// (ExtractPointCloud) on an all-fine grid via a one-directional Chamfer distance.
//
// Mirrors test_simpletsdf_variance.cpp's Context-skip pattern and example2/shape_fixtures.h
// reuse (see docs/superpowers/plans/2026-07-26-adaptive-voxel-grid.md, Tasks 1-3).
#include "Engine/Spatial/AdaptiveVoxelGrid.h"
#include "Engine/Spatial/SimpleTSDF.h"
#include "Engine/Core/Context.h"
#include "shape_fixtures.h"
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

using Engine::Spatial::AdaptiveVoxelGrid;

namespace {
    const float kVoxel = 0.05f, kTrunc = 0.15f;

    // Builds a Context once; skips the test if Vulkan is unavailable (mirrors
    // test_simpletsdf_variance.cpp's CtxHolder).
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

    void integrateCube(Engine::Core::Context &ctx, AdaptiveVoxelGrid &a, float voxel) {
        (void) ctx;
        for (const auto &v : fixtures::SampleViews(fixtures::Shape::Cube, voxel)) a.Integrate(v.points, v.camPos);
    }

    void integrateCubeSimple(Engine::Core::Context &ctx, Engine::Spatial::SimpleTSDF &s, float voxel) {
        (void) ctx;
        for (const auto &v : fixtures::SampleViews(fixtures::Shape::Cube, voxel)) s.Integrate(v.points, v.camPos);
    }

    // One-directional Chamfer distance: max over `from` of the nearest-neighbour distance to
    // `to` (brute-force -- cube MC vertex counts are small, so this stays fast).
    double maxNearestDist(const std::vector<Eigen::Vector3f> &from, const std::vector<Eigen::Vector3f> &to) {
        double worst = 0.0;
        for (const auto &p : from) {
            float best = std::numeric_limits<float>::max();
            for (const auto &q : to) best = std::min(best, (p - q).squaredNorm());
            worst = std::max(worst, double(std::sqrt(best)));
        }
        return worst;
    }
} // namespace

TEST(AdaptiveVoxelGrid, FineMatchesSimpleTSDF) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";
    Engine::Core::Context &ctx = *h.ctx;

    const auto views = fixtures::SampleViews(fixtures::Shape::Cube, kVoxel);
    ASSERT_FALSE(views.empty());

    Engine::Spatial::SimpleTSDF ref;
    ref.Build(ctx, kVoxel, kTrunc);
    for (const auto &v : views) ref.Integrate(v.points, v.camPos);

    AdaptiveVoxelGrid avg;
    avg.Build(ctx, kVoxel, kTrunc);
    for (const auto &v : views) avg.Integrate(v.points, v.camPos);

    // Force nothing to coarsen (theta=0 is unreachable by a non-negative variance mean)
    // so FineCount() reduces to the Task-1 fine-passthrough semantics.
    avg.SetVarianceThreshold(0.0f);
    EXPECT_EQ(avg.FineCount(), size_t(ref.FilledCount()));
}

TEST(AdaptiveVoxelGrid, CoarsensFlatKeepsMemoryLower) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";
    Engine::Core::Context &ctx = *h.ctx;

    AdaptiveVoxelGrid a;
    a.Build(ctx, kVoxel, kTrunc);
    integrateCube(ctx, a, kVoxel);
    a.SetVariancePercentile(0.6f); // coarsen the low-variance 60%

    const auto mixed = a.DownloadMixedVoxels();
    ASSERT_FALSE(mixed.empty());
    EXPECT_GT(a.CoarseCount(), 0u); // some flat blocks coarsened

    Engine::Spatial::SimpleTSDF allFine;
    allFine.Build(ctx, kVoxel, kTrunc);
    integrateCubeSimple(ctx, allFine, kVoxel);
    const size_t allFineCount = size_t(allFine.FilledCount());

    EXPECT_LT(a.FineCount() + a.CoarseCount(), allFineCount); // total voxels reduced vs all-fine
}

TEST(AdaptiveVoxelGrid, CoarseVoxelsAreFlat) { // coarse voxels have low variance by construction
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";
    Engine::Core::Context &ctx = *h.ctx;

    AdaptiveVoxelGrid a;
    a.Build(ctx, kVoxel, kTrunc);
    integrateCube(ctx, a, kVoxel);
    a.SetVariancePercentile(0.5f);

    for (const auto &mv : a.DownloadMixedVoxels())
        if (mv.level == 1) EXPECT_FLOAT_EQ(mv.size, 0.10f);
    SUCCEED();
}

TEST(AdaptiveVoxelGrid, CpuMcMatchesGpuMcAllFine) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";
    Engine::Core::Context &ctx = *h.ctx;

    AdaptiveVoxelGrid a;
    a.Build(ctx, kVoxel, kTrunc);
    integrateCube(ctx, a, kVoxel);
    a.SetVarianceThreshold(0.0f); // force all-fine (same rule as FineMatchesSimpleTSDF)
    const Engine::Spatial::AdaptiveMesh m = a.ExtractMesh();

    Engine::Spatial::SimpleTSDF s;
    s.Build(ctx, kVoxel, kTrunc);
    integrateCubeSimple(ctx, s, kVoxel);
    const auto ref = s.ExtractPointCloud().points; // GPU MC vertices

    ASSERT_FALSE(m.vertices.empty());
    ASSERT_FALSE(ref.empty());

    // One-directional Chamfer: every CPU-MC vertex has a close GPU-MC vertex.
    const double d = maxNearestDist(m.vertices, ref);
    EXPECT_LT(d, 0.05); // within one fine voxel
}

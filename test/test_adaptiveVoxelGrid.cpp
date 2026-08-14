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
#include "TSDF/Backends/SimpleTSDF.h"
#include "Engine/Core/Context.h"
#include "shape_fixtures.h"
#include <gtest/gtest.h>

#include <Eigen/Geometry> // Vector3f::cross
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

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

    // Grid-hash spot-check (Task 4 Step 1): true iff no two DISTINCT vertices lie strictly
    // closer than `eps` apart (identical/welded-together positions, distance ~0, are fine --
    // only *near*-duplicates that should have been welded but weren't are a failure).
    bool noNearDuplicates(const std::vector<Eigen::Vector3f> &verts, float eps) {
        if (verts.empty()) return true;
        const float cell = std::max(eps, 1e-6f);
        struct BinHash {
            size_t operator()(const std::array<int64_t, 3> &b) const noexcept {
                size_t h = std::hash<int64_t>()(b[0]);
                h = h * 31u + std::hash<int64_t>()(b[1]);
                h = h * 31u + std::hash<int64_t>()(b[2]);
                return h;
            }
        };
        auto binOf = [cell](const Eigen::Vector3f &p) {
            return std::array<int64_t, 3>{static_cast<int64_t>(std::floor(p.x() / cell)),
                                           static_cast<int64_t>(std::floor(p.y() / cell)),
                                           static_cast<int64_t>(std::floor(p.z() / cell))};
        };
        std::unordered_map<std::array<int64_t, 3>, std::vector<int>, BinHash> grid;
        for (int i = 0; i < static_cast<int>(verts.size()); ++i) {
            const auto b = binOf(verts[i]);
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        const std::array<int64_t, 3> nb{b[0] + dx, b[1] + dy, b[2] + dz};
                        const auto it = grid.find(nb);
                        if (it == grid.end()) continue;
                        for (int j : it->second) {
                            const float d = (verts[i] - verts[j]).norm();
                            if (d > 1e-9f && d < eps) return false;
                        }
                    }
            grid[b].push_back(i);
        }
        return true;
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

TEST(AdaptiveVoxelGrid, MeshAccuracyVsAllFine) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";
    Engine::Core::Context &ctx = *h.ctx;

    AdaptiveVoxelGrid fineOnly;
    fineOnly.Build(ctx, kVoxel, kTrunc);
    integrateCube(ctx, fineOnly, kVoxel);
    fineOnly.SetVarianceThreshold(0.0f); // force all-fine reference
    const Engine::Spatial::AdaptiveMesh refM = fineOnly.ExtractMesh();

    AdaptiveVoxelGrid adp;
    adp.Build(ctx, kVoxel, kTrunc);
    integrateCube(ctx, adp, kVoxel);
    adp.SetVariancePercentile(0.6f); // coarsen the flat majority
    const Engine::Spatial::AdaptiveMesh m = adp.ExtractMesh();

    ASSERT_GT(adp.CoarseCount(), 0u);
    ASSERT_FALSE(m.vertices.empty());
    ASSERT_FALSE(refM.vertices.empty());

    const double d = maxNearestDist(m.vertices, refM.vertices);
    EXPECT_LT(d, 0.10); // within one coarse voxel (2*kVoxel)
}

TEST(AdaptiveVoxelGrid, NoDegenerateOrDuplicateVerts) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";
    Engine::Core::Context &ctx = *h.ctx;

    AdaptiveVoxelGrid a;
    a.Build(ctx, kVoxel, kTrunc);
    integrateCube(ctx, a, kVoxel);
    a.SetVariancePercentile(0.6f);
    const Engine::Spatial::AdaptiveMesh m = a.ExtractMesh();

    ASSERT_FALSE(m.triangles.empty());
    for (const auto &t : m.triangles) { // no zero-area triangles
        const auto &A = m.vertices[t.x()];
        const auto &B = m.vertices[t.y()];
        const auto &C = m.vertices[t.z()];
        EXPECT_GT((B - A).cross(C - A).norm(), 1e-9f);
    }
    // No two welded vertices closer than the collapse epsilon (0.25*h) except identical.
    EXPECT_TRUE(noNearDuplicates(m.vertices, 0.25f * kVoxel));
}

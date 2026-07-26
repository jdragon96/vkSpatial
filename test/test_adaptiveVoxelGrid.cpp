// Task 1: AdaptiveVoxelGrid scaffold — composes SimpleTSDF for fine-resolution GPU
// integration. FineCount() must passthrough to SimpleTSDF::FilledCount() until Task 2
// implements the CPU variance merge (mixed-resolution grid).
//
// Mirrors test_simpletsdf_variance.cpp's Context-skip pattern and example2/shape_fixtures.h
// reuse (see docs/superpowers/plans/2026-07-26-adaptive-voxel-grid.md, Task 1).
#include "Engine/Spatial/AdaptiveVoxelGrid.h"
#include "Engine/Spatial/SimpleTSDF.h"
#include "Engine/Core/Context.h"
#include "shape_fixtures.h"
#include <gtest/gtest.h>

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

    EXPECT_EQ(avg.FineCount(), size_t(ref.FilledCount()));
}

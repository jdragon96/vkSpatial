#include <gtest/gtest.h>

#include "Engine/Core/Context.h"
#include "Engine/Spatial/BVHTypes.h"
#include "Engine/Spatial/BinaryLBVH.h"
#include "Engine/Spatial/SpatialIndex.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using namespace Engine::Spatial;

// Compile/shape guard for the interface header. Real backend behaviour is added
// in later tasks. This test only verifies the header is well-formed and the
// enum/params/defaults exist.
TEST(SpatialIndexInterface, EnumAndParamsExist) {
    EXPECT_EQ(BVHParams{}.maxLeafPrimitives, 4u);
    EXPECT_NE(static_cast<int>(BVHKind::BinaryLBVH),
              static_cast<int>(BVHKind::Wide));
}

namespace {
    // Builds a Context once; skips the test if Vulkan is unavailable.
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

    std::vector<PointPrim> randomPoints(uint32_t n, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(-20.0f, 20.0f);
        std::vector<PointPrim> pts(n);
        for (auto &p : pts) {
            p.x = d(rng);
            p.y = d(rng);
            p.z = d(rng);
        }
        return pts;
    }
} // namespace

TEST(BinaryLBVHTest, BuildProducesExpectedMetrics) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(512, 1);
    BinaryLBVH bvh(*h.ctx);
    bvh.Build(pts);

    EXPECT_EQ(bvh.Length(), 512u);
    EXPECT_EQ(bvh.NodeCount(), 2u * 512u - 1u);
    EXPECT_GT(bvh.MemoryBytes(), 0u);
}

TEST(BinaryLBVHTest, RejectsFewerThanTwoPrimitives) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";
    BinaryLBVH bvh(*h.ctx);
    std::vector<PointPrim> one{{0.0f, 0.0f, 0.0f}};
    EXPECT_THROW(bvh.Build(one), std::runtime_error);
}

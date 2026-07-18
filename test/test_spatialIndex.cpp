#include <gtest/gtest.h>

#include "Engine/Core/Context.h"
#include "Engine/Spatial/BVHTypes.h"
#include "Engine/Spatial/BinaryLBVH.h"
#include "Engine/Spatial/SpatialIndex.h"
#include "Engine/Spatial/WideBVH.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
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

    std::vector<uint32_t> cpuRadius(const std::vector<PointPrim> &pts,
                                    float cx, float cy, float cz, float r) {
        const float r2 = r * r;
        std::vector<uint32_t> out;
        for (uint32_t i = 0; i < pts.size(); ++i) {
            const float dx = pts[i].x - cx, dy = pts[i].y - cy, dz = pts[i].z - cz;
            if (dx * dx + dy * dy + dz * dz <= r2) out.push_back(i);
        }
        return out;
    }

    std::vector<uint32_t> cpuKNN(const std::vector<PointPrim> &pts,
                                 float cx, float cy, float cz, uint32_t k) {
        std::vector<std::pair<float, uint32_t>> d;
        d.reserve(pts.size());
        for (uint32_t i = 0; i < pts.size(); ++i) {
            const float dx = pts[i].x - cx, dy = pts[i].y - cy, dz = pts[i].z - cz;
            d.emplace_back(dx * dx + dy * dy + dz * dz, i);
        }
        std::sort(d.begin(), d.end());
        const uint32_t n = std::min<uint32_t>(k, static_cast<uint32_t>(d.size()));
        std::vector<uint32_t> out(n);
        for (uint32_t i = 0; i < n; ++i) out[i] = d[i].second;
        return out;
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

TEST(BinaryLBVHTest, RadiusMatchesCpu) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(512, 42);
    BinaryLBVH bvh(*h.ctx);
    bvh.Build(pts);

    auto gpu = bvh.RadiusSearch(1.0f, -2.0f, 0.5f, 7.5f);
    auto cpu = cpuRadius(pts, 1.0f, -2.0f, 0.5f, 7.5f);
    std::sort(gpu.begin(), gpu.end());
    EXPECT_EQ(gpu, cpu);
}

TEST(BinaryLBVHTest, KNNMatchesCpu) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(400, 99);
    BinaryLBVH bvh(*h.ctx);
    bvh.Build(pts);

    // KNN returns the correct k-nearest SET; internal order is unspecified
    // (cmd_knn.comp emits its max-heap array), so compare as sets by sorting
    // both sides by index — matching the reference test/test_bvhKNN.cpp.
    auto gpu = bvh.KNN(0.5f, -1.0f, 2.0f, 32);
    auto cpu = cpuKNN(pts, 0.5f, -1.0f, 2.0f, 32);
    std::sort(gpu.begin(), gpu.end());
    std::sort(cpu.begin(), cpu.end());
    EXPECT_EQ(gpu, cpu);
    EXPECT_THROW(bvh.KNN(0.0f, 0.0f, 0.0f, 0), std::runtime_error);
    EXPECT_THROW(bvh.KNN(0.0f, 0.0f, 0.0f, 65), std::runtime_error);
}

// ── Correctness across every registered backend ───────────────────────────────
// Implemented as a plain loop (not TEST_P): the anaconda-built GTest 1.11.0
// segfaults inside its parametrized-test machinery (null locale facet in
// num_get, EXC_BAD_ACCESS) during test discovery — this is the repo's first
// parametrized test, so that path had never been exercised. The loop gives the
// same coverage. Wide cases are appended to `cases` in Task 9.
struct BackendCase {
    BVHKind kind;
    uint32_t leaf;
    const char *label;
};

TEST(SpatialIndexBackend, RadiusAndKnnMatchCpu) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(512, 7);
    const std::vector<BackendCase> cases = {
            {BVHKind::BinaryLBVH, 0u, "BinaryLBVH"},
    };

    for (const auto &c : cases) {
        SCOPED_TRACE(c.label);
        auto idx = MakeSpatialIndex(*h.ctx, c.kind, BVHParams{c.leaf});
        idx->Build(pts);

        EXPECT_EQ(idx->Length(), 512u);
        EXPECT_GT(idx->NodeCount(), 0u);
        EXPECT_GT(idx->MemoryBytes(), 0u);

        auto gpuR = idx->RadiusSearch(1.0f, -2.0f, 0.5f, 7.5f);
        auto cpuR = cpuRadius(pts, 1.0f, -2.0f, 0.5f, 7.5f);
        std::sort(gpuR.begin(), gpuR.end());
        EXPECT_EQ(gpuR, cpuR);

        // KNN order is unspecified — compare as a set.
        auto gpuK = idx->KNN(0.5f, -1.0f, 2.0f, 16);
        auto cpuK = cpuKNN(pts, 0.5f, -1.0f, 2.0f, 16);
        std::sort(gpuK.begin(), gpuK.end());
        std::sort(cpuK.begin(), cpuK.end());
        EXPECT_EQ(gpuK, cpuK);
    }
}

TEST(EngineWideBVHTest, BuildProducesFewerNodesThanBinary) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(1024, 7);
    WideBVH bvh(*h.ctx, 4);
    bvh.Build(pts);

    EXPECT_EQ(bvh.Length(), 1024u);
    EXPECT_EQ(bvh.MaxLeafPrimitives(), 4u);
    EXPECT_GT(bvh.NodeCount(), 0u);
    EXPECT_LT(bvh.NodeCount(), 2u * 1024u - 1u);
    EXPECT_GT(bvh.MemoryBytes(), 0u);
}

TEST(EngineWideBVHTest, RejectsInvalidLeafSize) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";
    EXPECT_THROW(WideBVH(*h.ctx, 0), std::runtime_error);
    EXPECT_THROW(WideBVH(*h.ctx, 65), std::runtime_error);
}

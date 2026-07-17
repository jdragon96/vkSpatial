#include <gtest/gtest.h>

#include "vkCommon/vkContext.h"
#include "vkSpatial/vkWideBVH.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

using namespace vkCommon;
using namespace vkSpatial;

namespace {

    std::vector<uint32_t> cpuRadiusSearch(
            const std::vector<PointPrim> &points,
            float cx, float cy, float cz, float radius) {
        const float radius2 = radius * radius;
        std::vector<uint32_t> result;
        for (uint32_t i = 0; i < static_cast<uint32_t>(points.size()); ++i) {
            const float dx = points[i].x - cx;
            const float dy = points[i].y - cy;
            const float dz = points[i].z - cz;
            if (dx * dx + dy * dy + dz * dz <= radius2)
                result.push_back(i);
        }
        return result;
    }

    std::vector<uint32_t> cpuKNN(
            const std::vector<PointPrim> &points,
            float cx, float cy, float cz, uint32_t k) {
        std::vector<std::pair<float, uint32_t>> distances;
        distances.reserve(points.size());
        for (uint32_t i = 0; i < static_cast<uint32_t>(points.size()); ++i) {
            const float dx = points[i].x - cx;
            const float dy = points[i].y - cy;
            const float dz = points[i].z - cz;
            distances.emplace_back(dx * dx + dy * dy + dz * dz, i);
        }
        std::sort(distances.begin(), distances.end());

        const uint32_t resultCount =
                std::min(k, static_cast<uint32_t>(distances.size()));
        std::vector<uint32_t> result;
        result.reserve(resultCount);
        for (uint32_t i = 0; i < resultCount; ++i)
            result.push_back(distances[i].second);
        return result;
    }

    class WideBVHTest : public ::testing::Test {
    protected:
        void SetUp() override {
            try {
                ctx.init();
                initialized = true;
            } catch (const std::exception &e) {
                GTEST_SKIP() << "Vulkan context unavailable: " << e.what();
            }
        }

        void TearDown() override {
            if (initialized)
                ctx.shutdown();
        }

        VkContext ctx;
        bool initialized = false;
    };

} // namespace

TEST(WideBVHValidationTest, RejectsNullContext) {
    EXPECT_THROW(vkWideBVH(nullptr), std::runtime_error);
}

TEST(WideBVHValidationTest, RejectsInvalidLeafSize) {
    VkContext ctx;
    EXPECT_THROW(vkWideBVH(&ctx, 0), std::runtime_error);
    EXPECT_THROW(vkWideBVH(&ctx, 65), std::runtime_error);
}

TEST(WideBVHValidationTest, RejectsInvalidPrimitiveBoundsBeforeGpuWork) {
    VkContext ctx;
    vkWideBVH bvh(&ctx);

    std::vector<Primitive> invalidMinMax = {
            {0u, 2.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}};
    EXPECT_THROW(bvh.Build(invalidMinMax), std::runtime_error);

    std::vector<Primitive> invalidFinite = {
            {0u,
             0.0f,
             0.0f,
             0.0f,
             std::numeric_limits<float>::quiet_NaN(),
             1.0f,
             1.0f}};
    EXPECT_THROW(bvh.Build(invalidFinite), std::runtime_error);
}

TEST_F(WideBVHTest, SupportsSinglePrimitive) {
    const std::vector<PointPrim> points{{1.0f, 2.0f, 3.0f}};

    vkWideBVH bvh(&ctx);
    bvh.Build(points);

    EXPECT_EQ(bvh.Length(), 1u);
    EXPECT_EQ(bvh.NodeCount(), 1u);
    EXPECT_EQ(bvh.RadiusSearch(1.0f, 2.0f, 3.0f, 0.0f),
              std::vector<uint32_t>({0u}));
    EXPECT_EQ(bvh.KNN(0.0f, 0.0f, 0.0f, 4),
              std::vector<uint32_t>({0u}));
}

TEST_F(WideBVHTest, RadiusMatchesCpuReference) {
    constexpr uint32_t N = 512;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-20.0f, 20.0f);

    std::vector<PointPrim> points(N);
    for (PointPrim &point: points) {
        point.x = distribution(rng);
        point.y = distribution(rng);
        point.z = distribution(rng);
    }

    vkWideBVH bvh(&ctx);
    bvh.Build(points);

    auto gpuResult = bvh.RadiusSearch(1.0f, -2.0f, 0.5f, 7.5f);
    auto cpuResult = cpuRadiusSearch(points, 1.0f, -2.0f, 0.5f, 7.5f);
    std::sort(gpuResult.begin(), gpuResult.end());

    EXPECT_EQ(gpuResult, cpuResult);
}

TEST_F(WideBVHTest, KNNMatchesCpuReferenceInDistanceOrder) {
    constexpr uint32_t N = 400;
    constexpr uint32_t K = 32;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);

    std::vector<PointPrim> points(N);
    for (PointPrim &point: points) {
        point.x = distribution(rng);
        point.y = distribution(rng);
        point.z = distribution(rng);
    }

    vkWideBVH bvh(&ctx);
    bvh.Build(points);

    const auto gpuResult = bvh.KNN(0.5f, -1.0f, 2.0f, K);
    const auto cpuResult = cpuKNN(points, 0.5f, -1.0f, 2.0f, K);
    EXPECT_EQ(gpuResult, cpuResult);
}

TEST_F(WideBVHTest, UsesFewerNodesThanBinaryLayout) {
    constexpr uint32_t N = 1024;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    std::vector<PointPrim> points(N);
    for (PointPrim &point: points) {
        point.x = distribution(rng);
        point.y = distribution(rng);
        point.z = distribution(rng);
    }

    vkWideBVH bvh(&ctx, 4);
    bvh.Build(points);

    EXPECT_LT(bvh.NodeCount(), 2u * N - 1u);
    EXPECT_EQ(bvh.MaxLeafPrimitives(), 4u);
}

#include <gtest/gtest.h>

#include "vkBVH/common/vkContext.h"
#include "vkBVH/types.h"
#include "vkBVH/vkBVH.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#ifndef VKBVH_SHADER_DIR
#define VKBVH_SHADER_DIR "."
#endif

// CPU brute-force: PointPrim일 때 구-점 교차 (min==max인 AABB)
static std::vector<uint32_t> cpuRadiusSearch(const std::vector<PointPrim> &points,
                                             float cx, float cy, float cz, float r) {
    std::vector<uint32_t> result;
    const float r2 = r * r;
    for (uint32_t i = 0; i < static_cast<uint32_t>(points.size()); i++) {
        float dx = points[i].x - cx;
        float dy = points[i].y - cy;
        float dz = points[i].z - cz;
        if (dx * dx + dy * dy + dz * dz <= r2)
            result.push_back(i);
    }
    return result;
}

class RadiusSearchTest : public ::testing::Test {
protected:
    void SetUp() override { ctx.init(); }
    void TearDown() override { ctx.shutdown(); }
    VkContext ctx;
};

// [1] 반지름 내에 포인트가 없는 경우
TEST_F(RadiusSearchTest, NoPointsInRadius) {
    std::vector<PointPrim> points = {
            {10.f, 10.f, 10.f},
            {20.f, 20.f, 20.f},
    };
    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    auto result = bvh.RadiusSearch(0.f, 0.f, 0.f, 1.f);
    std::cout << "[NoPointsInRadius] result: " << result.size() << std::endl;
    EXPECT_TRUE(result.empty());
}

// [2] 모든 포인트가 반지름 내에 있는 경우
TEST_F(RadiusSearchTest, AllPointsInRadius) {
    std::vector<PointPrim> points = {
            {0.1f, 0.0f, 0.0f},
            {-0.1f, 0.0f, 0.0f},
            {0.0f, 0.1f, 0.0f},
            {0.0f, -0.1f, 0.0f},
    };
    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    auto result = bvh.RadiusSearch(0.f, 0.f, 0.f, 1.f);
    ASSERT_EQ(result.size(), points.size());

    std::cout << "[AllPointsInRadius] result: " << result.size() << std::endl;
    std::sort(result.begin(), result.end());
    for (uint32_t i = 0; i < static_cast<uint32_t>(points.size()); i++)
        EXPECT_EQ(result[i], i);
}

// [3] 일부 포인트만 반지름 내에 있는 경우 — CPU 레퍼런스와 비교
TEST_F(RadiusSearchTest, PartialPointsMatchCpuReference) {
    std::vector<PointPrim> points = {
            {0.5f, 0.0f, 0.0f}, // in  (dist=0.50)
            {5.0f, 0.0f, 0.0f}, // out
            {0.0f, 0.8f, 0.0f}, // in  (dist=0.80)
            {0.0f, 3.0f, 0.0f}, // out
            {0.7f, 0.7f, 0.0f}, // in  (dist≈0.99)
            {1.5f, 1.5f, 1.5f}, // out
    };
    const float cx = 0.f, cy = 0.f, cz = 0.f, r = 1.0f;

    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    auto gpuResult = bvh.RadiusSearch(cx, cy, cz, r);
    auto cpuResult = cpuRadiusSearch(points, cx, cy, cz, r);

    std::sort(gpuResult.begin(), gpuResult.end());
    std::sort(cpuResult.begin(), cpuResult.end());

    ASSERT_EQ(gpuResult.size(), cpuResult.size());
    for (size_t i = 0; i < cpuResult.size(); i++)
        EXPECT_EQ(gpuResult[i], cpuResult[i]) << "mismatch at pos " << i;
}

// [4] X축 위 정렬된 포인트 — 특정 반지름으로 정확한 인덱스 검출
TEST_F(RadiusSearchTest, AxisAlignedPointsCorrectSet) {
    // index i → position (i+1, 0, 0), dist from origin = i+1
    std::vector<PointPrim> points = {
            {1.f, 0.f, 0.f}, // dist=1
            {2.f, 0.f, 0.f}, // dist=2
            {3.f, 0.f, 0.f}, // dist=3
            {4.f, 0.f, 0.f}, // dist=4
            {5.f, 0.f, 0.f}, // dist=5
    };
    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    // r=3.5 → indices {0, 1, 2} 검출
    auto result = bvh.RadiusSearch(0.f, 0.f, 0.f, 3.5f);
    std::sort(result.begin(), result.end());
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 0u);
    EXPECT_EQ(result[1], 1u);
    EXPECT_EQ(result[2], 2u);
}

// [5] 랜덤 포인트 집합에서 CPU 브루트포스와 완전 일치 확인
TEST_F(RadiusSearchTest, RandomPointsMatchCpuBruteForce) {
    const uint32_t N = 500;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.f, 10.f);

    std::vector<PointPrim> points(N);
    for (auto &p: points) {
        p.x = dist(rng);
        p.y = dist(rng);
        p.z = dist(rng);
    }

    const float cx = 1.f, cy = -1.f, cz = 0.5f, r = 3.f;

    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    auto gpuResult = bvh.RadiusSearch(cx, cy, cz, r);
    auto cpuResult = cpuRadiusSearch(points, cx, cy, cz, r);

    std::sort(gpuResult.begin(), gpuResult.end());
    std::sort(cpuResult.begin(), cpuResult.end());

    ASSERT_EQ(gpuResult.size(), cpuResult.size()) << "count mismatch";
    for (size_t i = 0; i < cpuResult.size(); i++)
        EXPECT_EQ(gpuResult[i], cpuResult[i]) << "index mismatch at pos " << i;
}

// [6] 동일한 BVH에 대해 반지름을 늘릴수록 결과 수가 단조 증가
TEST_F(RadiusSearchTest, LargerRadiusReturnsMorePoints) {
    const uint32_t N = 200;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);

    std::vector<PointPrim> points(N);
    for (auto &p: points) {
        p.x = dist(rng);
        p.y = dist(rng);
        p.z = dist(rng);
    }

    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    const float cx = 0.f, cy = 0.f, cz = 0.f;
    const std::vector<float> radii = {0.5f, 1.0f, 2.0f, 4.0f, 8.0f};

    size_t prevCount = 0;
    for (float r: radii) {
        auto result = bvh.RadiusSearch(cx, cy, cz, r);
        EXPECT_GE(result.size(), prevCount)
                << "반지름 " << r << " 에서 이전보다 결과 수가 줄었습니다";
        prevCount = result.size();
    }
}

// [7] 여러 반지름 각각에서 CPU 레퍼런스와 일치
TEST_F(RadiusSearchTest, MultipleRadiiEachMatchCpu) {
    const uint32_t N = 150;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-8.f, 8.f);

    std::vector<PointPrim> points(N);
    for (auto &p: points) {
        p.x = dist(rng);
        p.y = dist(rng);
        p.z = dist(rng);
    }

    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    const float cx = 0.5f, cy = -0.5f, cz = 1.0f;
    for (float r: {0.5f, 1.5f, 3.0f, 6.0f}) {
        auto gpuResult = bvh.RadiusSearch(cx, cy, cz, r);
        auto cpuResult = cpuRadiusSearch(points, cx, cy, cz, r);

        std::sort(gpuResult.begin(), gpuResult.end());
        std::sort(cpuResult.begin(), cpuResult.end());

        EXPECT_EQ(gpuResult.size(), cpuResult.size())
                << "count mismatch at r=" << r;
        if (gpuResult.size() == cpuResult.size()) {
            for (size_t i = 0; i < cpuResult.size(); i++)
                EXPECT_EQ(gpuResult[i], cpuResult[i])
                        << "index mismatch at pos=" << i << " r=" << r;
        }
    }
}

// [8] 쿼리 중심에 정확히 위치한 포인트는 반드시 검출
TEST_F(RadiusSearchTest, PointAtQueryCenterAlwaysFound) {
    std::vector<PointPrim> points = {
            {0.f, 0.f, 0.f},   // 쿼리 중심에 위치
            {100.f, 0.f, 0.f}, // 멀리 있는 더미
    };
    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    for (float r: {0.0f, 0.001f, 1.0f}) {
        auto result = bvh.RadiusSearch(0.f, 0.f, 0.f, r);
        auto it = std::find(result.begin(), result.end(), 0u);
        EXPECT_NE(it, result.end())
                << "center point not found at r=" << r;
    }
}

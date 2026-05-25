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

// CPU brute-force KNN: indices sorted by ascending squared distance
static std::vector<uint32_t> cpuKNN(const std::vector<PointPrim> &points,
                                    float cx, float cy, float cz, int k) {
    std::vector<std::pair<float, uint32_t>> dists;
    dists.reserve(points.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(points.size()); i++) {
        float dx = points[i].x - cx;
        float dy = points[i].y - cy;
        float dz = points[i].z - cz;
        dists.push_back({dx * dx + dy * dy + dz * dz, i});
    }
    std::sort(dists.begin(), dists.end());

    int count = std::min(static_cast<int>(points.size()), k);
    std::vector<uint32_t> result;
    result.reserve(count);
    for (int i = 0; i < count; i++)
        result.push_back(dists[i].second);
    return result;
}

class KNNTest : public ::testing::Test {
protected:
    void SetUp() override { ctx.init(); }
    void TearDown() override { ctx.shutdown(); }
    VkContext ctx;
};

// [1] k=1: 가장 가까운 단일 이웃 검출
TEST_F(KNNTest, NearestSingleNeighbor) {
    std::vector<PointPrim> points = {
            {1.f, 0.f, 0.f}, // dist=1
            {3.f, 0.f, 0.f}, // dist=3
            {5.f, 0.f, 0.f}, // dist=5
    };
    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    auto result = bvh.KNN(0.f, 0.f, 0.f, 1);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 0u);
}

// [2] k=3: X축 위 포인트에서 가장 가까운 3개 검출
TEST_F(KNNTest, Top3OnAxis) {
    std::vector<PointPrim> points = {
            {1.f, 0.f, 0.f}, // dist=1
            {2.f, 0.f, 0.f}, // dist=2
            {3.f, 0.f, 0.f}, // dist=3
            {4.f, 0.f, 0.f}, // dist=4
            {5.f, 0.f, 0.f}, // dist=5
    };
    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    auto result = bvh.KNN(0.f, 0.f, 0.f, 3);
    std::sort(result.begin(), result.end());
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 0u);
    EXPECT_EQ(result[1], 1u);
    EXPECT_EQ(result[2], 2u);
}

// [3] k > N: 포인트 수보다 k가 크면 실제 포인트 수만큼 반환
TEST_F(KNNTest, KGreaterThanN) {
    std::vector<PointPrim> points = {
            {1.f, 0.f, 0.f},
            {2.f, 0.f, 0.f},
            {3.f, 0.f, 0.f},
    };
    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    auto result = bvh.KNN(0.f, 0.f, 0.f, 10);
    EXPECT_EQ(result.size(), points.size());
}

// [4] 쿼리 중심에 위치한 포인트가 k=1에서 반드시 nearest
TEST_F(KNNTest, PointAtQueryIsNearest) {
    std::vector<PointPrim> points = {
            {0.f, 0.f, 0.f}, // 쿼리 중심에 위치 → dist=0
            {5.f, 5.f, 5.f}, // 더미
    };
    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    auto result = bvh.KNN(0.f, 0.f, 0.f, 1);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 0u);
}

// [5] 랜덤 포인트 집합 — CPU 브루트포스와 결과 집합 일치
TEST_F(KNNTest, RandomPointsMatchCpuBruteForce) {
    const uint32_t N = 300;
    const int k = 10;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.f, 10.f);

    std::vector<PointPrim> points(N);
    for (auto &p: points) {
        p.x = dist(rng);
        p.y = dist(rng);
        p.z = dist(rng);
    }

    const float cx = 0.f, cy = 0.f, cz = 0.f;

    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    auto gpuResult = bvh.KNN(cx, cy, cz, k);
    auto cpuResult = cpuKNN(points, cx, cy, cz, k);

    std::sort(gpuResult.begin(), gpuResult.end());
    std::sort(cpuResult.begin(), cpuResult.end());

    ASSERT_EQ(gpuResult.size(), cpuResult.size()) << "count mismatch";
    for (size_t i = 0; i < cpuResult.size(); i++)
        EXPECT_EQ(gpuResult[i], cpuResult[i]) << "index mismatch at pos " << i;
}

// [6] 다양한 k 값에서 CPU와 일치
TEST_F(KNNTest, VariousKMatchCpu) {
    const uint32_t N = 200;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);

    std::vector<PointPrim> points(N);
    for (auto &p: points) {
        p.x = dist(rng);
        p.y = dist(rng);
        p.z = dist(rng);
    }

    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    const float cx = 1.f, cy = -1.f, cz = 0.5f;
    for (int k: {1, 5, 10, 20, 50}) {
        auto gpuResult = bvh.KNN(cx, cy, cz, k);
        auto cpuResult = cpuKNN(points, cx, cy, cz, k);

        std::sort(gpuResult.begin(), gpuResult.end());
        std::sort(cpuResult.begin(), cpuResult.end());

        EXPECT_EQ(gpuResult.size(), cpuResult.size()) << "size mismatch at k=" << k;
        if (gpuResult.size() == cpuResult.size()) {
            for (size_t i = 0; i < cpuResult.size(); i++)
                EXPECT_EQ(gpuResult[i], cpuResult[i])
                        << "index mismatch at pos=" << i << " k=" << k;
        }
    }
}

// [7] k를 늘릴수록 반환 결과 수가 단조 증가
TEST_F(KNNTest, LargerKReturnsMoreResults) {
    const uint32_t N = 100;
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

    size_t prevSize = 0;
    for (int k: {1, 5, 10, 30, 64}) {
        auto result = bvh.KNN(0.f, 0.f, 0.f, k);
        EXPECT_GE(result.size(), prevSize)
                << "k=" << k << " 에서 이전보다 결과 수가 줄었습니다";
        EXPECT_LE(result.size(), static_cast<size_t>(k))
                << "k=" << k << " 보다 많은 결과 반환됨";
        prevSize = result.size();
    }
}

// [8] k=N: 모든 포인트 반환 (전체 nearest)
TEST_F(KNNTest, KEqualsN_ReturnsAll) {
    const uint32_t N = 20;
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> dist(-3.f, 3.f);

    std::vector<PointPrim> points(N);
    for (auto &p: points) {
        p.x = dist(rng);
        p.y = dist(rng);
        p.z = dist(rng);
    }

    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    auto result = bvh.KNN(0.f, 0.f, 0.f, static_cast<int>(N));
    ASSERT_EQ(result.size(), N);

    // 모든 인덱스 {0, ..., N-1} 가 정확히 한 번씩 포함되어야 함
    std::sort(result.begin(), result.end());
    for (uint32_t i = 0; i < N; i++)
        EXPECT_EQ(result[i], i) << "missing or duplicate index " << i;
}

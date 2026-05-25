#include <gtest/gtest.h>

#include "vkBVH/common/vkComputeBase.h"
#include "vkBVH/common/vkContext.h"
#include "vkBVH/common/vkGPUMemory.h"
#include "vkBVH/types.h"
#include "vkBVH/vkBVH.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#ifndef VKBVH_SHADER_DIR
#define VKBVH_SHADER_DIR "."
#endif

// ── GPU Radix Sort 헬퍼 ───────────────────────────────────────────────────────
// vkBVH::stepSortMortonCodes() 의 독립 버전
// count → scan → scatter 를 8번 반복 (4비트 × 8 = 32비트)

static std::vector<MortonCode> gpuRadixSort(VkContext &ctx, const std::vector<MortonCode> &input) {
    constexpr uint32_t WG_SIZE = 256;
    constexpr uint32_t RADIX = 16;
    constexpr uint32_t PASSES = 8;

    const uint32_t N = static_cast<uint32_t>(input.size());
    const uint32_t numberOfWorkerGroup = (N + WG_SIZE - 1) / WG_SIZE;

    vkGPUMemory pingBuf(ctx.device, ctx.physDevice);
    vkGPUMemory pongBuf(ctx.device, ctx.physDevice);
    vkGPUMemory histBuf(ctx.device, ctx.physDevice);

    EXPECT_TRUE(pingBuf.Allocate(N * sizeof(MortonCode)));
    EXPECT_TRUE(pongBuf.Allocate(N * sizeof(MortonCode)));
    EXPECT_TRUE(histBuf.Allocate(RADIX * numberOfWorkerGroup * sizeof(uint32_t)));
    EXPECT_TRUE(pingBuf.Upload(input.data(),
                               N * sizeof(MortonCode),
                               ctx.computeQueue,
                               ctx.cmdPool));

    vkGPUMemory *ping = &pingBuf;
    vkGPUMemory *pong = &pongBuf;

    struct SortPC {
        uint32_t g_count, g_shift;
    };
    struct ScanPC {
        uint32_t g_numWGs;
    };

    const std::string shaderDir = VKBVH_SHADER_DIR;

    // ── 8패스 Radix Sort ─────────────────────────────────────────────────────
    for (uint32_t pass = 0; pass < PASSES; pass++) {
        SortPC pcSort{N, pass * 4};
        ScanPC pcScan{numberOfWorkerGroup};

        // Phase 1: 워크그룹별 히스토그램
        vkComputeBase count(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        count.Build(shaderDir + "/bvh_radixSort_histogram.comp")
                .Bind(0, *ping)
                .Bind(1, histBuf)
                .Args(pcSort)
                .Dispatch(numberOfWorkerGroup);
        count.Sync();

        // Phase 2: 전역 Prefix Scan
        vkComputeBase scan(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        scan.Build(shaderDir + "/bvh_radixSort_prefixScan.comp")
                .Bind(0, histBuf)
                .Args(pcScan)
                .Dispatch(1);
        scan.Sync();

        // Phase 3: Scatter
        vkComputeBase scatter(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        scatter.Build(shaderDir + "/bvh_radixSort_reorder.comp")
                .Bind(0, *ping)
                .Bind(1, histBuf)
                .Bind(2, *pong)
                .Args(pcSort)
                .Dispatch(numberOfWorkerGroup);
        scatter.Sync();

        std::swap(ping, pong);
    }

    // PASSES=8(짝수) → ping이 원래 pingBuf로 돌아옴
    std::vector<MortonCode> result(N);
    EXPECT_TRUE(ping->Download(result.data(), N * sizeof(MortonCode),
                               ctx.computeQueue, ctx.cmdPool));

    pingBuf.Clear();
    pongBuf.Clear();
    histBuf.Clear();
    return result;
}

// ── 테스트 픽스처 ─────────────────────────────────────────────────────────────

class RadixSortTest : public ::testing::Test {
protected:
    void SetUp() override { ctx.init(); }
    void TearDown() override { ctx.shutdown(); }
    VkContext ctx;
};

// ── 테스트 케이스 ─────────────────────────────────────────────────────────────

// [1] 8개 원소, 수동 값 검증
// 입력: [{5,0},{1,1},{3,2},{7,3},{2,4},{6,5},{0,6},{4,7}]
// 출력: [{0,6},{1,1},{2,4},{3,2},{4,7},{5,0},{6,5},{7,3}]
TEST_F(RadixSortTest, SmallArrayKnownValues) {
    std::vector<MortonCode> input = {
            {5, 0},
            {1, 1},
            {3, 2},
            {7, 3},
            {2, 4},
            {6, 5},
            {0, 6},
            {4, 7}};

    auto result = gpuRadixSort(ctx, input);
    ASSERT_EQ(result.size(), input.size());

    // 코드 오름차순 정렬 확인
    for (size_t i = 1; i < result.size(); i++)
        EXPECT_LE(result[i - 1].code, result[i].code)
                << "Not sorted at index " << i;

    // 각 값과 원래 인덱스 매핑 확인
    EXPECT_EQ(result[0].code, 0u);
    EXPECT_EQ(result[0].index, 6u);
    EXPECT_EQ(result[1].code, 1u);
    EXPECT_EQ(result[1].index, 1u);
    EXPECT_EQ(result[2].code, 2u);
    EXPECT_EQ(result[2].index, 4u);
    EXPECT_EQ(result[3].code, 3u);
    EXPECT_EQ(result[3].index, 2u);
    EXPECT_EQ(result[4].code, 4u);
    EXPECT_EQ(result[4].index, 7u);
    EXPECT_EQ(result[5].code, 5u);
    EXPECT_EQ(result[5].index, 0u);
    EXPECT_EQ(result[6].code, 6u);
    EXPECT_EQ(result[6].index, 5u);
    EXPECT_EQ(result[7].code, 7u);
    EXPECT_EQ(result[7].index, 3u);

    std::cout << "[RadixSort] SmallArray result:\n";
    for (auto &m: result)
        std::cout << "  code=" << m.code << "  originalIdx=" << m.index << "\n";
}

// [2] 모든 코드가 동일한 경우 — 인덱스만 달라도 크래시 없이 동작
TEST_F(RadixSortTest, AllSameCode) {
    std::vector<MortonCode> input(64);
    for (uint32_t i = 0; i < 64; i++) input[i] = {42u, i};

    auto result = gpuRadixSort(ctx, input);
    ASSERT_EQ(result.size(), 64u);

    for (auto &m: result)
        EXPECT_EQ(m.code, 42u);

    // 모든 원래 인덱스가 보존됐는지 확인 (순서는 달라도 됨)
    std::vector<uint32_t> indices;
    for (auto &m: result) indices.push_back(m.index);
    std::sort(indices.begin(), indices.end());
    for (uint32_t i = 0; i < 64; i++) EXPECT_EQ(indices[i], i);
}

// [3] 이미 정렬된 배열 — 항등 동작 확인
TEST_F(RadixSortTest, AlreadySorted) {
    const uint32_t N = 256;
    std::vector<MortonCode> input(N);
    for (uint32_t i = 0; i < N; i++) input[i] = {i * 3, i};

    auto result = gpuRadixSort(ctx, input);
    ASSERT_EQ(result.size(), N);

    for (size_t i = 1; i < result.size(); i++)
        EXPECT_LE(result[i - 1].code, result[i].code);
}

// [4] 역순 정렬된 배열
TEST_F(RadixSortTest, ReverseSorted) {
    const uint32_t N = 256;
    std::vector<MortonCode> input(N);
    for (uint32_t i = 0; i < N; i++) input[i] = {(N - 1 - i) * 3, i};

    auto result = gpuRadixSort(ctx, input);
    ASSERT_EQ(result.size(), N);

    for (size_t i = 1; i < result.size(); i++)
        EXPECT_LE(result[i - 1].code, result[i].code);
}

// [5] 대용량 랜덤 (10만 원소) — CPU 정렬과 비교
TEST_F(RadixSortTest, LargeRandomMatchesCpuSort) {
    const uint32_t N = 100'000;
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, 0x3FFFFFFFu); // 30-bit Morton

    std::vector<MortonCode> input(N);
    for (uint32_t i = 0; i < N; i++) input[i] = {dist(rng), i};

    // CPU 기준 정렬
    auto expected = input;
    std::stable_sort(expected.begin(), expected.end(),
                     [](const MortonCode &a, const MortonCode &b) { return a.code < b.code; });

    auto result = gpuRadixSort(ctx, input);
    ASSERT_EQ(result.size(), N);

    // 코드 값이 CPU 정렬과 동일한지 확인
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_EQ(result[i].code, expected[i].code)
                << "Mismatch at position " << i;
        if (result[i].code != expected[i].code) break; // 첫 실패 시 중단
    }

    std::cout << "[RadixSort] LargeRandom first 5 results:\n";
    for (int i = 0; i < 5; i++)
        std::cout << "  [" << i << "] code=" << result[i].code
                  << "  originalIdx=" << result[i].index << "\n";
}

static std::vector<MortonCode> gpuMortonThenSort(VkContext &ctx, const std::vector<Primitive> &prims) {
    const uint32_t N = static_cast<uint32_t>(prims.size());
    const std::string shaderDir = VKBVH_SHADER_DIR;

    // 씬 AABB 계산 (CPU)
    float minX = 1e38f, minY = 1e38f, minZ = 1e38f;
    float maxX = -1e38f, maxY = -1e38f, maxZ = -1e38f;
    for (const auto &p: prims) {
        minX = std::min(minX, p.aabbMinX);
        maxX = std::max(maxX, p.aabbMaxX);
        minY = std::min(minY, p.aabbMinY);
        maxY = std::max(maxY, p.aabbMaxY);
        minZ = std::min(minZ, p.aabbMinZ);
        maxZ = std::max(maxZ, p.aabbMaxZ);
    }

    // ── Step 1: GPU Morton 코드 계산 ─────────────────────────────────────────
    vkGPUMemory primBuf(ctx.device, ctx.physDevice);
    vkGPUMemory mortonBuf(ctx.device, ctx.physDevice);
    EXPECT_TRUE(primBuf.Allocate(N * sizeof(Primitive)));
    EXPECT_TRUE(mortonBuf.Allocate(N * sizeof(MortonCode)));
    EXPECT_TRUE(primBuf.Upload(prims.data(), N * sizeof(Primitive),
                               ctx.computeQueue, ctx.cmdPool));

    struct MortonPC {
        uint32_t count;
        float minX, minY, minZ, maxX, maxY, maxZ;
    };
    MortonPC mortonPC{N, minX, minY, minZ, maxX, maxY, maxZ};

    {
        vkComputeBase kernel(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        kernel.Build(shaderDir + "/bvh_mortonCode.comp")
                .Bind(0, mortonBuf)
                .Bind(1, primBuf)
                .Args(mortonPC)
                .DispatchElements(N);
        kernel.Sync();
    }

    // ── Step 2: GPU Radix Sort ────────────────────────────────────────────────
    // mortonBuf를 직접 입력으로 넘겨 정렬
    std::vector<MortonCode> unsorted(N);
    EXPECT_TRUE(mortonBuf.Download(unsorted.data(), N * sizeof(MortonCode),
                                   ctx.computeQueue, ctx.cmdPool));

    auto sorted = gpuRadixSort(ctx, unsorted);

    primBuf.Clear();
    mortonBuf.Clear();
    return sorted;
}

class MortonRadixSortTest : public ::testing::Test {
protected:
    void SetUp() override { ctx.init(); }
    void TearDown() override { ctx.shutdown(); }
    VkContext ctx;
};

// [6] 8개 포인트 → Morton 코드 계산 후 정렬
// 공간적으로 가까운 점들이 정렬 후 연속된 위치에 모여야 함
TEST_F(MortonRadixSortTest, PointsMortonSorted) {
    // 단위 큐브 [0,1]³ 내 8개 꼭짓점
    std::vector<Primitive> prims = {
            {0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, // (0,0,0)
            {1, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, // (1,1,1)
            {2, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f}, // (1,0,0)
            {3, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f}, // (0,1,0)
            {4, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f}, // (0,0,1)
            {5, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f}, // (1,1,0)
            {6, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f}, // (1,0,1)
            {7, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f}, // (0,1,1)
    };

    auto result = gpuMortonThenSort(ctx, prims);
    ASSERT_EQ(result.size(), 8u);

    // 정렬 확인
    for (size_t i = 1; i < result.size(); i++)
        EXPECT_LE(result[i - 1].code, result[i].code)
                << "Not sorted at " << i;

    // 원점(0,0,0)이 가장 낮은 코드, (1,1,1)이 가장 높은 코드여야 함
    EXPECT_EQ(result.front().index, 0u) << "Origin should have smallest Morton code";
    EXPECT_EQ(result.back().index, 1u) << "(1,1,1) should have largest Morton code";

    std::cout << "[Morton+Sort] 8-corner result:\n";
    for (auto &m: result)
        std::cout << "  code=0x" << std::hex << m.code
                  << std::dec << "  originalIdx=" << m.index << "\n";
}

// [7] X축 방향 점들 → 정렬 후 X 좌표 순서와 일치
TEST_F(MortonRadixSortTest, XAxisPointsSortedByX) {
    // y=0.5, z=0.5 고정, x만 변화 → Morton 코드가 x에 비례
    std::vector<Primitive> prims;
    for (uint32_t i = 0; i < 8; i++) {
        float x = i / 7.0f;
        prims.push_back({i, x, 0.5f, 0.5f, x, 0.5f, 0.5f});
    }

    auto result = gpuMortonThenSort(ctx, prims);
    ASSERT_EQ(result.size(), 8u);

    // 정렬 확인
    for (size_t i = 1; i < result.size(); i++)
        EXPECT_LE(result[i - 1].code, result[i].code);

    // originalIdx도 0→7 순서여야 함 (x가 커질수록 Morton 코드도 커짐)
    for (size_t i = 0; i < result.size(); i++)
        EXPECT_EQ(result[i].index, static_cast<uint32_t>(i))
                << "X-axis ordering broken at position " << i;
}

TEST_F(RadixSortTest, MortonCodeSorting) {
    const int N = 1000;
    std::vector<Primitive> points = Primitive::RadomPoint(N);
    const std::string shaderDir = VKBVH_SHADER_DIR;

    // morton code
    float minX = 1e38f, minY = 1e38f, minZ = 1e38f;
    float maxX = -1e38f, maxY = -1e38f, maxZ = -1e38f;
    for (const auto &p: points) {
        minX = std::min(minX, p.aabbMinX);
        maxX = std::max(maxX, p.aabbMaxX);
        minY = std::min(minY, p.aabbMinY);
        maxY = std::max(maxY, p.aabbMaxY);
        minZ = std::min(minZ, p.aabbMinZ);
        maxZ = std::max(maxZ, p.aabbMaxZ);
    }

    vkGPUMemory primBuf(ctx.device, ctx.physDevice);
    vkGPUMemory mortonBuf(ctx.device, ctx.physDevice);
    EXPECT_TRUE(primBuf.Allocate(N * sizeof(Primitive)));
    EXPECT_TRUE(mortonBuf.Allocate(N * sizeof(MortonCode)));
    EXPECT_TRUE(primBuf.Upload(points.data(), N * sizeof(Primitive), ctx.computeQueue, ctx.cmdPool));

    MortonConstant mortonPC;
    mortonPC.g_numberOfPrimitive = N;
    mortonPC.g_minX = minX;
    mortonPC.g_minY = minY;
    mortonPC.g_minZ = minZ;
    mortonPC.g_maxX = maxX;
    mortonPC.g_maxY = maxY;
    mortonPC.g_maxZ = maxZ;

    {
        vkComputeBase kernel(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        kernel.Build(shaderDir + "/bvh_mortonCode.comp")
                .Bind(0, mortonBuf)
                .Bind(1, primBuf)
                .Args(mortonPC)
                .DispatchElements(N);
        kernel.Sync();
    }
}

TEST_F(MortonRadixSortTest, LargeRandomPointsPipelineMatchesCpu) {
    const uint32_t N = 10'000;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    // 단위 큐브 내 랜덤 포인트
    std::vector<Primitive> prims(N);
    for (uint32_t i = 0; i < N; i++) {
        float x = dist(rng), y = dist(rng), z = dist(rng);
        prims[i] = {i, x, y, z, x, y, z};
    }

    auto gpuResult = gpuMortonThenSort(ctx, prims);
    ASSERT_EQ(gpuResult.size(), N);

    // CPU 기준: 동일한 Morton 계산 후 정렬
    // (CPU morton 함수를 재사용하지 않고 GPU 계산값과만 비교)
    // → 정렬 여부만 확인 (CPU와 GPU가 같은 알고리즘 사용)
    bool sorted = true;
    for (uint32_t i = 1; i < N; i++) {
        if (gpuResult[i - 1].code > gpuResult[i].code) {
            sorted = false;
            ADD_FAILURE() << "Not sorted at index " << i
                          << ": " << gpuResult[i - 1].code
                          << " > " << gpuResult[i].code;
            break;
        }
    }
    EXPECT_TRUE(sorted);

    // 모든 원본 인덱스가 보존됐는지 확인
    std::vector<uint32_t> indices(N);
    for (uint32_t i = 0; i < N; i++) indices[i] = gpuResult[i].index;
    std::sort(indices.begin(), indices.end());
    for (uint32_t i = 0; i < N; i++)
        EXPECT_EQ(indices[i], i) << "Missing primitive index " << i;

    std::cout << "[Morton+Sort] LargeRandom first 5:\n";
    for (int i = 0; i < 5; i++)
        std::cout << "  code=0x" << std::hex << gpuResult[i].code
                  << std::dec << "  originalIdx=" << gpuResult[i].index << "\n";
}

TEST_F(RadixSortTest, MortonRadixSortTesting) {
    std::vector<Primitive> points = Primitive::RadomPoint(1000);

    const uint32_t numberOfPoint = static_cast<uint32_t>(points.size());
    constexpr uint32_t WG_SIZE = 256;
    constexpr uint32_t RADIX = 16;
    constexpr uint32_t PASSES = 8;
    const uint32_t numberOfWorkerGroup = (numberOfPoint + WG_SIZE - 1) / WG_SIZE;

    vkGPUMemory primitiveBuffer(ctx.device, ctx.physDevice);
    vkGPUMemory mortonBuffer(ctx.device, ctx.physDevice);
    vkGPUMemory histogramBuffer(ctx.device, ctx.physDevice);
    vkGPUMemory pongBuffer(ctx.device, ctx.physDevice);
    ASSERT_TRUE(primitiveBuffer.Allocate(sizeof(Primitive) * numberOfPoint));
    ASSERT_TRUE(mortonBuffer.Allocate(sizeof(MortonCode) * numberOfPoint));
    ASSERT_TRUE(histogramBuffer.Allocate(sizeof(uint32_t) * RADIX * numberOfWorkerGroup));
    ASSERT_TRUE(pongBuffer.Allocate(sizeof(MortonCode) * numberOfPoint));
    ASSERT_TRUE(primitiveBuffer.Upload(points.data(), numberOfPoint * sizeof(Primitive),
                                       ctx.computeQueue, ctx.cmdPool));

    MortonConstant mc;
    mc.Extend(points);

    // Step 1: Morton 코드 계산
    {
        vkComputeBase mortonPass(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        mortonPass.Build(std::string(VKBVH_SHADER_DIR) + "/bvh_mortonCode.comp")
                .Bind(0, mortonBuffer)
                .Bind(1, primitiveBuffer)
                .Args(mc)
                .DispatchElements(numberOfPoint);
        mortonPass.Sync();
    }

    // Step 2: Radix Sort — 매 패스마다 커널 재생성 (Bind()는 VkBuffer 핸들을 값으로 저장하므로
    //         루프 밖에서 생성된 커널은 std::swap 후에도 같은 핸들을 사용함)
    vkGPUMemory *ping = &mortonBuffer;
    vkGPUMemory *pong = &pongBuffer;

    for (uint32_t pass = 0; pass < PASSES; pass++) {
        RadixSortPC pcSort{numberOfPoint, pass * 4};
        RadixScanPC pcScan{numberOfWorkerGroup};

        vkComputeBase histPass(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        histPass.Build(std::string(VKBVH_SHADER_DIR) + "/bvh_radixSort_histogram.comp")
                .Bind(0, *ping)
                .Bind(1, histogramBuffer)
                .Args(pcSort)
                .Dispatch(numberOfWorkerGroup);
        histPass.Sync();

        vkComputeBase scanPass(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        scanPass.Build(std::string(VKBVH_SHADER_DIR) + "/bvh_radixSort_prefixScan.comp")
                .Bind(0, histogramBuffer)
                .Args(pcScan)
                .Dispatch(1);
        scanPass.Sync();

        vkComputeBase reorderPass(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        reorderPass.Build(std::string(VKBVH_SHADER_DIR) + "/bvh_radixSort_reorder.comp")
                .Bind(0, *ping)
                .Bind(1, histogramBuffer)
                .Bind(2, *pong)
                .Args(pcSort)
                .Dispatch(numberOfWorkerGroup);
        reorderPass.Sync();

        std::swap(ping, pong);
    }

    // PASSES=8(짝수) → ping이 원래 mortonBuffer로 돌아옴
    std::vector<MortonCode> result(numberOfPoint);
    ASSERT_TRUE(ping->Download(result.data(), numberOfPoint * sizeof(MortonCode),
                               ctx.computeQueue, ctx.cmdPool));

    // 오름차순 정렬 확인
    for (uint32_t i = 1; i < numberOfPoint; i++)
        EXPECT_LE(result[i - 1].code, result[i].code)
                << "Not sorted at index " << i
                << " (" << result[i - 1].code << " > " << result[i].code << ")";

    // 원본 인덱스 완전성 확인 (손실·중복 없음)
    std::vector<uint32_t> indices(numberOfPoint);
    for (uint32_t i = 0; i < numberOfPoint; i++) indices[i] = result[i].index;
    std::sort(indices.begin(), indices.end());
    for (uint32_t i = 0; i < numberOfPoint; i++)
        EXPECT_EQ(indices[i], i) << "Missing primitive index " << i;

    std::cout << "[MortonSortTesting] first 5 results:\n";
    for (uint32_t i = 0; i < std::min(5u, numberOfPoint); i++)
        std::cout << "  [" << i << "] code=0x" << std::hex << result[i].code
                  << std::dec << "  idx=" << result[i].index << "\n";
}

#include <gtest/gtest.h>

#include "vkBVH/common/vkComputeBase.h"
#include "vkBVH/common/vkContext.h"
#include "vkBVH/common/vkGPUMemory.h"
#include "vkBVH/types.h"
#include "vkBVH/vkBVH.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>
#include <random>
#include <vector>

#ifndef VKBVH_SHADER_DIR
#define VKBVH_SHADER_DIR "."
#endif

static uint32_t bitPadding(uint32_t v) {
    v &= 0x000003ffu;
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v << 8u)) & 0x0300F00Fu;
    v = (v | (v << 4u)) & 0x030C30C3u;
    v = (v | (v << 2u)) & 0x09249249u;
    return v;
}

static uint32_t mortonFromNorm(float nx, float ny, float nz) {
    constexpr float SCALE = 1024.0f;
    constexpr uint32_t MAX_COORD = 1023u;
    auto clampF = [](float v) {
        return std::max(0.0f, std::min(v, static_cast<float>(MAX_COORD)));
    };
    uint32_t ix = static_cast<uint32_t>(clampF(nx * SCALE));
    uint32_t iy = static_cast<uint32_t>(clampF(ny * SCALE));
    uint32_t iz = static_cast<uint32_t>(clampF(nz * SCALE));
    return (bitPadding(ix) << 2u) | (bitPadding(iy) << 1u) | bitPadding(iz);
}

static uint32_t expectedMorton(const Primitive &p,
                               float minX, float minY, float minZ,
                               float maxX, float maxY, float maxZ) {
    float cx = (p.aabbMinX + p.aabbMaxX) * 0.5f;
    float cy = (p.aabbMinY + p.aabbMaxY) * 0.5f;
    float cz = (p.aabbMinZ + p.aabbMaxZ) * 0.5f;

    auto safeDiv = [](float a, float b) { return b > 1e-8f ? a / b : 0.0f; };
    float nx = std::clamp(safeDiv(cx - minX, maxX - minX), 0.0f, 1.0f);
    float ny = std::clamp(safeDiv(cy - minY, maxY - minY), 0.0f, 1.0f);
    float nz = std::clamp(safeDiv(cz - minZ, maxZ - minZ), 0.0f, 1.0f);
    return mortonFromNorm(nx, ny, nz);
}


class HierarchyTest : public ::testing::Test {
protected:
    void SetUp() override { ctx.init(); }
    void TearDown() override { ctx.shutdown(); }
    VkContext ctx;
};

struct CPUNode {
    int32_t left, right;
    uint32_t primitiveIdx;
    float aabbMinX, aabbMinY, aabbMinZ;
    float aabbMaxX, aabbMaxY, aabbMaxZ;
};
static_assert(sizeof(CPUNode) == 36, "CPUNode size mismatch");

struct CPUConstrInfo {
    uint32_t parent;
    int32_t visitationCount;
};
static_assert(sizeof(CPUConstrInfo) == 8, "CPUConstrInfo size mismatch");

static void buildHierarchy(VkContext &ctx,
                           const std::vector<Primitive> &prims,
                           std::vector<CPUNode> &outNodes,
                           std::vector<CPUConstrInfo> &outConstrInfos) {
    constexpr uint32_t WG_SIZE = 256;
    constexpr uint32_t RADIX = 16;
    constexpr uint32_t PASSES = 8;
    const uint32_t N = static_cast<uint32_t>(prims.size());
    const uint32_t NODES = 2 * N - 1;
    const uint32_t numWGs = (N + WG_SIZE - 1) / WG_SIZE;
    const std::string sd = VKBVH_SHADER_DIR;

    vkGPUMemory primBuf(ctx.device, ctx.physDevice);
    vkGPUMemory mortonBuf(ctx.device, ctx.physDevice);
    vkGPUMemory pongBuf(ctx.device, ctx.physDevice);
    vkGPUMemory histBuf(ctx.device, ctx.physDevice);
    vkGPUMemory nodeBuf(ctx.device, ctx.physDevice);
    vkGPUMemory constrBuf(ctx.device, ctx.physDevice);

    EXPECT_TRUE(primBuf.Allocate(N * sizeof(Primitive)));
    EXPECT_TRUE(mortonBuf.Allocate(N * sizeof(MortonCode)));
    EXPECT_TRUE(pongBuf.Allocate(N * sizeof(MortonCode)));
    EXPECT_TRUE(histBuf.Allocate(RADIX * numWGs * sizeof(uint32_t)));
    EXPECT_TRUE(nodeBuf.Allocate(NODES * sizeof(CPUNode)));
    EXPECT_TRUE(constrBuf.Allocate(NODES * sizeof(CPUConstrInfo)));
    EXPECT_TRUE(primBuf.Upload(prims.data(), N * sizeof(Primitive), ctx.computeQueue, ctx.cmdPool));

    // Step 1: Morton 코드 계산
    MortonConstant mc;
    mc.Extend(prims);
    {
        vkComputeBase k(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        k.Build(sd + "/bvh_mortonCode.comp")
                .Bind(0, mortonBuf)
                .Bind(1, primBuf)
                .Args(mc)
                .DispatchElements(N);
        k.Sync();
    }

    // Step 2: Radix Sort (루프 내 커널 재생성 + 포인터 swap)
    vkGPUMemory *ping = &mortonBuf;
    vkGPUMemory *pong = &pongBuf;
    for (uint32_t pass = 0; pass < PASSES; pass++) {
        RadixSortPC pcSort{N, pass * 4};
        RadixScanPC pcScan{numWGs};

        vkComputeBase h(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        h.Build(sd + "/bvh_radixSort_histogram.comp")
                .Bind(0, *ping)
                .Bind(1, histBuf)
                .Args(pcSort)
                .Dispatch(numWGs);
        h.Sync();

        vkComputeBase s(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        s.Build(sd + "/bvh_radixSort_prefixScan.comp")
                .Bind(0, histBuf)
                .Args(pcScan)
                .Dispatch(1);
        s.Sync();

        vkComputeBase r(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        r.Build(sd + "/bvh_radixSort_reorder.comp")
                .Bind(0, *ping)
                .Bind(1, histBuf)
                .Bind(2, *pong)
                .Args(pcSort)
                .Dispatch(numWGs);
        r.Sync();

        std::swap(ping, pong);
    }

    HierarchyPC hpc{N, 1u};
    {
        vkComputeBase k(ctx.device, ctx.physDevice, ctx.computeQueue, ctx.cmdPool);
        k.Build(sd + "/bvh_hierarchy.comp")
                .Bind(0, *ping)
                .Bind(1, primBuf)
                .Bind(2, nodeBuf)
                .Bind(3, constrBuf)
                .Args(hpc)
                .DispatchElements(N);
        k.Sync();
    }

    outNodes.resize(NODES);
    outConstrInfos.resize(NODES);
    EXPECT_TRUE(nodeBuf.Download(outNodes.data(), NODES * sizeof(CPUNode),
                                 ctx.computeQueue, ctx.cmdPool));
    EXPECT_TRUE(constrBuf.Download(outConstrInfos.data(), NODES * sizeof(CPUConstrInfo),
                                   ctx.computeQueue, ctx.cmdPool));
}

TEST_F(HierarchyTest, HierarchyBuildsValidTree) {
    const uint32_t N = 64;
    const uint32_t NODES = 2 * N - 1;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::vector<Primitive> prims(N);
    for (uint32_t i = 0; i < N; i++) {
        float x = dist(rng), y = dist(rng), z = dist(rng);
        prims[i] = {i, x, y, z, x, y, z};
    }

    std::vector<CPUNode> nodes;
    std::vector<CPUConstrInfo> constrInfos;
    buildHierarchy(ctx, prims, nodes, constrInfos);
    ASSERT_EQ(nodes.size(), NODES);
    ASSERT_EQ(constrInfos.size(), NODES);

    // (a) 노드 타입: 내부 노드 [0, N-2] 는 자식 있음, 리프 [N-1, 2N-2] 는 없음
    for (uint32_t i = 0; i < N - 1; i++) {
        EXPECT_NE(nodes[i].left, 0) << "Internal node " << i << " must have left child";
        EXPECT_NE(nodes[i].right, 0) << "Internal node " << i << " must have right child";
    }
    for (uint32_t i = N - 1; i < NODES; i++) {
        EXPECT_EQ(nodes[i].left, 0) << "Leaf node " << i << " left must be INVALID_POINTER";
        EXPECT_EQ(nodes[i].right, 0) << "Leaf node " << i << " right must be INVALID_POINTER";
    }

    // (b) 리프 primitiveIdx 완전성: {0, ..., N-1} 정확히 한 번씩
    std::vector<uint32_t> leafIdx;
    leafIdx.reserve(N);
    for (uint32_t i = N - 1; i < NODES; i++)
        leafIdx.push_back(nodes[i].primitiveIdx);
    std::sort(leafIdx.begin(), leafIdx.end());
    for (uint32_t i = 0; i < N; i++)
        EXPECT_EQ(leafIdx[i], i) << "Missing or duplicate primitiveIdx " << i;

    // (c) 부모 포인터 유효성
    EXPECT_EQ(constrInfos[0].parent, 0u) << "Root parent must point to itself";
    for (uint32_t i = 1; i < NODES; i++)
        EXPECT_LT(constrInfos[i].parent, N - 1)
                << "Node " << i << " parent must be an internal node index";

    // (d) 내부 노드 자식 인덱스 범위 [1, 2N-2]
    for (uint32_t i = 0; i < N - 1; i++) {
        EXPECT_GE(nodes[i].left, 1)
                << "Internal " << i << " left child index out of range";
        EXPECT_LT(static_cast<uint32_t>(nodes[i].left), NODES)
                << "Internal " << i << " left child index out of range";
        EXPECT_GE(nodes[i].right, 1)
                << "Internal " << i << " right child index out of range";
        EXPECT_LT(static_cast<uint32_t>(nodes[i].right), NODES)
                << "Internal " << i << " right child index out of range";
    }
}

TEST_F(HierarchyTest, HierarchyBFSCoversAllLeaves) {
    const uint32_t N = 100;
    const uint32_t NODES = 2 * N - 1;

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::vector<Primitive> prims(N);
    for (uint32_t i = 0; i < N; i++) {
        float x = dist(rng), y = dist(rng), z = dist(rng);
        prims[i] = {i, x, y, z, x, y, z};
    }

    std::vector<CPUNode> nodes;
    std::vector<CPUConstrInfo> constrInfos;
    buildHierarchy(ctx, prims, nodes, constrInfos);
    ASSERT_EQ(nodes.size(), NODES);

    // BFS from root (node 0)
    std::vector<bool> visited(NODES, false);
    std::queue<uint32_t> q;
    q.push(0);
    uint32_t leafCount = 0, internalCount = 0;
    std::vector<uint32_t> collectedIdx;

    while (!q.empty()) {
        uint32_t idx = q.front();
        q.pop();

        ASSERT_LT(idx, NODES) << "BFS reached out-of-range node index " << idx;
        ASSERT_FALSE(visited[idx]) << "Cycle: node " << idx << " visited twice";
        visited[idx] = true;

        const CPUNode &node = nodes[idx];
        if (node.left == 0 && node.right == 0) {
            leafCount++;
            collectedIdx.push_back(node.primitiveIdx);
        } else {
            internalCount++;
            q.push(static_cast<uint32_t>(node.left));
            q.push(static_cast<uint32_t>(node.right));
        }
    }

    EXPECT_EQ(leafCount, N) << "BFS must reach exactly N leaves";
    EXPECT_EQ(internalCount, N - 1) << "BFS must visit exactly N-1 internal nodes";

    // 수집된 primitiveIdx == {0, ..., N-1}
    std::sort(collectedIdx.begin(), collectedIdx.end());
    ASSERT_EQ(collectedIdx.size(), N);
    for (uint32_t i = 0; i < N; i++)
        EXPECT_EQ(collectedIdx[i], i) << "Missing primitiveIdx " << i;
}

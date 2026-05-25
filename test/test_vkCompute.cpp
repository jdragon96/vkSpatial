#include <gtest/gtest.h>

#include "vkBVH/common/vkComputeBase.h"
#include "vkBVH/common/vkContext.h"
#include "vkBVH/common/vkGPUMemory.h"

#include <numeric>
#include <vector>

class SumKernel : public vkComputeBase {
public:
    explicit SumKernel(const VkContext &ctx)
        : vkComputeBase(ctx.device, ctx.physDevice,
                        ctx.computeQueue, ctx.cmdPool) {}
};

class AtomicSumKernel : public vkComputeBase {
public:
    explicit AtomicSumKernel(const VkContext &ctx)
        : vkComputeBase(ctx.device, ctx.physDevice,
                        ctx.computeQueue, ctx.cmdPool) {}
};

// ── 테스트 픽스처 ─────────────────────────────────────────────────────────────

class VkComputeTest : public ::testing::Test {
protected:
    void SetUp() override { ctx.init(); }
    void TearDown() override { ctx.shutdown(); }

    VkContext ctx;
};

// ── 테스트 케이스 ─────────────────────────────────────────────────────────────

// 0 + 1 + 2 + ... + 10000 = 50,005,000
TEST_F(VkComputeTest, SumZeroToTenThousand) {
    constexpr uint32_t N = 10001u;
    constexpr uint32_t EXPECTED = 10000u * 10001u / 2u; // 50005000

    // 입력 데이터: 0, 1, 2, ..., 10000
    std::vector<uint32_t> inputData(N);
    std::iota(inputData.begin(), inputData.end(), 0u);

    // ── GPU 버퍼 ─────────────────────────────────────────────────────────────
    vkGPUMemory inputMem(ctx.device, ctx.physDevice);
    ASSERT_TRUE(inputMem.Allocate(N * sizeof(uint32_t)));
    ASSERT_TRUE(inputMem.Upload(inputData.data(), N * sizeof(uint32_t),
                                ctx.computeQueue, ctx.cmdPool));

    vkGPUMemory outputMem(ctx.device, ctx.physDevice);
    ASSERT_TRUE(outputMem.Allocate(sizeof(uint32_t)));

    SumKernel kernel(ctx);
    kernel.Build(R"GLSL(
#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PC {
    uint count;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer Input {
    uint g_data[];
};

layout(std430, set = 0, binding = 1) buffer Output {
    uint g_result;
};

void main() {
    uint sum = 0u;
    for (uint i = 0u; i < pc.count; i++) {
        sum += g_data[i];
    }
    g_result = sum;
}
)GLSL",
                 ShaderInput::GlslSrc)
            .Bind(0, inputMem)
            .Bind(1, outputMem)
            .Args(N)
            .Dispatch(1, 1, 1);

    kernel.Sync();

    // ── 결과 검증 ─────────────────────────────────────────────────────────────
    uint32_t result = 0;
    ASSERT_TRUE(outputMem.Download(&result, sizeof(uint32_t),
                                   ctx.computeQueue, ctx.cmdPool));

    EXPECT_EQ(result, EXPECTED)
            << "Expected 0+1+...+10000 = " << EXPECTED << ", got " << result;

    inputMem.Clear();
    outputMem.Clear();
}

// 256 스레드 × 40 workgroup = 10240 스레드로 병렬 atomicAdd
// 0 + 1 + 2 + ... + 10000 = 50,005,000
TEST_F(VkComputeTest, AtomicSumZeroToTenThousand) {
    constexpr uint32_t N = 10001u;
    constexpr uint32_t EXPECTED = 10000u * 10001u / 2u; // 50005000

    std::vector<uint32_t> inputData(N);
    std::iota(inputData.begin(), inputData.end(), 0u);

    // ── GPU 버퍼 ─────────────────────────────────────────────────────────────
    vkGPUMemory inputMem(ctx.device, ctx.physDevice);
    ASSERT_TRUE(inputMem.Allocate(N * sizeof(uint32_t)));
    ASSERT_TRUE(inputMem.Upload(inputData.data(), N * sizeof(uint32_t),
                                ctx.computeQueue, ctx.cmdPool));

    // atomicAdd 는 0에서 시작해야 하므로 명시적으로 0 업로드
    uint32_t zero = 0u;
    vkGPUMemory outputMem(ctx.device, ctx.physDevice);
    ASSERT_TRUE(outputMem.Allocate(sizeof(uint32_t)));
    ASSERT_TRUE(outputMem.Upload(&zero, sizeof(uint32_t),
                                 ctx.computeQueue, ctx.cmdPool));

    AtomicSumKernel kernel(ctx);
    kernel.Build(R"GLSL(
#version 460

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PC {
    uint count;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer Input {
    uint g_data[];
};

layout(std430, set = 0, binding = 1) buffer Output {
    uint g_result;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.count) return;
    atomicAdd(g_result, g_data[idx]);
}
)GLSL",
                 ShaderInput::GlslSrc)
            .Bind(0, inputMem)
            .Bind(1, outputMem)
            .Args(N)
            .DispatchElements(N); // gridX = ceil(10001 / 256) = 40

    kernel.Sync();

    // ── 결과 검증 ─────────────────────────────────────────────────────────────
    uint32_t result = 0;
    ASSERT_TRUE(outputMem.Download(&result, sizeof(uint32_t),
                                   ctx.computeQueue, ctx.cmdPool));

    EXPECT_EQ(result, EXPECTED)
            << "Expected 0+1+...+10000 = " << EXPECTED << ", got " << result;

    inputMem.Clear();
    outputMem.Clear();
}

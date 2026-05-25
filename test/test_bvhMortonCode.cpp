#include <gtest/gtest.h>

#include "vkSpatial/common/vkComputeBase.h"
#include "vkSpatial/common/vkContext.h"
#include "vkSpatial/common/vkGPUMemory.h"
#include "vkSpatial/types.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>

using namespace vkCommon;

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

// ── MortonKernel ───────────────────────────────────────────────────────────

class MortonKernel : public vkComputeBase {
public:
    explicit MortonKernel(const VkContext &ctx)
        : vkComputeBase(ctx.device, ctx.physDevice,
                        ctx.computeQueue, ctx.cmdPool) {}
};

// ── 테스트 픽스처 ─────────────────────────────────────────────────────────────

class MortonCodeTest : public ::testing::Test {
protected:
    void SetUp() override { ctx.init(); }
    void TearDown() override { ctx.shutdown(); }

    std::vector<MortonCode> runMorton(const std::vector<Primitive> &prims,
                                      float minX, float minY, float minZ,
                                      float maxX, float maxY, float maxZ) {
        const uint32_t N = static_cast<uint32_t>(prims.size());

        vkGPUMemory primMem(ctx.device, ctx.physDevice);
        EXPECT_TRUE(primMem.Allocate(N * sizeof(Primitive)));
        EXPECT_TRUE(primMem.Upload(prims.data(), N * sizeof(Primitive),
                                   ctx.computeQueue, ctx.cmdPool));

        vkGPUMemory outMem(ctx.device, ctx.physDevice);
        EXPECT_TRUE(outMem.Allocate(N * sizeof(MortonCode)));


        // FilesystemIncluder 경로 대신 AddInclude + GlslSrc 방식 사용
        // (VkComputeTest와 동일한 패턴 — 파일 기반 빌드의 크래시 우회)
        auto readFile = [](const std::string &p) -> std::string {
            std::ifstream ifs(p);
            return std::string(std::istreambuf_iterator<char>(ifs),
                               std::istreambuf_iterator<char>());
        };
        // shaderc(GlslSrc)가 크래시하므로 glslc로 사전 컴파일한 SPIR-V를 사용
        // VkComputeTest와 다르게 MortonCodeTest 셰이더는 런타임 컴파일 불가
        const std::string compPath = std::string(VKBVH_SHADER_DIR) + "/bvh_mortonCode.comp";
        const std::string spvPath = "/tmp/bvh_mortonCode.spv";

        // glslc로 SPIR-V 사전 컴파일
        {
            std::string cmd = "glslc --target-spv=spv1.5 " + compPath + " -o " + spvPath;
            int ret = std::system(cmd.c_str());
            if (ret != 0) {
                ADD_FAILURE() << "glslc failed: " << cmd;
                return {};
            }
        }

        MortonConstant pc{N, minX, minY, minZ, maxX, maxY, maxZ};

        MortonKernel kernel(ctx);
        kernel.Build(spvPath, ShaderInput::SpvFile)
                .Bind(0, outMem)
                .Bind(1, primMem)
                .Args(pc)
                .DispatchElements(N);

        kernel.Sync();

        std::vector<MortonCode> result(N);
        EXPECT_TRUE(outMem.Download(result.data(), N * sizeof(MortonCode),
                                    ctx.computeQueue, ctx.cmdPool));
        primMem.Clear();
        outMem.Clear();
        return result;
    }

    VkContext ctx;
};

// ── 테스트 케이스 ─────────────────────────────────────────────────────────────

// ── 테스트 케이스 ─────────────────────────────────────────────────────────────
// 주의: 이 시스템에서 shaderc(GlslSrc)가 크래시하므로
//       glslc로 사전 컴파일한 SPIR-V를 SpvFile로 로드하는 방식 사용

// [1] 원점 프리미티브 → Morton 코드 = 0
TEST_F(MortonCodeTest, OriginMortonIsZero) {
    Primitive p{0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    auto result = runMorton({p}, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f);

    EXPECT_EQ(result[0].code, 0u);
    EXPECT_EQ(result[0].index, 0u);
}

// [2] 최대 코너 프리미티브 → Morton 코드 = 0x3FFFFFFF (30비트 전체)
TEST_F(MortonCodeTest, MaxCornerMortonIsMax) {
    Primitive p{0, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    auto result = runMorton({p}, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f);

    EXPECT_EQ(result[0].code, 0x3FFFFFFFu);
    EXPECT_EQ(result[0].index, 0u);
}

// [3] CPU 레퍼런스와 비교 — 코드 + 인덱스 모두 검증
TEST_F(MortonCodeTest, MatchesCpuReference) {
    const float BND = 2.0f;
    std::vector<Primitive> prims = {
            {0, 0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 0.2f},
            {1, 0.8f, 0.8f, 0.8f, 1.0f, 1.0f, 1.0f},
            {2, 0.4f, 0.4f, 0.4f, 0.6f, 0.6f, 0.6f},
            {3, 0.0f, 0.8f, 0.0f, 0.2f, 1.0f, 0.2f},
            {4, 1.5f, 1.5f, 1.5f, 2.0f, 2.0f, 2.0f},
    };

    auto result = runMorton(prims, 0.f, 0.f, 0.f, BND, BND, BND);

    for (size_t i = 0; i < prims.size(); i++) {
        uint32_t exp = expectedMorton(prims[i], 0.f, 0.f, 0.f, BND, BND, BND);
        EXPECT_EQ(result[i].code, exp) << "Prim " << i << " code mismatch";
        EXPECT_EQ(result[i].index, (uint32_t) i) << "Prim " << i << " index mismatch";
    }
}

// [4] 공간 정렬 — X축으로 균등 배치 시 Morton 코드도 오름차순
TEST_F(MortonCodeTest, SpatialOrderingAlongX) {
    std::vector<Primitive> prims;
    for (int i = 0; i < 8; i++) {
        float x = i * 0.125f;
        prims.push_back({(uint32_t) i, x, 0.f, 0.f, x, 0.f, 0.f});
    }

    auto result = runMorton(prims, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f);

    for (size_t i = 1; i < result.size(); i++) {
        EXPECT_LE(result[i - 1].code, result[i].code)
                << "Ordering broken between prim " << i - 1 << " and " << i;
    }
}

// ── Morton 코드 디코딩 헬퍼 ───────────────────────────────────────────────────
// bitPadding 의 역연산: 3-비트 간격으로 퍼진 비트를 10비트로 압축

static uint32_t bitUnpadding(uint32_t v) {
    // 0000 1001 0010 0100 1001 0010 0100 1001
    v &= 0x09249249u;
    // 0000 0011 0000 1100 0011 0000 1100 0011
    v = (v | (v >> 2u)) & 0x030C30C3u;
    // 0000 0011 0000 0000 1111 0000 0000 1111
    v = (v | (v >> 4u)) & 0x0300F00Fu;
    // 0000 0011 0000 0000 0000 0000 1111 1111
    v = (v | (v >> 8u)) & 0x030000FFu;
    // 0000 0000 0000 0000 0000 0011 1111 1111
    v = (v | (v >> 16u)) & 0x000003FFu;
    return v;
}

// Morton 코드 → 정규화 좌표 [0, 1] 복원
// 셰이더: code = (bitPadding(ix) << 2) | (bitPadding(iy) << 1) | bitPadding(iz)
static float mortonDecodeX(uint32_t code) { return bitUnpadding(code >> 2u) / 1024.f; }
static float mortonDecodeY(uint32_t code) { return bitUnpadding(code >> 1u) / 1024.f; }
static float mortonDecodeZ(uint32_t code) { return bitUnpadding(code) / 1024.f; }

// [5] float → Morton → float 왕복 정확도 검증
// 10비트 양자화(1024단계)이므로 허용 오차 = 1/1024 ≈ 0.00098 (정규화 공간)
TEST_F(MortonCodeTest, FloatRoundtripAccuracy) {
    // 단위 bbox [0,1]³ 내 대표 좌표들
    const std::vector<std::array<float, 3>> testPoints = {
            {0.0f, 0.0f, 0.0f},     // 원점
            {1.0f, 1.0f, 1.0f},     // 최대 코너
            {0.5f, 0.5f, 0.5f},     // 중심
            {0.1f, 0.7f, 0.3f},     // 임의 좌표 ①
            {0.9f, 0.2f, 0.8f},     // 임의 좌표 ②
            {0.33f, 0.66f, 0.99f},  // 소수 좌표
            {0.001f, 0.999f, 0.5f}, // 극단값 근처
    };

    // 각 좌표를 점 프리미티브로 변환 (min == max)
    std::vector<Primitive> prims;
    for (uint32_t i = 0; i < testPoints.size(); i++) {
        float x = testPoints[i][0], y = testPoints[i][1], z = testPoints[i][2];
        prims.push_back({i, x, y, z, x, y, z});
    }

    // GPU에서 Morton 코드 계산 (전역 bbox = 단위 큐브 [0,1]³)
    auto result = runMorton(prims, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f);
    ASSERT_EQ(result.size(), prims.size());

    // 허용 오차: 정규화 공간에서 1/1024
    constexpr float EPS = 1.f / 1024.f;

    std::cout << "[Morton RoundTrip] float → code → float:\n";

    for (size_t i = 0; i < testPoints.size(); i++) {
        uint32_t code = result[i].code;

        // 디코딩
        float rx = mortonDecodeX(code);
        float ry = mortonDecodeY(code);
        float rz = mortonDecodeZ(code);

        float ox = testPoints[i][0];
        float oy = testPoints[i][1];
        float oz = testPoints[i][2];

        std::cout << "  [" << i << "] orig=(" << ox << "," << oy << "," << oz << ")"
                  << " → code=0x" << std::hex << code << std::dec
                  << " → recovered=(" << rx << "," << ry << "," << rz << ")"
                  << " err=(" << std::abs(rx - ox) << "," << std::abs(ry - oy) << "," << std::abs(rz - oz) << ")\n";

        EXPECT_NEAR(rx, ox, EPS) << "X 복원 오차 초과 (index=" << i << ")";
        EXPECT_NEAR(ry, oy, EPS) << "Y 복원 오차 초과 (index=" << i << ")";
        EXPECT_NEAR(rz, oz, EPS) << "Z 복원 오차 초과 (index=" << i << ")";
    }
}

// [6] 같은 값을 두 번 인코딩하면 동일한 코드가 나와야 함 (결정론적)
TEST_F(MortonCodeTest, EncodingIsDeterministic) {
    Primitive p{0, 0.314f, 0.271f, 0.577f, 0.314f, 0.271f, 0.577f};

    auto r1 = runMorton({p}, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f);
    auto r2 = runMorton({p}, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f);

    EXPECT_EQ(r1[0].code, r2[0].code) << "동일 입력 → 다른 Morton 코드";

    // 디코딩 정확도도 확인
    float rx = mortonDecodeX(r1[0].code);
    float ry = mortonDecodeY(r1[0].code);
    float rz = mortonDecodeZ(r1[0].code);

    EXPECT_NEAR(rx, 0.314f, 1.f / 1024.f);
    EXPECT_NEAR(ry, 0.271f, 1.f / 1024.f);
    EXPECT_NEAR(rz, 0.577f, 1.f / 1024.f);
}

TEST_F(MortonCodeTest, EncodeDecode) {
    constexpr float EPS = 1.f / 1024.f;

    std::vector<Primitive> vecPrimitives;
    vecPrimitives.push_back({1, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f});
    vecPrimitives.push_back({0, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f});

    auto result = runMorton(vecPrimitives, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f);
    ASSERT_EQ(result.size(), vecPrimitives.size());

    auto morton2point = [&](uint32_t code) {
        float x = mortonDecodeX(code);
        float y = mortonDecodeX(code);
        float z = mortonDecodeX(code);
        return Eigen::Vector3f(x, y, z);
    };

    for (auto r: result) {
        uint32_t code = r.code;
        Primitive prim = vecPrimitives[r.index];

        Eigen::Vector3f op = morton2point(code);
        Eigen::Vector3f center = prim.GetCenter();

        EXPECT_NEAR(op.x(), center.x(), EPS);
        EXPECT_NEAR(op.y(), center.y(), EPS);
        EXPECT_NEAR(op.z(), center.z(), EPS);
    }
}
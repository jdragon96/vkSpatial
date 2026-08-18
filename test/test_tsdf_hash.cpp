#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

TEST(ComputePipelineDefines, DefinitionsReachTheCompilerAndKeyTheCache) {
    Engine::Core::Context context;

    Engine::Core::ComputePipeline withDefinition(context);
    withDefinition.Define("PROBE_LOCAL_SIZE", "8").Build("kernel_define_probe.comp.glsl");
    EXPECT_EQ(withDefinition.GetLocalSize().width, 8u);

    // Same file, no definition. A cache keyed on the path alone would return the module built
    // above and report 8 -- that is the failure this test exists to catch.
    Engine::Core::ComputePipeline withoutDefinition(context);
    withoutDefinition.Build("kernel_define_probe.comp.glsl");
    EXPECT_EQ(withoutDefinition.GetLocalSize().width, 1u);

    // And the definition still applies after the undefined build populated the cache.
    Engine::Core::ComputePipeline again(context);
    again.Define("PROBE_LOCAL_SIZE", "4").Build("kernel_define_probe.comp.glsl");
    EXPECT_EQ(again.GetLocalSize().width, 4u);
}

#include "TSDF/Hash/HashStrategy.h"

TEST(TsdfHashStrategy, LinearProbeIsTheDefaultAndCarriesItsThreshold) {
    const TSDF::HashStrategy &linear = TSDF::HashStrategyByName("linear");
    EXPECT_STREQ(linear.name, "linear");
    EXPECT_FLOAT_EQ(linear.loadFactorLimit, 0.5f);
    EXPECT_STREQ(TSDF::LinearProbeStrategy().name, "linear");
}

TEST(TsdfHashStrategy, BucketedIsRegisteredWithItsOwnThreshold) {
    const TSDF::HashStrategy &bucketed = TSDF::HashStrategyByName("bucketed");
    EXPECT_STREQ(bucketed.name, "bucketed");
    EXPECT_STREQ(bucketed.macroName, "HASH_BUCKETED");
    EXPECT_FLOAT_EQ(bucketed.loadFactorLimit, 0.8f);
}

TEST(TsdfHashStrategy, UnknownNameFallsBackToLinear) {
    // Callers report the typo; silently running a different hash than requested would corrupt
    // an A/B comparison without any visible symptom.
    EXPECT_STREQ(TSDF::HashStrategyByName("no-such-hash").name, "linear");
}

#include "TSDF/Backends/TSDFBackend.h"

namespace {
    // The 19x19 plane at voxel 0.05 / truncation 0.15 -- roughly 1176 entries once integrated.
    // One generator so the geometry stays byte-identical across tests.
    //
    // The size is chosen for MARGIN against the two growth thresholds the tests below straddle at
    // 2048 slots (linear 0.5 -> 1024, bucketed 0.8 -> 1638), because the entry count is a property
    // of the integrator, not a constant: the earlier 17x17 patch sat at 1045, two percent above the
    // linear threshold, so a legitimate 3% change in band voxels (marching the point-to-plane band
    // along the normal instead of the ray) dropped it to 1014 and silently broke both growth tests.
    // 1176 clears linear by 15% and stays 28% under bucketed, and holds for both march directions.
    void NineteenByNineteenPlane(std::vector<Eigen::Vector3f> &points,
                                   std::vector<Eigen::Vector3f> &normals) {
        for (int i = -9; i <= 9; ++i)
            for (int j = -9; j <= 9; ++j) {
                points.emplace_back(float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

    std::unique_ptr<TSDFBackend> MakeWindow(Engine::Core::Context &context, uint32_t hashCapacity,
                                            const char *hashName) {
        TSDFBackendConfig config;
        config.voxelSize = 0.05f;
        config.truncation = 0.15f;
        config.hashCapacity = hashCapacity;
        config.hash = hashName;

        std::unique_ptr<TSDFBackend> backend = MakeTSDFBackend("advanced");
        backend->Build(context, config);
        return backend;
    }

    const Eigen::Vector3f kCamera(0.0f, 0.0f, 1.0f);
} // namespace

TEST(TsdfHashCounters, NormalIntegrationDropsNothing) {
    Engine::Core::Context context;
    std::unique_ptr<TSDFBackend> backend = MakeWindow(context, 1u << 16, "linear");

    std::vector<Eigen::Vector3f> points, normals;
    NineteenByNineteenPlane(points, normals);
    backend->Integrate(points, normals, kCamera);

    const TSDFBackendStats stats = backend->Stats();
    EXPECT_GT(stats.filledCount, 0u);
    EXPECT_EQ(stats.insertFailureCount, 0u) << "정상 조건에서 복셀이 드롭되면 임계값이 잘못된 것";
}

// maybeGrow runs once per Integrate, so one call can grow at most once. At 2048 slots the linear
// threshold is 1024: the first call sees occupancy 0 and inserts ~1176, the second sees 1176 >=
// 1024 and grows.
TEST(TsdfHashCounters, GrowCountRisesWhenTheTableIsTooSmall) {
    Engine::Core::Context context;
    std::unique_ptr<TSDFBackend> backend = MakeWindow(context, 1u << 11, "linear");

    std::vector<Eigen::Vector3f> points, normals;
    NineteenByNineteenPlane(points, normals);
    backend->Integrate(points, normals, kCamera);
    backend->Integrate(points, normals, kCamera);

    const TSDFBackendStats stats = backend->Stats();
    EXPECT_GT(stats.growCount, 0u);
    EXPECT_GT(stats.hashCapacity, 1u << 11);
    EXPECT_EQ(stats.insertFailureCount, 0u) << "성장이 제때 일어났다면 드롭은 없어야 한다";
}

TEST(TsdfHashStrategy, BucketedStoresTheSameEntriesAsLinear) {
    std::vector<Eigen::Vector3f> points, normals;
    NineteenByNineteenPlane(points, normals);

    struct Run {
        TSDFBackendStats stats;
        Engine::Core::OrientedPointCloud cloud;
    };

    auto runWith = [&](const char *hashName) {
        Engine::Core::Context context;
        std::unique_ptr<TSDFBackend> backend = MakeWindow(context, 1u << 16, hashName);
        backend->Integrate(points, normals, kCamera);

        Run run;
        run.stats = backend->Stats();
        run.cloud = backend->Extract();
        return run;
    };

    const Run linear = runWith("linear");
    const Run bucketed = runWith("bucketed");

    EXPECT_EQ(bucketed.stats.filledCount, linear.stats.filledCount);
    EXPECT_EQ(bucketed.stats.insertFailureCount, 0u);
    EXPECT_GT(linear.stats.filledCount, 0u);

    // findSlot failing on present keys drops zero-crossings, which shows up as a shorter cloud.
    ASSERT_GT(linear.cloud.points.size(), 0u) << "픽스처가 표면을 하나도 추출하지 못하면 비교가 무의미하다";
    EXPECT_EQ(bucketed.cloud.points.size(), linear.cloud.points.size())
            << "버킷 추출 점 수가 다르다면 findSlot이 저장된 키를 찾지 못하고 있다는 뜻이다";
}

TEST(TsdfHashStrategy, BucketedGrowsLaterThanLinear) {
    std::vector<Eigen::Vector3f> points, normals;
    NineteenByNineteenPlane(points, normals);

    auto capacityAfter = [&](const char *hashName) {
        Engine::Core::Context context;
        std::unique_ptr<TSDFBackend> backend = MakeWindow(context, 1u << 11, hashName);
        backend->Integrate(points, normals, kCamera);
        backend->Integrate(points, normals, kCamera);
        return backend->Stats();
    };

    const TSDFBackendStats linear = capacityAfter("linear");
    const TSDFBackendStats bucketed = capacityAfter("bucketed");

    EXPECT_GT(linear.growCount, 0u) << "이 픽스처는 성장을 강제해야 한다";
    EXPECT_LT(bucketed.hashCapacity, linear.hashCapacity)
            << "버킷은 임계값 0.8이라 선형탐사(0.5)보다 늦게 자라야 한다";
    EXPECT_EQ(bucketed.growCount, 0u) << "이 픽스처(~1176)는 버킷의 0.8*2048=1638 임계값을 넘지 않는다";
    EXPECT_EQ(bucketed.insertFailureCount, 0u);
}

// kernel_AdvancedTSDF.rehash.comp.glsl once hard-coded linear-probe addressing for the GROWN
// table regardless of strategy, so a bucketed entry landed at a slot no later bucketed probe
// would ever visit: an integrate touching the same voxel could not find it and inserted a
// duplicate. Growth is driven by many small separated patches so no single Integrate needs more
// room than the table currently has.
TEST(TsdfHashStrategy, BucketedSurvivesAGrowIntact) {
    Engine::Core::Context context;
    std::unique_ptr<TSDFBackend> backend = MakeWindow(context, 1u << 11, "bucketed");

    auto patch = [](float offsetX, std::vector<Eigen::Vector3f> &points,
                    std::vector<Eigen::Vector3f> &normals) {
        for (int i = -2; i <= 2; ++i)
            for (int j = -2; j <= 2; ++j) {
                points.emplace_back(offsetX + float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }
    };

    constexpr int kMaxPatches = 40;
    int patchIndex = 0;
    while (patchIndex < kMaxPatches && backend->Stats().growCount == 0) {
        // 0.3 apart -- beyond truncation (0.15) + voxelSize (0.05), so patches never share a voxel.
        const float offsetX = float(patchIndex) * 0.3f;
        std::vector<Eigen::Vector3f> points, normals;
        patch(offsetX, points, normals);
        backend->Integrate(points, normals, Eigen::Vector3f(offsetX, 0.0f, 1.0f));
        ++patchIndex;
    }

    ASSERT_GT(backend->Stats().growCount, 0u)
            << "이 픽스처는 실제로 성장을 강제해야 한다 (버킷 임계값 0.8을 넘겨야 함)";
    const uint64_t occupiedAfterGrow = backend->Stats().filledCount;

    // Re-touch the FIRST patch -- it predates the grow, so the rehash moved its entries.
    {
        std::vector<Eigen::Vector3f> points, normals;
        patch(0.0f, points, normals);
        backend->Integrate(points, normals, kCamera);
    }

    const TSDFBackendStats stats = backend->Stats();
    EXPECT_EQ(stats.insertFailureCount, 0u);
    EXPECT_EQ(stats.filledCount, occupiedAfterGrow)
            << "재통합이 filledCount를 늘렸다면, 그로우 이후 버킷 주소로 기존 엔트리를 찾지 못해 "
               "중복 삽입한 것이다";
}

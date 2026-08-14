#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include <gtest/gtest.h>

TEST(ComputePipelineDefines, DefinitionsReachTheCompilerAndKeyTheCache) {
    Engine::Core::Context context;

    Engine::Core::ComputePipeline withDefinition(context);
    withDefinition.Define("PROBE_LOCAL_SIZE", "8").Build("define_probe.comp.glsl");
    EXPECT_EQ(withDefinition.GetLocalSize().width, 8u);

    // Same file, no definition. A cache keyed on the path alone would return the module built
    // above and report 8 -- that is the failure this test exists to catch.
    Engine::Core::ComputePipeline withoutDefinition(context);
    withoutDefinition.Build("define_probe.comp.glsl");
    EXPECT_EQ(withoutDefinition.GetLocalSize().width, 1u);

    // And the definition still applies after the undefined build populated the cache.
    Engine::Core::ComputePipeline again(context);
    again.Define("PROBE_LOCAL_SIZE", "4").Build("define_probe.comp.glsl");
    EXPECT_EQ(again.GetLocalSize().width, 4u);
}

#include "TSDF/Memory/Hash/HashStrategy.h"

TEST(TsdfHashStrategy, LinearProbeIsTheDefaultAndCarriesItsThreshold) {
    const TSDF::HashStrategy &linear = TSDF::HashStrategyByName("linear");
    EXPECT_STREQ(linear.name, "linear");
    EXPECT_FLOAT_EQ(linear.loadFactorLimit, 0.5f);
    EXPECT_STREQ(TSDF::LinearProbeStrategy().name, "linear");
}

TEST(TsdfHashStrategy, UnknownNameFallsBackToLinear) {
    // Callers report the typo; silently running a different hash than requested would corrupt
    // an A/B comparison without any visible symptom.
    EXPECT_STREQ(TSDF::HashStrategyByName("no-such-hash").name, "linear");
}

#include "Engine/Compute/CommandBatch.h"
#include "TSDF/Memory/FlatStrategy.h"
#include "TSDF/Volume.h"

#include <vector>

TEST(TsdfHashCounters, NormalIntegrationDropsNothing) {
    Engine::Core::Context context;
    TSDF::FlatStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 16;
    strategy.Build(context, params);

    std::vector<Eigen::Vector3f> points, normals;
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) {
            points.emplace_back(float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
            normals.emplace_back(0.0f, 0.0f, 1.0f);
        }

    Engine::Compute::CommandBatch batch(context);
    strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
    batch.Submit();

    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_GT(stats.occupiedEntryCount, 0u);
    EXPECT_EQ(stats.insertFailureCount, 0u) << "정상 조건에서 복셀이 드롭되면 임계값이 잘못된 것";
}

// Ruling 2 (progress.md): maybeGrow runs once per Record call, so a single call can grow the
// table at most once. Uses the same 17x17 plane at voxel 0.05 / truncation 0.15 as the other
// tests in this repo (measured at roughly 1045 entries) with hashCapacity = 1u << 11 (2048 slots,
// so the 0.5 threshold is 1024), integrated via TWO Record calls each in its own CommandBatch.
// The first Record sees occupancy 0 and does not grow, inserting ~1045 (alpha ~= 0.51). The
// second sees 1045 >= 1024 and grows to 4096.
TEST(TsdfHashCounters, GrowCountRisesWhenTheTableIsTooSmall) {
    Engine::Core::Context context;
    TSDF::FlatStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 11; // 의도적으로 작게 -> 리해시를 강제한다
    strategy.Build(context, params);

    std::vector<Eigen::Vector3f> points, normals;
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) {
            points.emplace_back(float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
            normals.emplace_back(0.0f, 0.0f, 1.0f);
        }

    {
        Engine::Compute::CommandBatch batch(context);
        strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
        batch.Submit();
    }
    {
        Engine::Compute::CommandBatch batch(context);
        strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
        batch.Submit();
    }

    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_GT(stats.growCount, 0u);
    EXPECT_GT(stats.slotCapacity, 1u << 11);
    EXPECT_EQ(stats.insertFailureCount, 0u) << "성장이 제때 일어났다면 드롭은 없어야 한다";
}

TEST(TsdfHashStrategy, BucketedIsRegisteredWithItsOwnThreshold) {
    const TSDF::HashStrategy &bucketed = TSDF::HashStrategyByName("bucketed");
    EXPECT_STREQ(bucketed.name, "bucketed");
    EXPECT_STREQ(bucketed.macroName, "HASH_BUCKETED");
    EXPECT_FLOAT_EQ(bucketed.loadFactorLimit, 0.8f);
}

// 같은 스캔을 두 해시로 적분하면 저장된 엔트리 집합이 같아야 한다. 주소 지정만 다를 뿐
// 무엇을 저장하는지는 동일하기 때문이다. 다르면 버킷 구현이 키를 잃고 있다는 뜻이다.
TEST(TsdfHashStrategy, BucketedStoresTheSameEntriesAsLinear) {
    std::vector<Eigen::Vector3f> points, normals;
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) {
            points.emplace_back(float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
            normals.emplace_back(0.0f, 0.0f, 1.0f);
        }

    auto runWith = [&](const char *hashName) {
        Engine::Core::Context context;
        TSDF::FlatStrategy strategy;
        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 16;
        params.hashStrategy = hashName;
        strategy.Build(context, params);

        Engine::Compute::CommandBatch batch(context);
        strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
        batch.Submit();
        return strategy.Stats();
    };

    const TSDF::VolumeStats linear = runWith("linear");
    const TSDF::VolumeStats bucketed = runWith("bucketed");

    EXPECT_EQ(bucketed.occupiedEntryCount, linear.occupiedEntryCount);
    EXPECT_EQ(bucketed.insertFailureCount, 0u);
    EXPECT_GT(linear.occupiedEntryCount, 0u);
}

// Ruling 2 (progress.md): the plan's 81x81 plane at voxel 0.01 with hashCapacity = 1u << 12 would
// need bucketed's own inserts to grow the table within the SAME Record call that produced them --
// but maybeGrow decides from occupancy accumulated by EARLIER calls, so a single call can never
// demonstrate growth from its own inserts. Uses the same 17x17 plane at voxel 0.05 / truncation
// 0.15 fixture as the counters test above (measured at roughly 1045 entries) with
// hashCapacity = 1u << 11 (2048 slots), integrated via TWO Record calls each in its own
// CommandBatch. The first Record sees occupancy 0 and does not grow, inserting ~1045
// (alpha ~= 0.51). The second sees 1045 >= 1024 (linear's 0.5 * 2048) and grows linear to 4096,
// while bucketed's 0.8 * 2048 = 1638 threshold is not crossed and it stays at 2048.
TEST(TsdfHashStrategy, BucketedGrowsLaterThanLinear) {
    std::vector<Eigen::Vector3f> points, normals;
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) {
            points.emplace_back(float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
            normals.emplace_back(0.0f, 0.0f, 1.0f);
        }

    auto capacityAfter = [&](const char *hashName) {
        Engine::Core::Context context;
        TSDF::FlatStrategy strategy;
        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 11; // 의도적으로 작게 -> 리해시를 강제한다
        params.hashStrategy = hashName;
        strategy.Build(context, params);

        {
            Engine::Compute::CommandBatch batch(context);
            strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
            batch.Submit();
        }
        {
            Engine::Compute::CommandBatch batch(context);
            strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
            batch.Submit();
        }
        return strategy.Stats();
    };

    const TSDF::VolumeStats linear = capacityAfter("linear");
    const TSDF::VolumeStats bucketed = capacityAfter("bucketed");

    EXPECT_GT(linear.growCount, 0u) << "이 픽스처는 성장을 강제해야 한다";
    EXPECT_LE(bucketed.slotCapacity, linear.slotCapacity)
            << "버킷은 임계값 0.8이라 선형탐사(0.5)보다 늦게 자라야 한다";
    EXPECT_EQ(bucketed.insertFailureCount, 0u);
}

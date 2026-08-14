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

namespace {
    // Shared by every test in this file that wants "the 17x17 plane at voxel 0.05 / truncation
    // 0.15" -- measured at roughly 1045 entries once integrated (see the growth-fixture comments
    // below). A single generator keeps the geometry byte-identical across tests instead of four
    // copies of the same loop drifting apart.
    void SeventeenBySeventeenPlane(std::vector<Eigen::Vector3f> &points,
                                   std::vector<Eigen::Vector3f> &normals) {
        for (int i = -8; i <= 8; ++i)
            for (int j = -8; j <= 8; ++j) {
                points.emplace_back(float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

    // The origin sits exactly on a TileStrategy tile boundary (confirmed by
    // TsdfVolumeSwitching.EveryStrategyIntegratesTheSameScan's tableCount == 8 for the same-shaped
    // MakePlane fixture in test_tsdf_volume.cpp), so SeventeenBySeventeenPlane -- centered on the
    // origin -- gets split across up to 8 tiles under TileStrategy, diluting every tile's occupancy
    // far below a 2048-slot growth threshold. This is the same plane translated well clear of the
    // origin (offset validated tile-safe by test_tsdf_volume.cpp's MakeDenseSingleTilePlane, which
    // places points from 1.0 to 10.15 in one tile) so every point routes to ONE tile, giving that
    // tile the same ~1045-entry concentration BucketedGrowsLaterThanLinear relies on for
    // FlatStrategy's single window.
    void SeventeenBySeventeenPlaneInOneTile(std::vector<Eigen::Vector3f> &points,
                                            std::vector<Eigen::Vector3f> &normals) {
        constexpr float kOffset = 2.0f;
        for (int i = -8; i <= 8; ++i)
            for (int j = -8; j <= 8; ++j) {
                points.emplace_back(kOffset + float(i) * 0.0375f, kOffset + float(j) * 0.0375f, 0.0f);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }
} // namespace

TEST(TsdfHashCounters, NormalIntegrationDropsNothing) {
    Engine::Core::Context context;
    TSDF::FlatStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 16;
    strategy.Build(context, params);

    std::vector<Eigen::Vector3f> points, normals;
    SeventeenBySeventeenPlane(points, normals);

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
    SeventeenBySeventeenPlane(points, normals);

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
    SeventeenBySeventeenPlane(points, normals);

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
//
// Fix round 2: EXPECT_LE(bucketed, linear) alone cannot fail for the reason this test exists --
// it passes for ANY threshold that is >= linear's, including the stub case where bucketed's
// threshold is ignored entirely and both grow to 4096 (4096 <= 4096 still holds). The strict
// EXPECT_LT plus EXPECT_EQ(bucketed.growCount, 0u) below are what actually prove the threshold is
// wired through to the runtime decision -- not merely stored in the HashStrategy struct (that is
// BucketedIsRegisteredWithItsOwnThreshold's job, and it never runs maybeGrow at all).
TEST(TsdfHashStrategy, BucketedGrowsLaterThanLinear) {
    std::vector<Eigen::Vector3f> points, normals;
    SeventeenBySeventeenPlane(points, normals);

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
    EXPECT_LT(bucketed.slotCapacity, linear.slotCapacity)
            << "버킷은 임계값 0.8이라 선형탐사(0.5)보다 늦게 자라야 한다";
    EXPECT_EQ(bucketed.growCount, 0u) << "이 픽스처(~1045)는 버킷의 0.8*2048=1638 임계값을 넘지 않는다";
    EXPECT_EQ(bucketed.insertFailureCount, 0u);
}

// Fix round 1: AdvancedTSDF.rehash.comp.glsl used to hard-code linear-probe addressing for the
// GROWN table regardless of hash strategy, so a bucketed table's entries survived a grow at
// wangHash(key) % capacity -- a linear-probe slot -- instead of their bucketed
// wangHash(key) % bucketCount address. That slot generally sits outside every bucket a later
// bucketed findSlot/findOrInsert would ever probe for that key, so the entry becomes invisible to
// lookup: an integrate that later touches the same voxel cannot find it and inserts a duplicate.
// This test grows a bucketed table for real and then re-touches pre-grow voxels, which is exactly
// the scenario that trips the bug.
//
// Growth is driven by many small, spatially separated patches rather than one big plane so no
// single Record call ever needs more room than the table currently has: bucketed's 0.8 threshold
// sits comfortably below full capacity (1638 of 2048), so even the largest possible per-patch
// contribution cannot push occupancy past capacity before maybeGrow gets a chance to grow it
// first. Each patch keeps camera and points at the same relative offset (translated together) so
// every patch behaves identically regardless of its position in the sequence.
TEST(TsdfHashStrategy, BucketedSurvivesAGrowIntact) {
    Engine::Core::Context context;
    TSDF::FlatStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 11; // 2048 slots -> bucketed's grow threshold is 1638 (0.8 * 2048)
    params.hashStrategy = "bucketed";
    strategy.Build(context, params);

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
    while (patchIndex < kMaxPatches && strategy.Stats().growCount == 0) {
        // 0.3 apart -- comfortably beyond truncation (0.15) + voxelSize (0.05), so patches never
        // share a voxel and every call's points are genuinely new entries.
        const float offsetX = float(patchIndex) * 0.3f;
        std::vector<Eigen::Vector3f> points, normals;
        patch(offsetX, points, normals);
        Engine::Compute::CommandBatch batch(context);
        strategy.Record(points, normals, Eigen::Vector3f(offsetX, 0.0f, 1.0f), batch);
        batch.Submit();
        ++patchIndex;
    }

    ASSERT_GT(strategy.Stats().growCount, 0u)
            << "이 픽스처는 실제로 성장을 강제해야 한다 (버킷 임계값 0.8을 넘겨야 함)";
    const uint64_t occupiedAfterGrow = strategy.Stats().occupiedEntryCount;

    // Re-touch the FIRST patch -- it predates the grow, so its entries were the ones the rehash
    // moved. Correctly addressed, findOrInsert recognizes the existing keys and accumulates onto
    // them: occupiedEntryCount must not move. With the pre-fix kernel this re-integration cannot
    // find them and inserts brand-new duplicate slots instead.
    {
        std::vector<Eigen::Vector3f> points, normals;
        patch(0.0f, points, normals);
        Engine::Compute::CommandBatch batch(context);
        strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
        batch.Submit();
    }

    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_EQ(stats.insertFailureCount, 0u);
    EXPECT_EQ(stats.occupiedEntryCount, occupiedAfterGrow)
            << "재통합이 occupiedEntryCount를 늘렸다면, 그로우 이후 버킷 주소로 기존 엔트리를 찾지 못해 "
               "중복 삽입한 것이다";

    std::vector<TSDF::AdvancedEntry> downloaded;
    strategy.Download(downloaded);
    EXPECT_EQ(downloaded.size(), stats.occupiedEntryCount)
            << "다운로드된 엔트리 수와 occupiedEntryCount가 어긋나면 그로우 이후 테이블이 일관성을 잃은 것";
}

#include "TSDF/Memory/SubmapStrategy.h"
#include "TSDF/Memory/TileStrategy.h"

// Fix round 2: promoted from a throwaway smoke test. TiledDirectionalTSDF::GetTSDF picks between
// AdvancedTSDF's hash-aware Build and its plain 6-argument fallback via
// if constexpr (HasHashStrategyBuild<Backend>::value) -- and that trait fails SILENTLY. Every
// AdvancedTSDF::Build parameter is defaulted, so if the trait ever evaluated false the
// 6-argument call would still compile clean and quietly build LINEAR tables while
// params.hashStrategy == "bucketed": no compile error, no failing assertion, healthy-looking
// counters throughout. Task 6 measures tile and submap specifically under bucketed, so a
// silently-linear tiled/submap build would corrupt that headline number with no symptom
// anywhere else. This is regression protection for that branch, not a behavioural claim about
// bucketed's addressing -- BucketedStoresTheSameEntriesAsLinear and BucketedSurvivesAGrowIntact
// already cover that through FlatStrategy.
TEST(TsdfHashStrategy, TileAndSubmapBuildAndRunWithBucketed) {
    std::vector<Eigen::Vector3f> points, normals;
    SeventeenBySeventeenPlane(points, normals);

    {
        Engine::Core::Context context;
        TSDF::TileStrategy strategy;
        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 16;
        params.hashStrategy = "bucketed";
        strategy.Build(context, params);

        Engine::Compute::CommandBatch batch(context);
        strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
        batch.Submit();

        const TSDF::VolumeStats stats = strategy.Stats();
        EXPECT_GT(stats.occupiedEntryCount, 0u);
        EXPECT_EQ(stats.insertFailureCount, 0u);
    }
    {
        Engine::Core::Context context;
        TSDF::SubmapStrategy strategy;
        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 16;
        params.hashStrategy = "bucketed";
        strategy.Build(context, params);

        Engine::Compute::CommandBatch batch(context);
        strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
        batch.Submit();

        const TSDF::VolumeStats stats = strategy.Stats();
        EXPECT_GT(stats.occupiedEntryCount, 0u);
        EXPECT_EQ(stats.insertFailureCount, 0u);
    }
}

// Ruling 6 (controller, task 6): TileAndSubmapBuildAndRunWithBucketed above only asserts
// occupiedEntryCount > 0 and insertFailureCount == 0 -- both true whether the tile's own
// AdvancedTSDF::Build actually received the bucketed strategy or silently fell back to linear,
// because its hashCapacity (1<<16) is nowhere near either threshold. This test discriminates the
// same way BucketedGrowsLaterThanLinear does for FlatStrategy: force a grow at a small capacity and
// assert bucketed's slot count stays strictly below linear's. If TiledDirectionalTSDF::GetTSDF's
// HasHashStrategyBuild dispatch ever silently drops the strategy for the tile's Build call, both
// grow to the same capacity and EXPECT_LT fails.
//
// Uses SeventeenBySeventeenPlaneInOneTile, not the plain SeventeenBySeventeenPlane: the plain
// fixture straddles the origin tile boundary and splits across 8 tiles (see that helper's comment),
// which dilutes every tile's occupancy so far below a 2048-slot threshold that NEITHER hash grows --
// this was verified empirically (linear.growCount stayed 0). Concentrating the plane in one tile
// restores the single-table growth semantics BucketedGrowsLaterThanLinear depends on, which is what
// this test actually needs: proof that ONE tile's Build call was given the right strategy, not
// anything about routing across tiles.
TEST(TsdfHashStrategy, BucketedTileGrowsLaterThanLinear) {
    std::vector<Eigen::Vector3f> points, normals;
    SeventeenBySeventeenPlaneInOneTile(points, normals);

    auto capacityAfter = [&](const char *hashName) {
        Engine::Core::Context context;
        TSDF::TileStrategy strategy;
        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 11; // per-tile capacity, deliberately small -> forces a grow
        params.hashStrategy = hashName;
        strategy.Build(context, params);

        {
            Engine::Compute::CommandBatch batch(context);
            strategy.Record(points, normals, Eigen::Vector3f(2.0f, 2.0f, 1.0f), batch);
            batch.Submit();
        }
        {
            Engine::Compute::CommandBatch batch(context);
            strategy.Record(points, normals, Eigen::Vector3f(2.0f, 2.0f, 1.0f), batch);
            batch.Submit();
        }
        return strategy.Stats();
    };

    const TSDF::VolumeStats linear = capacityAfter("linear");
    const TSDF::VolumeStats bucketed = capacityAfter("bucketed");

    EXPECT_GT(linear.growCount, 0u) << "이 픽스처는 성장을 강제해야 한다";
    EXPECT_LT(bucketed.slotCapacity, linear.slotCapacity)
            << "타일 경로도 실제로 버킷 임계값 0.8을 쓴다면 선형탐사(0.5)보다 늦게 자라야 한다 -- "
               "같으면 타일이 조용히 선형으로 폴백한 것";
    EXPECT_EQ(bucketed.insertFailureCount, 0u);
    EXPECT_EQ(linear.insertFailureCount, 0u);
}

#include "TSDF/ComposedVolume.h"

// Task 6's Step 1 fixture: every registered strategy must implement Extract and return a
// non-empty, points/normals-parallel cloud for the same simple plane scan the other tests in this
// file use.
TEST(TsdfVolumeExtract, EveryStrategyExtractsAPointCloud) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();
    std::vector<Eigen::Vector3f> points, normals;
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) {
            points.emplace_back(float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
            normals.emplace_back(0.0f, 0.0f, 1.0f);
        }

    for (const std::string &name: registry.Names()) {
        Engine::Core::Context context;
        std::unique_ptr<TSDF::Volume> volume = registry.Create(name);
        ASSERT_NE(volume, nullptr) << name;

        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 16;
        volume->Build(context, params);
        volume->Integrate(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f));

        const Engine::Core::OrientedPointCloud cloud = volume->Extract(/*merge=*/true);
        EXPECT_GT(cloud.points.size(), 0u) << name;
        EXPECT_EQ(cloud.points.size(), cloud.normals.size()) << name;
    }
}

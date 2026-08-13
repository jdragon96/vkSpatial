#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/AdvancedTSDF.h"
#include "Engine/Spatial/SubmapAdvancedTSDF.h"
#include "Engine/Spatial/TiledAdvancedTSDF.h"
#include "TSDF/ComposedVolume.h"
#include "TSDF/Memory/FlatStrategy.h"
#include "TSDF/Memory/SubmapStrategy.h"
#include "TSDF/Memory/TileStrategy.h"
#include "TSDF/Volume.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <map>
#include <string>
#include <vector>

using Eigen::Vector3f;

namespace {

    // +Z를 향하는 평면 패치. span 폭을 (2*half+1)^2 점으로 채운다.
    void MakePlane(std::vector<Vector3f> &points, std::vector<Vector3f> &normals,
                   float span, int half) {
        points.clear();
        normals.clear();
        const float step = span / float(2 * half);
        for (int i = -half; i <= half; ++i)
            for (int j = -half; j <= half; ++j) {
                points.emplace_back(float(i) * step, float(j) * step, 0.0f);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

    // 한 타일 안에만 들어가는 조밀한 +Z 평면. 원점 중심이 아니라 양의 사분면으로 밀어 두었기
    // 때문에 (voxel 20..202, tile core 448, ghost 4) 모든 점이 정확히 타일 하나로 라우팅되고
    // 고스트 중복이 없다. 즉 이 클라우드는 통째로 단일 타일의 업로드 버퍼로 들어간다 --
    // maxPointsPerFrame 클램프가 살아 있으면 뒷부분이 잘려 나가는 유일한 조건.
    void MakeDenseSingleTilePlane(std::vector<Vector3f> &points, std::vector<Vector3f> &normals,
                                  float voxelSize, int side) {
        points.clear();
        normals.clear();
        for (int i = 0; i < side; ++i)
            for (int j = 0; j < side; ++j) {
                points.emplace_back(1.0f + float(i) * voxelSize, 1.0f + float(j) * voxelSize, 0.0f);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

} // namespace

TEST(TsdfAccessors, AdvancedReportsHashCapacity) {
    Engine::Core::Context context;
    Engine::Spatial::AdvancedTSDF tsdf;
    tsdf.Build(context, 0.05f, 0.15f, 1u << 16, 1u << 15);
    EXPECT_EQ(tsdf.HashCapacity(), 1u << 16);
}

TEST(TsdfAccessors, TiledSlotCapacityScalesWithTiles) {
    Engine::Core::Context context;
    Engine::Spatial::TiledAdvancedTSDF tsdf;
    tsdf.Build(context, 0.05f, 0.15f, 1u << 16, 1u << 15);
    EXPECT_EQ(tsdf.SlotCapacity(), 0u) << "타일이 아직 없으면 용량도 0";

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);
    tsdf.Integrate(points, normals, Vector3f(0.0f, 0.0f, 1.0f));

    EXPECT_GE(tsdf.TileCount(), 1u);
    EXPECT_EQ(tsdf.SlotCapacity(), uint64_t(tsdf.TileCount()) * (1u << 16));
}

TEST(TsdfAccessors, SubmapSumsBaseAndDetail) {
    Engine::Core::Context context;
    Engine::Spatial::SubmapAdvancedTSDF tsdf;
    tsdf.Build(context, 0.05f, 0.15f, 32, 4.0f, 1u << 16, 1u << 15);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);
    tsdf.Integrate(points, normals, Vector3f(0.0f, 0.0f, 1.0f));

    const uint64_t expected =
            uint64_t(tsdf.BaseTileCount() + tsdf.DetailTileCount()) * (1u << 16);
    EXPECT_EQ(tsdf.SlotCapacity(), expected);
    EXPECT_GT(tsdf.FilledCount(), 0u);
}

TEST(TsdfRegistry, UnknownNameReturnsNull) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();
    EXPECT_FALSE(registry.Has("no-such-volume"));
    EXPECT_EQ(registry.Create("no-such-volume"), nullptr);
}

TEST(TsdfRegistry, NamesAreSorted) {
    TSDF::VolumeRegistry registry;
    registry.Register("zulu", [] { return std::unique_ptr<TSDF::Volume>(); });
    registry.Register("alpha", [] { return std::unique_ptr<TSDF::Volume>(); });
    const std::vector<std::string> names = registry.Names();
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "alpha");
    EXPECT_EQ(names[1], "zulu");
}

TEST(TsdfFlatStrategy, StatsTrackFillAndCapacity) {
    Engine::Core::Context context;
    TSDF::FlatStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 16;
    strategy.Build(context, params);

    EXPECT_STREQ(strategy.Name(), "flat");
    EXPECT_EQ(strategy.Stats().occupiedEntryCount, 0u);
    EXPECT_EQ(strategy.Stats().slotCapacity, 1u << 16);
    EXPECT_EQ(strategy.Stats().tableCount, 1u);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);
    Engine::Compute::CommandBatch batch(context);
    strategy.Record(points, normals, Vector3f(0.0f, 0.0f, 1.0f), batch);
    batch.Submit();

    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_GT(stats.occupiedEntryCount, 0u);
    EXPECT_GT(stats.LoadFactor(), 0.0);
    EXPECT_LT(stats.LoadFactor(), 1.0);
    EXPECT_EQ(stats.deviceMemoryBytes, stats.slotCapacity * TSDF::kBytesPerHashSlot);

    std::vector<Engine::Spatial::AdvancedEntry> entries;
    strategy.Download(entries);
    EXPECT_EQ(entries.size(), stats.occupiedEntryCount);
}

TEST(TsdfFlatStrategy, StatsAreZeroBeforeBuild) {
    TSDF::FlatStrategy strategy;
    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_EQ(stats.slotCapacity, 0u);
    EXPECT_EQ(stats.LoadFactor(), 0.0);
    EXPECT_EQ(strategy.Device(), nullptr);
}

TEST(TsdfTileStrategy, TableCountFollowsTiles) {
    Engine::Core::Context context;
    TSDF::TileStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 16;
    strategy.Build(context, params);

    EXPECT_STREQ(strategy.Name(), "tile");
    EXPECT_EQ(strategy.Stats().tableCount, 0u) << "타일은 스캔이 닿을 때 생긴다";

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);
    Engine::Compute::CommandBatch batch(context);
    strategy.Record(points, normals, Vector3f(0.0f, 0.0f, 1.0f), batch);
    batch.Submit();

    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_GE(stats.tableCount, 1u);
    EXPECT_EQ(stats.slotCapacity, uint64_t(stats.tableCount) * (1u << 16));
    EXPECT_GT(stats.occupiedEntryCount, 0u);
    EXPECT_EQ(stats.deviceMemoryBytes, stats.slotCapacity * TSDF::kBytesPerHashSlot);

    std::vector<Engine::Spatial::AdvancedEntry> entries;
    strategy.Download(entries);
    EXPECT_GT(entries.size(), 0u);
}

TEST(TsdfSubmapStrategy, IntegratesDespiteSelfSubmittingBackend) {
    Engine::Core::Context context;
    TSDF::SubmapStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 16;
    strategy.Build(context, params);

    EXPECT_STREQ(strategy.Name(), "submap");

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);
    // 일부러 Submit하지 않는다: 이 전략은 batch에 아무것도 기록하지 않으므로 제출할 것이
    // 없고, 빈 batch 제출이 안전한지는 이 태스크가 검증할 대상이 아니다.
    Engine::Compute::CommandBatch batch(context);
    strategy.Record(points, normals, Vector3f(0.0f, 0.0f, 1.0f), batch);

    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_GT(stats.occupiedEntryCount, 0u) << "batch를 제출하지 않아도 적분은 끝나 있어야 한다";
    EXPECT_GE(stats.tableCount, 1u);
    EXPECT_EQ(stats.deviceMemoryBytes, stats.slotCapacity * TSDF::kBytesPerHashSlot);
}

TEST(TsdfRegistry, DefaultRegistersAllMemoryStrategies) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();
    EXPECT_EQ(registry.Names(), (std::vector<std::string>{"flat", "submap", "tile"}));
}

// 이름만 바꿔가며 같은 스캔을 적분하고 통계를 나란히 비교한다. 이것이 이 계획의 최종 산출물 --
// 메모리가 점유율(load factor)에 묶여 있는지 타일 개수에 묶여 있는지 판정하는 도구.
// 개별 전략이 정상인지만 보면 안 되고, 세 전략의 수치가 한 표에 모여야 판정이 된다.
TEST(TsdfVolumeSwitching, EveryStrategyIntegratesTheSameScan) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);

    std::map<std::string, TSDF::VolumeStats> byStrategy;

    std::printf("\n[ STRATEGY COMPARISON ] same %zu-point scan, hashCapacity=%u per table\n",
                points.size(), 1u << 16);
    std::printf("%-8s %10s %12s %8s %7s %14s\n", "name", "occupied", "slots", "load", "tables",
                "tableBytes");

    for (const std::string &name: registry.Names()) {
        Engine::Core::Context context;
        std::unique_ptr<TSDF::Volume> volume = registry.Create(name);
        ASSERT_NE(volume, nullptr) << name;
        EXPECT_EQ(std::string(volume->Name()), name);

        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 16;
        volume->Build(context, params);
        volume->Integrate(points, normals, Vector3f(0.0f, 0.0f, 1.0f));

        const TSDF::VolumeStats stats = volume->Stats();
        EXPECT_GT(stats.occupiedEntryCount, 0u) << name;
        EXPECT_GT(stats.slotCapacity, 0u) << name;
        EXPECT_GE(stats.tableCount, 1u) << name;
        // VACUOUS UNTIL THE HASH AXIS LANDS: no strategy increments insertFailureCount yet, so
        // this asserts a hard-coded 0, NOT that probing kept every voxel. Do not cite it as
        // evidence that nothing was dropped until a strategy actually reports probe failures.
        EXPECT_EQ(stats.insertFailureCount, 0u) << name << ": 복셀이 조용히 드롭되면 안 된다";

        std::vector<Engine::Spatial::AdvancedEntry> entries;
        volume->Download(entries);
        EXPECT_GT(entries.size(), 0u) << name;

        byStrategy[name] = stats;

        std::printf("%-8s %10llu %12llu %8.5f %7u %14llu\n", name.c_str(),
                    static_cast<unsigned long long>(stats.occupiedEntryCount),
                    static_cast<unsigned long long>(stats.slotCapacity), stats.LoadFactor(),
                    stats.tableCount,
                    static_cast<unsigned long long>(stats.deviceMemoryBytes));
        RecordProperty(name + ".occupiedEntryCount",
                       static_cast<int>(stats.occupiedEntryCount));
        RecordProperty(name + ".slotCapacity", static_cast<int>(stats.slotCapacity));
        RecordProperty(name + ".tableCount", static_cast<int>(stats.tableCount));
    }
    std::fflush(stdout);

    // 진짜 상대 비교 하나: 이 스캔은 단일 512^3 윈도우에 들어가므로 flat은 테이블 하나로 끝난다.
    // 같은 스캔을 타일로 나누면 타일 수만큼 테이블이 생기니, 타일 쪽 슬롯 수가 flat보다 적을 수는
    // 없다. 이 부등식이 깨지면 어느 한쪽이 스캔의 일부를 삼킨 것이다.
    ASSERT_EQ(byStrategy.count("flat"), 1u);
    ASSERT_EQ(byStrategy.count("tile"), 1u);
    EXPECT_GE(byStrategy["tile"].slotCapacity, byStrategy["flat"].slotCapacity)
            << "단일 윈도우에 들어가는 스캔에서 per-tile 테이블이 더 쌀 수는 없다";
}

// Configure가 실제로 GPU까지 닿는지 검증한다. currentFrame은 슬롯이 처음 채워질 때 GPU가
// 찍는 값이라, 되돌아온 모든 엔트리의 firstFrame이 7이면 Configure -> 전략 -> 백엔드 ->
// (타일 생성 포함) -> 셰이더 배선이 끝까지 살아 있다는 뜻이다. 이 배선은 이미 한 번
// 깨진 적이 있고(SubmapStrategy가 5개 중 2개만 전달), 커버리지가 없어 통과했었다.
TEST(TsdfVolumeSwitching, ConfigureReachesTheBackendOnEveryStrategy) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);

    constexpr int kDistinctiveFrame = 7; // 기본값 0과 구별되는 값이어야 의미가 있다

    for (const std::string &name: registry.Names()) {
        Engine::Core::Context context;
        std::unique_ptr<TSDF::Volume> volume = registry.Create(name);
        ASSERT_NE(volume, nullptr) << name;

        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 16;
        volume->Build(context, params);

        TSDF::IntegrationOptions options;
        options.currentFrame = kDistinctiveFrame;
        volume->Configure(options); // 첫 적분 전에 -- 지연 생성 타일은 소급 적용되지 않는다
        volume->Integrate(points, normals, Vector3f(0.0f, 0.0f, 1.0f));

        std::vector<Engine::Spatial::AdvancedEntry> entries;
        volume->Download(entries);
        // 먼저 비어 있지 않음을 확인해야 아래 루프가 공허하게 통과하지 않는다.
        ASSERT_GT(entries.size(), 0u) << name << ": 적분 결과가 비면 검사가 무의미하다";
        for (const Engine::Spatial::AdvancedEntry &entry: entries)
            ASSERT_EQ(entry.firstFrame, kDistinctiveFrame)
                    << name << ": Configure의 currentFrame이 GPU까지 전달되지 않았다";
    }
}

// 프레임이 maxPointsPerFrame보다 커도 점을 버리지 않는지 검증한다.
// maxPointsPerFrame은 업로드 버퍼 크기 힌트일 뿐 용량 상한이 아니다. 클램프하는 경로
// (AdvancedTSDF::RecordIntegrate)를 쓰면 한 타일에 32768점을 넘겨 보낼 때 뒷부분이 조용히
// 잘리고 occupiedEntryCount가 낮게 보고된다 -- 타일링이 실제보다 싸 보이는 방향의 편향.
// 같은 클라우드를 (a) 기본 힌트와 (b) 클라우드 전체를 담는 힌트로 각각 적분해 점유 수가
// 같아야 한다고 못 박는다. 클램프가 되살아나면 (a)만 줄어들어 즉시 깨진다.
TEST(TsdfVolumeSwitching, LargeFrameIsNotTruncatedByMaxPointsPerFrame) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();

    constexpr float kVoxelSize = 0.05f;
    std::vector<Vector3f> points, normals;
    MakeDenseSingleTilePlane(points, normals, kVoxelSize, 183); // 33489 점 > 기본 32768
    const TSDF::VolumeParams defaults;
    ASSERT_GT(points.size(), defaults.maxPointsPerFrame)
            << "클램프를 건드리려면 프레임이 기본 힌트보다 커야 한다";

    auto occupancyWithHint = [&](const std::string &name, uint32_t maxPointsPerFrame) {
        Engine::Core::Context context;
        std::unique_ptr<TSDF::Volume> volume = registry.Create(name);
        EXPECT_NE(volume, nullptr) << name;
        TSDF::VolumeParams params;
        params.voxelSize = kVoxelSize;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 18;
        params.maxPointsPerFrame = maxPointsPerFrame;
        volume->Build(context, params);
        volume->Integrate(points, normals, Vector3f(0.0f, 0.0f, 1.0f));
        return volume->Stats().occupiedEntryCount;
    };

    for (const std::string &name: registry.Names()) {
        const uint64_t clamped = occupancyWithHint(name, defaults.maxPointsPerFrame);
        const uint64_t roomy = occupancyWithHint(name, uint32_t(points.size()));
        EXPECT_GT(roomy, 0u) << name;
        EXPECT_EQ(clamped, roomy)
                << name << ": maxPointsPerFrame이 점을 삼켰다 (프레임 " << points.size()
                << "점, 힌트 " << defaults.maxPointsPerFrame << ")";
    }
}

TEST(TsdfVolumeSwitching, ResetEmptiesTheVolume) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);

    // 세 전략 모두 확인한다: Reset 후 남는 것은 전략마다 다르지만(단일 윈도우는 테이블을
    // 유지하고 타일 전략은 타일을 버린다) occupiedEntryCount == 0은 공통 계약이다.
    for (const std::string &name: registry.Names()) {
        Engine::Core::Context context;
        std::unique_ptr<TSDF::Volume> volume = registry.Create(name);
        ASSERT_NE(volume, nullptr) << name;

        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 16;
        volume->Build(context, params);

        volume->Integrate(points, normals, Vector3f(0.0f, 0.0f, 1.0f));
        ASSERT_GT(volume->Stats().occupiedEntryCount, 0u) << name;

        volume->Reset();
        EXPECT_EQ(volume->Stats().occupiedEntryCount, 0u) << name;
        EXPECT_EQ(volume->Download().size(), 0u) << name;
    }
}

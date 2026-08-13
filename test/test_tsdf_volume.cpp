#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/AdvancedTSDF.h"
#include "Engine/Spatial/SubmapAdvancedTSDF.h"
#include "Engine/Spatial/TiledAdvancedTSDF.h"
#include "TSDF/Memory/FlatStrategy.h"
#include "TSDF/Volume.h"

#include <gtest/gtest.h>

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

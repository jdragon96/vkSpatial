#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/AdvancedTSDF.h"
#include "Engine/Spatial/SubmapAdvancedTSDF.h"
#include "Engine/Spatial/TiledAdvancedTSDF.h"

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

#include "Engine/Core/Context.h"
#include "TSDF/TSDF.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using Eigen::Vector3f;

namespace {

    // A 0.05 m sphere sampled at ~0.0025 m -- the fixture DenseRegionClassify's own tests use for
    // "strong curvature, finely sampled". Same shape here so the pipeline test exercises a case
    // the classifier is already known to call dense.
    void MakeSphere(std::vector<Vector3f> &points, std::vector<Vector3f> &normals) {
        points.clear();
        normals.clear();
        const float radius = 0.05f;
        for (int a = 0; a < 180; ++a)
            for (int b = 0; b < 90; ++b) {
                const float theta = float(a) * float(M_PI) / 90.0f;
                const float phi = float(b) * float(M_PI) / 180.0f;
                const Vector3f direction(std::sin(phi) * std::cos(theta),
                                         std::sin(phi) * std::sin(theta), std::cos(phi));
                points.push_back(direction * radius);
                normals.push_back(direction);
            }
    }

    void SeventeenBySeventeenPlane(std::vector<Vector3f> &points, std::vector<Vector3f> &normals) {
        points.clear();
        normals.clear();
        for (int i = -8; i <= 8; ++i)
            for (int j = -8; j <= 8; ++j) {
                points.emplace_back(float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

    TSDFConfiguration MakeConfig(const std::string &splitter) {
        TSDFConfiguration config;
        config.splitter = splitter;
        config.backendConfig.voxelSize = 0.01f;
        config.backendConfig.truncation = 0.03f;
        config.backendConfig.hashCapacity = 1u << 16;
        config.backendConfig.maxPointPerFrame = 1u << 14;
        config.splitterConfig.baseResolution = 0.01f;
        config.splitterConfig.maxPointPerFrame = 1 << 14;
        return config;
    }

} // namespace

TEST(TSDFPipeline, IntegratesAndExtractsThroughTheSelectedBackend) {
    Engine::Core::Context context;
    TSDF tsdf;
    tsdf.Build(context, MakeConfig("dense"));

    std::vector<Vector3f> points, normals;
    MakeSphere(points, normals);
    tsdf.Integrate(points, normals);

    EXPECT_EQ(tsdf.LastDivision().baseIndex.size() + tsdf.LastDivision().detailIndex.size(),
              points.size())
            << "every point must land in exactly one level";
    EXPECT_EQ(tsdf.LastDivision().blockInsertFailureCount, 0u);
    EXPECT_EQ(tsdf.LastDivision().cellInsertFailureCount, 0u);
    EXPECT_GT(tsdf.Extract().points.size(), 0u);
}

// Switching the strategy changes where the points go, and nothing else about the pipeline.
TEST(TSDFPipeline, NoSubmapStrategySendsEveryPointToBase) {
    Engine::Core::Context context;
    TSDF tsdf;
    tsdf.Build(context, MakeConfig("none"));

    std::vector<Vector3f> points, normals;
    MakeSphere(points, normals);
    tsdf.Integrate(points, normals);

    EXPECT_EQ(tsdf.LastDivision().baseIndex.size(), points.size());
    EXPECT_TRUE(tsdf.LastDivision().detailIndex.empty());
    EXPECT_GT(tsdf.Extract().points.size(), 0u);
}

// The dense strategy is only worth its second level if it actually finds one.
TEST(TSDFPipeline, DenseStrategyRoutesSomePointsToTheDetailLevel) {
    Engine::Core::Context context;
    TSDF tsdf;
    tsdf.Build(context, MakeConfig("dense"));

    std::vector<Vector3f> points, normals;
    MakeSphere(points, normals);
    for (int frame = 0; frame < 2; ++frame) tsdf.Integrate(points, normals);

    EXPECT_GT(tsdf.LastDivision().denseBlockCount, 0u);
    EXPECT_GT(tsdf.LastDivision().detailIndex.size(), 0u);
    EXPECT_GT(tsdf.LastDivision().detailSlotEstimate, 0u);
}

// A single 512^3 window at 0.01 m reaches +/-2.56 m from its origin. Geometry past that needs a
// second window, and before routing existed it simply vanished.
TEST(TSDFPipeline, GeometryBeyondOneWindowGetsItsOwnWindow) {
    Engine::Core::Context context;
    TSDF tsdf;
    tsdf.Build(context, MakeConfig("none"));

    std::vector<Vector3f> points, normals;
    MakeSphere(points, normals);

    tsdf.Integrate(points, normals);
    const size_t nearWindows = tsdf.WindowCount();
    const size_t nearExtracted = tsdf.Extract().points.size();
    ASSERT_GT(nearExtracted, 0u);

    // 40 m out: far outside any window already built, and outside a single origin-centred one.
    std::vector<Vector3f> farPoints = points;
    for (Vector3f &p: farPoints) p += Vector3f(40.0f, 0.0f, 0.0f);
    tsdf.Integrate(farPoints, normals);

    EXPECT_GT(tsdf.WindowCount(), nearWindows) << "the far patch must open its own window";
    EXPECT_EQ(tsdf.WindowLimitRefusalCount(), 0u);

    const Engine::Core::OrientedPointCloud cloud = tsdf.Extract();
    EXPECT_GT(cloud.points.size(), nearExtracted) << "the far patch must survive extraction";

    const auto far = std::count_if(cloud.points.begin(), cloud.points.end(),
                                   [](const Vector3f &p) { return p.x() > 20.0f; });
    EXPECT_GT(far, 0) << "extracted geometry must include the far patch";
}

// The window ceiling must be observable, not a silent loss.
TEST(TSDFPipeline, WindowCeilingRefusesAndCounts) {
    Engine::Core::Context context;
    TSDF tsdf;
    TSDFConfiguration config = MakeConfig("none");
    config.maxResidentWindow = 1;
    tsdf.Build(context, config);

    std::vector<Vector3f> points, normals;
    MakeSphere(points, normals);
    tsdf.Integrate(points, normals);
    ASSERT_EQ(tsdf.WindowCount(), 1u);
    // The sphere straddles several windows already -- the world origin is a window CORNER -- so
    // some of it is refused on this frame too. The far patch's refusals are the delta.
    const uint32_t refusedNear = tsdf.WindowLimitRefusalCount();

    std::vector<Vector3f> farPoints = points;
    for (Vector3f &p: farPoints) p += Vector3f(40.0f, 0.0f, 0.0f);
    tsdf.Integrate(farPoints, normals);

    EXPECT_EQ(tsdf.WindowCount(), 1u);
    EXPECT_EQ(tsdf.WindowLimitRefusalCount() - refusedNear, farPoints.size())
            << "every refused point must be counted, none silently dropped";
}

TEST(TSDFPipeline, RegistriesAgreeWithWhatBuildAccepts) {
    Engine::Core::Context context;

    for (const std::string &name: TSDFBackendNames()) {
        TSDF tsdf;
        TSDFConfiguration config = MakeConfig("none");
        config.backend = name;
        EXPECT_NO_THROW(tsdf.Build(context, config)) << name;
    }
    for (const std::string &name: DataSplitterNames()) {
        TSDF tsdf;
        EXPECT_NO_THROW(tsdf.Build(context, MakeConfig(name))) << name;
    }

    TSDF unknownBackend;
    TSDFConfiguration config = MakeConfig("none");
    config.backend = "nope";
    EXPECT_THROW(unknownBackend.Build(context, config), std::runtime_error);

    TSDF unknownStrategy;
    EXPECT_THROW(unknownStrategy.Build(context, MakeConfig("nope")), std::runtime_error);
}

// The hash choice reaches every window the pipeline opens, not just a standalone backend.
TEST(TsdfHashStrategy, PipelineRunsWithBucketedAcrossWindows) {
    std::vector<Eigen::Vector3f> points, normals;
    SeventeenBySeventeenPlane(points, normals);

    auto runWith = [&](const char *hashName) {
        Engine::Core::Context context;
        TSDF tsdf;
        TSDFConfiguration config;
        config.splitter = "none";
        config.backendConfig.voxelSize = 0.05f;
        config.backendConfig.truncation = 0.15f;
        config.backendConfig.hashCapacity = 1u << 14;
        config.backendConfig.hash = hashName;
        tsdf.Build(context, config);

        tsdf.Integrate(points, normals);
        return tsdf.Stats();
    };

    const TSDFBackendStats linear = runWith("linear");
    const TSDFBackendStats bucketed = runWith("bucketed");

    EXPECT_GT(linear.tableCount, 0u);
    EXPECT_EQ(bucketed.tableCount, linear.tableCount);
    EXPECT_EQ(bucketed.filledCount, linear.filledCount);
    EXPECT_EQ(bucketed.insertFailureCount, 0u);
    EXPECT_EQ(linear.insertFailureCount, 0u);
}

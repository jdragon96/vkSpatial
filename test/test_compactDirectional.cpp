// Unit tests for CompactDirectionalTSDF: (1) directional-tier accuracy against an analytic
// cube ground truth (mirrors tsdf_benchmark.cpp's Compact-Directional row, which measures
// cube RMSE ~0.022mm -- see docs/MRHASH_VS_DIRECTIONAL_TSDF.md), and (2) the origin-relative
// movable window (Build's windowMinCorner) reproducing that SAME accuracy for a scene placed
// far from the world origin -- proving the 512^3-voxel hash window is no longer pinned at
// [-25.6,25.6]mm around (0,0,0).
//
// Reuses example2/shape_fixtures.h (render-free analytic cube/cylinder fixtures shared with
// test_simpletsdf_variance.cpp) instead of duplicating a sampler.
#include <gtest/gtest.h>

#include "shape_fixtures.h"

#include "Engine/Core/Context.h"
#include "TSDF/Backends/CompactDirectionalTSDF.h"

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

using fixtures::Shape;

namespace {
    // Builds a Context once; skips the test if Vulkan is unavailable (mirrors
    // test_spatialIndex.cpp's / test_simpletsdf_variance.cpp's CtxHolder).
    struct CtxHolder {
        std::unique_ptr<Engine::Core::Context> ctx;
        bool ok = false;
        CtxHolder() {
            try {
                ctx = std::make_unique<Engine::Core::Context>();
                ok = true;
            } catch (const std::exception &) {
                ok = false;
            }
        }
    };

    // Running mean + RMSE over per-point ground-truth errors (mirrors tsdf_benchmark.cpp's
    // ErrStats).
    struct ErrStats {
        double sum = 0.0, sumSq = 0.0;
        std::size_t count = 0;
        void add(double e) {
            sum += e;
            sumSq += e * e;
            ++count;
        }
        double mean() const { return count ? sum / double(count) : 0.0; }
        double rmse() const { return count ? std::sqrt(sumSq / double(count)) : 0.0; }
    };

    constexpr float kVoxel = 0.1f;
    constexpr float kTruncation = 0.3f;
    // maxDirections=3 matches tsdf_benchmark's default --maxdir; dirExponent=4,
    // viewAngleWeight=true matches RunCompactDirectional's SetIntegrationQuality call.
    const Engine::Spatial::IntegrationQuality kQuality{3, 4, true};

    // Directional tier threshold: tsdf_benchmark measures cube RMSE ~0.022mm for
    // Compact-Directional (docs/MRHASH_VS_DIRECTIONAL_TSDF.md), clearly below the
    // Simple(fine)/unweighted baseline's ~0.05-0.067mm. 0.035 gives headroom while still
    // failing if directional integration/extraction regresses toward the Simple tier.
    constexpr double kDirectionalRmseCeiling = 0.035;
} // namespace

TEST(CompactDirectionalTSDF, AccuracyNearDirectional) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    Engine::Spatial::CompactDirectionalTSDF cd;
    cd.Build(*h.ctx, kVoxel, kTruncation); // default window: origin voxel (-256,-256,-256)
    cd.SetIntegrationQuality(kQuality);
    EXPECT_EQ(cd.OriginVoxel(), Eigen::Vector3i(-256, -256, -256));

    const std::vector<fixtures::View> views = fixtures::SampleViews(Shape::Cube, kVoxel);
    ASSERT_FALSE(views.empty());
    for (const auto &v : views) cd.Integrate(v.points, v.normals, v.camPos);

    const Engine::Spatial::OrientedPointCloud cloud = cd.ExtractPointCloud();
    ASSERT_FALSE(cloud.points.empty()) << "extraction produced no surface points";

    ErrStats stats;
    for (const auto &p : cloud.points) stats.add(double(fixtures::NearestDistance(Shape::Cube, p)));

    EXPECT_LT(stats.rmse(), kDirectionalRmseCeiling)
            << "mean=" << stats.mean() << " rmse=" << stats.rmse() << " n=" << stats.count;
}

TEST(CompactDirectionalTSDF, OriginRelativeWindowWorks) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    // Same cube fixture, translated so it sits at world (100,100,100) -- entirely outside the
    // OLD fixed window ([-256,255] voxels == [-25.6,25.6]mm at voxel=0.1, around the origin).
    const Eigen::Vector3f shift(100.0f, 100.0f, 100.0f);
    std::vector<fixtures::View> views = fixtures::SampleViews(Shape::Cube, kVoxel);
    ASSERT_FALSE(views.empty());
    for (auto &v : views) {
        v.camPos += shift;
        for (auto &p : v.points) p += shift;
    }

    // windowMinCorner placed so the shifted cube (half-extent 1.5 + truncation band) fits
    // comfortably inside the 512^3-voxel (51.2mm-per-axis at voxel=0.1) window.
    const Eigen::Vector3f windowMin = shift - Eigen::Vector3f::Constant(2.5f);

    Engine::Spatial::CompactDirectionalTSDF cd;
    cd.Build(*h.ctx, kVoxel, kTruncation, 1u << 20, 1u << 15, windowMin);
    cd.SetIntegrationQuality(kQuality);

    // The window moved: origin voxel is no longer the old default (-256,-256,-256).
    EXPECT_NE(cd.OriginVoxel(), Eigen::Vector3i(-256, -256, -256));

    for (const auto &v : views) cd.Integrate(v.points, v.normals, v.camPos);

    const Engine::Spatial::OrientedPointCloud cloud = cd.ExtractPointCloud();
    ASSERT_FALSE(cloud.points.empty())
            << "movable-window extraction produced no surface points -- window did not move";

    ErrStats stats;
    for (const auto &p : cloud.points) {
        // Un-shift back to the fixture's native (origin-centred) frame before scoring against
        // fixtures::NearestDistance, which assumes a shape centred at (0,0,0).
        const Eigen::Vector3f local = p - shift;
        stats.add(double(fixtures::NearestDistance(Shape::Cube, local)));
    }

    // Same accuracy tier as the origin-centred case -- the movable window changes WHERE the
    // hash covers, not the integration/extraction logic, so accuracy should be unaffected.
    EXPECT_LT(stats.rmse(), kDirectionalRmseCeiling)
            << "mean=" << stats.mean() << " rmse=" << stats.rmse() << " n=" << stats.count;

    // Confirm the OLD fixed window (default Build(), origin pinned at world 0) would have
    // missed this scene entirely: every sample here falls outside [-25.6,25.6]mm, so
    // packDirKey's window-bounds check should skip every write and extraction should be empty.
    Engine::Spatial::CompactDirectionalTSDF cdOldWindow;
    cdOldWindow.Build(*h.ctx, kVoxel, kTruncation); // default window, NOT shifted
    cdOldWindow.SetIntegrationQuality(kQuality);
    for (const auto &v : views) cdOldWindow.Integrate(v.points, v.normals, v.camPos);
    const Engine::Spatial::OrientedPointCloud oldCloud = cdOldWindow.ExtractPointCloud();
    EXPECT_TRUE(oldCloud.points.empty())
            << "expected the OLD fixed [-25.6,25.6]mm window to miss a scene 100mm away "
               "from the origin; got "
            << oldCloud.points.size() << " points -- movable-window property not proven";
}

TEST(CompactDirectional, DirEntryIs24Bytes) {
    using Engine::Spatial::DirEntry;
    static_assert(sizeof(DirEntry) == 24, "DirEntry must be 24B (key+sumDW+sumW+sumN)");
    EXPECT_EQ(offsetof(DirEntry, sumW), 8u);
    EXPECT_EQ(offsetof(DirEntry, sumNx), 12u);
    EXPECT_EQ(offsetof(DirEntry, sumNz), 20u);
}

TEST(CompactDirectional, IntegrateAccumulatesStoredNormal) {
    Engine::Core::Context ctx;
    Engine::Spatial::CompactDirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.05f, 0.15f);           // voxel, truncation
    tsdf.SetIntegrationQuality({1, 4, true}); // single dominant dir, view weight on

    // A planar patch at z=0 with +Z normals, camera on +Z looking down.
    std::vector<Eigen::Vector3f> pts, nrm;
    for (int i = -6; i <= 6; ++i)
        for (int j = -6; j <= 6; ++j) { pts.emplace_back(i*0.02f, j*0.02f, 0.0f); nrm.emplace_back(0,0,1); }
    tsdf.Integrate(pts, nrm, Eigen::Vector3f(0, 0, 1));

    auto entries = tsdf.DownloadEntries();
    ASSERT_GT(entries.size(), 0u);
    int checked = 0;
    for (const auto& e : entries) {
        if (e.weight <= 0.0f) continue;
        EXPECT_NEAR(e.normal.z(), 1.0f, 1e-2f);
        EXPECT_NEAR(e.normal.x(), 0.0f, 1e-2f);
        ++checked;
    }
    EXPECT_GT(checked, 0);
}

TEST(CompactDirectional, ExtractUsesStoredGradientNormal) {
    Engine::Core::Context ctx;
    Engine::Spatial::CompactDirectionalTSDF tsdf;
    // Explicit symmetric window (voxel 0.05 -> 512*0.05=25.6m span, [-12.8,+12.8]).
    // The default windowMinCorner is calibrated for voxelSize=0.1 (span [-25.6,+25.6]);
    // at voxelSize=0.05 the same corner only reaches world [-25.6, 0.0), which would
    // clip this test's origin-straddling plane and make the zero-crossing unobservable
    // on the +side regardless of the normal source under test.
    tsdf.Build(ctx, 0.05f, 0.15f, 1u << 20, 1u << 15, Eigen::Vector3f(-12.8f, -12.8f, -12.8f));
    tsdf.SetIntegrationQuality({1, 4, true});

    std::vector<Eigen::Vector3f> pts, nrm;
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) { pts.emplace_back(i*0.02f, j*0.02f, 0.0f); nrm.emplace_back(0,0,1); }
    tsdf.Integrate(pts, nrm, Eigen::Vector3f(0, 0, 1));

    auto cloud = tsdf.ExtractPointCloud(1u << 18, /*merge=*/false);
    ASSERT_GT(cloud.normals.size(), 0u);
    double meanNz = 0.0;
    for (const auto& n : cloud.normals) meanNz += n.z();
    meanNz /= double(cloud.normals.size());
    EXPECT_GT(meanNz, 0.99); // stored-gradient normals ~ +Z, denoised
}

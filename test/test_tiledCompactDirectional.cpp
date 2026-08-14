#include "Engine/Core/Context.h"
#include "TSDF/Backends/CompactDirectionalTSDF.h"
#include "TSDF/Backends/TiledCompactDirectionalTSDF.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using TSDF::CompactDirectionalTSDF;
using Engine::Core::OrientedPointCloud;
using TSDF::TiledCompactDirectionalTSDF;
using Eigen::Vector3f;

namespace {

    // Axis-aligned +Z plane patch of side `span`, (2*half+1)^2 points, centered at `c`.
    void makePlane(std::vector<Vector3f> &pts, std::vector<Vector3f> &nrm, const Vector3f &c,
                   float span, int half) {
        pts.clear();
        nrm.clear();
        const float step = span / float(2 * half);
        for (int i = -half; i <= half; ++i)
            for (int j = -half; j <= half; ++j) {
                pts.emplace_back(c.x() + i * step, c.y() + j * step, c.z());
                nrm.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

    // Max over `a` of the nearest-neighbour distance to `b` (brute force; small clouds only).
    float maxNearest(const std::vector<Vector3f> &a, const std::vector<Vector3f> &b) {
        float worst = 0.0f;
        for (const auto &p : a) {
            float best = std::numeric_limits<float>::infinity();
            for (const auto &q : b) best = std::min(best, (p - q).squaredNorm());
            worst = std::max(worst, best);
        }
        return std::sqrt(worst);
    }

} // namespace

// floorDiv rounds toward -inf (documented contract), independent of GPU.
TEST(TiledCompact, FloorDivRoundsTowardNegInf) {
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(-1, 448), -1);
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(0, 448), 0);
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(447, 448), 0);
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(448, 448), 1);
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(-448, 448), -1);
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(-449, 448), -2);
}

// A patch that fits inside one tile core AND a single CompactDirectionalTSDF window must extract
// to the same surface through the tiled coordinator as through a bare backend (tiling adds no
// error where it isn't needed). voxel 0.05 -> tile-0 core [0,22.4); center the patch at (6,6,6),
// inside one core on EVERY axis (clear of the z=0 tile boundary + its G=4-voxel ghost band, which
// would otherwise split it across tiles) and inside the bare centered window [-12.8,12.8).
TEST(TiledCompact, TilingMatchesSingleWindowWhereItFits) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(6.0f, 6.0f, 6.0f), 4.0f, 30);

    TiledCompactDirectionalTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(pts, nrm, Vector3f(6.0f, 6.0f, 7.0f));
    const Engine::Core::OrientedPointCloud tiledCloud = tiled.ExtractPointCloud(/*merge=*/false);

    CompactDirectionalTSDF single;
    single.Build(ctx, 0.05f, 0.15f, 1u << 20, 1u << 15, Vector3f(-12.8f, -12.8f, -12.8f));
    single.Integrate(pts, nrm, Vector3f(6.0f, 6.0f, 7.0f));
    const Engine::Core::OrientedPointCloud singleCloud = single.ExtractPointCloud(1u << 18, /*merge=*/false);

    ASSERT_GT(tiledCloud.points.size(), 100u);
    ASSERT_GT(singleCloud.points.size(), 100u);
    EXPECT_EQ(tiled.TileCount(), 1u);
    // Same lattice -> counts match closely and every point coincides.
    EXPECT_NEAR(double(tiledCloud.points.size()), double(singleCloud.points.size()),
                0.02 * double(singleCloud.points.size()));
    EXPECT_LT(maxNearest(tiledCloud.points, singleCloud.points), 0.05f);
    EXPECT_LT(maxNearest(singleCloud.points, tiledCloud.points), 0.05f);
}

// A plane wider than one core spans >=2 tiles and must extract seam-free (no gap band).
TEST(TiledCompact, PlaneSpanningTilesIsSeamFree) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(0.0f, 0.0f, 0.0f), 30.0f, 200); // x,y in [-15,15]

    TiledCompactDirectionalTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(pts, nrm, Vector3f(0.0f, 0.0f, 1.0f));
    const Engine::Core::OrientedPointCloud cloud = tiled.ExtractPointCloud(/*merge=*/false);

    ASSERT_GT(cloud.points.size(), 1000u);
    EXPECT_GE(tiled.TileCount(), 2u); // crosses the x=0 / y=0 tile boundary
    // Every 0.5-wide x-bin across the interior [-14,14] is populated -> no seam gap.
    const int nbins = 56; // (14 - (-14)) / 0.5
    std::vector<int> bin(nbins, 0);
    for (const auto &p : cloud.points) {
        if (p.x() < -14.0f || p.x() >= 14.0f) continue;
        bin[int((p.x() + 14.0f) / 0.5f)]++;
    }
    for (int b = 0; b < nbins; ++b) EXPECT_GT(bin[b], 0) << "empty x-bin " << b << " (seam gap)";
}

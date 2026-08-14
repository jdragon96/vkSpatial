#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "TSDF/Backends/AdvancedTSDF.h"
#include "TSDF/Backends/TiledAdvancedTSDF.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <tuple>
#include <vector>

using TSDF::AdvancedEntry;
using TSDF::AdvancedTSDF;
using Engine::Core::OrientedPointCloud;
using TSDF::TiledAdvancedTSDF;
using Eigen::Vector3f;

namespace {

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

// Tiling adds no error where a single AdvancedTSDF window already fits. Patch centered at (6,6,6):
// inside tile-0's core on every axis (clear of tile boundaries + ghost bands, so TileCount==1) and
// inside the centered default window [-12.8,12.8) at voxel 0.05.
TEST(TiledAdvanced, TilingMatchesSingleWindowWhereItFits) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(6.0f, 6.0f, 6.0f), 4.0f, 30);

    TiledAdvancedTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(pts, nrm, Vector3f(6.0f, 6.0f, 7.0f));
    const Engine::Core::OrientedPointCloud tiledCloud = tiled.ExtractPointCloud(/*merge=*/false);

    AdvancedTSDF single;
    single.Build(ctx, 0.05f, 0.15f); // default centered window covers [-12.8, 12.8)
    single.Integrate(pts, nrm, Vector3f(6.0f, 6.0f, 7.0f));
    const Engine::Core::OrientedPointCloud singleCloud = single.ExtractPointCloud(1u << 18, /*merge=*/false);

    ASSERT_GT(tiledCloud.points.size(), 100u);
    ASSERT_GT(singleCloud.points.size(), 100u);
    EXPECT_EQ(tiled.TileCount(), 1u);
    EXPECT_NEAR(double(tiledCloud.points.size()), double(singleCloud.points.size()),
                0.02 * double(singleCloud.points.size()));
    EXPECT_LT(maxNearest(tiledCloud.points, singleCloud.points), 0.05f);
    EXPECT_LT(maxNearest(singleCloud.points, tiledCloud.points), 0.05f);
}

// A plane wider than one core spans >=2 tiles with no seam gap.
TEST(TiledAdvanced, PlaneSpanningTilesIsSeamFree) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(0.0f, 0.0f, 0.0f), 30.0f, 200); // x,y in [-15,15]

    TiledAdvancedTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(pts, nrm, Vector3f(0.0f, 0.0f, 1.0f));
    const Engine::Core::OrientedPointCloud cloud = tiled.ExtractPointCloud(/*merge=*/false);

    ASSERT_GT(cloud.points.size(), 1000u);
    EXPECT_GE(tiled.TileCount(), 2u);
    const int nbins = 56;
    std::vector<int> bin(nbins, 0);
    for (const auto &p : cloud.points) {
        if (p.x() < -14.0f || p.x() >= 14.0f) continue;
        bin[int((p.x() + 14.0f) / 0.5f)]++;
    }
    for (int b = 0; b < nbins; ++b) EXPECT_GT(bin[b], 0) << "empty x-bin " << b << " (seam gap)";
}

// Only tiles a point (or its ghost) touches are allocated; two separated patches -> exactly 2.
TEST(TiledAdvanced, OnlyTouchedTilesAllocated) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> a, an, b, bn;
    makePlane(a, an, Vector3f(6.0f, 6.0f, 6.0f), 4.0f, 20);    // tile (0,0,0)
    makePlane(b, bn, Vector3f(106.0f, 6.0f, 6.0f), 4.0f, 20);  // x~2120 vox -> tile (4,0,0)

    TiledAdvancedTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(a, an, Vector3f(6.0f, 6.0f, 7.0f));
    tiled.Integrate(b, bn, Vector3f(106.0f, 6.0f, 7.0f));
    EXPECT_EQ(tiled.TileCount(), 2u);
}

// A1 (confidence weight) and A2 (Hermite) reach the tiles: changing each setter SHIFTS the
// extracted surface. Uses a multi-view cylinder — on a curved surface each voxel is observed with
// a different tsdf per view, so the settings are observable. (A flat single-view plane is scale-
// invariant to both: sumDW/sumW cancels the weight, and a constant gradient makes the Hermite root
// equal the linear one — and the point count never changes either way, since zero-crossing voxels
// are geometry-fixed.) We therefore assert the surface SHIFTS via nearest-neighbour distance, not
// that the count changes. If a setter failed to forward, both runs would use the tile default and
// the clouds would coincide (maxNearest ~ 0). Measured margins: A1(projective) ~2.6e-3, A2 ~6e-4.
TEST(TiledAdvanced, A1A2SettersReachTiles) {
    Engine::Core::Context ctx;
    const Vector3f ctr(6.0f, 6.0f, 6.0f);
    std::vector<Vector3f> pts, nrm; // cylinder (radial normals) centered off any tile boundary
    for (int a = 0; a < 120; ++a) {
        const float th = 2.0f * float(M_PI) * float(a) / 120.0f;
        const float c = std::cos(th), s = std::sin(th);
        for (int k = 0; k <= 30; ++k) {
            const float z = -0.4f + 0.8f * float(k) / 30.0f;
            pts.emplace_back(ctr.x() + 0.3f * c, ctr.y() + 0.3f * s, ctr.z() + z);
            nrm.emplace_back(c, s, 0.0f);
        }
    }
    const std::vector<Vector3f> camDirs = {{1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};

    auto run = [&](bool p2p, float conf, bool hermite) {
        TiledAdvancedTSDF t;
        t.Build(ctx, 0.02f, 0.06f);
        t.SetIntegrationQuality({2, 4, true});
        t.SetPointToPlane(p2p);
        t.SetConfidenceWeight(conf);
        t.SetHermitePosition(hermite);
        for (const auto &cd : camDirs) {
            std::vector<Vector3f> vp, vn;
            for (size_t i = 0; i < pts.size(); ++i)
                if (nrm[i].dot(cd) > 0.3f) { vp.push_back(pts[i]); vn.push_back(nrm[i]); }
            t.Integrate(vp, vn, ctr + cd * 5.0f);
        }
        return t.ExtractPointCloud(/*merge=*/false);
    };

    // A1: projective mode makes each voxel's tsdf view-dependent, so confidence weighting shifts
    // the fused surface measurably.
    const Engine::Core::OrientedPointCloud confOff = run(/*p2p=*/false, 0.0f, false);
    const Engine::Core::OrientedPointCloud confOn = run(/*p2p=*/false, 0.8f, false);
    ASSERT_GT(confOff.points.size(), 100u);
    ASSERT_GT(confOn.points.size(), 100u);
    EXPECT_GT(maxNearest(confOff.points, confOn.points), 1e-4f)
            << "SetConfidenceWeight had no effect -> A1 not reaching tiles";

    // A2: Hermite moves the sub-voxel zero-crossing on the curved surface.
    const Engine::Core::OrientedPointCloud hermOff = run(/*p2p=*/true, 0.5f, false);
    const Engine::Core::OrientedPointCloud hermOn = run(/*p2p=*/true, 0.5f, true);
    ASSERT_GT(hermOff.points.size(), 100u);
    ASSERT_GT(hermOn.points.size(), 100u);
    EXPECT_GT(maxNearest(hermOff.points, hermOn.points), 1e-4f)
            << "SetHermitePosition had no effect -> A2 not reaching tiles";
}

// DownloadEntries aggregates all tiles, core-only: every returned (voxel,direction) is unique
// (ghost overlap between tiles must not double-count).
TEST(TiledAdvanced, DownloadEntriesAggregatesTilesCoreOnly) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(0.0f, 0.0f, 6.0f), 30.0f, 200); // x,y in [-15,15] -> >=2 tiles

    TiledAdvancedTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(pts, nrm, Vector3f(0.0f, 0.0f, 7.0f));
    const std::vector<AdvancedEntry> entries = tiled.DownloadEntries();

    ASSERT_GT(entries.size(), 1000u);
    EXPECT_GE(tiled.TileCount(), 2u);
    std::set<std::tuple<int, int, int, uint32_t>> keys;
    for (const AdvancedEntry &e: entries)
        keys.emplace(int(std::lround(e.center.x() / 0.05f)), int(std::lround(e.center.y() / 0.05f)),
                     int(std::lround(e.center.z() / 0.05f)), e.direction);
    EXPECT_EQ(keys.size(), entries.size())
            << "duplicate (voxel,dir) across tiles -> core-only dedup broken";
}

// Reset drops all tiles: TileCount 0 and DownloadEntries empty afterwards.
TEST(TiledAdvanced, ResetClearsTiles) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(0.0f, 0.0f, 6.0f), 30.0f, 150);

    TiledAdvancedTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(pts, nrm, Vector3f(0.0f, 0.0f, 7.0f));
    ASSERT_GT(tiled.TileCount(), 0u);
    ASSERT_GT(tiled.DownloadEntries().size(), 0u);

    tiled.Reset();
    EXPECT_EQ(tiled.TileCount(), 0u);
    EXPECT_TRUE(tiled.DownloadEntries().empty());
}

// One core box per touched tile.
TEST(TiledAdvanced, CoreBoxesMatchTileCount) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(0.0f, 0.0f, 6.0f), 30.0f, 150);

    TiledAdvancedTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(pts, nrm, Vector3f(0.0f, 0.0f, 7.0f));
    EXPECT_EQ(tiled.CoreBoxes().size(), std::size_t(tiled.TileCount()));
}

// The batched Integrate overload (all touched tiles recorded into one CommandBatch, one submit)
// yields the same occupied set as the self-submitting overload. Uses a multi-tile plane.
TEST(TiledAdvanced, BatchedIntegrateMatchesSelfSubmit) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(0.0f, 0.0f, 0.0f), 30.0f, 200); // spans >= 2 tiles
    const Vector3f cam(0.0f, 0.0f, 1.0f);

    TiledAdvancedTSDF a, b;
    a.Build(ctx, 0.05f, 0.15f);
    b.Build(ctx, 0.05f, 0.15f);

    a.Integrate(pts, nrm, cam); // self-submitting
    {
        Engine::Compute::CommandBatch batch(ctx);
        b.Integrate(pts, nrm, cam, batch); // batched
        batch.Submit();
    }
    ASSERT_GE(a.TileCount(), 2u);
    EXPECT_EQ(a.TileCount(), b.TileCount());
    EXPECT_EQ(a.DownloadEntries().size(), b.DownloadEntries().size());
    EXPECT_GT(a.DownloadEntries().size(), 1000u);
}

// The GPU-route IntegrateGPU (whole cloud uploaded ONCE; each tile's shader window-filters it) yields
// the same tiles + occupied set as the CPU-route Integrate -- so GPU-side routing is seam-free and
// loses no contribution. Plane kept under the per-tile maxPoints so neither path clamps.
TEST(TiledAdvanced, GpuRouteMatchesCpuRoute) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(0.0f, 0.0f, 0.0f), 30.0f, 100); // 201^2 pts, spans >= 2 tiles
    const Vector3f cam(0.0f, 0.0f, 1.0f);

    TiledAdvancedTSDF cpu, gpu;
    cpu.Build(ctx, 0.05f, 0.15f);
    gpu.Build(ctx, 0.05f, 0.15f);

    cpu.Integrate(pts, nrm, cam);    // CPU route: per-tile point copies
    gpu.IntegrateGPU(pts, nrm, cam); // GPU route: shared cloud + shader window filter

    ASSERT_GE(cpu.TileCount(), 2u);
    EXPECT_EQ(cpu.TileCount(), gpu.TileCount());
    EXPECT_EQ(cpu.DownloadEntries().size(), gpu.DownloadEntries().size());
    EXPECT_GT(gpu.DownloadEntries().size(), 1000u);
}

// Each voxel's AdvancedEntry::firstFrame is GPU-stamped with SetCurrentFrame's value the FIRST time
// the voxel is filled -- so the CPU recovers "first-seen frame" (and "new this frame") without a
// tracker. Two non-overlapping patches integrated at different frames must carry their own stamps.
TEST(TiledAdvanced, GpuFirstFrameStamp) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pA, nA, pB, nB;
    makePlane(pA, nA, Vector3f(0.0f, 0.0f, 0.0f), 4.0f, 20);   // patch near x in [-2, 2]
    makePlane(pB, nB, Vector3f(10.0f, 0.0f, 0.0f), 4.0f, 20);  // patch near x in [8, 12]
    const Vector3f cam(0.0f, 0.0f, 1.0f);

    TiledAdvancedTSDF t;
    t.Build(ctx, 0.05f, 0.15f);
    t.SetCurrentFrame(0);
    t.IntegrateGPU(pA, nA, cam);
    t.SetCurrentFrame(5);
    t.IntegrateGPU(pB, nB, cam); // pB voxels are new at frame 5; pA voxels keep firstFrame 0

    const std::vector<AdvancedEntry> entries = t.DownloadEntries();
    ASSERT_GT(entries.size(), 200u);
    int a = 0, b = 0, bad = 0;
    for (const AdvancedEntry &e: entries) {
        if (e.center.x() < 5.0f) { // patch A -> filled at frame 0
            (e.firstFrame == 0) ? ++a : ++bad;
        } else { // patch B -> filled at frame 5
            (e.firstFrame == 5) ? ++b : ++bad;
        }
    }
    EXPECT_EQ(bad, 0);
    EXPECT_GT(a, 50);
    EXPECT_GT(b, 50);
}

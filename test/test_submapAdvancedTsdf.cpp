#include "Engine/Core/Context.h"
#include "TSDF/Backends/SubmapAdvancedTSDF.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using Engine::Core::OrientedPointCloud;
using TSDF::SubmapAdvancedTSDF;
using Eigen::Vector3f;

namespace {

    // +Z plane patch of side `span` centered at `c`, points on a grid of pitch `spacing`.
    void makePlane(std::vector<Vector3f> &pts, std::vector<Vector3f> &nrm, const Vector3f &c,
                   float span, float spacing) {
        const int half = std::max(1, int(span / (2.0f * spacing)));
        for (int i = -half; i <= half; ++i)
            for (int j = -half; j <= half; ++j) {
                pts.emplace_back(c.x() + i * spacing, c.y() + j * spacing, c.z());
                nrm.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

    // Min nearest-neighbour distance over the cloud (brute force; small clouds).
    float minSpacing(const std::vector<Vector3f> &p) {
        float best = std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < p.size(); ++i)
            for (std::size_t j = i + 1; j < p.size(); ++j)
                best = std::min(best, (p[i] - p[j]).squaredNorm());
        return std::sqrt(best);
    }

} // namespace

// A locally dense sub-patch flips to dense ONLINE (during integration, no pre-scan) and gets a detail
// submap; extracted points there reach half-voxel spacing a base-only map cannot.
TEST(SubmapAdvanced, DenseRegionGetsDetail) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> coarseP, coarseN, denseP, denseN;
    makePlane(coarseP, coarseN, Vector3f(6, 6, 6), 8.0f, 0.05f); // ~1 pt / base voxel -> sparse
    makePlane(denseP, denseN, Vector3f(6, 6, 6), 1.2f, 0.012f);  // dense within one block

    SubmapAdvancedTSDF s;
    s.Build(ctx, 0.05f, 0.15f, 32, 4.0f);
    s.Integrate(coarseP, coarseN, Vector3f(6, 6, 7)); // still sparse -> no dense block yet
    EXPECT_EQ(s.DenseBlockCount(), 0u);
    s.Integrate(denseP, denseN, Vector3f(6, 6, 7)); // the dense patch tips the block over the threshold
    EXPECT_GT(s.DenseBlockCount(), 0u);
    EXPECT_GT(s.DetailTileCount(), 0u);

    const Engine::Core::OrientedPointCloud cloud = s.ExtractPointCloud(/*merge=*/false);
    ASSERT_GT(cloud.points.size(), 100u);
    EXPECT_LT(minSpacing(cloud.points), 0.035f); // detail (0.025) present
}

// A uniformly sparse scene never flips any block dense; extraction is base-only resolution.
TEST(SubmapAdvanced, SparseSceneNoDetail) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> p, n;
    makePlane(p, n, Vector3f(6, 6, 6), 8.0f, 0.05f); // ~1 pt / base voxel

    SubmapAdvancedTSDF s;
    s.Build(ctx, 0.05f, 0.15f, 32, 4.0f);
    s.Integrate(p, n, Vector3f(6, 6, 7));
    EXPECT_EQ(s.DenseBlockCount(), 0u);
    EXPECT_EQ(s.DetailTileCount(), 0u);

    const Engine::Core::OrientedPointCloud cloud = s.ExtractPointCloud(/*merge=*/false);
    ASSERT_GT(cloud.points.size(), 100u);
    EXPECT_GT(minSpacing(cloud.points), 0.035f); // base-only ~0.05
}

// Precedence dedup: in dense blocks the base surface is REPLACED by detail, not added on top.
// Detail (half voxel) ~4x base points for a plane; without dedup it would be ~5x (base+detail).
TEST(SubmapAdvanced, BaseReplacedNotAddedInDenseBlocks) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> denseP, denseN;
    makePlane(denseP, denseN, Vector3f(6, 6, 6), 1.2f, 0.012f);

    auto run = [&](float k) {
        SubmapAdvancedTSDF s;
        s.Build(ctx, 0.05f, 0.15f, 32, k); // detailK = k
        s.Integrate(denseP, denseN, Vector3f(6, 6, 7)); // online density flips (or not) this frame
        return s.ExtractPointCloud(/*merge=*/false).points.size();
    };
    const double nBase = double(run(1.0e9f)); // k huge -> never flips dense -> base only
    const double nSub = double(run(4.0f));     // flips dense -> detail replaces base
    ASSERT_GT(nBase, 50.0);
    EXPECT_GT(nSub, 2.5 * nBase); // detail present (finer)
    EXPECT_LT(nSub, 4.7 * nBase); // base dropped in the dense block (not ~5x = base+detail)
}

// DownloadEntries aggregates base+detail; DenseBlockBoxes/BaseCoreBoxes match the counts (online).
TEST(SubmapAdvanced, DownloadEntriesAndBoxes) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> coarseP, coarseN, denseP, denseN;
    makePlane(coarseP, coarseN, Vector3f(6, 6, 6), 8.0f, 0.05f);
    makePlane(denseP, denseN, Vector3f(6, 6, 6), 1.2f, 0.012f);

    SubmapAdvancedTSDF s;
    s.Build(ctx, 0.05f, 0.15f, 32, 4.0f);
    s.Integrate(coarseP, coarseN, Vector3f(6, 6, 7));
    s.Integrate(denseP, denseN, Vector3f(6, 6, 7));

    EXPECT_GT(s.DownloadEntries().size(), 100u);
    EXPECT_EQ(s.DenseBlockBoxes().size(), std::size_t(s.DenseBlockCount()));
    EXPECT_EQ(s.BaseCoreBoxes().size(), std::size_t(s.BaseTileCount()));
}

// Reset drops both levels AND the learned dense set: online density is re-learned from the stream on a
// replay-from-frame-0, unlike the old pre-scan which kept a fixed dense set.
TEST(SubmapAdvanced, ResetClearsDensity) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> denseP, denseN;
    makePlane(denseP, denseN, Vector3f(6, 6, 6), 1.2f, 0.012f);

    SubmapAdvancedTSDF s;
    s.Build(ctx, 0.05f, 0.15f, 32, 4.0f);
    s.Integrate(denseP, denseN, Vector3f(6, 6, 7));
    ASSERT_GT(s.DenseBlockCount(), 0u);
    ASSERT_GT(s.DetailTileCount(), 0u);

    s.Reset();
    EXPECT_EQ(s.BaseTileCount(), 0u);
    EXPECT_EQ(s.DetailTileCount(), 0u);
    EXPECT_TRUE(s.DownloadEntries().empty());
    EXPECT_EQ(s.DenseBlockCount(), 0u); // density cleared -> re-learned on the next stream
}

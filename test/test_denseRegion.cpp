#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "TSDF/Structure/DenseRegionClassifier.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

using Eigen::Vector3f;

namespace {

    // A planar +Z patch of `count` x `count` samples spaced `spacing` apart, offset by `blockOffset`
    // on every axis so it sits in the middle of block (0,0,0) instead of on the world origin.
    // `floor(position / blockWorld)` puts the world origin on a block CORNER, so an origin-centred
    // patch straddles up to four blocks; the offset keeps the whole patch inside one. Spacing is the
    // knob every density assertion turns.
    void MakePlane(std::vector<Vector3f> &points, std::vector<Vector3f> &normals,
                   float spacing, int count, float blockOffset) {
        points.clear();
        normals.clear();
        const float half = 0.5f * spacing * float(count - 1);
        for (int i = 0; i < count; ++i)
            for (int j = 0; j < count; ++j) {
                points.emplace_back(float(i) * spacing - half + blockOffset,
                                     float(j) * spacing - half + blockOffset,
                                     blockOffset);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

} // namespace

TEST(DenseRegionAccumulate, CountsPointsAndOccupancyPerBlock) {
    Engine::Core::Context context;
    const float baseVoxel = 0.01f;
    const int blockVoxels = 32;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, baseVoxel, blockVoxels, /*maxPointPerFrame=*/1u << 15);

    const float blockOffset = baseVoxel * float(blockVoxels) * 0.5f; // centre of block (0,0,0)
    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, /*spacing=*/0.0025f, /*count=*/64, blockOffset); // 4096 points, one block

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    batch.Submit();

    const std::vector<TSDF::BlockRecord> blocks = classifier.ReadBlocks();
    ASSERT_EQ(blocks.size(), 1u) << "a 0.16 m patch at 0.32 m blocks must land in one block";
    EXPECT_EQ(blocks[0].pointCount, points.size());
    EXPECT_GT(blocks[0].fineOccupied, 0u);
    EXPECT_GT(blocks[0].coarseOccupied, 0u);
    EXPECT_GE(blocks[0].fineOccupied, blocks[0].coarseOccupied)
            << "a finer grid can never have fewer occupied cells";
    EXPECT_EQ(classifier.BlockInsertFailureCount(), 0u)
            << "a healthy run must not report a full block table";
}

// The ratio is the whole spacing estimate: occupiedFine/occupiedCoarse = min(4, (v/s)^2).
// At s = v/4 the fine grid is fully resolved (ratio ~4); at s = 2v neither grid is (ratio ~1).
TEST(DenseRegionAccumulate, OccupancyRatioTracksSampleSpacing) {
    Engine::Core::Context context;
    const float baseVoxel = 0.01f;
    const int blockVoxels = 32;
    const float blockOffset = baseVoxel * float(blockVoxels) * 0.5f; // centre of block (0,0,0)

    auto ratioAt = [&](float spacing, int count) {
        TSDF::DenseRegionClassifier classifier;
        classifier.Build(context, baseVoxel, blockVoxels, 1u << 15);
        std::vector<Vector3f> points, normals;
        MakePlane(points, normals, spacing, count, blockOffset);
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(points, normals, batch);
        batch.Submit();
        const std::vector<TSDF::BlockRecord> blocks = classifier.ReadBlocks();
        EXPECT_EQ(blocks.size(), 1u) << "spacing " << spacing;
        return double(blocks[0].fineOccupied) / double(blocks[0].coarseOccupied);
    };

    EXPECT_NEAR(ratioAt(baseVoxel * 0.25f, 64), 4.0, 0.6) << "s = v/4 resolves the fine grid";
    EXPECT_NEAR(ratioAt(baseVoxel * 2.0f, 8), 1.0, 0.3) << "s = 2v resolves neither grid";
}

// Regression for a packBlockKey collision review found: under the old XOR-with-overlapping-shifts
// formula, blocks (-32,-32,32) and (-32,-31,0) -- 10+ metres apart at 0.32 m blocks -- both packed
// to 0x7e10fc20, silently merging their two records into one (corrupting point count, normal sum,
// and, through the shared blockSlot, occupancy). Confirmed to fail against the old packBlockKey
// before the disjoint-field fix landed; must keep passing after.
TEST(DenseRegionAccumulate, DistantBlocksDoNotCollide) {
    Engine::Core::Context context;
    const float baseVoxel = 0.01f;
    const int blockVoxels = 32;
    const float blockWorld = baseVoxel * float(blockVoxels);

    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, baseVoxel, blockVoxels, /*maxPointPerFrame=*/1u << 12);

    auto blockCentre = [&](int blockX, int blockY, int blockZ) {
        return Vector3f((float(blockX) + 0.5f) * blockWorld, (float(blockY) + 0.5f) * blockWorld,
                        (float(blockZ) + 0.5f) * blockWorld);
    };

    // Cluster A: 5 points near the centre of block (-32,-32,32).
    // Cluster B: 3 points near the centre of block (-32,-31,0) -- the reviewer's colliding pair.
    std::vector<Vector3f> points, normals;
    const Vector3f centreA = blockCentre(-32, -32, 32);
    const Vector3f centreB = blockCentre(-32, -31, 0);
    for (int i = 0; i < 5; ++i) {
        points.push_back(centreA + Vector3f(float(i) * 0.001f, 0.0f, 0.0f));
        normals.emplace_back(0.0f, 0.0f, 1.0f);
    }
    for (int i = 0; i < 3; ++i) {
        points.push_back(centreB + Vector3f(float(i) * 0.001f, 0.0f, 0.0f));
        normals.emplace_back(0.0f, 0.0f, 1.0f);
    }

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    batch.Submit();

    const std::vector<TSDF::BlockRecord> blocks = classifier.ReadBlocks();
    ASSERT_EQ(blocks.size(), 2u) << "two blocks 10+ m apart must not collide onto one record";
    std::vector<uint32_t> pointCounts;
    for (const auto &block : blocks) pointCounts.push_back(block.pointCount);
    std::sort(pointCounts.begin(), pointCounts.end());
    EXPECT_EQ(pointCounts, (std::vector<uint32_t>{3u, 5u}))
            << "point counts must match the two clusters exactly -- a merge would sum them to 8";
}

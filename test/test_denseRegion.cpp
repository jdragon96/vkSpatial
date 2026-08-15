#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "TSDF/Structure/DenseRegionClassifier.h"

#include <gtest/gtest.h>

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

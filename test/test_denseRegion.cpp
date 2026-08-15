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
    // knob every density assertion turns. `blockOffset` defaults to 0 (origin-centred, so the patch
    // may straddle several blocks) for callers that only care about per-block classification, not
    // about which or how many blocks the points land in.
    void MakePlane(std::vector<Vector3f> &points, std::vector<Vector3f> &normals,
                   float spacing, int count, float blockOffset = 0.0f) {
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

// A dense flat plane must NOT be refined: it resolves the fine grid, but there is no geometry
// detail to recover. This is the condition the old redundancy heuristic could not see.
TEST(DenseRegionClassify, DenseFlatPlaneIsNotRefined) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.0025f, 64);

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    batch.Submit();

    EXPECT_EQ(classifier.DenseBlockCount(), 0u)
            << "normal coherence ~1 on a plane must veto refinement";
}

// A densely scanned sphere patch has both the sampling and the curvature, so it must be refined.
TEST(DenseRegionClassify, DenseCurvedSurfaceIsRefined) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    // A 0.05 m sphere sampled at ~0.0025 m: strong curvature across one block.
    std::vector<Vector3f> points, normals;
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

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    batch.Submit();

    EXPECT_GT(classifier.DenseBlockCount(), 0u);
    EXPECT_GT(classifier.DetailSlotEstimate(), 0u)
            << "a refined block must report the slots its detail table will need";
}

// Sparse sampling fails the spacing condition however curved the surface is.
TEST(DenseRegionClassify, SparseCurvedSurfaceIsNotRefined) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    std::vector<Vector3f> points, normals;
    const float radius = 0.05f;
    for (int a = 0; a < 24; ++a)
        for (int b = 0; b < 12; ++b) {
            const float theta = float(a) * float(M_PI) / 12.0f;
            const float phi = float(b) * float(M_PI) / 24.0f;
            const Vector3f direction(std::sin(phi) * std::cos(theta),
                                     std::sin(phi) * std::sin(theta), std::cos(phi));
            points.push_back(direction * radius);
            normals.push_back(direction);
        }

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    batch.Submit();

    EXPECT_EQ(classifier.DenseBlockCount(), 0u) << "spacing must veto refinement";
}

namespace {

    // A small paraboloid patch z = curvature*(x^2+y^2), centred in the middle of block (0,0,0) (see
    // MakePlane's comment on why: the world origin sits on a block corner). halfWidth, curvature and
    // pointsPerSide are tuned -- not arbitrary -- so that a SINGLE frame clears resolvesFineGrid,
    // hasDetail and keepsSignal with real margin while its fine-cell footprint (fineOccupied on that
    // one frame) sits comfortably under DensityCriteria::minimumFineOccupied (measured: 56 vs 64).
    // See RevisitedSmallFootprintIsNotRefined for why that specific combination matters.
    void MakeSmallCurvedPatch(std::vector<Vector3f> &points, std::vector<Vector3f> &normals,
                              float blockOffset) {
        points.clear();
        normals.clear();
        const float halfWidth = 0.0125f;
        const float curvature = 31.0f;
        const int pointsPerSide = 40;
        for (int i = 0; i < pointsPerSide; ++i)
            for (int j = 0; j < pointsPerSide; ++j) {
                const float x = -halfWidth + 2.0f * halfWidth * float(i) / float(pointsPerSide - 1);
                const float y = -halfWidth + 2.0f * halfWidth * float(j) / float(pointsPerSide - 1);
                const float z = curvature * (x * x + y * y);
                points.emplace_back(blockOffset + x, blockOffset + y, blockOffset + z);
                const Vector3f gradient(-2.0f * curvature * x, -2.0f * curvature * y, 1.0f);
                normals.push_back(gradient.normalized());
            }
    }

} // namespace

// Regression for a review finding: hasSurface must read fineOccupiedMax, not the cumulative
// fineOccupied. fineOccupied re-counts a physical cell once per frame (the per-frame cell hash is
// wiped every frame), so it is as much a revisit counter as a footprint measure; a block with a
// genuinely small footprint can walk it past minimumFineOccupied purely by being looked at enough
// times, with no growth in real extent -- exactly what minimumFineOccupied exists to prevent. A
// single Record+Classify cannot catch this: fineOccupied, fineOccupiedFrame and fineOccupiedMax are
// all numerically equal after one frame. This replays the SAME small patch across several
// independent frames -- each its own CommandBatch, submitted, matching how a real scan integrates
// -- and checks the verdict never flips.
TEST(DenseRegionClassify, RevisitedSmallFootprintIsNotRefined) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    const float blockOffset = 0.01f * 32.0f * 0.5f; // centre of block (0,0,0)
    std::vector<Vector3f> points, normals;
    MakeSmallCurvedPatch(points, normals, blockOffset);

    for (int frame = 0; frame < 4; ++frame) {
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(points, normals, batch);
        classifier.Classify(batch);
        batch.Submit();

        EXPECT_EQ(classifier.DenseBlockCount(), 0u)
                << "frame " << frame
                << ": revisiting the same small footprint must not accumulate into a false latch";
    }
}

// The partition must be exhaustive: every input point lands in exactly one level. A point silently
// dropped here vanishes from the reconstruction with no counter to show it.
TEST(DenseRegionPartition, EveryPointLandsInExactlyOneLevel) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    std::vector<Vector3f> points, normals;
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

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    classifier.Partition(batch);
    batch.Submit();

    std::vector<uint32_t> base, detail;
    classifier.ReadPartition(base, detail);
    EXPECT_EQ(base.size() + detail.size(), points.size());
    EXPECT_GT(detail.size(), 0u) << "a curved dense surface must send points to the detail level";
}

// The verdict is a latch: a block that became dense stays dense on later frames.
TEST(DenseRegionPartition, DenseVerdictDoesNotRevert) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    std::vector<Vector3f> dense, denseNormals;
    const float radius = 0.05f;
    for (int a = 0; a < 180; ++a)
        for (int b = 0; b < 90; ++b) {
            const float theta = float(a) * float(M_PI) / 90.0f;
            const float phi = float(b) * float(M_PI) / 180.0f;
            const Vector3f direction(std::sin(phi) * std::cos(theta),
                                     std::sin(phi) * std::sin(theta), std::cos(phi));
            dense.push_back(direction * radius);
            denseNormals.push_back(direction);
        }

    {
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(dense, denseNormals, batch);
        classifier.Classify(batch);
        batch.Submit();
    }
    const uint32_t afterDenseFrame = classifier.DenseBlockCount();
    ASSERT_GT(afterDenseFrame, 0u);

    // A sparse second frame over the same region would fail the criteria on its own.
    std::vector<Vector3f> sparse, sparseNormals;
    for (size_t i = 0; i < dense.size(); i += 40) {
        sparse.push_back(dense[i]);
        sparseNormals.push_back(denseNormals[i]);
    }
    {
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(sparse, sparseNormals, batch);
        classifier.Classify(batch);
        batch.Submit();
    }
    EXPECT_GE(classifier.DenseBlockCount(), afterDenseFrame) << "the verdict must not revert";
}

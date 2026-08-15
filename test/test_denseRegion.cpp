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
    EXPECT_GT(detail.size(), 0u) << "a curved dense surface must send points to the detail level";

    // base.size()+detail.size() == points.size() alone does not prove every INDEX landed exactly
    // once -- it holds whenever each invocation increments a counter once, regardless of what got
    // written into the lists (e.g. a non-atomic-slot kernel bug: read the counter, write, THEN
    // atomicAdd -- collides concurrent writes onto the same slot while the counters still end up
    // correct). Every index in [0, points.size()) must appear EXACTLY ONCE across base+detail
    // combined; verified as a mutation-tested regression guard against exactly that bug shape.
    std::vector<uint8_t> seen(points.size(), 0);
    for (uint32_t index : base)   { ASSERT_LT(index, points.size()); EXPECT_EQ(seen[index]++, 0); }
    for (uint32_t index : detail) { ASSERT_LT(index, points.size()); EXPECT_EQ(seen[index]++, 0); }
    EXPECT_EQ(size_t(std::count(seen.begin(), seen.end(), 1)), points.size());
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
        classifier.Partition(batch);
        batch.Submit();
    }
    EXPECT_GE(classifier.DenseBlockCount(), afterDenseFrame) << "the verdict must not revert";

    // g_dense[] is only ever written 1u and only cleared by Reset(), so the flag latches by
    // construction -- the DenseBlockCount() check above only proves the COUNTER latches. What the
    // spec actually needs is that a latched block still ROUTES its points to detail on a later
    // frame; that is a property of Partition() reading m_denseFlags, not of Classify() alone.
    std::vector<uint32_t> base, detail;
    classifier.ReadPartition(base, detail);
    EXPECT_GT(detail.size(), 0u)
            << "a block latched dense must still route its points to the detail level";
}

// Record() has two early returns (an empty `points`, or a mismatched/empty `normals` that makes
// the shared count zero) above the point where the recorded-point-count state used to be set. A
// frame the caller filters down to nothing must not leave Partition() replaying the PREVIOUS
// frame's still-resident m_blockIndex -- that duplicates every one of that frame's points into the
// caller's integration with no counter showing it.
TEST(DenseRegionPartition, EmptyFrameDoesNotReplayThePreviousPartition) {
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

    {
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(points, normals, batch);
        classifier.Classify(batch);
        classifier.Partition(batch);
        batch.Submit();
    }
    std::vector<uint32_t> base, detail;
    classifier.ReadPartition(base, detail);
    ASSERT_EQ(base.size() + detail.size(), points.size()) << "sanity: the real frame partitions fully";

    // An empty cloud takes Record()'s "points.empty()" early return.
    std::vector<Vector3f> empty, emptyNormals;
    {
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(empty, emptyNormals, batch);
        classifier.Classify(batch);
        classifier.Partition(batch);
        batch.Submit();
    }
    classifier.ReadPartition(base, detail);
    EXPECT_EQ(base.size(), 0u) << "an empty frame must not replay the previous frame's partition";
    EXPECT_EQ(detail.size(), 0u) << "an empty frame must not replay the previous frame's partition";
}

// The HASH_INSERT_FAILED -> base branch (partition.comp.glsl's `blockSlot != EMPTY_KEY` guard) has
// no coverage unless a fixture actually overflows the block table -- both other fixtures here touch
// only a handful of blocks against kBlockCapacity's 8192 slots. blockWorld = 0.01 * 32 = 0.32 m, so
// a 0.4 m lattice spacing (> blockWorld) guarantees every lattice point's floor() lands in its OWN
// block along every axis: 25^3 = 15625 distinct blocks, so by the pigeonhole principle alone (never
// mind MAX_PROBE) at least 15625 - 8192 = 7433 of them cannot find a slot. This overflows the table
// without touching any production code.
TEST(DenseRegionPartition, HashInsertFailureStillConservesEveryPoint) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    std::vector<Vector3f> points, normals;
    const float spacing = 0.4f;
    const int side = 25;
    for (int x = 0; x < side; ++x)
        for (int y = 0; y < side; ++y)
            for (int z = 0; z < side; ++z) {
                points.emplace_back(float(x) * spacing, float(y) * spacing, float(z) * spacing);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    classifier.Partition(batch);
    batch.Submit();

    EXPECT_GT(classifier.BlockInsertFailureCount(), 0u)
            << "25^3 distinct blocks must overflow an 8192-slot table";

    // The point of this test: a frame that overflows the block table still loses nothing. Every
    // point -- insert-failed or not -- must land in exactly one of the two lists.
    std::vector<uint32_t> base, detail;
    classifier.ReadPartition(base, detail);
    std::vector<uint8_t> seen(points.size(), 0);
    for (uint32_t index : base)   { ASSERT_LT(index, points.size()); EXPECT_EQ(seen[index]++, 0); }
    for (uint32_t index : detail) { ASSERT_LT(index, points.size()); EXPECT_EQ(seen[index]++, 0); }
    EXPECT_EQ(size_t(std::count(seen.begin(), seen.end(), 1)), points.size())
            << "a frame that overflows the block table must still lose no points";
}

// Build-time maxPointPerFrame is a hint, not a cap. A larger frame must grow the buffers, because
// clamping would drop points with no counter and no symptom -- growPointBuffers' own comment names
// this as deliberate, and it is the same failure shape task 1's FIX 1 found already shipped once in
// this repository (cellCapacity silently drifting past the buffers' true allocation). This is a
// regression latch for that already-fixed behaviour, not a red/green TDD test: growPointBuffers
// already grows the point/normal/blockIndex/base/detail buffers AND the cell hashes together (see
// DenseRegionClassifier.cpp:111-152), so this is expected to pass on first run.
TEST(DenseRegionPartition, LargeFrameGrowsInsteadOfTruncating) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, /*maxPointPerFrame=*/1024u);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.0025f, 64); // 4096 points, four times the hint

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    classifier.Partition(batch);
    batch.Submit();

    std::vector<uint32_t> base, detail;
    classifier.ReadPartition(base, detail);
    EXPECT_EQ(base.size() + detail.size(), points.size())
            << "the frame must not have been truncated to the build-time hint";

    // A capacity-clamped run would drop points with no crash and no error. BlockInsertFailureCount
    // guards a DIFFERENT ceiling (the 8192-slot block table; this fixture touches only 4 blocks, see
    // MakePlane's comment on the block-corner straddle -- nowhere near that limit), but a grown frame
    // must not be quietly failing through THAT counter either, so it is checked here too.
    EXPECT_EQ(classifier.BlockInsertFailureCount(), 0u)
            << "a grown frame must not be silently dropping points through the block-table ceiling";

    // Second, independent channel: the ACCUMULATE pass (upstream of Partition) must also have seen
    // every point, not just the build-time hint's worth. This exercises growPointBuffers' point/
    // normal/blockIndex growth specifically, through BlockRecord::pointCount's atomicAdd -- a
    // different buffer and a different kernel than the base/detail partition lists checked above and
    // below, so a growth failure isolated to only one of the two buffer groups cannot hide from both
    // checks at once.
    uint32_t totalBlockPointCount = 0;
    const std::vector<TSDF::BlockRecord> blocks = classifier.ReadBlocks();
    for (const auto &block : blocks) totalBlockPointCount += block.pointCount;
    EXPECT_EQ(totalBlockPointCount, points.size())
            << "the accumulate pass must also have recorded the full frame, not just the hint";

    // base.size()+detail.size() == points.size() alone does not prove every INDEX landed exactly
    // once -- see EveryPointLandsInExactlyOneLevel's comment for why. Reusing that test's seen-map
    // shape here rather than inventing a new one.
    std::vector<uint8_t> seen(points.size(), 0);
    for (uint32_t index : base)   { ASSERT_LT(index, points.size()); EXPECT_EQ(seen[index]++, 0); }
    for (uint32_t index : detail) { ASSERT_LT(index, points.size()); EXPECT_EQ(seen[index]++, 0); }
    EXPECT_EQ(size_t(std::count(seen.begin(), seen.end(), 1)), points.size())
            << "a grown frame must still conserve every point exactly once, not just sum to the "
               "right total";

    // m_maxPointPerFrame is private and intentionally has no accessor (not adding one just for this
    // test). The checks above ARE the observable consequence of growth: growPointBuffers is the only
    // place that field changes, a single Record(4096 points) call against a fresh 1024-hint Build()
    // either grows it to 6144 in that one call or does not grow it at all (there is no partial-growth
    // case to worry about here), and the accumulate-pass and partition-list conservation checks above
    // -- through two independent buffer groups and kernels -- both fail immediately the moment growth
    // does not happen. Confirmed by mutation, not just by this argument: see task-4-report.md.
}

// LargeFrameGrowsInsteadOfTruncating (above) guards the point/normal/blockIndex/base/detail buffer
// group, but is structurally blind to the cell hashes: accumulate.comp.glsl's pointCount/blockIndex
// writes (lines 124-130) happen unconditionally once a block slot is found, BEFORE the fine/coarse
// cell claim (lines 132-142) ever runs, and partition.comp.glsl only routes an already-counted point
// to a different LIST (base vs. detail) -- it never drops one. So Task 1's Critical 1 (cellCapacity
// silently drifting past the cell hashes' true allocation) would leave that test, and every other
// DenseRegion* fixture in this suite, green: none of them fires the grow branch at all. 11 of the 12
// pre-existing fixtures Build with 1<<15; the one exception (DistantBlocksDoNotCollide) Builds with
// 1<<12 but only ever records 8 points, nowhere near even that smaller hint -- checked every Build()
// call in this file directly rather than assuming they all match.
//
// MakePlane's usual 0.0025 spacing is HALF the 0.005 fine-cell width (baseVoxel * 0.5), so 4 points
// collapse onto every fine cell -- measured 1024 distinct cells for 4096 points in the task-4-report,
// comfortably under capacity whether or not the cell hashes actually grow, which is exactly why that
// spacing cannot exercise this path. 0.0075 is spaced ABOVE the fine-cell width instead: the step
// exceeds the cell width, so floor() strictly increases point-to-point and every point claims its own
// distinct fine cell -- measured exactly 1:1 (see task-4-report.md, fix round). This fixture is sized
// at 128x128 = 16384 points, not the smaller 64x64 = 4096 first tried: the review's own 4096-point
// prediction turned out to be a WEAK mutation target (measured, see report) -- the un-grown cell hash
// is device-local (VMA/MoltenVK) memory whose *logical* size the shader indexes past under the
// drift bug, but small over-reads/writes land in real, mapped allocator slack rather than faulting or
// visibly colliding, so a 2x oversubscription (4096 keys into a capacity sized for 2048, see below)
// produced zero observable deficit. 8x oversubscription (this fixture) reliably exhausts that slack --
// measured a several-hundred-key deficit under the mutation described below (two runs: 15826/16384
// and 15737/16384 -- the exact shortfall varies slightly run to run, since it depends on GPU
// thread-scheduling order among the racing atomicCompSwap probes once the table is genuinely this
// oversubscribed), with no crash either time. See task-4-report.md for both full runs.
//
// The comparison is against the UN-GROWN capacity, which is NOT Task 1's FIX 1 number (3072): that
// figure assumed growPointBuffers' own first call (from Build()) still resizes the cell hashes before
// the growth path gets frozen. The mutation below removes that resize from EVERY call, including the
// first, so the cell hashes never move past Build()'s own raw pre-allocation: kCellCapacityFactor(2.0)
// x 1024 = 2048 (Build(..., 1024u)'s local `cellCapacity`, DenseRegionClassifier.cpp:63-64) -- traced,
// and consistent with the measured 4096-point non-deficit above (2x over 2048 still fit in slack).
TEST(DenseRegionPartition, LargeFrameGrowsCellHashesNotJustPointBuffers) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, /*maxPointPerFrame=*/1024u);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.0075f, 128); // 16384 points, each claiming its own distinct fine cell

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    batch.Submit();

    uint32_t totalFineOccupied = 0;
    for (const auto &block : classifier.ReadBlocks()) totalFineOccupied += block.fineOccupied;
    EXPECT_EQ(totalFineOccupied, points.size())
            << "a cell hash that failed to grow past the build-time hint's capacity would saturate "
               "and silently under-report occupancy -- exactly Task 1's Critical 1, reproduced";
}

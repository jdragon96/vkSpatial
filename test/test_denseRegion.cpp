#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "TSDF/Structure/DenseRegionClassifier.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
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
    // The occupancy numbers just asserted are only meaningful if every cell claim actually landed.
    // A probe-exhausted claim returns the SAME false as "already counted this frame", so without
    // this counter an under-reported fineOccupied/coarseOccupied is indistinguishable from a
    // correctly deduplicated one -- and both flow straight into the ratio the verdict rests on.
    EXPECT_EQ(classifier.CellInsertFailureCount(), 0u)
            << "a healthy run must not exhaust a cell hash probe chain";
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

    // Spec section 8 asks for all four spacings, and the two that were dropped are the informative
    // ones: v/2 is the CEILING boundary (the last spacing that should still read 4, because a fine
    // cell is v/2 wide, so one sample per fine cell is the most the ratio can see) and v is the
    // first spacing that should read 1 (one sample per COARSE cell, so both grids report the same
    // count). Point counts are chosen to keep each patch's footprint inside the one block while
    // spanning a comparable area.
    EXPECT_NEAR(ratioAt(baseVoxel * 0.25f, 64), 4.0, 0.6) << "s = v/4 resolves the fine grid";
    EXPECT_NEAR(ratioAt(baseVoxel * 0.5f, 32), 4.0, 0.6) << "s = v/2 is the last spacing that does";
    EXPECT_NEAR(ratioAt(baseVoxel * 1.0f, 16), 1.0, 0.3) << "s = v is the first that does not";
    EXPECT_NEAR(ratioAt(baseVoxel * 2.0f, 8), 1.0, 0.3) << "s = 2v resolves neither grid";
}

// blockVoxels is a public parameter but only [1, 32] is representable: the fine cell key packs 6
// bits per axis and reserves bits 18+ for the block slot, so at blockVoxels = 64 the fine cell index
// runs to 127, overflows its field, and aliases distinct cells onto one key -- ACROSS blocks, since
// the overflow runs into the slot field. Occupancy then under-reports and verdicts change with no
// symptom whatsoever. Rejected at the door rather than clamped, because there is nothing to clamp to.
TEST(DenseRegionAccumulate, BuildRejectsABlockWiderThanTheCellKey) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    EXPECT_THROW(classifier.Build(context, 0.01f, 64, 1u << 12), std::runtime_error);
    EXPECT_THROW(classifier.Build(context, 0.01f, 0, 1u << 12), std::runtime_error);
    EXPECT_NO_THROW(classifier.Build(context, 0.01f, 32, 1u << 12)) << "32 is the supported maximum";
}

// Record() used to fall back to min(points.size(), normals.size()) on a mismatch, which is a silent
// truncation of the caller's frame: the points past the shorter array vanish from the reconstruction
// with no counter and no error. There is no correct count to guess, so the caller has to hear about
// it -- matching DirectionalTSDF::Integrate, which throws on the same condition.
TEST(DenseRegionAccumulate, RecordRejectsAPointNormalSizeMismatch) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 12);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.01f, 8);
    normals.pop_back();

    Engine::Compute::CommandBatch batch(context);
    EXPECT_THROW(classifier.Record(points, normals, batch), std::runtime_error);
}

// Reset() is reachable on a default-constructed classifier -- every other public method guards for
// that, and this one dereferenced m_blockRecords and *m_context unconditionally.
TEST(DenseRegionAccumulate, ResetOnAnUnbuiltClassifierIsANoOp) {
    TSDF::DenseRegionClassifier classifier;
    classifier.Reset();
    EXPECT_EQ(classifier.BlockCount(), 0u);
    EXPECT_EQ(classifier.DenseBlockCount(), 0u);
}

// Reset() zeroed the records, the block count, the insert-failure counter and the dense latch -- but
// not the two readback buffers the accessors serve. DenseBlockCount(), DetailSlotEstimate() and
// ReadPartition() read those straight back with no recomputation, so a Reset() followed by a query
// answered with the PRE-Reset scene: the caller sees dense blocks in a scene that has none, and a
// partition of a frame that belongs to a discarded scene.
TEST(DenseRegionAccumulate, ResetClearsEveryCounterTheAccessorsServe) {
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
    ASSERT_GT(classifier.DenseBlockCount(), 0u) << "sanity: the scene being discarded had verdicts";

    classifier.Reset();

    EXPECT_EQ(classifier.BlockCount(), 0u);
    EXPECT_EQ(classifier.DenseBlockCount(), 0u) << "a reset scene has no dense blocks";
    EXPECT_EQ(classifier.DetailSlotEstimate(), 0u) << "a reset scene needs no detail slots";
    std::vector<uint32_t> base, detail;
    classifier.ReadPartition(base, detail);
    EXPECT_EQ(base.size(), 0u) << "a reset scene has no partitioned frame to hand back";
    EXPECT_EQ(detail.size(), 0u) << "a reset scene has no partitioned frame to hand back";
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

// DistantBlocksDoNotCollide (above) pins packBlockKey against ALIASING inside its range. This pins
// the range itself, which was the last capacity ceiling with no counter: the packed key keeps 10
// bits per axis, so `& 0x3FFu` wraps anything outside [-512, 511] onto a near block and merges the
// two regions' statistics into one record -- silently, and with the merged verdict then routing
// both regions.
//
// The range is not academic, because baseVoxel is the caller's: the extent is
// 512 x baseVoxel x blockVoxels, so at spec section 5's own regime (50 um detail, hence 100 um
// base) blockWorld is 3.2 mm and the range collapses to +/-1.638 m -- smaller than a single room.
// This fixture uses exactly that regime. Blocks +531 and -493 are the concrete colliding pair:
// (531 + 512) & 0x3FF == 19 == (-493 + 512) & 0x3FF, so they pack to the identical key while
// sitting 3.28 m apart.
//
// The fix routes an out-of-range block through the SAME counted-failure branch as a full table --
// the point still reaches the base level, so nothing is dropped, and the ceiling becomes visible
// through the counter that already existed.
TEST(DenseRegionAccumulate, BlocksOutsideThePackableRangeAreCountedNotAliased) {
    Engine::Core::Context context;
    const float baseVoxel = 1e-4f; // 100 um base voxel: blockWorld = 3.2 mm, range = +/-1.638 m
    const int blockVoxels = 32;
    const float blockWorld = baseVoxel * float(blockVoxels);

    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, baseVoxel, blockVoxels, /*maxPointPerFrame=*/1u << 12);

    // Block centres, so a float rounding wobble cannot move a point into a neighbouring block.
    auto blockCentre = [&](int blockX) {
        return Vector3f((float(blockX) + 0.5f) * blockWorld, 0.5f * blockWorld, 0.5f * blockWorld);
    };

    std::vector<Vector3f> points, normals;
    const int outsideBlock = 531;  // x ~ +1.7008 m -- past the +511 ceiling
    const int insideBlock = -493;  // x ~ -1.5760 m -- inside, and the block +531 aliases onto
    for (int i = 0; i < 5; ++i) {
        points.push_back(blockCentre(outsideBlock) + Vector3f(0.0f, 0.0f, float(i) * 1e-5f));
        normals.emplace_back(0.0f, 0.0f, 1.0f);
    }
    for (int i = 0; i < 3; ++i) {
        points.push_back(blockCentre(insideBlock) + Vector3f(0.0f, 0.0f, float(i) * 1e-5f));
        normals.emplace_back(0.0f, 0.0f, 1.0f);
    }

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    classifier.Partition(batch);
    batch.Submit();

    EXPECT_EQ(classifier.BlockInsertFailureCount(), 5u)
            << "every point in a block outside the packable range must be COUNTED, not folded onto "
               "some other block's key";

    const std::vector<TSDF::BlockRecord> blocks = classifier.ReadBlocks();
    ASSERT_EQ(blocks.size(), 1u) << "only the in-range block may own a record";
    EXPECT_EQ(blocks[0].pointCount, 3u)
            << "the in-range block's statistics must not be polluted by the far block's 5 points -- "
               "a merge would read 8";

    // Counted is not dropped: the never-lose-a-point contract has to hold through this branch too,
    // exactly as HashInsertFailureStillConservesEveryPoint pins it for the full-table branch.
    std::vector<uint32_t> base, detail;
    classifier.ReadPartition(base, detail);
    std::vector<uint8_t> seen(points.size(), 0);
    for (uint32_t index : base)   { ASSERT_LT(index, points.size()); EXPECT_EQ(seen[index]++, 0); }
    for (uint32_t index : detail) { ASSERT_LT(index, points.size()); EXPECT_EQ(seen[index]++, 0); }
    EXPECT_EQ(size_t(std::count(seen.begin(), seen.end(), 1)), points.size())
            << "an out-of-range block must still lose no points";
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

    // A block sampled NON-UNIFORMLY: ~90% of its surface area swept coarsely at s = 2v, plus a small
    // strongly curved feature sampled at s = v/8. Both live inside block (0,0,0), which spans
    // [0, 0.32)^3 at baseVoxel 0.01.
    //
    // This is the one shape that separates condition 1 (resolvesFineGrid) from condition 3
    // (keepsSignal). On any UNIFORMLY sampled surface the two are not independent: with spacing s
    // over area A, pointCount = A/s^2 and fineOccupied = min(pointCount, 4A/v^2), so keepsSignal
    // (P >= 3F) forces F != P, hence s < v/2, hence F = 4A/v^2 and F/coarseOccupied = 4 >= 3.2.
    // keepsSignal IMPLIES resolvesFineGrid, and the pair collapses to the redundancy heuristic
    // (pointCount/coarseOccupied >= 12) that this component exists to replace. Every other fixture
    // in this file is uniform, so none of them can tell the two apart.
    //
    // Non-uniform breaks the implication. The dense feature is 256x more densely sampled per unit
    // area than the sweep, so it supplies almost every point (high redundancy -> keepsSignal passes
    // easily) while covering almost none of the area (poor coverage -> the coverage-weighted ratio
    // stays near the sweep's own value of 1). That is exactly the case the redundancy heuristic gets
    // wrong, and exactly what spec sections 1-2 claim over it.
    void MakeNonUniformlySampledBlock(std::vector<Vector3f> &points,
                                      std::vector<Vector3f> &normals) {
        points.clear();
        normals.clear();

        // 1. The coarse sweep: a 0.20 x 0.20 m flat patch at s = 2v = 0.02. Offset by half a fine
        //    cell so no sample sits exactly on a cell boundary. s > v > v/2, so every point claims
        //    its own fine AND its own coarse cell -- a local ratio of exactly 1.
        const int sweepCount = 11;
        const float sweepSpacing = 0.02f;
        const float sweepOrigin = 0.0525f;
        for (int i = 0; i < sweepCount; ++i)
            for (int j = 0; j < sweepCount; ++j) {
                points.emplace_back(sweepOrigin + float(i) * sweepSpacing,
                                    sweepOrigin + float(j) * sweepSpacing, 0.2825f);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }

        // 2. The fine feature: a sphere sampled at s = v/8 = 0.00125, sized so its surface area is
        //    1/9 of the sweep's -- i.e. the sweep is 90% of the sampled area. A full sphere rather
        //    than a gentle bulge because the feature supplies ~97% of the points, so the block's
        //    normal coherence is essentially the feature's own: a gentle one would read coherent and
        //    condition 2 would veto, hiding the effect this fixture exists to show.
        const float radius = 0.0188f; // 4*pi*r^2 = 0.00444 m^2 = (0.20 m)^2 / 9
        const Vector3f centre(0.14f, 0.14f, 0.10f);
        const int latitudeCount = 47;  // pi*r / 0.00125
        const int longitudeCount = 94; // 2*pi*r / 0.00125
        for (int a = 0; a < longitudeCount; ++a)
            for (int b = 0; b < latitudeCount; ++b) {
                const float theta = float(a) * 2.0f * float(M_PI) / float(longitudeCount);
                const float phi = float(b) * float(M_PI) / float(latitudeCount);
                const Vector3f direction(std::sin(phi) * std::cos(theta),
                                         std::sin(phi) * std::sin(theta), std::cos(phi));
                points.push_back(centre + direction * radius);
                normals.push_back(direction);
            }
    }

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

    // A SPARSE curved glimpse of block (0,0,0): a 7x7 grid at 0.02 m spacing over a strongly curved
    // paraboloid. The spacing is what matters -- 0.02 is above both the coarse cell (v = 0.01) and
    // the fine cell (v/2 = 0.005), so every one of the 49 points claims its own fine AND its own
    // coarse cell, giving an exactly predictable per-frame footprint of 49: under the 64 floor, and
    // by a margin no float wobble can close.
    //
    // Curvature 10 over a 0.06 m half-width puts the normal coherence at ~0.69 (computed, then
    // measured): the frame reads as genuinely curved, so hasDetail passes. That combination -- a
    // frame that passes the curvature test while failing the footprint floor -- is the whole point;
    // see SparseGlimpseCannotLatchOnAnEarlierFramesFootprint.
    void MakeSparseCurvedGlimpse(std::vector<Vector3f> &points, std::vector<Vector3f> &normals,
                                 float blockOffset) {
        points.clear();
        normals.clear();
        const float spacing = 0.02f;
        const float curvature = 10.0f;
        const int pointsPerSide = 7;
        const float half = 0.5f * spacing * float(pointsPerSide - 1); // 0.06
        for (int i = 0; i < pointsPerSide; ++i)
            for (int j = 0; j < pointsPerSide; ++j) {
                const float x = float(i) * spacing - half;
                const float y = float(j) * spacing - half;
                // Based at z = 0.05 rather than at blockOffset, so the 0.072 m of curvature rise
                // still lands inside the block alongside the plane this glimpse follows.
                points.emplace_back(blockOffset + x, blockOffset + y,
                                    0.05f + curvature * (x * x + y * y));
                const Vector3f gradient(-2.0f * curvature * x, -2.0f * curvature * y, 1.0f);
                normals.push_back(gradient.normalized());
            }
    }

} // namespace

// Regression for a review finding: hasSurface must read a PER-FRAME occupancy, not the cumulative
// fineOccupied. fineOccupied re-counts a physical cell once per frame (the per-frame cell hash is
// wiped every frame), so it is as much a revisit counter as a footprint measure; a block with a
// genuinely small footprint can walk it past minimumFineOccupied purely by being looked at enough
// times, with no growth in real extent -- exactly what minimumFineOccupied exists to prevent. A
// single Record+Classify cannot catch this: fineOccupied, fineOccupiedFrame and fineOccupiedMax are
// all numerically equal after one frame. This replays the SAME small patch across several
// independent frames -- each its own CommandBatch, submitted, matching how a real scan integrates
// -- and checks the verdict never flips.
//
// The per-frame field it reads was fineOccupiedMax when this test was written and is now
// fineOccupiedFrame (the whole-branch review's C1 fix moved it, so that the extent condition and the
// curvature condition describe the same frame). Either satisfies THIS test -- the cumulative field
// is all it rules out -- so the move itself is pinned separately, by
// SparseGlimpseCannotLatchOnAnEarlierFramesFootprint.
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

// The regression latch for hasSurface reading fineOccupiedFrame rather than fineOccupiedMax. That
// move landed with the C1 fix so the two single-viewpoint conditions would describe the SAME frame,
// but nothing pinned it: RevisitedSmallFootprintIsNotRefined measures 56 under either field, so
// reverting the line left every test green. On a branch that mutation-tested every other
// load-bearing condition this was the one exception, and the boundary it leaves uncovered is the
// same failure class as C1 by a different route.
//
// The uncovered case: a block seen WELL once and then GLIMPSED sparsely with scattered normals.
// Under fineOccupiedMax the floor is satisfied by the first frame's footprint while the curvature
// test is satisfied by the second frame's normals, so the block latches on evidence no single
// viewpoint ever produced -- a flat wall earning a detail level off one noisy glimpse, which is
// exactly what this component exists to prevent.
//
// Frame 1 is that dense flat plane: a big footprint, but coherence 1.0, so hasDetail vetoes and
// nothing latches. Frame 2 is the sparse curved glimpse: only 49 fine cells this frame -- under the
// 64 floor -- while frame 1's accumulated statistics keep the two cumulative conditions passing.
TEST(DenseRegionClassify, SparseGlimpseCannotLatchOnAnEarlierFramesFootprint) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    const TSDF::DensityCriteria criteria;
    classifier.Build(context, 0.01f, 32, 1u << 15, criteria);

    const float blockOffset = 0.01f * 32.0f * 0.5f; // centre of block (0,0,0)

    // Frame 1: the block is seen well. Big footprint, no curvature.
    std::vector<Vector3f> densePlane, densePlaneNormals;
    MakePlane(densePlane, densePlaneNormals, 0.0025f, 64, blockOffset);
    {
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(densePlane, densePlaneNormals, batch);
        classifier.Classify(batch);
        batch.Submit();
    }
    ASSERT_EQ(classifier.DenseBlockCount(), 0u)
            << "sanity: a dense flat plane is vetoed by coherence, so nothing is latched yet";
    const std::vector<TSDF::BlockRecord> afterFrame1 = classifier.ReadBlocks();
    ASSERT_EQ(afterFrame1.size(), 1u);
    ASSERT_GE(afterFrame1[0].fineOccupiedMax, criteria.minimumFineOccupied)
            << "sanity: frame 1 must leave a remembered footprint ABOVE the floor -- that memory is "
               "the thing this test proves cannot be borrowed";

    // Frame 2: the same block, glimpsed sparsely, with normals that do read as curvature.
    std::vector<Vector3f> glimpse, glimpseNormals;
    MakeSparseCurvedGlimpse(glimpse, glimpseNormals, blockOffset);
    {
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(glimpse, glimpseNormals, batch);
        classifier.Classify(batch);
        batch.Submit();
    }

    const std::vector<TSDF::BlockRecord> blocks = classifier.ReadBlocks();
    ASSERT_EQ(blocks.size(), 1u) << "both frames must land in the one block being classified";
    const TSDF::BlockRecord &record = blocks[0];

    const double occupancyRatio = double(record.fineOccupied) / double(record.coarseOccupied);
    const double coherence =
            std::sqrt(double(record.sumNormalFrameX) * double(record.sumNormalFrameX) +
                      double(record.sumNormalFrameY) * double(record.sumNormalFrameY) +
                      double(record.sumNormalFrameZ) * double(record.sumNormalFrameZ)) /
            10000.0 / double(record.pointCountFrame);

    // The three conditions that must PASS on frame 2, so the verdict is attributable to the fourth.
    EXPECT_GE(occupancyRatio, criteria.occupancyRatio)
            << "condition 1 must pass: frame 1's accumulated coverage still carries the ratio";
    EXPECT_LT(coherence, criteria.normalCoherence)
            << "condition 2 must pass: this frame's normals do read as curvature";
    EXPECT_GE(double(record.pointCount), criteria.samplesPerFineCell * double(record.fineOccupied))
            << "condition 3 must pass: the cumulative sampling is still redundant";
    // And the one that must FAIL -- but only when it is read per frame.
    EXPECT_LT(record.fineOccupiedFrame, criteria.minimumFineOccupied)
            << "condition 4 must fail: THIS frame's footprint is under the floor";
    EXPECT_GE(record.fineOccupiedMax, criteria.minimumFineOccupied)
            << "and must fail only per-frame: the remembered maximum is still well over the floor, "
               "which is what a fineOccupiedMax reading would borrow";

    EXPECT_EQ(classifier.DenseBlockCount(), 0u)
            << "a block must earn its detail level from ONE viewpoint that saw both the extent and "
               "the curvature -- not from extent remembered off an earlier frame and curvature "
               "measured on this one";
}

// Regression for the branch's one critical review finding: the normal sum used to accumulate across
// frames in an int32, and COHERENT normals -- the flat wall -- maximise it, so the flat wall is what
// overflows first. Wrap-around drives |sumNormal| / pointCount down through normalCoherence,
// hasDetail flips true, the other three conditions already hold on a dense flat plane, and the block
// latches dense PERMANENTLY. The one condition that exists to keep flat walls out inverts into the
// condition that lets them in.
//
// Arithmetic for this exact fixture (4096 points, normal (0,0,1), x10000 fixed point, so 4.096e7 per
// frame): int32 saturates partway through frame 53 (2.147e9 / 4.096e7 = 52.4), giving coherence
// 0.9785 -- still vetoed -- and by frame 56 the wrapped sum reads -2,001,207,296, i.e. coherence
// 0.8725 < 0.9. At that point the other three all hold: ratio 57344/14336 = 4 >= 3.2; pointCount
// 229376 >= 3 x 57344; fineOccupied 1024 >= 64. Confirmed RED at exactly frame 56 before the fix.
// 60 frames, not 56, so the assertion has margin past the measured flip.
//
// On real data the wrap is far sooner: a 0.32 m block face at ~600k points/m^2 takes ~61k
// points/frame, so the first wrap lands in ~4 frames. Two tests already replayed frames; neither
// replayed more than 4, which is why this shipped through four task reviews.
//
// The fix makes the normal sum per-frame, matching what spec section 5 already decided for the
// occupancy half: single-viewpoint measurement, because averaging normals ACROSS viewpoints does not
// give a more confident estimate, it gives a mixture of differently-misregistered ones.
TEST(DenseRegionClassify, ReplayedFlatPlaneNeverOverflowsIntoADenseLatch) {
    Engine::Core::Context context;
    const float baseVoxel = 0.01f;
    const int blockVoxels = 32;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, baseVoxel, blockVoxels, 1u << 15);

    const float blockOffset = baseVoxel * float(blockVoxels) * 0.5f; // centre of block (0,0,0)
    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, /*spacing=*/0.0025f, /*count=*/64, blockOffset); // 4096, one block

    for (int frame = 0; frame < 60; ++frame) {
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(points, normals, batch);
        classifier.Classify(batch);
        batch.Submit();

        ASSERT_EQ(classifier.DenseBlockCount(), 0u)
                << "frame " << frame
                << ": a flat wall re-scanned from one viewpoint must stay vetoed by normal "
                   "coherence -- an overflowing accumulator inverts that veto and the latch makes "
                   "it permanent";
    }

    // The veto has to still be the NORMAL one, not an accident of the fixture drifting sparse: this
    // is the same plane every frame, so its coherence must read ~1 the whole way.
    const std::vector<TSDF::BlockRecord> blocks = classifier.ReadBlocks();
    ASSERT_EQ(blocks.size(), 1u);
    const double coherence =
            std::sqrt(double(blocks[0].sumNormalFrameX) * double(blocks[0].sumNormalFrameX) +
                      double(blocks[0].sumNormalFrameY) * double(blocks[0].sumNormalFrameY) +
                      double(blocks[0].sumNormalFrameZ) * double(blocks[0].sumNormalFrameZ)) /
            10000.0 / double(blocks[0].pointCountFrame);
    EXPECT_NEAR(coherence, 1.0, 0.01)
            << "the per-frame normal sum must still read a coherent plane after 60 replays";
}

// Condition 1 (resolvesFineGrid) is the spec's core decision, and until this fixture it had no
// evidence: every other fixture in this file samples uniformly, where condition 3 implies condition
// 1 (see MakeNonUniformlySampledBlock's comment for the proof), so setting occupancyRatio to 0 left
// all of them green. The claim over the redundancy heuristic was argued, never shown.
//
// This is the block that separates them: a coarse sweep over 90% of the area plus a small densely
// sampled sphere. Conditions 2, 3 and 4 all pass with margin -- the block IS curved, IS redundantly
// sampled, and DOES have real extent -- and the only thing standing between it and a second full
// tile hierarchy is that the sampling does not COVER the block. The old heuristic
// (pointCount/coarseOccupied >= 12) reads ~27 here and refines.
//
// The sub-condition assertions below are not decoration: they are what makes the verdict attributable
// to condition 1 rather than to the fixture accidentally failing something else.
TEST(DenseRegionClassify, NonUniformlySampledBlockIsNotRefined) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    const TSDF::DensityCriteria criteria;
    classifier.Build(context, 0.01f, 32, 1u << 15, criteria);

    std::vector<Vector3f> points, normals;
    MakeNonUniformlySampledBlock(points, normals);

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    batch.Submit();

    const std::vector<TSDF::BlockRecord> blocks = classifier.ReadBlocks();
    ASSERT_EQ(blocks.size(), 1u) << "both regions must land in the one block being classified";
    const TSDF::BlockRecord &record = blocks[0];

    const double occupancyRatio = double(record.fineOccupied) / double(record.coarseOccupied);
    const double coherence =
            std::sqrt(double(record.sumNormalFrameX) * double(record.sumNormalFrameX) +
                      double(record.sumNormalFrameY) * double(record.sumNormalFrameY) +
                      double(record.sumNormalFrameZ) * double(record.sumNormalFrameZ)) /
            10000.0 / double(record.pointCountFrame);

    // The three conditions that must PASS, so the verdict is attributable to the fourth.
    EXPECT_LT(coherence, criteria.normalCoherence)
            << "condition 2 must pass: the block is genuinely curved";
    EXPECT_GE(double(record.pointCount), criteria.samplesPerFineCell * double(record.fineOccupied))
            << "condition 3 must pass: the sampling is genuinely redundant";
    EXPECT_GE(record.fineOccupiedFrame, criteria.minimumFineOccupied)
            << "condition 4 must pass: the block has genuine extent";
    // And the one that must FAIL.
    EXPECT_LT(occupancyRatio, criteria.occupancyRatio)
            << "condition 1 must fail: the sampling does not cover the block";

    // The heuristic being replaced would refine this block; the point of condition 1 is that it
    // does not.
    EXPECT_GE(double(record.pointCount) / double(record.coarseOccupied), 12.0)
            << "sanity: this is a block the old redundancy heuristic sends to the detail level";

    EXPECT_EQ(classifier.DenseBlockCount(), 0u)
            << "a block whose sampling does not cover it must not earn a detail level, however "
               "redundantly its one small feature was sampled";
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

// Record() has an early return for an empty `points` above the point where the recorded-point-count
// state used to be set (a mismatched `normals` now throws instead; see
// RecordRejectsAPointNormalSizeMismatch). A frame the caller filters down to nothing must not leave
// Partition() replaying the PREVIOUS frame's still-resident m_blockIndex -- that duplicates every
// one of that frame's points into the caller's integration with no counter showing it.
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
    EXPECT_EQ(classifier.CellInsertFailureCount(), 0u)
            << "nor through the cell-hash ceiling, which grows in the same call";

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

// LargeFrameGrowsInsteadOfTruncating (above) covers a larger FRAME against a fixed Build. This
// covers a larger BUILD: Build() is not documented as one-shot, and a second Build() with a bigger
// hint used to leave m_maxPointPerFrame claiming the new size while the four point-sized buffers
// stayed at the old one. growPointBuffers' early return then fired for every subsequent frame --
// the member already said 32768 -- so Record() memcpy'd the frame into a mapped allocation sized for
// the FIRST hint, and partition wrote its index lists past the same ceiling. The smaller-hint
// direction was considered and deferred during the build; the larger-hint direction is the one that
// corrupts memory, and it was missed.
TEST(DenseRegionPartition, RebuildWithALargerHintReallocatesThePointBuffers) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, /*maxPointPerFrame=*/1024u); // point buffers hold 1536
    classifier.Build(context, 0.01f, 32, /*maxPointPerFrame=*/32768u);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.0025f, 64); // 4096 points: inside the new hint, past the old buffers

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    classifier.Partition(batch);
    batch.Submit();

    uint32_t totalBlockPointCount = 0;
    for (const auto &block : classifier.ReadBlocks()) totalBlockPointCount += block.pointCount;
    EXPECT_EQ(totalBlockPointCount, points.size())
            << "the second Build's hint must actually reach the point buffers, not just the member "
               "that guards them";

    std::vector<uint32_t> base, detail;
    classifier.ReadPartition(base, detail);
    std::vector<uint8_t> seen(points.size(), 0);
    for (uint32_t index : base)   { ASSERT_LT(index, points.size()); EXPECT_EQ(seen[index]++, 0); }
    for (uint32_t index : detail) { ASSERT_LT(index, points.size()); EXPECT_EQ(seen[index]++, 0); }
    EXPECT_EQ(size_t(std::count(seen.begin(), seen.end(), 1)), points.size())
            << "a frame recorded after a re-Build must still conserve every point exactly once";
}

// LargeFrameGrowsInsteadOfTruncating (above) guards the point/normal/blockIndex/base/detail buffer
// group, but is structurally blind to the cell hashes: accumulate.comp.glsl's pointCount/blockIndex
// writes (lines 124-130) happen unconditionally once a block slot is found, BEFORE the fine/coarse
// cell claim (lines 132-142) ever runs, and partition.comp.glsl only routes an already-counted point
// to a different LIST (base vs. detail) -- it never drops one. So Task 1's Critical 1 (cellCapacity
// silently drifting past the cell hashes' true allocation) would leave that test, and every other
// DenseRegion* fixture in this suite, green. 10 of the 12 pre-existing fixtures Build with 1<<15 and
// record nowhere near that many points, so they never fire the grow branch at all. There are TWO
// exceptions, not one: DistantBlocksDoNotCollide Builds with 1<<12 but records only 8 points, so it
// does not grow either; and LargeFrameGrowsInsteadOfTruncating (1024u hint, 4096 points) DOES fire
// the grow branch -- it just cannot see this bug, because its 4-points-per-cell spacing collapses to
// 1024 distinct fine cells, under even the frozen 2048 capacity. That last one is the sharp case: the
// one fixture that exercises growth is still blind to the half of it this test covers. Counted every
// Build() call in this file directly rather than assuming they match.
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
    // The sharpest place for this counter: this is the one fixture that genuinely oversubscribes a
    // cell hash under the mutation it guards against, so it is where a probe-exhaustion deficit
    // would appear first. Under the un-grown-hash mutation the counter is what NAMES the deficit
    // that totalFineOccupied only shows as a number that is slightly too small.
    EXPECT_EQ(classifier.CellInsertFailureCount(), 0u)
            << "a grown cell hash must not be exhausting its probe chains";
}

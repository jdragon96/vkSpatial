#include <gtest/gtest.h>

#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalHostStore.h"
#include "Engine/Spatial/DirectionalTSDF.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

using namespace Engine::Spatial;

TEST(DirectionalGroupKeyTest, EqualityComparesAllFields) {
    DirectionalGroupKey a{1, 2, 3, 4};
    DirectionalGroupKey b{1, 2, 3, 4};
    DirectionalGroupKey c{1, 2, 3, 5};
    DirectionalGroupKey d{-1, 2, 3, 4};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);
}

TEST(DirectionalGroupKeyTest, HashSpreadsAcrossDirectionsAndCoords) {
    DirectionalGroupKeyHash hash;
    std::unordered_set<size_t> seen;
    for (int g = -2; g <= 2; ++g)
        for (uint8_t d = 0; d < kNumDirections; ++d)
            seen.insert(hash(DirectionalGroupKey{g, -g, g * 3, d}));
    // 30 keys — expect essentially no collisions from a decent hash.
    EXPECT_GT(seen.size(), 25u);
}

TEST(DirectionalTSDFTypesTest, MetaPackRoundTrips) {
    uint32_t packed = PackMeta(5, SlotState::ResidentDirty, true, true);
    EXPECT_EQ(MetaDirection(packed), 5);
    EXPECT_EQ(MetaState(packed), SlotState::ResidentDirty);
    EXPECT_TRUE(MetaDirty(packed));
    EXPECT_TRUE(MetaValid(packed));

    packed = PackMeta(0, SlotState::Free, false, false);
    EXPECT_EQ(MetaDirection(packed), 0);
    EXPECT_EQ(MetaState(packed), SlotState::Free);
    EXPECT_FALSE(MetaDirty(packed));
    EXPECT_FALSE(MetaValid(packed));
}

TEST(DirectionalTSDFTypesTest, IndexGridOffsetMatchesLayout) {
    // Layout: ((lz*50 + ly)*50 + lx)*6 + dir
    EXPECT_EQ(IndexGridOffset(0, 0, 0, 0), 0u);
    EXPECT_EQ(IndexGridOffset(0, 0, 0, 5), 5u);
    EXPECT_EQ(IndexGridOffset(1, 0, 0, 0), 6u);
    EXPECT_EQ(IndexGridOffset(0, 1, 0, 0), 50u * 6u);
    EXPECT_EQ(IndexGridOffset(0, 0, 1, 0), 50u * 50u * 6u);
    EXPECT_EQ(IndexGridOffset(49, 49, 49, 5), kIndexGridCells - 1u);
}

TEST(DirectionalHostStoreTest, GetOrCreateInitializesZeroedGroup) {
    DirectionalHostStore store;
    DirectionalGroupKey key{10, -3, 7, 2};
    EXPECT_FALSE(store.Contains(key));

    auto &group = store.GetOrCreate(key);
    EXPECT_TRUE(store.Contains(key));
    EXPECT_EQ(store.Size(), 1u);
    for (const auto &v : group) {
        EXPECT_EQ(v.value, 0.0f);
        EXPECT_EQ(v.weight, 0.0f);
    }
}

TEST(DirectionalHostStoreTest, PutGetRoundTrip) {
    DirectionalHostStore store;
    DirectionalGroupKey key{0, 0, 0, 0};

    DirectionalHostStore::Group group{};
    group[0] = {0.5f, 2.0f};
    group[511] = {-0.25f, 1.0f};
    store.Put(key, group);

    const auto &loaded = store.Get(key);
    EXPECT_FLOAT_EQ(loaded[0].value, 0.5f);
    EXPECT_FLOAT_EQ(loaded[0].weight, 2.0f);
    EXPECT_FLOAT_EQ(loaded[511].value, -0.25f);
    EXPECT_FLOAT_EQ(loaded[511].weight, 1.0f);
}

TEST(DirectionalHostStoreTest, GetMissingKeyThrows) {
    DirectionalHostStore store;
    EXPECT_THROW(store.Get(DirectionalGroupKey{1, 1, 1, 1}), std::runtime_error);
}

TEST(DirectionalHostStoreTest, KeysWithDifferentDirectionsAreDistinct) {
    DirectionalHostStore store;
    store.GetOrCreate(DirectionalGroupKey{3, 3, 3, 0})[0] = {0.1f, 1.0f};
    store.GetOrCreate(DirectionalGroupKey{3, 3, 3, 1})[0] = {0.9f, 1.0f};
    EXPECT_EQ(store.Size(), 2u);
    EXPECT_FLOAT_EQ(store.Get(DirectionalGroupKey{3, 3, 3, 0})[0].value, 0.1f);
    EXPECT_FLOAT_EQ(store.Get(DirectionalGroupKey{3, 3, 3, 1})[0].value, 0.9f);
}

TEST(DirectionalTSDFTest, BeginFrameResetsIndexGridAndComputesLocalBase) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/256);
    tsdf.BeginFrame(Eigen::Vector3f::Zero());

    // groupWorldSize = 0.1 * 8 = 0.8; floor(0/0.8) - 25 = -25 per axis.
    EXPECT_EQ(tsdf.LocalBase().x(), -25);
    EXPECT_EQ(tsdf.LocalBase().y(), -25);
    EXPECT_EQ(tsdf.LocalBase().z(), -25);
    EXPECT_FLOAT_EQ(tsdf.GroupWorldSize(), 0.8f);

    auto grid = tsdf.DebugDownloadIndexGrid();
    ASSERT_EQ(grid.size(), kIndexGridCells);
    size_t invalid = 0;
    for (uint32_t v : grid)
        if (v == kInvalidPoolIndex) ++invalid;
    EXPECT_EQ(invalid, grid.size());
}

TEST(DirectionalTSDFTest, SingleFrameUploadRegistersIndexGrid) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/256);
    tsdf.BeginFrame(Eigen::Vector3f::Zero());

    std::vector<DirectionalGroupKey> required = {
            {0, 0, 0, 0},
            {0, 0, 0, 3}, // same spatial group, different direction → separate slot
            {-1, 4, 2, 5},
    };
    tsdf.EnsureResident(required);

    auto grid = tsdf.DebugDownloadIndexGrid();
    std::unordered_set<uint32_t> slots;
    for (const auto &key : required) {
        uint32_t lx = uint32_t(key.gx - tsdf.LocalBase().x());
        uint32_t ly = uint32_t(key.gy - tsdf.LocalBase().y());
        uint32_t lz = uint32_t(key.gz - tsdf.LocalBase().z());
        uint32_t poolIndex = grid[IndexGridOffset(lx, ly, lz, key.direction)];
        EXPECT_NE(poolIndex, kInvalidPoolIndex);
        EXPECT_LT(poolIndex, 256u);
        slots.insert(poolIndex);
    }
    EXPECT_EQ(slots.size(), required.size()); // invariant #1: no duplicate slots

    // A key never requested stays invalid.
    EXPECT_EQ(tsdf.DebugQueryPoolIndex({5, 5, 5, 1}), kInvalidPoolIndex);

    auto stats = tsdf.LastFrameStats();
    EXPECT_EQ(stats.missingCount, 3u);
    EXPECT_EQ(stats.residentCount, 3u);
    EXPECT_EQ(stats.h2dBytes, uint32_t(3u * kVoxelsPerGroup * sizeof(GpuTsdfVoxel)));
}

TEST(DirectionalTSDFTest, EnsureResidentSkipsAlreadyResidentKeys) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 64);
    tsdf.BeginFrame(Eigen::Vector3f::Zero());

    DirectionalGroupKey key{1, 1, 1, 0};
    tsdf.EnsureResident({key});
    uint32_t slotBefore = tsdf.DebugQueryPoolIndex(key);

    tsdf.EnsureResident({key}); // second call: nothing new to upload
    EXPECT_EQ(tsdf.DebugQueryPoolIndex(key), slotBefore);
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 1u);
    EXPECT_EQ(tsdf.LastFrameStats().residentCount, 1u);
}

TEST(DirectionalTSDFTest, HostStoreValuesRoundTripThroughGpuPool) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 64);

    DirectionalGroupKey key{1, 2, 3, 4};
    DirectionalHostStore::Group group{};
    group[7] = {0.5f, 2.0f};
    group[200] = {-0.75f, 4.0f};
    tsdf.HostStore().Put(key, group);

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident({key});

    auto out = tsdf.DebugDownloadGroupVoxels(key);
    EXPECT_NEAR(out[7].value, 0.5f, 1e-3f);
    EXPECT_NEAR(out[7].weight, 2.0f, 1e-3f);
    EXPECT_NEAR(out[200].value, -0.75f, 1e-3f);
    EXPECT_NEAR(out[200].weight, 4.0f, 1e-3f);
    EXPECT_EQ(out[0].weight, 0.0f);
}

TEST(DirectionalTSDFTest, PoolExhaustionThrows) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/2);
    tsdf.BeginFrame(Eigen::Vector3f::Zero());

    std::vector<DirectionalGroupKey> required = {
            {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}};
    EXPECT_THROW(tsdf.EnsureResident(required), std::runtime_error);
}

TEST(DirectionalTSDFTest, KeyOutsideLocalWindowThrows) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 64);
    tsdf.BeginFrame(Eigen::Vector3f::Zero()); // window covers groups [-25, 25)

    EXPECT_THROW(tsdf.EnsureResident({DirectionalGroupKey{100, 0, 0, 0}}),
                 std::runtime_error);
}

TEST(DirectionalTSDFPhase2Test, RepeatedFrameSameAABBReusesEverything) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/256);

    std::vector<DirectionalGroupKey> required = {
            {0, 0, 0, 0}, {1, 0, 0, 2}, {-2, 3, 1, 5}};

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident(required);
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 3u);

    std::vector<uint32_t> slotsBefore;
    for (const auto &key : required)
        slotsBefore.push_back(tsdf.DebugQueryPoolIndex(key));

    // Same AABB again: everything must be reused, nothing uploaded.
    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident(required);

    auto stats = tsdf.LastFrameStats();
    EXPECT_EQ(stats.missingCount, 0u);
    EXPECT_EQ(stats.h2dBytes, 0u);
    EXPECT_EQ(stats.residentCount, 3u);
    EXPECT_FLOAT_EQ(stats.overlapRatio, 1.0f);
    EXPECT_EQ(tsdf.DebugLastClassifyCounts().reusable, 3u);

    for (size_t i = 0; i < required.size(); ++i)
        EXPECT_EQ(tsdf.DebugQueryPoolIndex(required[i]), slotsBefore[i]);
}

TEST(DirectionalTSDFPhase2Test, OverlapRatioReflectsPartialReuse) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 256);

    // Frame 1: groups x ∈ [0, 9]. Frame 2: x ∈ [5, 14] → 5 reused, 5 missing.
    std::vector<DirectionalGroupKey> frame1, frame2;
    for (int x = 0; x <= 9; ++x) frame1.push_back({x, 0, 0, 0});
    for (int x = 5; x <= 14; ++x) frame2.push_back({x, 0, 0, 0});

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident(frame1);
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 10u);

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident(frame2);

    auto stats = tsdf.LastFrameStats();
    EXPECT_EQ(stats.missingCount, 5u);
    EXPECT_EQ(stats.h2dBytes, uint32_t(5u * kVoxelsPerGroup * sizeof(GpuTsdfVoxel)));
    EXPECT_FLOAT_EQ(stats.overlapRatio, 0.5f);
    // All 10 frame-1 groups are still inside the window → all reusable.
    EXPECT_EQ(tsdf.DebugLastClassifyCounts().reusable, 10u);
    // Resident = 10 reused + 5 newly uploaded.
    EXPECT_EQ(stats.residentCount, 15u);
}

// Window shift eviction → return: a clean group whose slot was NOT reused must be
// revived from the still-intact pool data (no re-upload) when the window returns.
TEST(DirectionalTSDFPhase2Test, EvictedCleanGroupRevivesWhenWindowReturns) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/8);

    DirectionalGroupKey key{0, 0, 0, 0};
    DirectionalHostStore::Group group{};
    group[42] = {0.25f, 3.0f};
    tsdf.HostStore().Put(key, group);

    // Frame 1: window at origin, upload the group.
    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident({key});
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 1u);

    // Frame 2: window far away (base.x = floor(80/0.8)-25 = 75) → key outside → clean-freed.
    tsdf.BeginFrame(Eigen::Vector3f(80.0f, 0.0f, 0.0f));
    EXPECT_EQ(tsdf.DebugLastClassifyCounts().reusable, 0u);
    EXPECT_EQ(tsdf.DebugQueryPoolIndex(key), kInvalidPoolIndex); // outside window

    // Frame 3: window returns; slot was never reused, so classify revives it.
    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident({key});

    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 0u); // no re-upload needed
    EXPECT_EQ(tsdf.DebugLastClassifyCounts().reusable, 1u);
    auto out = tsdf.DebugDownloadGroupVoxels(key);
    EXPECT_NEAR(out[42].value, 0.25f, 1e-3f);
    EXPECT_NEAR(out[42].weight, 3.0f, 1e-3f);
}

// Tiny pool forces the evicted slot to be stolen; the group must then reload from the
// host store when the window returns (clean groups need no write-back — invariant #8).
TEST(DirectionalTSDFPhase2Test, StolenSlotGroupReloadsFromHostStore) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/2);

    DirectionalGroupKey k1{0, 0, 0, 0};
    DirectionalHostStore::Group group{};
    group[7] = {-0.5f, 2.0f};
    tsdf.HostStore().Put(k1, group);

    // Frame 1: upload k1.
    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident({k1});

    // Frame 2: window far away; two new groups consume both pool slots (k1's included).
    tsdf.BeginFrame(Eigen::Vector3f(80.0f, 0.0f, 0.0f));
    tsdf.EnsureResident({DirectionalGroupKey{100, 0, 0, 0},
                         DirectionalGroupKey{101, 0, 0, 0}});
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 2u);

    // Frame 3: window back at origin; k1's slot was stolen → must reload from host store.
    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident({k1});

    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 1u);
    auto out = tsdf.DebugDownloadGroupVoxels(k1);
    EXPECT_NEAR(out[7].value, -0.5f, 1e-3f);
    EXPECT_NEAR(out[7].weight, 2.0f, 1e-3f);
}

// Invariant #1 sweep: a window sliding one group per frame must never produce a
// duplicate pool slot in the indexGrid, the per-frame missing count must match the
// analytic expectation, and non-invalid cell count must equal residentCount.
TEST(DirectionalTSDFPhase2Test, DuplicateActiveSlotNeverOccurs) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/512);

    for (int f = 0; f < 6; ++f) {
        // Shift the window by exactly one group (0.8mm) per frame.
        tsdf.BeginFrame(Eigen::Vector3f(0.8f * float(f), 0.0f, 0.0f));

        // Required: groups gx ∈ [f-2, f+2] × directions {0, 1}.
        std::vector<DirectionalGroupKey> required;
        for (int dx = -2; dx <= 2; ++dx)
            for (uint8_t d = 0; d < 2; ++d)
                required.push_back({f + dx, 0, 0, d});
        tsdf.EnsureResident(required);

        // Analytic expectation: frame 0 uploads all 10; every later frame reuses
        // gx ∈ [f-2, f+1] (8 keys) and uploads only gx = f+2 (2 directions).
        if (f == 0) {
            EXPECT_EQ(tsdf.LastFrameStats().missingCount, 10u);
        } else {
            EXPECT_EQ(tsdf.LastFrameStats().missingCount, 2u);
            EXPECT_FLOAT_EQ(tsdf.LastFrameStats().overlapRatio, 0.8f);
        }

        auto grid = tsdf.DebugDownloadIndexGrid();
        std::unordered_set<uint32_t> seen;
        uint32_t nonInvalid = 0;
        for (uint32_t v : grid) {
            if (v == kInvalidPoolIndex) continue;
            ++nonInvalid;
            EXPECT_TRUE(seen.insert(v).second) << "duplicate pool slot " << v
                                               << " at frame " << f;
        }
        EXPECT_EQ(nonInvalid, tsdf.LastFrameStats().residentCount) << "frame " << f;
    }
}

// The full-reload H2D baseline vs steady-state: after the first frame, per-frame upload
// bytes must be a small fraction of the initial full upload (the doc's core claim,
// hardware-independent form).
TEST(DirectionalTSDFPhase2Test, SteadyStateUploadIsSmallFractionOfFullReload) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 512);

    auto requiredAt = [](int f) {
        std::vector<DirectionalGroupKey> keys;
        for (int dx = -4; dx <= 4; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                keys.push_back({f + dx, dy, 0, 0});
        return keys; // 27 groups, sliding one group per frame
    };

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident(requiredAt(0));
    const uint32_t fullReloadBytes = tsdf.LastFrameStats().h2dBytes;
    ASSERT_GT(fullReloadBytes, 0u);

    uint32_t steadyBytesMax = 0;
    for (int f = 1; f <= 5; ++f) {
        tsdf.BeginFrame(Eigen::Vector3f(0.8f * float(f), 0.0f, 0.0f));
        tsdf.EnsureResident(requiredAt(f));
        steadyBytesMax = std::max(steadyBytesMax, tsdf.LastFrameStats().h2dBytes);
    }

    // 27-group window sliding 1 column/frame → 3 new groups/frame = 1/9 of full reload.
    EXPECT_EQ(steadyBytesMax, uint32_t(3u * kVoxelsPerGroup * sizeof(GpuTsdfVoxel)));
    EXPECT_LT(float(steadyBytesMax) / float(fullReloadBytes), 0.15f);
}

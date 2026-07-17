#include <gtest/gtest.h>

#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalHostStore.h"
#include "Engine/Spatial/DirectionalTSDF.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"

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

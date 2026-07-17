#include <gtest/gtest.h>

#include "Engine/Spatial/DirectionalTSDFTypes.h"

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

#include "VoxelFillDebug.h"

#include <gtest/gtest.h>

#include <array>
#include <set>

using TSDF::AdvancedEntry;
using Eigen::Vector3f;
using namespace voxdbg;

namespace {
    AdvancedEntry ent(float cx, float cy, float cz, uint32_t dir, float tsdf, float weight) {
        AdvancedEntry e;
        e.center = Vector3f(cx, cy, cz);
        e.direction = dir;
        e.tsdf = tsdf;
        e.weight = weight;
        e.normal = Vector3f(0, 0, 1);
        return e;
    }
} // namespace

TEST(VoxelFillDebug, KeyOfQuantizesToVoxel) {
    // Two centers inside the same 0.05 voxel + same dir -> identical key; different dir -> different.
    const auto a = keyOf(ent(0.101f, 0.0f, 0.0f, 4, 0, 1), 0.05f);
    const auto b = keyOf(ent(0.099f, 0.0f, 0.0f, 4, 0, 1), 0.05f);
    const auto c = keyOf(ent(0.101f, 0.0f, 0.0f, 5, 0, 1), 0.05f);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST(VoxelFillDebug, FillTrackerMarksNewAndRecordsFirstFrame) {
    FillTracker t(0.05f);
    std::vector<AdvancedEntry> f0 = {ent(0, 0, 0, 4, 0, 1), ent(1, 0, 0, 4, 0, 1)};
    auto n0 = t.update(f0, 0);
    EXPECT_EQ(n0.size(), 2u);
    EXPECT_EQ(n0[0], 1);
    EXPECT_EQ(n0[1], 1);
    EXPECT_EQ(t.size(), 2u);

    // Frame 1: one repeat (0,0,0) + one new (2,0,0). Only the new one is marked; firstFrame kept.
    std::vector<AdvancedEntry> f1 = {ent(0, 0, 0, 4, 0, 1), ent(2, 0, 0, 4, 0, 1)};
    auto n1 = t.update(f1, 1);
    EXPECT_EQ(n1[0], 0); // (0,0,0) already seen at frame 0
    EXPECT_EQ(n1[1], 1); // (2,0,0) new at frame 1
    EXPECT_EQ(t.size(), 3u);
    EXPECT_EQ(t.firstFrame(keyOf(ent(0, 0, 0, 4, 0, 1), 0.05f)), 0);
    EXPECT_EQ(t.firstFrame(keyOf(ent(2, 0, 0, 4, 0, 1), 0.05f)), 1);
}

TEST(VoxelFillDebug, FillTrackerResetReMarksEverything) {
    FillTracker t(0.05f);
    std::vector<AdvancedEntry> f = {ent(0, 0, 0, 4, 0, 1)};
    t.update(f, 0);
    t.reset();
    EXPECT_EQ(t.size(), 0u);
    auto n = t.update(f, 5);
    EXPECT_EQ(n[0], 1);
    EXPECT_EQ(t.firstFrame(keyOf(ent(0, 0, 0, 4, 0, 1), 0.05f)), 5);
}

TEST(VoxelFillDebug, TsdfColorSurfaceBandAndSign) {
    const float trunc = 0.15f;
    const Rgba surf = tsdfColor(0.0f, trunc); // |d| < 0.1*trunc -> white
    EXPECT_EQ(surf, (Rgba{255, 255, 255, 255}));
    const Rgba pos = tsdfColor(0.12f, trunc); // + -> red dominant
    EXPECT_EQ(pos[0], 255);
    EXPECT_LT(pos[2], 255);
    const Rgba neg = tsdfColor(-0.12f, trunc); // - -> blue dominant
    EXPECT_EQ(neg[2], 255);
    EXPECT_LT(neg[0], 255);
}

TEST(VoxelFillDebug, WeightColorEndpoints) {
    const Rgba lo = weightColor(0.0f, 10.0f);  // low -> blue
    EXPECT_EQ(lo[2], 255);
    EXPECT_EQ(lo[0], 0);
    const Rgba hi = weightColor(10.0f, 10.0f); // high -> red
    EXPECT_EQ(hi[0], 255);
    EXPECT_EQ(hi[2], 0);
}

TEST(VoxelFillDebug, FillFrameColorGreyForUnseenAndSpreads) {
    EXPECT_EQ(fillFrameColor(-1, 10), (Rgba{90, 90, 90, 255})); // unseen
    EXPECT_NE(fillFrameColor(0, 10), fillFrameColor(9, 10));    // early != late
}

TEST(VoxelFillDebug, DirectionColorSixDistinct) {
    std::set<Rgba> s;
    for (uint8_t d = 0; d < 6; ++d) s.insert(directionColor(d));
    EXPECT_EQ(s.size(), 6u);
}

TEST(VoxelFillDebug, BelowThresholdBoundary) {
    EXPECT_TRUE(belowThreshold(0.9f, 1.0f));
    EXPECT_FALSE(belowThreshold(1.0f, 1.0f));
    EXPECT_FALSE(belowThreshold(1.5f, 1.0f));
}

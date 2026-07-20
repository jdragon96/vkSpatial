#include "Engine/Registration/Downsample.h"
#include "Engine/Registration/RegistrationTypes.h"
#include <gtest/gtest.h>
using namespace Engine::Registration;

TEST(Registration, VoxelDownsampleReducesAndKeepsExtent) {
    PointCloud in;
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j) {
            in.points.emplace_back(i * 0.25f, j * 0.25f, 0.0f); // dense 10x10mm plane, 0.25mm spacing
            in.normals.emplace_back(0, 0, 1);
        }
    PointCloud out = DownsampleVoxel(in, 1.0f); // 1mm cells → ~10x10 = ~100 pts
    EXPECT_LT(out.points.size(), in.points.size());
    EXPECT_GT(out.points.size(), 50u);
    EXPECT_EQ(out.normals.size(), out.points.size());
    // normals preserved (all +Z)
    for (auto& n : out.normals) EXPECT_NEAR(n.z(), 1.0f, 1e-3f);
}

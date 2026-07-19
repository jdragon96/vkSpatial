#include "Engine/Spatial/DirectionalIntegrationQuality.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial;

TEST(IntegrationQuality, K1MatchesDominantAxis) {
    IntegrationQuality q; // {1,4,false}
    DirWeight out[6];
    // +Z dominant
    int n = TopKDirections(Eigen::Vector3f(0.1f, -0.2f, 0.9f).normalized(), q, out);
    ASSERT_EQ(n, 1);
    EXPECT_EQ(out[0].direction, 4u); // +Z
    EXPECT_FLOAT_EQ(out[0].relWeight, 1.0f);
}

TEST(IntegrationQuality, AxisAlignedK2StillSingle) {
    IntegrationQuality q; q.maxDirections = 2;
    DirWeight out[6];
    int n = TopKDirections(Eigen::Vector3f(0, 0, 1), q, out); // pure +Z
    ASSERT_EQ(n, 1) << "secondary axes have r=0, excluded by minRelWeight";
    EXPECT_EQ(out[0].direction, 4u);
}

TEST(IntegrationQuality, DiagonalK2SplitsAcrossTwoLayers) {
    IntegrationQuality q; q.maxDirections = 2;
    DirWeight out[6];
    Eigen::Vector3f n(1, 0, 1); n.normalize(); // 45° between +X and +Z
    int cnt = TopKDirections(n, q, out);
    ASSERT_EQ(cnt, 2);
    EXPECT_EQ(out[0].relWeight, 1.0f);          // dominant normalized to 1
    EXPECT_NEAR(out[1].relWeight, 1.0f, 1e-4f); // equal split at exactly 45°
    // both are +X(0) and +Z(4)
    uint8_t a = out[0].direction, b = out[1].direction;
    EXPECT_TRUE((a == 0u && b == 4u) || (a == 4u && b == 0u));
}

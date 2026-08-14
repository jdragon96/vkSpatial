#include "TSDF/Backends/DirectionalIntegrationQuality.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial;

#include "Engine/Core/Context.h"
#include "TSDF/Backends/DirectionalTSDF.h"

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

TEST(IntegrationQuality, NonDominantTieHonorsXYZOrder) {
    IntegrationQuality q; q.maxDirections = 2;
    DirWeight out[6];
    // |x|=|y|=1 tied, |z|=2 dominant. Tie must resolve x before y.
    int cnt = TopKDirections(Eigen::Vector3f(1, 1, 2), q, out);
    ASSERT_EQ(cnt, 2);
    EXPECT_EQ(out[0].direction, 4u); // +Z dominant
    EXPECT_EQ(out[1].direction, 0u); // +X wins the tie over +Y
}

TEST(IntegrationQuality, MultiDirectionWritesTwoLayers) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf; tsdf.Build(ctx);
    IntegrationQuality q; q.maxDirections = 2; tsdf.SetIntegrationQuality(q);
    // one sample near origin, normal at 45° between +X and +Z
    std::vector<Eigen::Vector3f> p{Eigen::Vector3f(0, 0, 0)};
    std::vector<Eigen::Vector3f> n{Eigen::Vector3f(1, 0, 1).normalized()};
    tsdf.Integrate(p, n, Eigen::Vector3f(0, 0, 5), Eigen::Vector3f::Zero());
    // the group containing the origin voxel should have nonzero weight in BOTH +X and +Z layers
    Eigen::Vector3i b = tsdf.LocalBase();
    // owner group of voxel (0,0,0): g=(0,0,0)
    auto gx = tsdf.DebugDownloadGroupVoxels({0,0,0,0}); // +X
    auto gz = tsdf.DebugDownloadGroupVoxels({0,0,0,4}); // +Z
    auto anyWeighted = [](const auto &grp){ for (auto &v : grp) if (v.weight > 0.0f) return true; return false; };
    EXPECT_TRUE(anyWeighted(gx)) << "+X layer got no contribution";
    EXPECT_TRUE(anyWeighted(gz)) << "+Z layer got no contribution";
}

// Two surfaces at the same voxel neighbourhood, normals 90° apart, must remain
// distinct points (never blurred into one averaged normal). NOTE: the brief's original
// sample set was two 1D lines of 9 points each; that under-samples the GPU marching
// extraction (directional_tsdf_extract.comp requires an axis-adjacent sign crossing
// within a single direction layer, built up from overlapping ray writes) and yields
// ZERO raw candidates regardless of the merge logic, so it can't exercise this guard.
// Widened to small 2D patches (matching the density existing extraction tests use,
// e.g. makePlane) so real candidates reach mergeCandidates.
TEST(IntegrationQuality, StrongSplitKeepsPerpendicularSurfaces) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf; tsdf.Build(ctx);
    IntegrationQuality q; q.maxDirections = 2; tsdf.SetIntegrationQuality(q);
    // a thin corner: +X-facing and +Z-facing patches meeting near a shared voxel
    std::vector<Eigen::Vector3f> p, n;
    for (int i = -4; i <= 4; ++i)
        for (int j = -4; j <= 4; ++j) { p.emplace_back(j*0.05f, i*0.05f, 0.02f); n.emplace_back(0,0,1); }
    for (int i = -4; i <= 4; ++i)
        for (int j = -4; j <= 4; ++j) { p.emplace_back(0.02f, i*0.05f, j*0.05f); n.emplace_back(1,0,0); }
    tsdf.Integrate(p, n, Eigen::Vector3f(1,0,1), Eigen::Vector3f::Zero());
    // Discriminating check — scope to the shared CORNER only. Near (|x|,|z| small) a +Z
    // candidate (~(0,0,0.02)) and a +X candidate (~(0.02,0,0)) sit ~0.028 apart, i.e. within
    // posThresh (0.6*0.1=0.06), so the merge actually decides whether to fuse them. Correct
    // behavior keeps them as separate ~+Z and ~+X normals; a broken split would blur them to
    // a ~45° normal (neither component > 0.7). Far-field points on each patch are excluded so
    // they cannot mask a corner-merge regression (the flaw the global check had).
    bool cornerZ=false, cornerX=false;
    for (auto &pt : tsdf.PointCloud()) {
        if (std::abs(pt.position.x()) > 0.08f || std::abs(pt.position.z()) > 0.08f) continue;
        if (pt.normal.z() > 0.7f) cornerZ = true;
        if (pt.normal.x() > 0.7f) cornerX = true;
    }
    EXPECT_TRUE(cornerZ && cornerX)
        << "position-close perpendicular candidates at the corner were blurred into one normal";
}

// Deterministic CI regression anchor for the §7 coverage win (the chair benchmark proves this
// on real data but needs external scans). Two planes with ~20° off-axis normals: under K=1 both
// fall in the +Z layer, but under K=2 each also writes its secondary ±X layer, so multi-direction
// extracts strictly MORE surface. If multi-direction integration regressed to single, multi would
// equal single and this fails.
TEST(IntegrationQuality, MultiDirectionExtractsMoreSurfaceThanSingle) {
    auto run = [](IntegrationQuality q) {
        Engine::Core::Context ctx;
        DirectionalTSDF tsdf; tsdf.Build(ctx); tsdf.SetIntegrationQuality(q);
        std::vector<Eigen::Vector3f> p, n;
        // ~30° off-axis: secondary relWeight (0.5/0.87)^4 = 0.109 >= 0.05 cutoff, so under
        // K=2 each sample genuinely writes its secondary ±X layer (dominant stays +Z). At a
        // shallower ~20° the secondary would fall below the cutoff and K would be inert — the
        // extra coverage must come from the multi-DIRECTION write, not just view weighting.
        for (int i = -6; i <= 6; ++i)
            for (int j = -6; j <= 6; ++j) {
                Eigen::Vector3f na(0.5f, 0.0f, 0.87f); na.normalize();
                p.emplace_back(i*0.05f, j*0.05f, 0.0f);  n.push_back(na);
                Eigen::Vector3f nb(-0.5f, 0.0f, 0.87f); nb.normalize();
                p.emplace_back(i*0.05f, j*0.05f, 0.12f); n.push_back(nb);
            }
        tsdf.Integrate(p, n, Eigen::Vector3f(0, 0, 5), Eigen::Vector3f::Zero());
        return tsdf.PointCloud().size();
    };
    const size_t single = run(IntegrationQuality{1, 4, false});
    const size_t multi  = run(IntegrationQuality{2, 4, true});
    EXPECT_GT(single, 0u);
    EXPECT_GT(multi, single)
        << "multi-direction should extract more surface (off-axis normals write a 2nd layer); "
        << "single=" << single << " multi=" << multi;
}

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "TSDF/Backends/AdvancedTSDF.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using Engine::Spatial::AdvancedTSDF;
using Engine::Spatial::AdvancedEntry;
using Eigen::Vector3f;

namespace {

    // A planar +Z patch straddling the origin, with +Z normals.
    void makePlane(std::vector<Vector3f> &pts, std::vector<Vector3f> &nrm, float span, int half) {
        pts.clear();
        nrm.clear();
        const float step = span / float(2 * half);
        for (int i = -half; i <= half; ++i)
            for (int j = -half; j <= half; ++j) {
                pts.emplace_back(i * step, j * step, 0.0f);
                nrm.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

    // Dense cylinder side (axis = Z), radial outward normals.
    void makeCylinderSide(std::vector<Vector3f> &pts, std::vector<Vector3f> &nrm, float radius,
                          float halfZ, int nAng, int nZ) {
        pts.clear();
        nrm.clear();
        for (int a = 0; a < nAng; ++a) {
            const float th = 2.0f * float(M_PI) * float(a) / float(nAng);
            const float c = std::cos(th), s = std::sin(th);
            for (int k = 0; k <= nZ; ++k) {
                const float z = -halfZ + 2.0f * halfZ * float(k) / float(nZ);
                pts.emplace_back(radius * c, radius * s, z);
                nrm.emplace_back(c, s, 0.0f); // radial outward
            }
        }
    }

} // namespace

TEST(AdvancedTSDF, LayoutIs24Bytes) {
    static_assert(sizeof(Engine::Spatial::AdvDirEntry) == 24, "24B");
    EXPECT_EQ(offsetof(Engine::Spatial::AdvDirEntry, sumNx), 12u);
}

// The window-default footgun fix: the 512^3 window is centred on the origin at ANY voxelSize
// (originVoxel = -256), unlike CompactDirectionalTSDF's fixed corner (only valid at 0.1).
TEST(AdvancedTSDF, DefaultWindowCentredAtAnyVoxelSize) {
    Engine::Core::Context ctx;
    for (float v : {0.1f, 0.05f, 0.01f}) {
        AdvancedTSDF t;
        t.Build(ctx, v, 3.0f * v);
        EXPECT_EQ(t.OriginVoxel(), Eigen::Vector3i(-256, -256, -256)) << "voxel " << v;
    }
}

TEST(AdvancedTSDF, IntegratePlaneStoresAndExtractsNormal) {
    Engine::Core::Context ctx;
    AdvancedTSDF tsdf;
    tsdf.Build(ctx, 0.05f, 0.15f); // default centred window covers [-12.8, 12.8)
    tsdf.SetIntegrationQuality({1, 4, true});

    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, 0.6f, 8);
    tsdf.Integrate(pts, nrm, Vector3f(0, 0, 1));

    // (a) stored gradient recovered at readback.
    auto entries = tsdf.DownloadEntries();
    ASSERT_GT(entries.size(), 0u);
    int checked = 0;
    for (const auto &e : entries) {
        if (e.weight <= 0.0f) continue;
        EXPECT_NEAR(e.normal.z(), 1.0f, 1e-2f);
        ++checked;
    }
    EXPECT_GT(checked, 0);

    // (b) extracted oriented cloud uses the stored-gradient normal.
    auto cloud = tsdf.ExtractPointCloud(1u << 18, /*merge=*/false);
    ASSERT_GT(cloud.normals.size(), 0u);
    double meanNz = 0.0;
    for (const auto &n : cloud.normals) meanNz += n.z();
    meanNz /= double(cloud.normals.size());
    EXPECT_GT(meanNz, 0.99);
}

// Discriminator on curved geometry: on a cylinder, the stored-gradient normal (weighted mean of
// exact input normals) stays radial (in-plane, outward). A coarse central-difference gradient
// would leak into Z and misalign — so this exercises that the stored-gradient branch is real.
TEST(AdvancedTSDF, CurvedCylinderNormalsAreRadial) {
    Engine::Core::Context ctx;
    AdvancedTSDF tsdf;
    tsdf.Build(ctx, 0.02f, 0.06f);
    tsdf.SetIntegrationQuality({2, 4, true});

    std::vector<Vector3f> pts, nrm;
    makeCylinderSide(pts, nrm, 0.3f, 0.4f, 180, 40);

    // Integrate in 4 views so every side is seen front-facing; camera far along each axis.
    const std::vector<Vector3f> camDirs = {{1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
    for (const auto &cd : camDirs) {
        std::vector<Vector3f> vp, vn;
        for (size_t i = 0; i < pts.size(); ++i)
            if (nrm[i].dot(cd) > 0.3f) { vp.push_back(pts[i]); vn.push_back(nrm[i]); }
        tsdf.Integrate(vp, vn, cd * 5.0f);
    }

    auto cloud = tsdf.ExtractPointCloud(1u << 19, /*merge=*/false);
    ASSERT_GT(cloud.points.size(), 100u);

    int radialOutward = 0, total = 0;
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        const Vector3f &p = cloud.points[i];
        const Vector3f &n = cloud.normals[i];
        Eigen::Vector2f pxy(p.x(), p.y());
        if (pxy.norm() < 1e-3f) continue;
        Eigen::Vector2f nxy(n.x(), n.y());
        ++total;
        // radial (small Z) and outward (aligned with position's XY direction)
        if (std::abs(n.z()) < 0.20f && nxy.normalized().dot(pxy.normalized()) > 0.9f)
            ++radialOutward;
    }
    ASSERT_GT(total, 100);
    EXPECT_GT(double(radialOutward) / double(total), 0.9)
            << radialOutward << "/" << total << " radial-outward";
}

// Point-to-plane (default) and projective both reconstruct a plane; toggle must not crash and
// must keep normals correct.
TEST(AdvancedTSDF, PointToPlaneToggle) {
    Engine::Core::Context ctx;
    for (bool p2p : {true, false}) {
        AdvancedTSDF tsdf;
        tsdf.Build(ctx, 0.05f, 0.15f);
        tsdf.SetIntegrationQuality({1, 4, true});
        tsdf.SetPointToPlane(p2p);

        std::vector<Vector3f> pts, nrm;
        makePlane(pts, nrm, 0.6f, 8);
        tsdf.Integrate(pts, nrm, Vector3f(0, 0, 1));

        auto cloud = tsdf.ExtractPointCloud(1u << 18, /*merge=*/false);
        ASSERT_GT(cloud.normals.size(), 0u) << "p2p=" << p2p;
        double meanNz = 0.0, meanZ = 0.0;
        for (size_t i = 0; i < cloud.normals.size(); ++i) {
            meanNz += cloud.normals[i].z();
            meanZ += cloud.points[i].z();
        }
        meanNz /= double(cloud.normals.size());
        meanZ /= double(cloud.points.size());
        EXPECT_GT(meanNz, 0.99) << "p2p=" << p2p;      // stored-gradient normal
        EXPECT_NEAR(meanZ, 0.0, 0.02) << "p2p=" << p2p; // surface at z=0
    }
}

// RecordIntegrate into a caller-owned CommandBatch produces the same occupied set as the
// self-submitting Integrate (the mapped-upload + batched-dispatch path is result-equivalent).
TEST(AdvancedTsdf, RecordIntegrateMatchesIntegrate) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, 0.4f, 10);
    const Vector3f cam(0, 0, 1);

    AdvancedTSDF a, b;
    a.Build(ctx, 0.02f, 0.06f);
    b.Build(ctx, 0.02f, 0.06f);

    a.Integrate(pts, nrm, cam); // self-submitting
    {
        Engine::Compute::CommandBatch batch(ctx);
        b.RecordIntegrate(pts, nrm, cam, batch); // recorded, then submitted once
        batch.Submit();
    }

    const auto ea = a.DownloadEntries();
    const auto eb = b.DownloadEntries();
    ASSERT_GT(ea.size(), 100u);
    EXPECT_EQ(ea.size(), eb.size());
}

// Within the Build-time capacity, IntegrateGPU is result-equivalent to Integrate (it takes the same
// upload+dispatch path, just without the clamp/grow kicking in).
TEST(AdvancedTsdf, IntegrateGpuMatchesIntegrateWithinCapacity) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, 0.4f, 10); // 21*21 = 441 points, well under the default maxPoints
    const Vector3f cam(0, 0, 1);

    AdvancedTSDF a, b;
    a.Build(ctx, 0.02f, 0.06f);
    b.Build(ctx, 0.02f, 0.06f);
    a.Integrate(pts, nrm, cam);
    b.IntegrateGPU(pts, nrm, cam);

    const auto ea = a.DownloadEntries();
    const auto eb = b.DownloadEntries();
    ASSERT_GT(ea.size(), 100u);
    EXPECT_EQ(ea.size(), eb.size());
}

// The real-time win: a frame LARGER than the Build-time maxPoints. Plain Integrate clamps (drops the
// excess -> less coverage); IntegrateGPU grows the upload buffers to fit, recovering the full frame.
TEST(AdvancedTsdf, IntegrateGpuGrowsBeyondMaxPoints) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, 0.6f, 8); // 17*17 = 289 points
    const Vector3f cam(0, 0, 1);
    const uint32_t small = 64;    // << 289: forces the clamp/grow divergence

    // Reference: whole frame integrated with a generous capacity.
    AdvancedTSDF ref;
    ref.Build(ctx, 0.05f, 0.15f, 1u << 20, 1u << 15);
    ref.Integrate(pts, nrm, cam);
    const uint32_t refFilled = ref.FilledCount();
    ASSERT_GT(refFilled, 0u);

    // Same small capacity: Integrate clamps to the first 64 points -> fewer voxels.
    AdvancedTSDF clamped;
    clamped.Build(ctx, 0.05f, 0.15f, 1u << 20, small);
    clamped.Integrate(pts, nrm, cam);
    EXPECT_LT(clamped.FilledCount(), refFilled);

    // Same small capacity: IntegrateGPU grows to fit -> matches the full-capacity reference.
    AdvancedTSDF grown;
    grown.Build(ctx, 0.05f, 0.15f, 1u << 20, small);
    grown.IntegrateGPU(pts, nrm, cam);
    EXPECT_EQ(grown.FilledCount(), refFilled);
}

// Hash auto-grow: a tile that fills its initial hash doubles + GPU-rehashes instead of overflowing
// (probe chains exceeding MAX_PROBE silently dropped voxels AND slowed integrate as the map grew). A
// map that STARTS with a hash far too small must, fed like a stream, end with the SAME occupied set as
// a generously-sized one -- no loss -- and the rehash must preserve the accumulators (normals recover).
TEST(AdvancedTsdf, HashAutoGrowMatchesLargeHashNoLoss) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, 0.6f, 20); // 41*41 points -> thousands of band voxels, forcing several grows
    const Vector3f cam(0, 0, 1);

    // Reference: a generously-sized hash never grows or overflows.
    AdvancedTSDF big;
    big.Build(ctx, 0.02f, 0.06f, 1u << 20);
    big.SetIntegrationQuality({1, 4, true});
    big.IntegrateGPU(pts, nrm, cam);
    const uint32_t refFilled = big.FilledCount();
    ASSERT_GT(refFilled, 1024u); // must exceed the tiny start below so grows are actually forced

    // Start with a hash far too small and feed the cloud in chunks (the streaming regime): the
    // per-chunk delta stays well under capacity, so auto-grow (checked before each integrate) keeps the
    // load bounded and no insert overflows. The final occupied set must match the big-hash reference.
    AdvancedTSDF grow;
    grow.Build(ctx, 0.02f, 0.06f, 1024u);
    grow.SetIntegrationQuality({1, 4, true});
    const std::size_t chunk = pts.size() / 20u + 1u;
    for (std::size_t s = 0; s < pts.size(); s += chunk) {
        const std::size_t e = std::min(pts.size(), s + chunk);
        const std::vector<Vector3f> cp(pts.begin() + s, pts.begin() + e);
        const std::vector<Vector3f> cn(nrm.begin() + s, nrm.begin() + e);
        grow.IntegrateGPU(cp, cn, cam);
    }
    EXPECT_EQ(grow.FilledCount(), refFilled); // no overflow drops -> identical occupied set

    // Rehash preserved the per-slot accumulators + gradient: the plane's normal still recovers as +Z.
    auto cloud = grow.ExtractPointCloud(1u << 18, /*merge=*/false);
    ASSERT_GT(cloud.normals.size(), 100u);
    double meanNz = 0.0;
    for (const auto &n : cloud.normals) meanNz += n.z();
    meanNz /= double(cloud.normals.size());
    EXPECT_GT(meanNz, 0.99);
}

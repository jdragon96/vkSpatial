#include "Engine/Pipeline/Registration/GpuIcp.h"
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <random>
#include <vector>
using Engine::Pipeline::LocalGrid;
using Eigen::Vector3f;

// Brute-force nearest within radius (reference).
static int bruteNearest(const std::vector<Vector3f>& pts, const Vector3f& q, float radius) {
    int best = -1; float bestD2 = radius * radius;
    for (int i = 0; i < (int)pts.size(); ++i) {
        const float d2 = (pts[i] - q).squaredNorm();
        if (d2 < bestD2) { bestD2 = d2; best = i; }
    }
    return best;
}

TEST(LocalGrid, NearestMatchesBruteForce) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<Vector3f> pts;
    for (int i = 0; i < 2000; ++i) pts.emplace_back(u(rng), u(rng), u(rng));
    const float cell = 0.1f, radius = 0.1f;
    LocalGrid grid(pts, cell);
    for (int i = 0; i < 500; ++i) {
        const Vector3f q(u(rng), u(rng), u(rng));
        const int g = grid.Nearest(q, radius);
        const int b = bruteNearest(pts, q, radius);
        if (b < 0) { EXPECT_LT(g, 0); continue; }
        // Grid and brute may pick different indices only if equidistant; compare distances.
        ASSERT_GE(g, 0);
        EXPECT_NEAR((pts[g] - q).norm(), (pts[b] - q).norm(), 1e-5f);
    }
}

#include "Engine/Core/Context.h"
#include "Engine/Pipeline/Registration/GpuIcp.h"
#include "Engine/Registration/RegistrationTypes.h"

// CPU reference: point-to-plane H,b in T's frame, over grid-NN correspondences, CENTRED on tgt centroid.
static void cpuAccumulate(const std::vector<Vector3f>& src, const Engine::Registration::PointCloud& tgt,
                          const Eigen::Matrix4f& T, float maxCorr,
                          Eigen::Matrix<double,6,6>& H, Eigen::Matrix<double,6,1>& b, int& inliers) {
    Vector3f c = Vector3f::Zero();
    for (auto& q : tgt.points) c += q; c /= float(std::max<size_t>(1, tgt.points.size()));
    std::vector<Vector3f> tc(tgt.points.size());
    for (size_t i=0;i<tc.size();++i) tc[i] = tgt.points[i] - c;
    LocalGrid grid(tc, maxCorr);
    H.setZero(); b.setZero(); inliers = 0;
    const Eigen::Matrix3f R = T.block<3,3>(0,0); const Vector3f t = T.block<3,1>(0,3);
    for (auto& s : src) {
        const Vector3f p = R * (s - c) + t;
        const int qi = grid.Nearest(p, maxCorr);
        if (qi < 0) continue;
        const Vector3f& q = tc[qi]; const Vector3f& n = tgt.normals[qi];
        const float e = (p - q).dot(n);
        Eigen::Matrix<float,6,1> J; J.head<3>() = p.cross(n); J.tail<3>() = n;
        H += (J * J.transpose()).cast<double>(); b += (-J * e).cast<double>(); ++inliers;
    }
}

TEST(GpuIcp, AccumulateMatchesCpu) {
    Engine::Core::Context ctx;
    // A small +Z plane patch as source; a matching plane as target (with +Z normals).
    std::vector<Vector3f> src; Engine::Registration::PointCloud tgt;
    for (int i=-15;i<=15;++i) for (int j=-15;j<=15;++j) {
        src.emplace_back(i*0.02f, j*0.02f, 0.01f);          // 1 cm above the target plane
        tgt.points.emplace_back(i*0.02f, j*0.02f, 0.0f);
        tgt.normals.emplace_back(0,0,1);
    }
    const Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    const float maxCorr = 0.05f;

    Eigen::Matrix<double,6,6> Hc; Eigen::Matrix<double,6,1> bc; int nc;
    cpuAccumulate(src, tgt, T, maxCorr, Hc, bc, nc);
    ASSERT_GT(nc, 100);

    Engine::Pipeline::GpuPointToPlaneIcp gpu(ctx);
    const auto out = gpu.Accumulate(src, tgt, T, maxCorr);
    EXPECT_EQ(out.inliers, nc);
    EXPECT_TRUE(((out.H - Hc).array().abs() < 1e-2 * (1.0 + Hc.array().abs())).all()) << out.H << "\n---\n" << Hc;
    EXPECT_TRUE(((out.b - bc).array().abs() < 1e-2 * (1.0 + bc.array().abs())).all()) << out.b << "\n---\n" << bc;
}

#include "Engine/Registration/Icp.h"
#include <Eigen/Geometry>

TEST(GpuIcp, SolveMatchesCpuOnCorner) {
    Engine::Core::Context ctx;
    // A 3-plane corner target (constrains all 6 DoF); source = target perturbed by a small transform.
    Engine::Registration::PointCloud tgt; std::vector<Vector3f> src;
    auto addPlane = [&](const Vector3f& o, const Vector3f& u, const Vector3f& v, const Vector3f& n){
        for (int i=-10;i<=10;++i) for (int j=-10;j<=10;++j) {
            const Vector3f p = o + u*(i*0.03f) + v*(j*0.03f);
            tgt.points.push_back(p); tgt.normals.push_back(n);
        }
    };
    addPlane({0,0,0},{1,0,0},{0,1,0},{0,0,1});
    addPlane({0,0,0},{0,1,0},{0,0,1},{1,0,0});
    addPlane({0,0,0},{1,0,0},{0,0,1},{0,1,0});
    Eigen::Isometry3f perturb = Eigen::Isometry3f::Identity();
    perturb.translate(Vector3f(0.02f, -0.015f, 0.01f));
    perturb.rotate(Eigen::AngleAxisf(0.03f, Vector3f::UnitZ()));
    for (const auto& q : tgt.points) src.push_back(perturb * q); // source is the model, moved

    Engine::Registration::RegistrationParam params; params.maxCorrDist = 0.1f; params.maxIters = 30;
    const auto cpu = Engine::Registration::AlignPointToPlaneIcp(src, tgt, Eigen::Matrix4f::Identity(), params);
    ASSERT_TRUE(cpu.valid);

    Engine::Pipeline::GpuPointToPlaneIcp gpu(ctx);
    const auto g = gpu.Solve(src, tgt, Eigen::Matrix4f::Identity(), params);
    ASSERT_TRUE(g.valid);
    // Both should recover ~perturb⁻¹ (align source back onto target). Compare the two poses directly.
    EXPECT_TRUE(((g.T - cpu.T).array().abs() < 5e-3f).all()) << "gpu:\n" << g.T << "\ncpu:\n" << cpu.T;
}

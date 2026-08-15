#include "Engine/Backend/LoopClosure.h"

#include "Engine/Features/FpfhSignature.h"
#include "BVH/NeighborQuery.h"

#include <Eigen/Geometry> // Eigen::umeyama

#include <algorithm>
#include <random>
#include <vector>

namespace Engine::Backend {

    using Engine::Features::ComputeFPFH;
    using Engine::Spatial::CpuGridNeighborhood;
    using Engine::Features::FpfhSignature;
    using Engine::Features::FPFH_DIM;
    using Engine::Core::OrientedPointCloud;

    namespace {

        float fpfhDist2(const FpfhSignature &a, const FpfhSignature &b) {
            float s = 0.0f;
            for (int c = 0; c < FPFH_DIM; ++c) {
                const float d = a.hist[c] - b.hist[c];
                s += d * d;
            }
            return s;
        }

        // Rigid transform (no scale) mapping src points onto dst points, via SVD (Umeyama).
        SE3 solveRigid(const std::vector<Eigen::Vector3d> &src,
                       const std::vector<Eigen::Vector3d> &dst) {
            const int n = static_cast<int>(src.size());
            Eigen::Matrix<double, 3, Eigen::Dynamic> S(3, n), D(3, n);
            for (int i = 0; i < n; ++i) {
                S.col(i) = src[i];
                D.col(i) = dst[i];
            }
            const Eigen::Matrix4d T = Eigen::umeyama(S, D, /*with_scaling=*/false);
            SE3 out;
            out.R = T.block<3, 3>(0, 0);
            out.t = T.block<3, 1>(0, 3);
            return out;
        }

    } // namespace

    RegistrationResult RegisterPointClouds(const Engine::Core::OrientedPointCloud &source,
                                           const Engine::Core::OrientedPointCloud &target,
                                           const RegistrationConfig &cfg) {
        RegistrationResult res;
        const size_t ns = source.size(), nt = target.size();
        if (ns < 3 || nt < 3) return res;

        // Points as double for the SE(3) math.
        std::vector<Eigen::Vector3d> sp(ns), tp(nt);
        for (size_t i = 0; i < ns; ++i) sp[i] = source.points[i].cast<double>();
        for (size_t i = 0; i < nt; ++i) tp[i] = target.points[i].cast<double>();

        // ── 1. FPFH on both clouds ──
        const auto fs = ComputeFPFH(source, {cfg.fpfhRadius});
        const auto ft = ComputeFPFH(target, {cfg.fpfhRadius});

        // ── 2. Feature correspondences: each source point -> nearest target in FPFH space ──
        struct Corr { int s, t; };
        std::vector<Corr> corr;
        corr.reserve(ns);
        for (size_t i = 0; i < ns; ++i) {
            float best = std::numeric_limits<float>::max();
            int bestJ = -1;
            for (size_t j = 0; j < nt; ++j) {
                const float d = fpfhDist2(fs[i], ft[j]);
                if (d < best) { best = d; bestJ = static_cast<int>(j); }
            }
            if (bestJ >= 0) corr.push_back({static_cast<int>(i), bestJ});
        }
        if (corr.size() < 3) return res;

        const double thr = cfg.inlierThreshold;
        const double thr2 = thr * thr;

        // ── 3. RANSAC over 3-correspondence samples ──
        std::mt19937 rng(cfg.seed);
        std::uniform_int_distribution<int> pick(0, static_cast<int>(corr.size()) - 1);
        auto edgeOk = [&](int a, int b) {
            const double ls = (sp[corr[a].s] - sp[corr[b].s]).norm();
            const double lt = (tp[corr[a].t] - tp[corr[b].t]).norm();
            if (ls < 1e-6 || lt < 1e-6) return false;
            const double r = ls / lt;
            return r > cfg.edgeLengthTol && r < 1.0 / cfg.edgeLengthTol;
        };

        int bestInliers = 0;
        SE3 bestT = SE3::Identity();
        std::vector<Eigen::Vector3d> s3(3), t3(3);
        for (int iter = 0; iter < cfg.ransacIterations; ++iter) {
            int a = pick(rng), b = pick(rng), c = pick(rng);
            if (a == b || b == c || a == c) continue;
            if (!edgeOk(a, b) || !edgeOk(b, c) || !edgeOk(a, c)) continue;

            s3 = {sp[corr[a].s], sp[corr[b].s], sp[corr[c].s]};
            t3 = {tp[corr[a].t], tp[corr[b].t], tp[corr[c].t]};
            const SE3 T = solveRigid(s3, t3);

            int inl = 0;
            for (const Corr &m : corr)
                if ((T * sp[m.s] - tp[m.t]).squaredNorm() < thr2) ++inl;
            if (inl > bestInliers) {
                bestInliers = inl;
                bestT = T;
            }
        }
        if (bestInliers < 3) return res;

        // Refit on all RANSAC inliers.
        {
            std::vector<Eigen::Vector3d> si, ti;
            for (const Corr &m : corr)
                if ((bestT * sp[m.s] - tp[m.t]).squaredNorm() < thr2) {
                    si.push_back(sp[m.s]);
                    ti.push_back(tp[m.t]);
                }
            if (si.size() >= 3) bestT = solveRigid(si, ti);
        }

        // ── 4. Point-to-plane ICP refinement against the target ──
        // Point-to-plane (residual = n_target · (T p_source − p_target)) rather than
        // point-to-point, because on smooth surfaces point-to-point ICP slides
        // tangentially — low residual but a biased transform (KinectFusion, doc 02 §3.1).
        // Linearised per KinectFusion: G = [n; q×n], update T ← Exp(δ)·T (left perturb).
        CpuGridNeighborhood targetNN(target.points, std::max(cfg.inlierThreshold * 3.0f, cfg.fpfhRadius));
        const float searchR = static_cast<float>(cfg.inlierThreshold * 3.0);
        std::vector<uint32_t> nnIdx;
        std::vector<float> nnDist;
        std::vector<Eigen::Vector3d> tn(nt);
        for (size_t i = 0; i < nt; ++i) tn[i] = target.normals[i].cast<double>();
        for (int it = 0; it < cfg.icpIterations; ++it) {
            Mat6 A = Mat6::Zero();
            Vec6 bb = Vec6::Zero();
            int used = 0;
            for (size_t i = 0; i < ns; ++i) {
                const Eigen::Vector3d q = bestT * sp[i];
                targetNN.Radius(q.cast<float>(), searchR, nnIdx, nnDist);
                int bestK = -1;
                double bestD = static_cast<double>(searchR) * searchR;
                for (const uint32_t k : nnIdx) {
                    const double d2 = (tp[k] - q).squaredNorm();
                    if (d2 < bestD) { bestD = d2; bestK = static_cast<int>(k); }
                }
                if (bestK < 0) continue;
                const Eigen::Vector3d n = tn[bestK];
                const double r0 = n.dot(q - tp[bestK]);
                Vec6 G;
                G.head<3>() = n;
                G.tail<3>() = q.cross(n);
                A += G * G.transpose();
                bb += G * r0;
                ++used;
            }
            if (used < 6) break;
            const Vec6 d = A.ldlt().solve(-bb);
            bestT = SE3Exp(d) * bestT; // left perturbation
            if (d.norm() < 1e-8) break; // converged
        }

        // ── 5. Final inlier / fitness verification (dense, over all source points) ──
        int inliers = 0;
        double sse = 0.0;
        for (size_t i = 0; i < ns; ++i) {
            const Eigen::Vector3d q = bestT * sp[i];
            targetNN.Radius(q.cast<float>(), searchR, nnIdx, nnDist);
            double bestD2 = thr2;
            bool hit = false;
            for (const uint32_t k : nnIdx) {
                const double d2 = (tp[k] - q).squaredNorm();
                if (d2 < bestD2) { bestD2 = d2; hit = true; }
            }
            if (hit) { ++inliers; sse += bestD2; }
        }

        res.T_source_to_target = bestT;
        res.inliers = inliers;
        res.fitness = double(inliers) / double(std::min(ns, nt));
        res.inlierRmse = inliers > 0 ? std::sqrt(sse / inliers) : 0.0;
        res.success = inliers >= cfg.minInliers && res.fitness >= cfg.minFitness;
        return res;
    }

} // namespace Engine::Backend

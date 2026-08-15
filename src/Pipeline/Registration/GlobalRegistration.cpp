#include "Pipeline/Registration/GlobalRegistration.h"

#include "Engine/Features/Downsample.h"
#include "Engine/Features/FeatureMatching.h"
#include "Engine/Features/Fpfh.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Geometry>
#include <random>

using namespace Engine::Features;

namespace Engine::Registration {

    Eigen::Matrix4f SolveRigidUmeyama(const std::vector<Eigen::Vector3f> &src, const std::vector<Eigen::Vector3f> &dst) {
        const size_t n = src.size();
        // 3 non-collinear points are the minimum for a well-posed rigid (rotation +
        // translation) fit; below that umeyama() is degenerate.
        if (n == 0 || dst.size() != n || n < 3) return Eigen::Matrix4f::Identity();

        Eigen::Matrix3Xf S(3, Eigen::Index(n)), D(3, Eigen::Index(n));
        for (size_t i = 0; i < n; ++i) {
            S.col(Eigen::Index(i)) = src[i];
            D.col(Eigen::Index(i)) = dst[i];
        }

        // with_scaling = false: rigid (SE(3)) fit only, no scale term.
        return Eigen::umeyama(S, D, false);
    }

    namespace {

        // Fitness floor below which a RANSAC estimate is rejected even if it happens to
        // clear the numInliers>=3 floor (e.g. 3 inliers out of 4 total correspondences is
        // not a trustworthy coarse alignment). Not part of RegistrationConfig (Task 1's
        // frozen interface) -- kept as an internal implementation constant.
        constexpr float kMinFitness = 0.1f;

        // ‖T*p_src - p_tgt‖ < inlierThr, evaluated over every entry in `corr` (not just the
        // 3-point sample T was fit from). Optionally records which correspondence indices
        // were inliers.
        size_t CountInliers(const Eigen::Matrix4f &T, const PointCloud &src, const PointCloud &tgt,
                             const std::vector<Correspondence> &corr, float inlierThr,
                             std::vector<int> *inlierCorrIdx = nullptr) {
            if (inlierCorrIdx) inlierCorrIdx->clear();
            const Eigen::Matrix3f R = T.block<3, 3>(0, 0);
            const Eigen::Vector3f t = T.block<3, 1>(0, 3);

            size_t count = 0;
            for (size_t i = 0; i < corr.size(); ++i) {
                const Correspondence &c = corr[i];
                const Eigen::Vector3f p = R * src.points[c.srcIdx] + t;
                const float d = (p - tgt.points[c.tgtIdx]).norm();
                if (d < inlierThr) {
                    ++count;
                    if (inlierCorrIdx) inlierCorrIdx->push_back(int(i));
                }
            }
            return count;
        }

        // Shared downsample->FPFH->match->RANSAC pipeline behind both EstimateRansac() and
        // Estimate(). In addition to the RegistrationResult, this also hands back the
        // downsampled clouds, matched correspondences, and inlier threshold used to reach
        // it, so Estimate() can feed the coarse inlier point pairs to Ceres and recount
        // inliers/fitness under the refined transform -- without re-running downsample/FPFH/
        // matching/RANSAC a second time.
        struct CoarsePipelineResult {
            RegistrationResult result;
            PointCloud srcDs, tgtDs;
            std::vector<Correspondence> corr;
            float inlierThr = 0.0f;
            std::vector<Eigen::Vector3f> inlierSrc, inlierDst; // T-aligned pairs, coarse inlier set
        };

        CoarsePipelineResult RunCoarsePipeline(const PointCloud &src, const PointCloud &tgt,
                                                const RegistrationConfig &cfg) {
            CoarsePipelineResult out;
            RegistrationResult &result = out.result;

            out.srcDs = DownsampleVoxel(src, cfg.voxelSize);
            out.tgtDs = DownsampleVoxel(tgt, cfg.voxelSize);
            const PointCloud &srcDs = out.srcDs;
            const PointCloud &tgtDs = out.tgtDs;

            // FPFH requires per-point normals.
            if (srcDs.normals.size() != srcDs.points.size() || tgtDs.normals.size() != tgtDs.points.size())
                return out;

            const float normalRadius = cfg.normalRadiusGain * cfg.voxelSize;
            const float fpfhRadius = cfg.fpfhRadiusGain * cfg.voxelSize;

            const std::vector<Fpfh33> srcF = ComputeFpfh(srcDs, normalRadius, fpfhRadius);
            const std::vector<Fpfh33> tgtF = ComputeFpfh(tgtDs, normalRadius, fpfhRadius);

            out.corr = MatchFeatures(srcF, tgtF, 0.95f, cfg.numMaxCorr);
            const std::vector<Correspondence> &corr = out.corr;
            if (corr.size() < 3) return out; // can't even form one 3-point sample

            out.inlierThr = cfg.ransacInlierGain * cfg.voxelSize;
            const float inlierThr = out.inlierThr;

            // Deterministic RNG (fixed seed) so this test is reproducible despite RANSAC's
            // random sampling.
            std::mt19937 rng(12345u);
            std::uniform_int_distribution<int> pick(0, int(corr.size()) - 1);

            Eigen::Matrix4f bestT = Eigen::Matrix4f::Identity();
            size_t bestInliers = 0;

            std::vector<Eigen::Vector3f> sampleSrc(3), sampleDst(3);
            for (int iter = 0; iter < cfg.ransacIters; ++iter) {
                int i0 = pick(rng);
                int i1 = pick(rng);
                while (i1 == i0) i1 = pick(rng);
                int i2 = pick(rng);
                while (i2 == i0 || i2 == i1) i2 = pick(rng);

                sampleSrc[0] = srcDs.points[corr[i0].srcIdx];
                sampleDst[0] = tgtDs.points[corr[i0].tgtIdx];
                sampleSrc[1] = srcDs.points[corr[i1].srcIdx];
                sampleDst[1] = tgtDs.points[corr[i1].tgtIdx];
                sampleSrc[2] = srcDs.points[corr[i2].srcIdx];
                sampleDst[2] = tgtDs.points[corr[i2].tgtIdx];

                const Eigen::Matrix4f T = SolveRigidUmeyama(sampleSrc, sampleDst);
                const size_t inliers = CountInliers(T, srcDs, tgtDs, corr, inlierThr);
                if (inliers > bestInliers) {
                    bestInliers = inliers;
                    bestT = T;
                }
            }

            std::vector<int> inlierCorrIdx;
            CountInliers(bestT, srcDs, tgtDs, corr, inlierThr, &inlierCorrIdx);

            out.inlierSrc.reserve(inlierCorrIdx.size());
            out.inlierDst.reserve(inlierCorrIdx.size());
            for (int idx: inlierCorrIdx) {
                out.inlierSrc.push_back(srcDs.points[corr[idx].srcIdx]);
                out.inlierDst.push_back(tgtDs.points[corr[idx].tgtIdx]);
            }

            // Final refit over the full inlier set (tighter than the 3-point sample fit).
            result.T = out.inlierSrc.size() >= 3 ? SolveRigidUmeyama(out.inlierSrc, out.inlierDst) : bestT;
            result.numInliers = inlierCorrIdx.size();
            result.fitness = corr.empty() ? 0.0f : float(result.numInliers) / float(corr.size());
            result.valid = result.numInliers >= 3 && result.fitness > kMinFitness;

            return out;
        }

        // AutoDiff residual for one src->tgt correspondence under the rigid transform being
        // solved for: r = R(q)*p_src + t - p_tgt (3-vector). `q` is a Ceres quaternion
        // ([w,x,y,z], see ceres/rotation.h), consumed via ceres::QuaternionRotatePoint so the
        // convention is guaranteed consistent with ceres::QuaternionManifold's parameter
        // layout (no manual R(q) construction to get wrong).
        struct RigidResidual {
            RigidResidual(const Eigen::Vector3f &pSrc, const Eigen::Vector3f &pTgt)
                : pSrc_(pSrc.cast<double>()), pTgt_(pTgt.cast<double>()) {}

            template<typename T>
            bool operator()(const T *const q, const T *const t, T *residual) const {
                const T p[3] = {T(pSrc_.x()), T(pSrc_.y()), T(pSrc_.z())};
                T rp[3];
                ceres::QuaternionRotatePoint(q, p, rp);
                residual[0] = rp[0] + t[0] - T(pTgt_.x());
                residual[1] = rp[1] + t[1] - T(pTgt_.y());
                residual[2] = rp[2] + t[2] - T(pTgt_.z());
                return true;
            }

            static ceres::CostFunction *Create(const Eigen::Vector3f &pSrc, const Eigen::Vector3f &pTgt) {
                return new ceres::AutoDiffCostFunction<RigidResidual, 3, 4, 3>(new RigidResidual(pSrc, pTgt));
            }

            Eigen::Vector3d pSrc_, pTgt_;
        };

    } // namespace

    RegistrationResult EstimateRansac(const PointCloud &src, const PointCloud &tgt, const RegistrationConfig &cfg) {
        return RunCoarsePipeline(src, tgt, cfg).result;
    }

    RegistrationResult Estimate(const PointCloud &src, const PointCloud &tgt, const RegistrationConfig &cfg) {
        CoarsePipelineResult coarse = RunCoarsePipeline(src, tgt, cfg);
        if (!coarse.result.valid) return coarse.result; // don't refine garbage

        // Seed q/t from the coarse T. Ceres quaternion convention is [w,x,y,z]
        // (ceres/rotation.h); Eigen::Quaterniond's w()/x()/y()/z() accessors map directly.
        const Eigen::Matrix3f R0 = coarse.result.T.block<3, 3>(0, 0);
        const Eigen::Vector3f t0 = coarse.result.T.block<3, 1>(0, 3);
        Eigen::Quaterniond quat0(Eigen::Matrix3d(R0.cast<double>()));
        quat0.normalize();

        double q[4] = {quat0.w(), quat0.x(), quat0.y(), quat0.z()};
        double t[3] = {double(t0.x()), double(t0.y()), double(t0.z())};

        ceres::Problem problem;
        problem.AddParameterBlock(q, 4, new ceres::QuaternionManifold());
        problem.AddParameterBlock(t, 3);

        const double lossScale = double(cfg.ceresLossGain * cfg.voxelSize);
        for (size_t i = 0; i < coarse.inlierSrc.size(); ++i) {
            problem.AddResidualBlock(RigidResidual::Create(coarse.inlierSrc[i], coarse.inlierDst[i]),
                                      new ceres::CauchyLoss(lossScale), q, t);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.logging_type = ceres::SILENT;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        Eigen::Quaterniond quatR(q[0], q[1], q[2], q[3]);
        quatR.normalize();
        const Eigen::Matrix3f Rr = quatR.toRotationMatrix().cast<float>();
        const Eigen::Vector3f tr{float(t[0]), float(t[1]), float(t[2])};

        RegistrationResult refined = coarse.result;
        refined.T = Eigen::Matrix4f::Identity();
        refined.T.block<3, 3>(0, 0) = Rr;
        refined.T.block<3, 1>(0, 3) = tr;

        // Recount inliers/fitness under the refined transform, over the same
        // correspondence set the coarse pipeline matched.
        refined.numInliers = CountInliers(refined.T, coarse.srcDs, coarse.tgtDs, coarse.corr, coarse.inlierThr);
        refined.fitness = coarse.corr.empty() ? 0.0f : float(refined.numInliers) / float(coarse.corr.size());
        refined.valid = refined.numInliers >= 3 && refined.fitness > kMinFitness;

        return refined;
    }

} // namespace Engine::Registration

#include "Engine/Registration/GlobalRegistration.h"

#include "Engine/Registration/Downsample.h"
#include "Engine/Registration/FeatureMatching.h"
#include "Engine/Registration/Fpfh.h"

#include <Eigen/Geometry>
#include <random>

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

    } // namespace

    RegistrationResult EstimateRansac(const PointCloud &src, const PointCloud &tgt, const RegistrationConfig &cfg) {
        RegistrationResult result;

        const PointCloud srcDs = DownsampleVoxel(src, cfg.voxelSize);
        const PointCloud tgtDs = DownsampleVoxel(tgt, cfg.voxelSize);

        // FPFH requires per-point normals.
        if (srcDs.normals.size() != srcDs.points.size() || tgtDs.normals.size() != tgtDs.points.size()) return result;

        const float normalRadius = cfg.normalRadiusGain * cfg.voxelSize;
        const float fpfhRadius = cfg.fpfhRadiusGain * cfg.voxelSize;

        const std::vector<Fpfh33> srcF = ComputeFpfh(srcDs, normalRadius, fpfhRadius);
        const std::vector<Fpfh33> tgtF = ComputeFpfh(tgtDs, normalRadius, fpfhRadius);

        const std::vector<Correspondence> corr = MatchFeatures(srcF, tgtF, 0.95f, cfg.numMaxCorr);
        if (corr.size() < 3) return result; // can't even form one 3-point sample

        const float inlierThr = cfg.ransacInlierGain * cfg.voxelSize;

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

        std::vector<Eigen::Vector3f> inlierSrc, inlierDst;
        inlierSrc.reserve(inlierCorrIdx.size());
        inlierDst.reserve(inlierCorrIdx.size());
        for (int idx: inlierCorrIdx) {
            inlierSrc.push_back(srcDs.points[corr[idx].srcIdx]);
            inlierDst.push_back(tgtDs.points[corr[idx].tgtIdx]);
        }

        // Final refit over the full inlier set (tighter than the 3-point sample fit).
        result.T = inlierSrc.size() >= 3 ? SolveRigidUmeyama(inlierSrc, inlierDst) : bestT;
        result.numInliers = inlierCorrIdx.size();
        result.fitness = corr.empty() ? 0.0f : float(result.numInliers) / float(corr.size());
        result.valid = result.numInliers >= 3 && result.fitness > kMinFitness;

        return result;
    }

} // namespace Engine::Registration

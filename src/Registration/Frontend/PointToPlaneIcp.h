#pragma once

#include "Registration/RegistrationTypes.h"
#include "Common/PointCloud.h"
#include "Registration/RegistrationParam.h"
#include "Registration/RegistrationResult.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace Registration {

    namespace detail {

        class IcpGridNN {
        public:
            IcpGridNN(const std::vector<Eigen::Vector3f> &pts, float cell)
                : m_pts(pts), m_cell(cell > 1e-8f ? cell : 1e-8f) {
                for (int i = 0; i < static_cast<int>(pts.size()); ++i) m_grid[key(cellOf(pts[i]))].push_back(i);
            }

            // Index of the nearest point within `radius`, or -1.
            int Nearest(const Eigen::Vector3f &q, float radius) const {
                const Eigen::Vector3i c = cellOf(q);
                const float r2 = radius * radius;
                int best = -1;
                float bestD2 = r2;
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const auto it = m_grid.find(key(Eigen::Vector3i(c.x() + dx, c.y() + dy, c.z() + dz)));
                            if (it == m_grid.end()) continue;
                            for (int idx: it->second) {
                                const float d2 = (q - m_pts[idx]).squaredNorm();
                                if (d2 < bestD2) {
                                    bestD2 = d2;
                                    best = idx;
                                }
                            }
                        }
                return best;
            }

        private:
            Eigen::Vector3i cellOf(const Eigen::Vector3f &p) const {
                return Eigen::Vector3i(static_cast<int>(std::floor(p.x() / m_cell)),
                                       static_cast<int>(std::floor(p.y() / m_cell)),
                                       static_cast<int>(std::floor(p.z() / m_cell)));
            }
            static int64_t key(const Eigen::Vector3i &c) {
                return (int64_t(c.x()) & 0x1FFFFF) | ((int64_t(c.y()) & 0x1FFFFF) << 21) |
                       ((int64_t(c.z()) & 0x1FFFFF) << 42);
            }

            const std::vector<Eigen::Vector3f> &m_pts;
            float m_cell;
            std::unordered_map<int64_t, std::vector<int>> m_grid;
        };

    } // namespace detail


    // `sourceNormals` (sensor/source-frame, pre-pose) may be empty to skip the normal-compatibility
    // rejection entirely (e.g. when the caller has no per-point source normals); when non-empty it MUST
    // be index-aligned with `src` (sourceNormals[i] corresponds to src[i]).
    inline Registration::RegistrationResult AlignPointToPlaneIcp(const std::vector<Eigen::Vector3f> &src,
                                                   const std::vector<Eigen::Vector3f> &sourceNormals,
                                                   const Common::PointCloud &tgt,
                                                   const Eigen::Matrix4f &priorT,
                                                   const Registration::RegistrationParam &params = {}) {
        Registration::RegistrationResult res;
        res.T = priorT;
        if (src.empty() || tgt.points.size() < 3 || tgt.normals.size() != tgt.points.size()) return res;
        if (!sourceNormals.empty() && sourceNormals.size() != src.size()) return res;

        // Grid built ONCE per solve, at the WIDEST (coarsest) distance -- params.maxCorrDist. This is
        // the per-solve hoist: coarse-to-fine annealing (below) only narrows the per-iteration QUERY
        // radius passed to grid.Nearest(); it never rebuilds the grid. Because the grid cell equals
        // the widest distance and the 3x3x3 neighbor scan covers a full cell width in every
        // direction, any query radius <= that width (i.e. every annealed value, since
        // minCorrespondenceDistance < maxCorrDist by construction) is still fully covered.
        const detail::IcpGridNN grid(tgt.points, params.maxCorrDist);
        Eigen::Matrix4f T = priorT;

        for (int iter = 0; iter < params.maxIters; ++iter) {
            // Coarse-to-fine: shrinks geometrically from maxCorrDist (iter 0) to
            // minCorrespondenceDistance (final iter); huberScale anneals by the same ratio. A no-op
            // (returns the fixed params.maxCorrDist / params.huberScale) when
            // minCorrespondenceDistance is 0 (default) -- identical to pre-Task-5 behaviour.
            const AnnealedIcpIterationParams annealed = AnnealIcpIteration(params, iter);
            Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
            Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
            const Eigen::Matrix3f R = T.block<3, 3>(0, 0);
            const Eigen::Vector3f t = T.block<3, 1>(0, 3);
            int inliers = 0;
            float sumOfSquaredResiduals = 0.0f;

            // 1. Accumulate the point-to-plane normal equations over current correspondences.
            for (size_t sourceIndex = 0; sourceIndex < src.size(); ++sourceIndex) {
                const Eigen::Vector3f &s = src[sourceIndex];
                const Eigen::Vector3f p = R * s + t; // src point in the current frame
                const int qi = grid.Nearest(p, annealed.maxCorrespondenceDistance);
                if (qi < 0) continue;
                const Eigen::Vector3f &q = tgt.points[qi];
                const Eigen::Vector3f &n = tgt.normals[qi];
                const float e = (p - q).dot(n); // point-to-plane residual

                // Normal-compatibility rejection: drop correspondences whose surfaces face too
                // differently (e.g. opposite sides of a thin structure), even if close in distance.
                if (!sourceNormals.empty()) {
                    const Eigen::Vector3f transformedSourceNormal = R * sourceNormals[sourceIndex];
                    if (transformedSourceNormal.dot(n) < params.normalCompatibilityCosine) continue;
                }

                Eigen::Matrix<float, 6, 1> J;
                J.head<3>() = p.cross(n); // rotation part
                J.tail<3>() = n;          // translation part

                // Huber (robust) weight: full trust inside the knee, down-weighted (not hard-rejected)
                // beyond it, so a few gross outliers cannot dominate the normal equations.
                const float absoluteResidual = std::abs(e);
                const float robustWeight = absoluteResidual <= annealed.huberScale
                                               ? 1.0f
                                               : annealed.huberScale / absoluteResidual;
                H += robustWeight * (J * J.transpose());
                b += robustWeight * (-J * e);
                sumOfSquaredResiduals += e * e; // RMSE stays unweighted (a true fit metric)
                ++inliers;
            }
            if (inliers < params.minInliers) break;

            // 2. Solve H x = b for the incremental twist x = [rot(3), trans(3)].
            const Eigen::Matrix<float, 6, 1> x = H.ldlt().solve(b);

            // 3. Compose the incremental transform onto T (delta applied in the current frame).
            const Eigen::Matrix3f Rd = (Eigen::AngleAxisf(x[2], Eigen::Vector3f::UnitZ()) *
                                        Eigen::AngleAxisf(x[1], Eigen::Vector3f::UnitY()) *
                                        Eigen::AngleAxisf(x[0], Eigen::Vector3f::UnitX()))
                                               .toRotationMatrix();
            Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
            delta.block<3, 3>(0, 0) = Rd;
            delta.block<3, 1>(0, 3) = x.tail<3>();
            T = delta * T;

            res.numInliers = static_cast<std::size_t>(inliers);
            res.fitness = float(inliers) / float(src.size());
            res.rmse = inliers > 0 ? std::sqrt(sumOfSquaredResiduals / float(inliers)) : 0.0f;
            if (x.norm() < params.convEps) break;
        }

        res.T = T;
        res.valid = res.numInliers >= static_cast<std::size_t>(params.minInliers);
        return res;
    }

} // namespace Registration

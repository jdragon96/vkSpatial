#pragma once

#include "Engine/Eval/SyntheticSurface.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace Engine::Eval {

    inline float AccuracyRMSE(const std::vector<Eigen::Vector3f> &reconPoints,
                              const Surface &surface) {
        if (reconPoints.empty()) return std::numeric_limits<float>::infinity();
        double sumSq = 0.0;
        for (const auto &p: reconPoints) {
            const float d = surface.Distance(p);
            sumSq += double(d) * double(d);
        }
        return float(std::sqrt(sumSq / double(reconPoints.size())));
    }

    // Brute force, kept for the accelerated version below to be checked against. O(|from|x|to|):
    // usable only for small clouds -- see NearestNeighbourRMSE.
    inline float NearestNeighbourRMSEBruteForce(const std::vector<Eigen::Vector3f> &from,
                                                const std::vector<Eigen::Vector3f> &to) {
        if (from.empty() || to.empty()) return std::numeric_limits<float>::infinity();
        double sumSq = 0.0;
        for (const auto &a: from) {
            float best = std::numeric_limits<float>::max();
            for (const auto &b: to) {
                const float d2 = (a - b).squaredNorm();
                if (d2 < best) best = d2;
            }
            sumSq += double(best);
        }
        return float(std::sqrt(sumSq / double(from.size())));
    }

    namespace rmse_detail {

        // Uniform grid over `to`, searched in expanding Chebyshev rings. EXACT nearest neighbour,
        // not an approximation: a ring at distance r cannot hold a point closer than (r-1)*cell to
        // a query in the centre cell, so once the best distance found is within r*cell the rings
        // beyond r cannot improve on it and the search stops.
        class NearestPointGrid {
        public:
            NearestPointGrid(const std::vector<Eigen::Vector3f> &points, float cell)
                : m_points(points), m_cell(cell > 0.0f ? cell : 1.0f) {
                Eigen::Vector3f minimum = points[0], maximum = points[0];
                for (const Eigen::Vector3f &p: points) {
                    minimum = minimum.cwiseMin(p);
                    maximum = maximum.cwiseMax(p);
                }
                m_origin = minimum;
                for (int axis = 0; axis < 3; ++axis)
                    m_dims[axis] = std::max(1, int((maximum[axis] - minimum[axis]) / m_cell) + 1);

                const std::size_t cellCount = std::size_t(m_dims.x()) * m_dims.y() * m_dims.z();
                m_bucketStart.assign(cellCount + 1, 0);
                for (const Eigen::Vector3f &p: points) ++m_bucketStart[cellIndex(cellOf(p)) + 1];
                for (std::size_t i = 0; i < cellCount; ++i) m_bucketStart[i + 1] += m_bucketStart[i];

                m_bucketPoints.resize(points.size());
                std::vector<std::uint32_t> next(m_bucketStart.begin(), m_bucketStart.end() - 1);
                for (std::size_t i = 0; i < points.size(); ++i)
                    m_bucketPoints[next[cellIndex(cellOf(points[i]))]++] = std::uint32_t(i);
            }

            float SquaredDistanceToNearest(const Eigen::Vector3f &query) const {
                const Eigen::Vector3i centre = clampToGrid(cellOf(query));
                float best = std::numeric_limits<float>::max();
                const int maxRing = m_dims.maxCoeff();

                for (int ring = 0; ring <= maxRing; ++ring) {
                    scanRing(query, centre, ring, best);

                    // Having finished ring r, every cell in ring r+1 or beyond is separated from
                    // the query's own cell by at least r whole cells, so nothing out there can be
                    // nearer than r*cell. Checked AFTER the scan, and against r rather than r+1:
                    // the query sits somewhere inside the centre cell, not at its centre, so one
                    // cell of slack is what makes the bound safe.
                    const float unreachable = float(ring) * m_cell;
                    if (best <= unreachable * unreachable) break;
                }
                return best;
            }

        private:
            Eigen::Vector3i cellOf(const Eigen::Vector3f &p) const {
                return Eigen::Vector3i(int(std::floor((p.x() - m_origin.x()) / m_cell)),
                                       int(std::floor((p.y() - m_origin.y()) / m_cell)),
                                       int(std::floor((p.z() - m_origin.z()) / m_cell)));
            }
            Eigen::Vector3i clampToGrid(Eigen::Vector3i c) const {
                for (int axis = 0; axis < 3; ++axis)
                    c[axis] = std::min(std::max(c[axis], 0), m_dims[axis] - 1);
                return c;
            }
            std::size_t cellIndex(const Eigen::Vector3i &c) const {
                const Eigen::Vector3i g = clampToGrid(c);
                return (std::size_t(g.z()) * m_dims.y() + g.y()) * m_dims.x() + g.x();
            }

            // Visits every cell whose Chebyshev distance from `centre` is exactly `ring`.
            void scanRing(const Eigen::Vector3f &query, const Eigen::Vector3i &centre, int ring,
                          float &best) const {
                for (int dz = -ring; dz <= ring; ++dz)
                    for (int dy = -ring; dy <= ring; ++dy)
                        for (int dx = -ring; dx <= ring; ++dx) {
                            if (std::max(std::max(std::abs(dx), std::abs(dy)), std::abs(dz)) != ring)
                                continue;
                            const Eigen::Vector3i c(centre.x() + dx, centre.y() + dy, centre.z() + dz);
                            if ((c.array() < 0).any() || (c.array() >= m_dims.array()).any()) continue;
                            const std::size_t index = cellIndex(c);
                            for (std::uint32_t s = m_bucketStart[index]; s < m_bucketStart[index + 1]; ++s) {
                                const float d2 = (query - m_points[m_bucketPoints[s]]).squaredNorm();
                                if (d2 < best) best = d2;
                            }
                        }
            }

            const std::vector<Eigen::Vector3f> &m_points;
            float m_cell;
            Eigen::Vector3f m_origin;
            Eigen::Vector3i m_dims;
            std::vector<std::uint32_t> m_bucketStart;
            std::vector<std::uint32_t> m_bucketPoints;
        };

        // Roughly one point per cell: the ring search then touches a handful of cells per query,
        // while a much finer grid pays for empty rings and a much coarser one degenerates towards
        // the brute-force scan.
        inline float ChooseCellSize(const std::vector<Eigen::Vector3f> &points) {
            Eigen::Vector3f minimum = points[0], maximum = points[0];
            for (const Eigen::Vector3f &p: points) {
                minimum = minimum.cwiseMin(p);
                maximum = maximum.cwiseMax(p);
            }
            const Eigen::Vector3f extent = maximum - minimum;
            const float volume = std::max(extent.x(), 1e-6f) * std::max(extent.y(), 1e-6f) *
                                 std::max(extent.z(), 1e-6f);
            const float cell = std::cbrt(volume / float(points.size()));
            return cell > 1e-6f ? cell : 1e-6f;
        }

    } // namespace rmse_detail

    // EXACT mean nearest-neighbour distance, grid-accelerated.
    //
    // This was a brute-force double loop, which is O(|from|x|to|). On icp_quality_diag's own
    // comparison at voxel 0.01 the two reconstructions are ~1.9M points each, so one direction is
    // 3.6e12 distance evaluations and the run never finished -- measured at over seven hours of
    // wall clock with no output, because both tracker runs had already completed into a stdio
    // buffer that was never flushed. A single tracker skips the comparison and finishes in 90 s,
    // which is what made this look like a tracker problem rather than a metric one.
    inline float NearestNeighbourRMSE(const std::vector<Eigen::Vector3f> &from,
                                      const std::vector<Eigen::Vector3f> &to) {
        if (from.empty() || to.empty()) return std::numeric_limits<float>::infinity();

        const rmse_detail::NearestPointGrid grid(to, rmse_detail::ChooseCellSize(to));
        double sumSq = 0.0;
        for (const Eigen::Vector3f &a: from) sumSq += double(grid.SquaredDistanceToNearest(a));
        return float(std::sqrt(sumSq / double(from.size())));
    }

    inline float CompletenessRMSE(const std::vector<Eigen::Vector3f> &gtDense,
                                  const std::vector<Eigen::Vector3f> &reconPoints) {
        return NearestNeighbourRMSE(gtDense, reconPoints);
    }

    inline float ReconToGtNnRMSE(const std::vector<Eigen::Vector3f> &reconPoints,
                                 const std::vector<Eigen::Vector3f> &gtDense) {
        return NearestNeighbourRMSE(reconPoints, gtDense);
    }

} // namespace Engine::Eval

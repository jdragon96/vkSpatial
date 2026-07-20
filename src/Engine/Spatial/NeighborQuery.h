#pragma once

// Radius-neighbourhood query over a fixed point set, abstracted behind an interface so that
// feature code (FPFH) is agnostic to the backend. A CPU uniform-grid implementation ships now;
// a GPU BVH-backed implementation (Engine::Spatial::SpatialIndex::RadiusSearch) can adopt the
// same interface later once the Engine::Core large-N determinism issue is resolved.

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace Engine::Spatial {

    struct NeighborQuery {
        virtual ~NeighborQuery() = default;

        // Neighbours of `p` within `radius`. Appends point indices into the underlying set and
        // their Euclidean distances to `p`. Whether `p`'s own index is returned is up to the
        // caller's data (a query point coincident with a set point yields distance 0).
        virtual void Radius(const Eigen::Vector3f &p, float radius,
                            std::vector<uint32_t> &outIdx, std::vector<float> &outDist) const = 0;
    };

    // Uniform spatial-hash grid with cell size = query radius. Average O(1) per neighbour via a
    // 3x3x3 cell scan. Holds a non-owning reference to `points`, which must outlive the object.
    class CpuGridNeighborhood : public NeighborQuery {
    public:
        CpuGridNeighborhood(const std::vector<Eigen::Vector3f> &points, float cellSize)
            : m_points(points), m_cell(cellSize > 1e-12f ? cellSize : 1e-12f) {
            if (points.empty()) {
                m_origin = Eigen::Vector3f::Zero();
                return;
            }
            m_origin = points[0];
            for (const auto &p : points) m_origin = m_origin.cwiseMin(p);
            for (uint32_t i = 0; i < points.size(); ++i)
                m_grid[cellKey(cellCoord(points[i]))].push_back(i);
        }

        void Radius(const Eigen::Vector3f &p, float radius,
                    std::vector<uint32_t> &outIdx, std::vector<float> &outDist) const override {
            outIdx.clear();
            outDist.clear();
            const float r2 = radius * radius;
            const Eigen::Vector3i c = cellCoord(p);
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        auto it = m_grid.find(cellKey(c + Eigen::Vector3i(dx, dy, dz)));
                        if (it == m_grid.end()) continue;
                        for (uint32_t idx : it->second) {
                            const float d2 = (m_points[idx] - p).squaredNorm();
                            if (d2 <= r2) {
                                outIdx.push_back(idx);
                                outDist.push_back(std::sqrt(d2));
                            }
                        }
                    }
        }

    private:
        Eigen::Vector3i cellCoord(const Eigen::Vector3f &p) const {
            const Eigen::Vector3f q = (p - m_origin) / m_cell;
            return Eigen::Vector3i(int(std::floor(q.x())), int(std::floor(q.y())), int(std::floor(q.z())));
        }
        // Pack (biased) cell coords into a 63-bit key; bias keeps them non-negative for the
        // origin=bbox-min case while still tolerating a small amount of underflow near the edge.
        static uint64_t cellKey(const Eigen::Vector3i &c) {
            constexpr int64_t kBias = 1 << 20;      // ±1M cells/axis
            constexpr uint64_t kMask = (1ull << 21) - 1;
            const uint64_t x = uint64_t(int64_t(c.x()) + kBias) & kMask;
            const uint64_t y = uint64_t(int64_t(c.y()) + kBias) & kMask;
            const uint64_t z = uint64_t(int64_t(c.z()) + kBias) & kMask;
            return x | (y << 21) | (z << 42);
        }

        const std::vector<Eigen::Vector3f> &m_points; // non-owning
        float m_cell;
        Eigen::Vector3f m_origin;
        std::unordered_map<uint64_t, std::vector<uint32_t>> m_grid;
    };

} // namespace Engine::Spatial

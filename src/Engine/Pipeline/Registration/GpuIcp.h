#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <vector>

namespace Engine::Pipeline {

    // Dense uniform grid over a point set's AABB (cell = correspondence radius). Buckets are stored as
    // a CSR-style pair (bucketStart prefix-sum + bucketIdx grouped indices) so the SAME arrays upload
    // straight to the GPU. Built once per ICP solve over the CROPPED (local) target -> small + cheap.
    class LocalGrid {
    public:
        LocalGrid(const std::vector<Eigen::Vector3f> &pts, float cell);
        int Nearest(const Eigen::Vector3f &q, float radius) const;

        Eigen::Vector3f m_origin;                // AABB min
        Eigen::Vector3i m_dims{1, 1, 1};         // cells per axis
        float m_cell = 1.0f;
        std::vector<uint32_t> m_bucketStart;     // size dims.prod()+1 (prefix sum)
        std::vector<uint32_t> m_bucketIdx;       // size pts.size() (pt indices grouped by cell)
        const std::vector<Eigen::Vector3f> &m_pts;

    private:
        int cellIndex(const Eigen::Vector3i &c) const {
            return (c.z() * m_dims.y() + c.y()) * m_dims.x() + c.x();
        }
        Eigen::Vector3i cellOf(const Eigen::Vector3f &p) const {
            return Eigen::Vector3i(int(std::floor((p.x() - m_origin.x()) / m_cell)),
                                   int(std::floor((p.y() - m_origin.y()) / m_cell)),
                                   int(std::floor((p.z() - m_origin.z()) / m_cell)));
        }
    };

} // namespace Engine::Pipeline

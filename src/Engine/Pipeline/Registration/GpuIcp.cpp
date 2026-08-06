#include "Engine/Pipeline/Registration/GpuIcp.h"
#include <algorithm>
#include <cmath>

namespace Engine::Pipeline {

    LocalGrid::LocalGrid(const std::vector<Eigen::Vector3f> &pts, float cell) : m_pts(pts) {
        m_cell = cell > 1e-8f ? cell : 1e-8f;
        if (pts.empty()) { m_bucketStart.assign(2, 0); return; }
        Eigen::Vector3f mn = pts[0], mx = pts[0];
        for (const auto &p : pts) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
        m_origin = mn;
        for (int a = 0; a < 3; ++a)
            m_dims[a] = std::max(1, int(std::floor((mx[a] - mn[a]) / m_cell)) + 1);
        const int nCells = m_dims.x() * m_dims.y() * m_dims.z();

        // Counting sort of point indices by cell -> CSR (bucketStart prefix sum, bucketIdx grouped).
        m_bucketStart.assign(nCells + 1, 0);
        for (const auto &p : pts) ++m_bucketStart[cellIndex(cellOf(p)) + 1];
        for (int i = 0; i < nCells; ++i) m_bucketStart[i + 1] += m_bucketStart[i];
        m_bucketIdx.resize(pts.size());
        std::vector<uint32_t> cursor(m_bucketStart.begin(), m_bucketStart.end() - 1);
        for (int i = 0; i < (int)pts.size(); ++i)
            m_bucketIdx[cursor[cellIndex(cellOf(pts[i]))]++] = uint32_t(i);
    }

    int LocalGrid::Nearest(const Eigen::Vector3f &q, float radius) const {
        if (m_pts.empty()) return -1;
        const Eigen::Vector3i c = cellOf(q);
        const float r2 = radius * radius;
        int best = -1; float bestD2 = r2;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const Eigen::Vector3i cc(c.x() + dx, c.y() + dy, c.z() + dz);
                    if ((cc.array() < 0).any() || (cc.array() >= m_dims.array()).any()) continue;
                    const int ci = cellIndex(cc);
                    for (uint32_t k = m_bucketStart[ci]; k < m_bucketStart[ci + 1]; ++k) {
                        const int idx = int(m_bucketIdx[k]);
                        const float d2 = (q - m_pts[idx]).squaredNorm();
                        if (d2 < bestD2) { bestD2 = d2; best = idx; }
                    }
                }
        return best;
    }

} // namespace Engine::Pipeline

#include "Engine/Spatial/AdaptiveVoxelGrid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>

namespace Engine::Spatial {

    void AdaptiveVoxelGrid::Build(Engine::Core::Context &ctx, float fineVoxelSize, float truncation,
                                  uint32_t hashCapacity, uint32_t maxPoints) {
        m_h = fineVoxelSize;
        m_trunc = truncation;
        m_fine.Build(ctx, fineVoxelSize, truncation, hashCapacity, maxPoints);
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::Integrate(const std::vector<Eigen::Vector3f> &points,
                                      const Eigen::Vector3f &cameraPos) {
        m_fine.Integrate(points, cameraPos);
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::Reset() {
        m_fine.Reset();
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::SetVarianceThreshold(float sigma2) {
        m_threshold = sigma2;
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::SetVariancePercentile(float p) {
        m_threshold = -1.0f; // re-enable percentile mode
        m_percentile = p;
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::SetMinOccupancy(uint32_t n) {
        m_minOcc = n;
        m_mixedDirty = true;
    }

    std::vector<MixedVoxel> AdaptiveVoxelGrid::DownloadMixedVoxels() {
        if (m_mixedDirty) buildMixed();
        return m_mixed;
    }

    size_t AdaptiveVoxelGrid::FineCount() const {
        if (m_mixedDirty) const_cast<AdaptiveVoxelGrid *>(this)->buildMixed();
        return m_fineCount;
    }

    size_t AdaptiveVoxelGrid::CoarseCount() const {
        if (m_mixedDirty) const_cast<AdaptiveVoxelGrid *>(this)->buildMixed();
        return m_coarseCount;
    }

    AdaptiveMesh AdaptiveVoxelGrid::ExtractMesh() {
        // Task 4
        return {};
    }

    // Recovers each fine voxel's integer coordinate from its world-space centre, buckets
    // fine voxels into 2x2x2 coarse blocks (floor-divided coord, correct for negatives), and
    // coarsens a block to a single weight-averaged voxel iff it is sufficiently observed
    // (count >= m_minOcc) AND sufficiently flat (mean per-voxel variance < theta). theta is
    // either the fixed m_threshold (>=0) or, in percentile mode (m_threshold < 0), the
    // m_percentile-th quantile of the observed per-voxel variances.
    void AdaptiveVoxelGrid::buildMixed() {
        const std::vector<VoxelStat> vox = m_fine.DownloadVoxels();

        auto vcoord = [&](const Eigen::Vector3f &c) {
            return Eigen::Vector3i(
                    static_cast<int>(std::lround(c.x() / m_h - 0.5f)),
                    static_cast<int>(std::lround(c.y() / m_h - 0.5f)),
                    static_cast<int>(std::lround(c.z() / m_h - 0.5f)));
        };

        float theta = m_threshold;
        if (theta < 0.0f) {
            std::vector<float> s;
            s.reserve(vox.size());
            for (const auto &v : vox) s.push_back(v.variance);
            std::sort(s.begin(), s.end());
            if (s.empty()) {
                theta = 0.0f;
            } else {
                const size_t idx = std::min(s.size() - 1, static_cast<size_t>(m_percentile * s.size()));
                const float qv = s[idx];
                // Escape any exact-tie plateau at the p-th quantile value: real fixture data
                // has a large mass of EXACTLY-zero-variance fine voxels (voxels touched by a
                // single observation have zero sample variance by definition), so for common
                // percentiles qv==0 and a literal "meanVar < qv" would coarsen nothing --
                // every tied (zero-variance) block fails a strict "< 0" test. Percentile mode
                // means "coarsen the low-variance p-fraction, ties included", so theta is
                // redefined as the smallest observed variance strictly greater than qv; the
                // comparison below stays a single, uniform "meanVar < theta" (ties at/below qv
                // now satisfy it), matching fixed-threshold mode's semantics exactly when there
                // are no ties.
                const auto it = std::upper_bound(s.begin(), s.end(), qv);
                theta = (it != s.end()) ? *it : std::nextafter(qv, std::numeric_limits<float>::infinity());
            }
        }

        // Floor division (a / b rounded toward -infinity), correct for negative coords --
        // plain C++ integer division truncates toward zero, which would bucket e.g. -1 and 0
        // into different-signed halves of the same coarse block.
        auto fdiv = [](int a, int b) {
            const int q = a / b, r = a % b;
            return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
        };

        struct Blk {
            std::vector<size_t> idx;
        };
        std::map<std::array<int, 3>, Blk> blocks;
        for (size_t i = 0; i < vox.size(); ++i) {
            const Eigen::Vector3i vc = vcoord(vox[i].center);
            blocks[{fdiv(vc.x(), 2), fdiv(vc.y(), 2), fdiv(vc.z(), 2)}].idx.push_back(i);
        }

        m_mixed.clear();
        m_fineCount = 0;
        m_coarseCount = 0;
        for (const auto &[ck, blk] : blocks) {
            double vs = 0.0;
            float wsum = 0.0f, dwsum = 0.0f;
            for (size_t i : blk.idx) {
                vs += double(vox[i].variance);
                wsum += vox[i].weight;
                dwsum += vox[i].weight * vox[i].tsdf;
            }
            const float meanVar = static_cast<float>(vs / double(blk.idx.size()));
            if (blk.idx.size() >= m_minOcc && meanVar < theta) {
                // Coarse voxel: centre of the 2x2x2 fine block, weight-averaged value.
                const Eigen::Vector3f cc(
                        (ck[0] * 2 + 1) * m_h, (ck[1] * 2 + 1) * m_h, (ck[2] * 2 + 1) * m_h);
                m_mixed.push_back({cc, dwsum / std::max(wsum, 1e-6f), wsum, 2.0f * m_h, 1});
                ++m_coarseCount;
            } else {
                for (size_t i : blk.idx) {
                    m_mixed.push_back({vox[i].center, vox[i].tsdf, vox[i].weight, m_h, 0});
                    ++m_fineCount;
                }
            }
        }
        m_mixedDirty = false;
    }

} // namespace Engine::Spatial

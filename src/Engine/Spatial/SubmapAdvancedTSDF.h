#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalIntegrationQuality.h"
#include "Engine/Spatial/OrientedPointCloud.h"
#include "Engine/Spatial/TiledAdvancedTSDF.h"

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Engine::Spatial {

    // Density-adaptive 2-level detail submap: a base TiledAdvancedTSDF at baseVoxel (ALL points)
    // plus a detail TiledAdvancedTSDF at baseVoxel/2 populated only in dense blocks, for detail
    // recovery where points are dense. Batch 2-pass:
    //   AddDensity(all pts) -> FinalizeDensity() -> Integrate(all frames) -> ExtractPointCloud().
    // A block (cube of blockVoxels base-voxels) is dense if its average points-per-occupied-base-
    // voxel >= k (default 4): if a base voxel is averaging >= k observed points, halving the voxel
    // resolves them; flat/sparse blocks stay coarse. Extraction is precedence dedup: detail is kept
    // whole and base points inside a dense block are dropped (detail wins there).
    class SubmapAdvancedTSDF {
    public:
        void Build(Engine::Core::Context &ctx, float baseVoxel, float truncation,
                   int blockVoxels = 32, float detailPtsPerVoxel = 4.0f,
                   uint32_t tileHashPerTile = 1u << 20, uint32_t maxPointsPerFrame = 1u << 17) {
            m_baseVoxel = baseVoxel;
            m_blockWorld = baseVoxel * float(blockVoxels);
            m_detailK = detailPtsPerVoxel;
            m_finalized = false;
            m_base.Build(ctx, baseVoxel, truncation, tileHashPerTile, maxPointsPerFrame);
            m_detail.Build(ctx, baseVoxel * 0.5f, truncation, tileHashPerTile, maxPointsPerFrame);
            m_count.clear();
            m_occ.clear();
            m_dense.clear();
        }

        void SetIntegrationQuality(const IntegrationQuality &q) {
            m_base.SetIntegrationQuality(q);
            m_detail.SetIntegrationQuality(q);
        }
        void SetPointToPlane(bool on) {
            m_base.SetPointToPlane(on);
            m_detail.SetPointToPlane(on);
        }
        void SetConfidenceWeight(float lambda) {
            m_base.SetConfidenceWeight(lambda);
            m_detail.SetConfidenceWeight(lambda);
        }
        void SetHermitePosition(bool on) {
            m_base.SetHermitePosition(on);
            m_detail.SetHermitePosition(on);
        }

        // Pass 1: accumulate per-block point count + distinct occupied base-voxel keys.
        void AddDensity(const std::vector<Eigen::Vector3f> &pts) {
            for (const Eigen::Vector3f &p : pts) {
                const BlockKey b = blockOf(p);
                ++m_count[b];
                m_occ[b].insert(voxelKey(p));
            }
        }

        // Mark dense blocks (avg pts / occupied base-voxel >= k); free the accumulators.
        void FinalizeDensity() {
            for (const auto &kv : m_count) {
                auto it = m_occ.find(kv.first);
                const std::size_t occ =
                        (it != m_occ.end() && !it->second.empty()) ? it->second.size() : 1;
                if (float(kv.second) / float(occ) >= m_detailK) m_dense.insert(kv.first);
            }
            m_count.clear();
            m_occ.clear();
            m_finalized = true;
        }

        // Pass 2: base gets all points; detail gets only points whose block is dense.
        void Integrate(const std::vector<Eigen::Vector3f> &pts,
                       const std::vector<Eigen::Vector3f> &nrm,
                       const Eigen::Vector3f &cam = Eigen::Vector3f::Zero()) {
            m_base.Integrate(pts, nrm, cam);
            if (!m_finalized || m_dense.empty()) return;
            std::vector<Eigen::Vector3f> dp, dn;
            const std::size_t n = std::min(pts.size(), nrm.size());
            dp.reserve(n);
            dn.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
                if (m_dense.count(blockOf(pts[i]))) {
                    dp.push_back(pts[i]);
                    dn.push_back(nrm[i]);
                }
            if (!dp.empty()) m_detail.Integrate(dp, dn, cam);
        }

        // Detail (whole) + base (points inside dense blocks dropped) -> precedence dedup.
        OrientedPointCloud ExtractPointCloud(bool merge = true) const {
            OrientedPointCloud out = m_detail.ExtractPointCloud(merge);
            const OrientedPointCloud base = m_base.ExtractPointCloud(merge);
            const std::size_t m = std::min(base.points.size(), base.normals.size());
            for (std::size_t i = 0; i < m; ++i) {
                if (m_dense.count(blockOf(base.points[i]))) continue; // detail covers this block
                out.points.push_back(base.points[i]);
                out.normals.push_back(base.normals[i]);
            }
            return out;
        }

        uint32_t DenseBlockCount() const { return static_cast<uint32_t>(m_dense.size()); }
        uint32_t BaseTileCount() const { return m_base.TileCount(); }
        uint32_t DetailTileCount() const { return m_detail.TileCount(); }

        // Aggregate base+detail occupied entries with precedence dedup (detail inside dense blocks,
        // base elsewhere) -- the same rule as ExtractPointCloud, on raw voxel entries. For the
        // voxel_fill_debugger's per-frame readout.
        std::vector<AdvancedEntry> DownloadEntries() const {
            std::vector<AdvancedEntry> out = m_detail.DownloadEntries();
            for (const AdvancedEntry &e : m_base.DownloadEntries())
                if (!m_dense.count(blockOf(e.center))) out.push_back(e);
            return out;
        }

        // Drop all tiles in both levels (e.g. to replay integration on scrub-back). The finalized
        // dense-block set is preserved, so re-integration refills base+detail consistently.
        void Reset() {
            m_base.Reset();
            m_detail.Reset();
        }

        // World-space AABB of each dense block -- the region where the detail submap is active.
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> DenseBlockBoxes() const {
            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> out;
            out.reserve(m_dense.size());
            for (const BlockKey &b : m_dense) {
                const Eigen::Vector3f mn(float(b.x) * m_blockWorld, float(b.y) * m_blockWorld,
                                         float(b.z) * m_blockWorld);
                out.emplace_back(mn, mn + Eigen::Vector3f::Constant(m_blockWorld));
            }
            return out;
        }

        // Base-level tile core boxes (the coarse 512^3 windows).
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> BaseCoreBoxes() const {
            return m_base.CoreBoxes();
        }

    private:
        struct BlockKey {
            int x, y, z;
            bool operator==(const BlockKey &o) const { return x == o.x && y == o.y && z == o.z; }
        };
        struct BlockKeyHash {
            std::size_t operator()(const BlockKey &k) const {
                std::size_t h = std::hash<int>()(k.x);
                h ^= std::hash<int>()(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h ^= std::hash<int>()(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                return h;
            }
        };

        BlockKey blockOf(const Eigen::Vector3f &p) const {
            return BlockKey{static_cast<int>(std::floor(p.x() / m_blockWorld)),
                            static_cast<int>(std::floor(p.y() / m_blockWorld)),
                            static_cast<int>(std::floor(p.z() / m_blockWorld))};
        }
        int64_t voxelKey(const Eigen::Vector3f &p) const {
            const int64_t vx = static_cast<int64_t>(std::floor(p.x() / m_baseVoxel)) & 0x1FFFFF;
            const int64_t vy = static_cast<int64_t>(std::floor(p.y() / m_baseVoxel)) & 0x1FFFFF;
            const int64_t vz = static_cast<int64_t>(std::floor(p.z() / m_baseVoxel)) & 0x1FFFFF;
            return vx | (vy << 21) | (vz << 42);
        }

        float m_baseVoxel = 0.01f;
        float m_blockWorld = 0.32f;
        float m_detailK = 4.0f;
        bool m_finalized = false;
        TiledAdvancedTSDF m_base;
        TiledAdvancedTSDF m_detail;
        std::unordered_map<BlockKey, uint32_t, BlockKeyHash> m_count;
        std::unordered_map<BlockKey, std::unordered_set<int64_t>, BlockKeyHash> m_occ;
        std::unordered_set<BlockKey, BlockKeyHash> m_dense;
    };

} // namespace Engine::Spatial

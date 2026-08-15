#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/OrientedPointCloud.h"
#include "TSDF/Backends/DirectionalIntegrationQuality.h"
#include "TSDF/Backends/TiledAdvancedTSDF.h"

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TSDF {

    class SubmapAdvancedTSDF {
    public:
        void Build(Engine::Core::Context &ctx,
                   float baseVoxel,
                   float truncation,
                   int blockVoxels = 32,
                   float detailPtsPerVoxel = 4.0f,
                   uint32_t tileHashPerTile = 1u << 20,
                   uint32_t maxPointsPerFrame = 1u << 17,
                   float detailTruncVoxels = 3.0f,
                   const HashStrategy &hash = LinearProbeStrategy()) {
            m_ctx = &ctx;
            m_baseVoxel = baseVoxel;
            m_blockWorld = baseVoxel * float(blockVoxels);
            m_detailK = detailPtsPerVoxel;
            m_base.Build(ctx, baseVoxel, truncation, tileHashPerTile, maxPointsPerFrame, hash);
            const float detailVoxel = baseVoxel * 0.5f;
            const float detailTrunc = std::min(truncation, detailVoxel * detailTruncVoxels);
            m_detail.Build(ctx, detailVoxel, detailTrunc, tileHashPerTile, maxPointsPerFrame, hash);
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

        void SetDownsample(bool on) { m_downsample = on; }

        void SetCurrentFrame(int frame) {
            m_base.SetCurrentFrame(frame);
            m_detail.SetCurrentFrame(frame);
        }

        void Integrate(const std::vector<Eigen::Vector3f> &pts,
                       const std::vector<Eigen::Vector3f> &nrm,
                       const Eigen::Vector3f &cam = Eigen::Vector3f::Zero()) {

            // online: learn dense blocks from this frame BEFORE splitting the cloud
            updateDensity(pts);
            std::vector<Eigen::Vector3f> dsP, dsN;

            // 1. Downsample points
            if (m_downsample) {
                voxelDownsample(pts, nrm, m_baseVoxel * 0.5f, dsP, dsN);
            }
            const std::vector<Eigen::Vector3f> &p = m_downsample ? dsP : pts;
            const std::vector<Eigen::Vector3f> &n = m_downsample ? dsN : nrm;
            if (std::min(p.size(), n.size()) == 0) return;
            Engine::Compute::CommandBatch batch(*m_ctx);
            std::vector<Eigen::Vector3f> basePts, baseNrm, detailPts, detailNrm;

            // 2. Split the cloud per level (base = non-dense + seam blocks, detail = dense blocks)
            if (!SplitPointDenseOrDetail(
                        p,
                        n,
                        basePts,
                        baseNrm,
                        detailPts,
                        detailNrm)) {
                // no detail level -> whole cloud to base
                m_base.Integrate(p, n, cam, batch);
            } else {
                if (!basePts.empty()) m_base.Integrate(basePts, baseNrm, cam, batch);
                if (!detailPts.empty()) m_detail.Integrate(detailPts, detailNrm, cam, batch);
            }
            batch.Submit(); // one submit for base + detail
        }

        void IntegrateGPU(const std::vector<Eigen::Vector3f> &pts,
                          const std::vector<Eigen::Vector3f> &nrm,
                          const Eigen::Vector3f &cam = Eigen::Vector3f::Zero()) {
            updateDensity(pts); // online: learn dense blocks from this frame BEFORE splitting the cloud
            std::vector<Eigen::Vector3f> dsP, dsN;
            if (m_downsample) voxelDownsample(pts, nrm, m_baseVoxel * 0.5f, dsP, dsN);
            const std::vector<Eigen::Vector3f> &p = m_downsample ? dsP : pts;
            const std::vector<Eigen::Vector3f> &n = m_downsample ? dsN : nrm;
            if (std::min(p.size(), n.size()) == 0) return;
            Engine::Compute::CommandBatch batch(*m_ctx);
            std::vector<Eigen::Vector3f> basePts, baseNrm, detailPts, detailNrm;
            if (!SplitPointDenseOrDetail(p, n, basePts, baseNrm, detailPts, detailNrm)) {
                m_base.IntegrateGPU(p, n, cam, batch); // no detail level -> whole cloud to base
            } else {
                if (!basePts.empty()) m_base.IntegrateGPU(basePts, baseNrm, cam, batch);
                if (!detailPts.empty()) m_detail.IntegrateGPU(detailPts, detailNrm, cam, batch);
            }
            batch.Submit(); // one submit for base + detail
        }

        void PreWarm() {
            const std::vector<Eigen::Vector3f> p{Eigen::Vector3f::Zero()};
            const std::vector<Eigen::Vector3f> q{Eigen::Vector3f::UnitZ()};
            IntegrateGPU(p, q, Eigen::Vector3f::UnitZ());
            std::vector<AdvancedEntry> scratch;
            DownloadEntries(scratch);
            Reset();
        }

        // Detail (whole) + base (points inside dense blocks dropped) -> precedence dedup.
        Engine::Core::OrientedPointCloud ExtractPointCloud(bool merge = true) const {
            Engine::Core::OrientedPointCloud out = m_detail.ExtractPointCloud(merge);
            const Engine::Core::OrientedPointCloud base = m_base.ExtractPointCloud(merge);
            const std::size_t m = std::min(base.points.size(), base.normals.size());
            for (std::size_t i = 0; i < m; ++i) {
                if (m_dense.count(blockOf(base.points[i]))) continue; // detail covers this block
                out.points.push_back(base.points[i]);
                out.normals.push_back(base.normals[i]);
            }
            return out;
        }

        uint32_t DenseBlockCount() const { return static_cast<uint32_t>(m_dense.size()); }

        // Base + detail combined. Both levels are real storage, so a memory comparison must see
        // the sum rather than either level alone.
        uint32_t FilledCount() const { return m_base.FilledCount() + m_detail.FilledCount(); }

        uint64_t SlotCapacity() const { return m_base.SlotCapacity() + m_detail.SlotCapacity(); }

        uint32_t BaseTileCount() const { return m_base.TileCount(); }
        uint32_t DetailTileCount() const { return m_detail.TileCount(); }

        // 64-bit like the tiled levels they sum: both levels are full tile hierarchies, so the
        // totals are the two largest tile counts in the system added together.
        uint64_t InsertFailureCount() const {
            return m_base.InsertFailureCount() + m_detail.InsertFailureCount();
        }

        uint64_t GrowCount() const { return m_base.GrowCount() + m_detail.GrowCount(); }

        void DownloadEntries(std::vector<AdvancedEntry> &out) const {
            if (m_dense.empty()) {
                m_base.DownloadEntries(out);
                return;
            }
            m_detail.DownloadEntries(out);         // detail into out (reuses out's capacity)
            m_base.DownloadEntries(m_baseScratch); // base into a reused scratch (warm across frames)
            appendBaseOutsideDenseBlocks(out);     // base entries where detail does NOT already win
        }

        std::vector<AdvancedEntry> DownloadEntries() const {
            std::vector<AdvancedEntry> out;
            DownloadEntries(out);
            return out;
        }

        void Reset() {
            m_base.Reset();
            m_detail.Reset();
            m_count.clear();
            m_occ.clear();
            m_dense.clear();
        }

        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> DenseBlockBoxes() const {
            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> out;
            out.reserve(m_dense.size());
            for (const BlockKey &b: m_dense) {
                const Eigen::Vector3f mn(float(b.x) * m_blockWorld, float(b.y) * m_blockWorld,
                                         float(b.z) * m_blockWorld);
                out.emplace_back(mn, mn + Eigen::Vector3f::Constant(m_blockWorld));
            }
            return out;
        }

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

        static constexpr std::size_t kParallelDedupMin = 1u << 15; // below this, filter serially
        static constexpr unsigned kMaxDedupThreads = 8u;           // cap workers for the base dedup

        void appendBaseOutsideDenseBlocks(std::vector<AdvancedEntry> &out) const {
            const std::size_t n = m_baseScratch.size();
            if (n == 0) return;

            auto keepInto = [this](std::size_t lo, std::size_t hi,
                                   std::vector<AdvancedEntry> &into) {
                for (std::size_t i = lo; i < hi; ++i)
                    if (!m_dense.count(blockOf(m_baseScratch[i].center))) into.push_back(m_baseScratch[i]);
            };

            const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
            const unsigned nThreads = (n < kParallelDedupMin) ? 1u : std::min(hw, kMaxDedupThreads);
            if (nThreads == 1u) {
                out.reserve(out.size() + n);
                keepInto(0, n, out);
                return;
            }

            std::vector<std::vector<AdvancedEntry>> local(nThreads);
            std::vector<std::thread> workers;
            workers.reserve(nThreads - 1);
            const std::size_t chunk = (n + nThreads - 1) / nThreads;
            for (unsigned w = 1; w < nThreads; ++w) {
                const std::size_t lo = std::min(n, w * chunk), hi = std::min(n, lo + chunk);
                local[w].reserve(hi - lo);
                workers.emplace_back([&, lo, hi, w] { keepInto(lo, hi, local[w]); });
            }
            local[0].reserve(std::min(n, chunk));
            keepInto(0, std::min(n, chunk), local[0]); // this thread filters the first chunk
            for (auto &t: workers) t.join();

            std::size_t kept = 0;
            for (const auto &l: local) kept += l.size();
            out.reserve(out.size() + kept);
            for (const auto &l: local) out.insert(out.end(), l.begin(), l.end());
        }

        // Online density: accumulate this frame's per-block point count + distinct occupied base-voxel
        // count, and flip a block to dense the moment its running (avg points / occupied voxel) crosses
        // detailK. Dense is MONOTONIC (never un-flips), so a block is decided once and only detail covers
        // it thereafter. A flipped block stops being tracked (count/occ erased) -> memory tracks only the
        // still-undecided blocks, bounded for arbitrarily long streams. No pre-scan / future frames.
        void updateDensity(const std::vector<Eigen::Vector3f> &pts) {
            for (const Eigen::Vector3f &p: pts) {
                const BlockKey b = blockOf(p);
                if (m_dense.count(b)) continue; // already decided -> not tracked
                auto &occ = m_occ[b];
                occ.insert(voxelKey(p));
                const uint32_t c = ++m_count[b];
                if (float(c) / float(occ.size()) >= m_detailK) {
                    m_dense.insert(b); // enough overlap -> detail from now on
                    m_count.erase(b);
                    m_occ.erase(b);
                }
            }
        }

        // Split the frame per level using the CURRENTLY-known dense set (updateDensity ran first this
        // frame). Detail gets dense-block points; base gets the rest. A block that has flipped dense is
        // dropped from base entirely from that frame on (its earlier base voxels are discarded on
        // download, where detail wins) -- the online form of interior-dense-skip. Returns false when no
        // block is dense yet, so the caller integrates the whole cloud into base (the early-stream ramp).
        bool SplitPointDenseOrDetail(const std::vector<Eigen::Vector3f> &originalPoint,
                                     const std::vector<Eigen::Vector3f> &origialNormal,
                                     std::vector<Eigen::Vector3f> &basePoints,
                                     std::vector<Eigen::Vector3f> &baseNormals,
                                     std::vector<Eigen::Vector3f> &detailVoxelPoint,
                                     std::vector<Eigen::Vector3f> &detailVoxelNormal) const {
            if (m_dense.empty()) return false;
            const std::size_t n = std::min(originalPoint.size(), origialNormal.size());
            basePoints.reserve(n);
            baseNormals.reserve(n);
            detailVoxelPoint.reserve(n);
            detailVoxelNormal.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                if (m_dense.count(blockOf(originalPoint[i]))) {
                    detailVoxelPoint.push_back(originalPoint[i]); // dense -> detail only (base stops)
                    detailVoxelNormal.push_back(origialNormal[i]);
                } else {
                    basePoints.push_back(originalPoint[i]); // non-dense -> base
                    baseNormals.push_back(origialNormal[i]);
                }
            }
            return true;
        }
        int64_t voxelKey(const Eigen::Vector3f &p) const {
            const int64_t vx = static_cast<int64_t>(std::floor(p.x() / m_baseVoxel)) & 0x1FFFFF;
            const int64_t vy = static_cast<int64_t>(std::floor(p.y() / m_baseVoxel)) & 0x1FFFFF;
            const int64_t vz = static_cast<int64_t>(std::floor(p.z() / m_baseVoxel)) & 0x1FFFFF;
            return vx | (vy << 21) | (vz << 42);
        }

        static void voxelDownsample(const std::vector<Eigen::Vector3f> &pts,
                                    const std::vector<Eigen::Vector3f> &nrm, float voxel,
                                    std::vector<Eigen::Vector3f> &outP,
                                    std::vector<Eigen::Vector3f> &outN) {
            const float inv = 1.0f / voxel;
            const std::size_t n = std::min(pts.size(), nrm.size());
            std::unordered_set<int64_t> seen;
            seen.reserve(n);
            outP.reserve(n);
            outN.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                const int64_t vx = static_cast<int64_t>(std::floor(pts[i].x() * inv)) & 0x1FFFFF;
                const int64_t vy = static_cast<int64_t>(std::floor(pts[i].y() * inv)) & 0x1FFFFF;
                const int64_t vz = static_cast<int64_t>(std::floor(pts[i].z() * inv)) & 0x1FFFFF;
                if (seen.insert(vx | (vy << 21) | (vz << 42)).second) {
                    outP.push_back(pts[i]);
                    outN.push_back(nrm[i]);
                }
            }
        }

        Engine::Core::Context *m_ctx = nullptr;
        float m_baseVoxel = 0.01f;
        float m_blockWorld = 0.32f;
        float m_detailK = 4.0f;
        // off by default; the pipeline/debugger opt in via SetDownsample
        bool m_downsample = false;
        // reused base readback (dense-path download)
        mutable std::vector<AdvancedEntry> m_baseScratch;
        TiledAdvancedTSDF m_base;
        TiledAdvancedTSDF m_detail;
        std::unordered_map<BlockKey, uint32_t, BlockKeyHash> m_count;
        std::unordered_map<BlockKey, std::unordered_set<int64_t>, BlockKeyHash> m_occ;
        std::unordered_set<BlockKey, BlockKeyHash> m_dense; // blocks decided dense (monotonic, online)
    };

} // namespace TSDF

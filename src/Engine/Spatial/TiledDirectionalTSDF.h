#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalIntegrationQuality.h"
#include "Engine/Spatial/OrientedPointCloud.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace Engine::Spatial {

    // CPU coordinator that lifts a single-window directional-TSDF Backend's 512^3 limit by TILING
    // space: it partitions the world voxel grid into fixed cubic tiles, each backed by one Backend
    // instance (a 512^3 window). Only touched tiles are ever allocated (lazy, on first
    // integration), so memory is proportional to the scanned surface, and the scene can grow
    // indefinitely within VRAM.
    //
    // GEOMETRY (fixed): core side C = kCore = 448 voxels; per-tile 512^3 window = core + ghost
    // margin G on every side (G = ceil(trunc/voxel)+1, covers the full truncation band); requires
    // C + 2G <= 512. Global voxel origin O = 0; tile of voxel v is floorDiv(v - O, C) per axis.
    //
    // GHOST ROUTING: a point near a tile boundary is integrated into BOTH the owning tile and the
    // adjacent tile(s), so every tile's core carries the full truncation band (no seam).
    // CORE-ONLY EXTRACTION: each tile emits only points whose voxel lies in its own core, so the
    // ghost overlap never produces duplicates across tiles.
    //
    // Backend contract: Build(ctx, voxel, trunc, hashCap, maxPoints, windowMinCorner),
    // SetIntegrationQuality, SetPointToPlane, Integrate(points, normals, cameraPos),
    // ExtractPointCloud(maxCandidates, merge) -> OrientedPointCloud, FilledCount. Backend-specific
    // configuration (e.g. AdvancedTSDF's A1/A2 setters) is forwarded via m_configureHook.
    template <class Backend>
    class TiledDirectionalTSDF {
    public:
        TiledDirectionalTSDF() = default;
        virtual ~TiledDirectionalTSDF() = default;

        // Store params + ctx. Tiles are created lazily on first integration. hashCapacityPerTile is
        // the per-tile Backend hash size; maxPointsPerFrame the per-tile per-Integrate point cap.
        void Build(Engine::Core::Context &ctx,
                   float voxelSize,
                   float truncation,
                   uint32_t hashCapacityPerTile = 1u << 22,
                   uint32_t maxPointsPerFrame = 1u << 17) {
            m_ctx = &ctx;
            m_voxelSize = voxelSize;
            m_truncation = truncation;
            m_hashCapacityPerTile = hashCapacityPerTile;
            m_maxPointsPerFrame = maxPointsPerFrame;
            m_origin = Eigen::Vector3i::Zero();
            // G must cover the full truncation band (band radius = ceil(trunc/voxel) voxels), +1.
            m_ghost = static_cast<int>(std::ceil(truncation / voxelSize)) + 1;
            if (kCore + 2 * m_ghost > 512) {
                throw std::runtime_error(
                        "TiledDirectionalTSDF: C + 2G exceeds the 512^3 window "
                        "(truncation too large for voxelSize)");
            }
            m_tiles.clear();
        }

        // Applied to each tile's Backend on creation.
        void SetIntegrationQuality(const IntegrationQuality &q) { m_quality = q; }

        // Integrate SDF form, forwarded to every tile (default true = point-to-plane).
        void SetPointToPlane(bool on) { m_pointToPlane = on; }

        // Route each point to its owning tile plus any adjacent tile whose ghost band it falls in,
        // then integrate the per-tile sublists (lazily creating+building touched tiles).
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero()) {
            const size_t n = std::min(points.size(), normals.size());
            if (n == 0) return;

            struct SubList {
                std::vector<Eigen::Vector3f> pts, nrm;
            };
            std::unordered_map<TileKey, SubList, TileKeyHash> routed;

            for (size_t i = 0; i < n; ++i) {
                const Eigen::Vector3f &p = points[i];
                const Eigen::Vector3i v(static_cast<int>(std::floor(p.x() / m_voxelSize)),
                                        static_cast<int>(std::floor(p.y() / m_voxelSize)),
                                        static_cast<int>(std::floor(p.z() / m_voxelSize)));
                const Eigen::Vector3i home = tileOf(v);
                const Eigen::Vector3i local = (v - m_origin) - home * kCore; // 0..C-1 per axis

                int offs[3][2];
                int nOff[3];
                for (int a = 0; a < 3; ++a) {
                    offs[a][0] = 0;
                    nOff[a] = 1;
                    if (local[a] < m_ghost) {
                        offs[a][1] = -1;
                        nOff[a] = 2;
                    } else if (local[a] >= kCore - m_ghost) {
                        offs[a][1] = +1;
                        nOff[a] = 2;
                    }
                }

                for (int ix = 0; ix < nOff[0]; ++ix)
                    for (int iy = 0; iy < nOff[1]; ++iy)
                        for (int iz = 0; iz < nOff[2]; ++iz) {
                            const TileKey key{home.x() + offs[0][ix], home.y() + offs[1][iy],
                                              home.z() + offs[2][iz]};
                            SubList &s = routed[key];
                            s.pts.push_back(p);
                            s.nrm.push_back(normals[i]);
                        }
            }

            for (auto &kv : routed) {
                Backend *tile = tileFor(kv.first);
                tile->Integrate(kv.second.pts, kv.second.nrm, cameraPos);
            }
        }

        // Extract each tile, KEEP ONLY points whose voxel lies in that tile's core (drop ghost
        // duplicates), and concatenate. merge is forwarded to Backend::ExtractPointCloud.
        OrientedPointCloud ExtractPointCloud(bool merge = true) const {
            OrientedPointCloud out;
            for (const auto &kv : m_tiles) {
                const TileKey &key = kv.first;
                const Eigen::Vector3i tile(key.x, key.y, key.z);
                const Eigen::Vector3i coreMin = m_origin + tile * kCore;
                const Eigen::Vector3i coreMax = coreMin + Eigen::Vector3i::Constant(kCore);

                const OrientedPointCloud tileCloud = kv.second->ExtractPointCloud(1u << 21, merge);
                const size_t m = std::min(tileCloud.points.size(), tileCloud.normals.size());
                for (size_t i = 0; i < m; ++i) {
                    const Eigen::Vector3f &p = tileCloud.points[i];
                    const Eigen::Vector3i v(static_cast<int>(std::floor(p.x() / m_voxelSize)),
                                            static_cast<int>(std::floor(p.y() / m_voxelSize)),
                                            static_cast<int>(std::floor(p.z() / m_voxelSize)));
                    if (v.x() >= coreMin.x() && v.x() < coreMax.x() && v.y() >= coreMin.y() &&
                        v.y() < coreMax.y() && v.z() >= coreMin.z() && v.z() < coreMax.z()) {
                        out.points.push_back(p);
                        out.normals.push_back(tileCloud.normals[i]);
                    }
                }
            }
            return out;
        }

        // Sum of tiles' FilledCount. NOTE: ghost overlap makes this slightly MORE than the true
        // occupied-voxel set -- that surplus is the honest tiling overhead.
        uint32_t FilledCount() const {
            uint32_t total = 0;
            for (const auto &kv : m_tiles) total += kv.second->FilledCount();
            return total;
        }

        uint32_t TileCount() const { return static_cast<uint32_t>(m_tiles.size()); }

        // Fixed core side C and the derived ghost margin G (0 before Build). Diagnostics.
        int CoreVoxels() const { return kCore; }
        int GhostVoxels() const { return m_ghost; }

        // floor(a / b) with rounding toward -inf for negatives (callers pass b = C > 0).
        static int floorDiv(int a, int b) {
            int q = a / b;
            int r = a % b;
            if (r != 0 && ((r < 0) != (b < 0))) --q;
            return q;
        }

    protected:
        // Extension point: applied to each tile right after the common config (quality + p2p), on
        // creation. Subclasses set this to forward Backend-specific settings (e.g. A1/A2).
        std::function<void(Backend &)> m_configureHook;

    private:
        static constexpr int kCore = 448; // C: voxels per tile core axis

        struct TileKey {
            int x, y, z;
            bool operator==(const TileKey &o) const { return x == o.x && y == o.y && z == o.z; }
        };
        struct TileKeyHash {
            std::size_t operator()(const TileKey &k) const {
                std::size_t h = std::hash<int>()(k.x);
                h ^= std::hash<int>()(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h ^= std::hash<int>()(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                return h;
            }
        };

        Eigen::Vector3i tileOf(const Eigen::Vector3i &v) const {
            return Eigen::Vector3i(floorDiv(v.x() - m_origin.x(), kCore),
                                   floorDiv(v.y() - m_origin.y(), kCore),
                                   floorDiv(v.z() - m_origin.z(), kCore));
        }

        Backend *tileFor(const TileKey &key) {
            auto it = m_tiles.find(key);
            if (it != m_tiles.end()) return it->second.get();

            const Eigen::Vector3i tile(key.x, key.y, key.z);
            const Eigen::Vector3i originVoxel =
                    m_origin + tile * kCore - Eigen::Vector3i::Constant(m_ghost);
            const Eigen::Vector3f windowMinCorner = originVoxel.cast<float>() * m_voxelSize;

            auto tsdf = std::make_unique<Backend>();
            tsdf->Build(*m_ctx, m_voxelSize, m_truncation, m_hashCapacityPerTile,
                        m_maxPointsPerFrame, windowMinCorner);
            tsdf->SetIntegrationQuality(m_quality);
            tsdf->SetPointToPlane(m_pointToPlane);
            if (m_configureHook) m_configureHook(*tsdf);
            Backend *ptr = tsdf.get();
            m_tiles.emplace(key, std::move(tsdf));
            return ptr;
        }

        Engine::Core::Context *m_ctx = nullptr;
        float m_voxelSize = 0.001f;
        float m_truncation = 0.003f;
        uint32_t m_hashCapacityPerTile = 1u << 22;
        uint32_t m_maxPointsPerFrame = 1u << 17;
        int m_ghost = 0;                                    // G (set in Build)
        Eigen::Vector3i m_origin = Eigen::Vector3i::Zero(); // O
        IntegrationQuality m_quality;
        bool m_pointToPlane = true;

        std::unordered_map<TileKey, std::unique_ptr<Backend>, TileKeyHash> m_tiles;
    };

} // namespace Engine::Spatial

#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Spatial/CompactDirectionalTSDF.h"
#include "Engine/Spatial/DirectionalIntegrationQuality.h"
#include "Engine/Spatial/OrientedPointCloud.h"

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Engine::Spatial {

    // CPU wrapper that lifts CompactDirectionalTSDF's single 512^3-voxel window limit by TILING
    // space: it partitions the world voxel grid into fixed-size cubic tiles, each backed by one
    // CompactDirectionalTSDF instance (a 512^3 window). Scenes larger than 512 voxels/axis (e.g.
    // a chair at 1mm voxels ~= 827 voxels/axis) that no single CompactDirectionalTSDF can hold are
    // reconstructed here at low, tile-count-proportional memory -- only touched tiles are ever
    // allocated (lazy, on first integration).
    //
    // GEOMETRY (fixed):
    //   - tile CORE side C = kCore = 448 voxels; the "owned" region of a tile.
    //   - per-tile 512^3 window = core + a ghost margin G on every side (G = ceil(trunc/voxel)+1,
    //     enough to cover the full truncation band). Requires C + 2G <= 512.
    //   - global voxel origin O = ivec3::Zero(); tile of a voxel v is floorDiv(v - O, C) per axis.
    //
    // GHOST ROUTING (band correctness): a point near a tile boundary is integrated into BOTH the
    // owning tile and the adjacent tile(s), so every tile's core carries the full truncation band
    // (no seam). CORE-ONLY EXTRACTION (dedup): each tile emits only points whose voxel lies in its
    // own core, so the ghost overlap never produces duplicates across tiles.
    class TiledCompactDirectionalTSDF {
    public:
        TiledCompactDirectionalTSDF() = default;

        // Store params + ctx. Tiles are created lazily on first integration. hashCapacityPerTile
        // is the per-tile CompactDirectionalTSDF hash size; maxPointsPerFrame the per-tile per-
        // Integrate point cap.
        void Build(Engine::Core::Context &ctx,
                   float voxelSize,
                   float truncation,
                   uint32_t hashCapacityPerTile = 1u << 22,
                   uint32_t maxPointsPerFrame = 1u << 17);

        // Applied to each tile's CompactDirectionalTSDF on creation.
        void SetIntegrationQuality(const IntegrationQuality &q) { m_quality = q; }

        // Integrate SDF form, forwarded to every tile (default true = point-to-plane).
        void SetPointToPlane(bool on) { m_pointToPlane = on; }

        // Route each point to its owning tile plus any adjacent tile whose ghost band it falls in,
        // then integrate the per-tile sublists (lazily creating+building touched tiles).
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero());

        // Extract each tile, KEEP ONLY points whose voxel lies in that tile's core (drop ghost
        // duplicates), and concatenate. merge is forwarded to CompactDirectionalTSDF::ExtractPointCloud.
        OrientedPointCloud ExtractPointCloud(bool merge = true) const;

        // Sum of tiles' FilledCount. NOTE: ghost overlap means this is slightly MORE than the true
        // occupied-voxel set -- that surplus is the honest tiling overhead.
        uint32_t FilledCount() const;

        uint32_t TileCount() const { return static_cast<uint32_t>(m_tiles.size()); }

        // Fixed core side C and the derived ghost margin G (0 before Build). Diagnostics.
        int CoreVoxels() const { return kCore; }
        int GhostVoxels() const { return m_ghost; }

        // floor(a / b) with correct rounding toward -inf for negatives (callers pass b = C > 0).
        static int floorDiv(int a, int b);

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

        Eigen::Vector3i tileOf(const Eigen::Vector3i &v) const;
        CompactDirectionalTSDF *tileFor(const TileKey &key); // lazy create + Build + SetQuality

        Engine::Core::Context *m_ctx = nullptr;
        float m_voxelSize = 0.001f;
        float m_truncation = 0.003f;
        uint32_t m_hashCapacityPerTile = 1u << 22;
        uint32_t m_maxPointsPerFrame = 1u << 17;
        int m_ghost = 0;                                    // G (set in Build)
        Eigen::Vector3i m_origin = Eigen::Vector3i::Zero(); // O
        IntegrationQuality m_quality;
        bool m_pointToPlane = true; // forwarded to each tile (matches CompactDirectionalTSDF default)

        std::unordered_map<TileKey, std::unique_ptr<CompactDirectionalTSDF>, TileKeyHash> m_tiles;
    };

} // namespace Engine::Spatial

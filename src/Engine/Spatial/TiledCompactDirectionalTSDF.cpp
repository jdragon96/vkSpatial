#include "Engine/Spatial/TiledCompactDirectionalTSDF.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace Engine::Spatial {

    int TiledCompactDirectionalTSDF::floorDiv(int a, int b) {
        int q = a / b;
        int r = a % b;
        if (r != 0 && ((r < 0) != (b < 0))) --q;
        return q;
    }

    Eigen::Vector3i TiledCompactDirectionalTSDF::tileOf(const Eigen::Vector3i &v) const {
        return Eigen::Vector3i(floorDiv(v.x() - m_origin.x(), kCore),
                               floorDiv(v.y() - m_origin.y(), kCore),
                               floorDiv(v.z() - m_origin.z(), kCore));
    }

    void TiledCompactDirectionalTSDF::Build(Engine::Core::Context &ctx,
                                            float voxelSize,
                                            float truncation,
                                            uint32_t hashCapacityPerTile,
                                            uint32_t maxPointsPerFrame) {
        m_ctx = &ctx;
        m_voxelSize = voxelSize;
        m_truncation = truncation;
        m_hashCapacityPerTile = hashCapacityPerTile;
        m_maxPointsPerFrame = maxPointsPerFrame;
        m_origin = Eigen::Vector3i::Zero();
        // G must cover the full truncation band (band radius = ceil(trunc/voxel) voxels), plus one.
        m_ghost = static_cast<int>(std::ceil(truncation / voxelSize)) + 1;
        assert(kCore + 2 * m_ghost <= 512 && "ghost margin too large: C + 2G must fit the 512 window");
        if (kCore + 2 * m_ghost > 512) {
            throw std::runtime_error(
                    "TiledCompactDirectionalTSDF: C + 2G exceeds the 512^3 window "
                    "(truncation too large for voxelSize)");
        }
        m_tiles.clear();
    }

    CompactDirectionalTSDF *TiledCompactDirectionalTSDF::tileFor(const TileKey &key) {
        auto it = m_tiles.find(key);
        if (it != m_tiles.end()) return it->second.get();

        const Eigen::Vector3i tile(key.x, key.y, key.z);
        // 512 window covers [tile*C - G, tile*C - G + 512) voxels: core [tile*C, tile*C+C) plus a
        // G-voxel ghost band on each side (with slack of 512 - C - 2G on the high side).
        const Eigen::Vector3i originVoxel =
                m_origin + tile * kCore - Eigen::Vector3i::Constant(m_ghost);
        const Eigen::Vector3f windowMinCorner = originVoxel.cast<float>() * m_voxelSize;

        auto tsdf = std::make_unique<CompactDirectionalTSDF>();
        tsdf->Build(*m_ctx, m_voxelSize, m_truncation, m_hashCapacityPerTile, m_maxPointsPerFrame,
                    windowMinCorner);
        tsdf->SetIntegrationQuality(m_quality);
        tsdf->SetPointToPlane(m_pointToPlane);
        CompactDirectionalTSDF *ptr = tsdf.get();
        m_tiles.emplace(key, std::move(tsdf));
        return ptr;
    }

    void TiledCompactDirectionalTSDF::Integrate(const std::vector<Eigen::Vector3f> &points,
                                                const std::vector<Eigen::Vector3f> &normals,
                                                const Eigen::Vector3f &cameraPos) {
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

            // Per-axis tile offsets this point contributes to: always the owning tile (0), plus the
            // -1 neighbour if within G of the low edge, or the +1 neighbour if within G of the high
            // edge (a point can't be near both since C >> 2G).
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
            CompactDirectionalTSDF *tile = tileFor(kv.first);
            tile->Integrate(kv.second.pts, kv.second.nrm, cameraPos);
        }
    }

    OrientedPointCloud TiledCompactDirectionalTSDF::ExtractPointCloud(bool merge) const {
        OrientedPointCloud out;
        for (const auto &kv : m_tiles) {
            const TileKey &key = kv.first;
            const Eigen::Vector3i tile(key.x, key.y, key.z);
            const Eigen::Vector3i coreMin = m_origin + tile * kCore;
            const Eigen::Vector3i coreMax = coreMin + Eigen::Vector3i::Constant(kCore); // exclusive

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

    uint32_t TiledCompactDirectionalTSDF::FilledCount() const {
        uint32_t total = 0;
        for (const auto &kv : m_tiles) total += kv.second->FilledCount();
        return total;
    }

} // namespace Engine::Spatial

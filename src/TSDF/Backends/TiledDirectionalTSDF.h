#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "TSDF/Backends/DirectionalIntegrationQuality.h"
#include "Engine/Core/OrientedPointCloud.h"
#include "TSDF/Memory/Hash/HashStrategy.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TSDF {

    template<class T, class = void>
    struct HasSetCurrentFrame : std::false_type {};
    template<class T>
    struct HasSetCurrentFrame<T, std::void_t<decltype(std::declval<T &>().SetCurrentFrame(0))>>
        : std::true_type {};

    // Only AdvancedTSDF's Build takes a trailing HashStrategy; the frozen CompactDirectionalTSDF
    // (also instantiated through this template, via TiledCompactDirectionalTSDF) does not and must
    // not be touched. GetTSDF below picks the right overload with this instead of hard-coding the
    // hash-aware call, which would fail to compile for that instantiation.
    template<class T, class = void>
    struct HasHashStrategyBuild : std::false_type {};
    template<class T>
    struct HasHashStrategyBuild<
            T, std::void_t<decltype(std::declval<T &>().Build(
                       std::declval<Engine::Core::Context &>(), 0.0f, 0.0f, uint32_t(0), uint32_t(0),
                       std::declval<const Eigen::Vector3f &>(), std::declval<const HashStrategy &>()))>>
        : std::true_type {};

    template<class Backend>
    class TiledDirectionalTSDF {
    public:
        TiledDirectionalTSDF() = default;
        virtual ~TiledDirectionalTSDF() = default;

        void Build(Engine::Core::Context &ctx,
                   float voxelSize,
                   float truncation,
                   uint32_t hashCapacityPerTile = 1u << 22,
                   uint32_t maxPointsPerFrame = 1u << 17,
                   const TSDF::HashStrategy &hash = TSDF::LinearProbeStrategy()) {
            m_ctx = &ctx;
            m_voxelSize = voxelSize;
            m_truncation = truncation;
            m_hashCapacityPerTile = hashCapacityPerTile;
            m_maxPointsPerFrame = maxPointsPerFrame;
            m_hash = &hash;
            m_origin = Eigen::Vector3i::Zero();
            m_ghost = static_cast<int>(std::ceil(truncation / voxelSize)) + 1;
            if (kCore + 2 * m_ghost > 512) {
                throw std::runtime_error(
                        "TiledDirectionalTSDF: C + 2G exceeds the 512^3 window "
                        "(truncation too large for voxelSize)");
            }
            m_tiles.clear();
            AllocateReusePointCloudeBuffer(m_maxPointsPerFrame);
        }

        void SetIntegrationQuality(const IntegrationQuality &q) { m_quality = q; }

        void SetPointToPlane(bool on) { m_pointToPlane = on; }

        void SetCurrentFrame(int frame) {
            m_currentFrame = frame;
            if constexpr (HasSetCurrentFrame<Backend>::value)
                for (auto &kv: m_tiles) kv.second->SetCurrentFrame(frame);
        }

        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero()) {
            for (auto &kv: splitPointsToTiles(points, normals)) {
                Backend *tile = GetTSDF(kv.first);
                tile->Integrate(kv.second.pts, kv.second.nrm, cameraPos);
            }
        }

        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos,
                       Engine::Compute::CommandBatch &batch) {
            for (auto &kv: splitPointsToTiles(points, normals)) {
                Backend *tile = GetTSDF(kv.first);
                tile->RecordIntegrate(kv.second.pts, kv.second.nrm, cameraPos, batch);
            }
        }

        // Same per-tile routing as the batched Integrate, but each tile grows its upload buffers
        // to the cloud it was routed instead of clamping to maxPointsPerFrame. Use this wherever
        // dropping points would corrupt a measurement -- the clamping form truncates silently.
        // Like SlotCapacity(), this is instantiated lazily: a Backend without RecordIntegrateGPU
        // only fails if this is actually called on that instantiation.
        void RecordIntegrateGPU(const std::vector<Eigen::Vector3f> &points,
                                const std::vector<Eigen::Vector3f> &normals,
                                const Eigen::Vector3f &cameraPosition,
                                Engine::Compute::CommandBatch &batch) {
            for (auto &kv: splitPointsToTiles(points, normals)) {
                Backend *tile = GetTSDF(kv.first);
                tile->RecordIntegrateGPU(kv.second.pts, kv.second.nrm, cameraPosition, batch);
            }
        }

        void IntegrateGPU(const std::vector<Eigen::Vector3f> &points,
                          const std::vector<Eigen::Vector3f> &normals,
                          const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero()) {
            if (points.empty()) return;
            Engine::Compute::CommandBatch batch(*m_ctx);
            IntegrateGPU(points, normals, cameraPos, batch);
            batch.Submit();
        }

        void IntegrateGPU(const std::vector<Eigen::Vector3f> &points,
                          const std::vector<Eigen::Vector3f> &normals,
                          const Eigen::Vector3f &cameraPos,
                          Engine::Compute::CommandBatch &batch) {
            const std::size_t n = std::min(points.size(), normals.size());
            if (n == 0) return;
            UploadReuseBuffer(points, normals, static_cast<uint32_t>(n));
            for (const TileKey &key: GetTilesAffectedByFrame(points, normals)) {
                Backend *tsdf = GetTSDF(key);
                tsdf->RecordIntegrateShared(
                        *m_reusePointBuffer,
                        *m_reuseNormalBuffer,
                        static_cast<uint32_t>(n),
                        cameraPos,
                        batch);
            }
        }

        Engine::Core::OrientedPointCloud ExtractPointCloud(bool merge = true) const {
            Engine::Core::OrientedPointCloud out;
            for (const auto &kv: m_tiles) {
                const TileKey &key = kv.first;
                const Eigen::Vector3i tile(key.x, key.y, key.z);
                const Eigen::Vector3i coreMin = m_origin + tile * kCore;
                const Eigen::Vector3i coreMax = coreMin + Eigen::Vector3i::Constant(kCore);

                const Engine::Core::OrientedPointCloud tileCloud = kv.second->ExtractPointCloud(1u << 21, merge);
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

        uint32_t FilledCount() const {
            uint32_t total = 0;
            for (const auto &kv: m_tiles) total += kv.second->FilledCount();
            return total;
        }

        // Total slots across every live tile. Mirrors FilledCount(); the two together give the
        // load factor. Instantiated lazily, so a Backend without HashCapacity() only fails if
        // this is actually called on that instantiation.
        uint64_t SlotCapacity() const {
            uint64_t total = 0;
            for (const auto &kv: m_tiles) total += kv.second->HashCapacity();
            return total;
        }

        uint32_t TileCount() const { return static_cast<uint32_t>(m_tiles.size()); }

        // Summed across live tiles, mirroring FilledCount()/SlotCapacity(). Accumulated in 64 bits
        // (each tile's own counter is 32-bit): a fine-voxel scan runs thousands of tiles, so a
        // 32-bit total could wrap and report a healthy 0 for a volume that dropped billions of
        // observations. VolumeStats stores both as 64-bit for the same reason.
        uint64_t InsertFailureCount() const {
            uint64_t total = 0;
            for (const auto &kv: m_tiles) total += kv.second->InsertFailureCount();
            return total;
        }

        uint64_t GrowCount() const {
            uint64_t total = 0;
            for (const auto &kv: m_tiles) total += kv.second->GrowCount();
            return total;
        }

        void Reset() { m_tiles.clear(); }

        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> CoreBoxes() const {
            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> out;
            out.reserve(m_tiles.size());
            for (const auto &kv: m_tiles) {
                const Eigen::Vector3i tile(kv.first.x, kv.first.y, kv.first.z);
                const Eigen::Vector3i coreMin = m_origin + tile * kCore;
                const Eigen::Vector3i coreMax = coreMin + Eigen::Vector3i::Constant(kCore);
                out.emplace_back(coreMin.cast<float>() * m_voxelSize,
                                 coreMax.cast<float>() * m_voxelSize);
            }
            return out;
        }

        int CoreVoxels() const { return kCore; }
        int GhostVoxels() const { return m_ghost; }

        static int floorDiv(int a, int b) {
            int q = a / b;
            int r = a % b;
            if (r != 0 && ((r < 0) != (b < 0))) --q;
            return q;
        }

    protected:
        std::function<void(Backend &)> m_configureHook;

        float voxelSize() const { return m_voxelSize; }
        Engine::Core::Context *contextPtr() const { return m_ctx; }            // for subclass batched readback
        uint32_t hashCapacityPerTile() const { return m_hashCapacityPerTile; } // sizes shared scratch

        template<class Fn>
        void forEachTileCore(Fn &&fn) const {
            for (const auto &kv: m_tiles) {
                const Eigen::Vector3i tile(kv.first.x, kv.first.y, kv.first.z);
                const Eigen::Vector3i coreMin = m_origin + tile * kCore;
                const Eigen::Vector3i coreMax = coreMin + Eigen::Vector3i::Constant(kCore);
                fn(*kv.second, coreMin, coreMax);
            }
        }

    private:
        static constexpr int kCore = 448;                          // C: voxels per tile core axis
        static constexpr std::size_t kParallelRouteMin = 1u << 14; // below this, bin tiles serially
        static constexpr unsigned kMaxRouteThreads = 8u;           // cap worker threads for tile binning

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

        struct SubList {
            std::vector<Eigen::Vector3f> pts, nrm;
        };

        int GetTilesAffectedByPoint(const Eigen::Vector3f &p, TileKey out[8]) const {
            const Eigen::Vector3i v(static_cast<int>(std::floor(p.x() / m_voxelSize)),
                                    static_cast<int>(std::floor(p.y() / m_voxelSize)),
                                    static_cast<int>(std::floor(p.z() / m_voxelSize)));
            const Eigen::Vector3i home = tileOf(v);
            const Eigen::Vector3i local = (v - m_origin) - home * kCore;
            int offs[3][2], nOff[3];
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
            int k = 0;
            for (int ix = 0; ix < nOff[0]; ++ix)
                for (int iy = 0; iy < nOff[1]; ++iy)
                    for (int iz = 0; iz < nOff[2]; ++iz)
                        out[k++] = TileKey{home.x() + offs[0][ix], home.y() + offs[1][iy],
                                           home.z() + offs[2][iz]};
            return k;
        }

        std::vector<TileKey> GetTilesAffectedByFrame(const std::vector<Eigen::Vector3f> &points,
                                                     const std::vector<Eigen::Vector3f> &normals) const {
            const std::size_t n = std::min(points.size(), normals.size());
            std::vector<TileKey> out;
            if (n == 0) return out;

            auto binRange = [&](std::size_t lo,
                                std::size_t hi,
                                std::unordered_set<TileKey, TileKeyHash> &into) {
                TileKey tgt[8];
                for (std::size_t i = lo; i < hi; ++i) {
                    const int m = GetTilesAffectedByPoint(points[i], tgt);
                    for (int t = 0; t < m; ++t) into.insert(tgt[t]);
                }
            };

            const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
            const unsigned nThreads = (n < kParallelRouteMin) ? 1u : std::min(hw, kMaxRouteThreads);
            if (nThreads == 1u) {
                std::unordered_set<TileKey, TileKeyHash> seen;
                binRange(0, n, seen);
                out.assign(seen.begin(), seen.end());
                return out;
            }

            std::vector<std::unordered_set<TileKey, TileKeyHash>> local(nThreads);
            std::vector<std::thread> workers;
            workers.reserve(nThreads - 1);
            const std::size_t chunk = (n + nThreads - 1) / nThreads;
            for (unsigned w = 1; w < nThreads; ++w) {
                const std::size_t lo = std::min(n, w * chunk);
                const std::size_t hi = std::min(n, lo + chunk);
                workers.emplace_back([&, lo, hi, w] { binRange(lo, hi, local[w]); });
            }
            binRange(0, std::min(n, chunk), local[0]);
            for (auto &t: workers) t.join();

            std::unordered_set<TileKey, TileKeyHash> merged;
            for (const auto &s: local) merged.insert(s.begin(), s.end());
            out.assign(merged.begin(), merged.end());
            return out;
        }

        void UploadReuseBuffer(const std::vector<Eigen::Vector3f> &points,
                               const std::vector<Eigen::Vector3f> &normals,
                               uint32_t n) {
            AllocateReusePointCloudeBuffer(n);
            std::memcpy(m_reusePointBuffer->MappedPtr(), points.data(), n * 3u * sizeof(float));
            std::memcpy(m_reuseNormalBuffer->MappedPtr(), normals.data(), n * 3u * sizeof(float));
            m_reusePointBuffer->MakeVisibleToGPU(n * 3u * sizeof(float));
            m_reuseNormalBuffer->MakeVisibleToGPU(n * 3u * sizeof(float));
        }

        void AllocateReusePointCloudeBuffer(uint32_t n) {
            if (m_reusePointBuffer && n <= m_reuseBufferSize) return;
            m_reuseBufferSize = std::max(n + n / 2u, m_maxPointsPerFrame);
            m_reusePointBuffer = std::make_unique<Engine::Core::Buffer>(*m_ctx);
            m_reuseNormalBuffer = std::make_unique<Engine::Core::Buffer>(*m_ctx);
            m_reusePointBuffer->AllocateHostVisible(m_reuseBufferSize * 3u * sizeof(float));
            m_reuseNormalBuffer->AllocateHostVisible(m_reuseBufferSize * 3u * sizeof(float));
        }

        /*
        Point Cloud
        │
        ├── point p
        │
        ▼
        Voxel 좌표 계산
        │
        ▼
        어느 Tile에 속하는지 계산
        │
        ├── Tile 내부 중앙 → 해당 Tile에만 추가
        │
        └── Tile 경계 근처
                │
                ├── 현재 Tile
                └── 이웃 Tile에도 추가
        */
        std::unordered_map<TileKey, SubList, TileKeyHash>
        splitPointsToTiles(const std::vector<Eigen::Vector3f> &points,
                           const std::vector<Eigen::Vector3f> &normals) const {
            std::unordered_map<TileKey, SubList, TileKeyHash> routed;
            const size_t n = std::min(points.size(), normals.size());
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
            return routed;
        }

        Eigen::Vector3i tileOf(const Eigen::Vector3i &v) const {
            return Eigen::Vector3i(floorDiv(v.x() - m_origin.x(), kCore),
                                   floorDiv(v.y() - m_origin.y(), kCore),
                                   floorDiv(v.z() - m_origin.z(), kCore));
        }

        Backend *GetTSDF(const TileKey &key) {
            auto it = m_tiles.find(key);
            if (it != m_tiles.end()) return it->second.get();

            // Calculate position of tile
            const Eigen::Vector3i tile(key.x, key.y, key.z);
            const Eigen::Vector3i originVoxel =
                    m_origin + tile * kCore - Eigen::Vector3i::Constant(m_ghost);
            const Eigen::Vector3f windowMinCorner = originVoxel.cast<float>() * m_voxelSize;

            // Build TSDF
            auto tsdf = std::make_unique<Backend>();
            if constexpr (HasHashStrategyBuild<Backend>::value) {
                tsdf->Build(
                        *m_ctx,
                        m_voxelSize,
                        m_truncation,
                        m_hashCapacityPerTile,
                        m_maxPointsPerFrame,
                        windowMinCorner,
                        *m_hash);
            } else {
                tsdf->Build(
                        *m_ctx,
                        m_voxelSize,
                        m_truncation,
                        m_hashCapacityPerTile,
                        m_maxPointsPerFrame,
                        windowMinCorner);
            }
            tsdf->SetIntegrationQuality(m_quality);
            tsdf->SetPointToPlane(m_pointToPlane);
            if constexpr (HasSetCurrentFrame<Backend>::value) tsdf->SetCurrentFrame(m_currentFrame);
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
        const TSDF::HashStrategy *m_hash = &TSDF::LinearProbeStrategy();
        int m_ghost = 0;                                    // G (set in Build)
        Eigen::Vector3i m_origin = Eigen::Vector3i::Zero(); // O
        IntegrationQuality m_quality;
        bool m_pointToPlane = true;
        // forwarded to tiles for the per-slot first-fill stamp
        int m_currentFrame = 0;

        std::unordered_map<TileKey, std::unique_ptr<Backend>, TileKeyHash> m_tiles;

        std::unique_ptr<Engine::Core::Buffer> m_reusePointBuffer, m_reuseNormalBuffer;
        uint32_t m_reuseBufferSize = 0;
    };

} // namespace TSDF

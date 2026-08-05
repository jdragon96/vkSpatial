#include "Engine/Spatial/AdvancedTSDF.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace Engine::Spatial {

    static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFFu;
    // Matches #define TSDF_SCALE 10000.0 in advanced_tsdf_integrate.vert.glsl /
    // advanced_tsdf_extract.vert.glsl (extract's MIN_WEIGHT gate = TSDF_SCALE/2).
    static constexpr int32_t kTsdfScale = 10000;
    static constexpr size_t kMaxClustersPerVoxel = 6; // max distinct orientations per voxel

    namespace {
        // Must match the push_constant block in advanced_tsdf_integrate.vert.glsl (all 4-byte
        // scalars -> tightly packed). g_pointToPlane is the last field.
        struct IntegratePC {
            uint32_t numPoints;
            uint32_t hashCapacity;
            float voxelSize;
            float truncation;
            float camX;
            float camY;
            float camZ;
            uint32_t maxDirections;
            uint32_t dirExponent;
            uint32_t viewAngleWeight;
            int32_t originX;
            int32_t originY;
            int32_t originZ;
            uint32_t pointToPlane;
            float confWeight;
            int32_t currentFrame;
        };

        // Must match the push_constant block in advanced_tsdf_extract.vert.glsl.
        struct ExtractPC {
            float voxelSize;
            uint32_t hashCapacity;
            uint32_t maxCandidates;
            int32_t originX;
            int32_t originY;
            int32_t originZ;
            float truncation;
            uint32_t hermite;
        };

        // Must match the push_constant block in advanced_tsdf_compact.comp.glsl. The kernel decodes on
        // the GPU (local key -> world centre), so it needs this tile's origin + voxel size + the core
        // bounds (in LOCAL voxel coords) that gate which voxels it appends.
        struct CompactPC {
            uint32_t hashCapacity;
            uint32_t maxOut;
            float voxelSize;
            int32_t originX, originY, originZ;
            int32_t coreMinX, coreMinY, coreMinZ;
            int32_t coreMaxX, coreMaxY, coreMaxZ;
        };

        // floor(worldMinCorner / voxelSize), component-wise.
        Eigen::Vector3i FloorToVoxel(const Eigen::Vector3f &worldMinCorner, float voxelSize) {
            return Eigen::Vector3i(static_cast<int32_t>(std::floor(worldMinCorner.x() / voxelSize)),
                                   static_cast<int32_t>(std::floor(worldMinCorner.y() / voxelSize)),
                                   static_cast<int32_t>(std::floor(worldMinCorner.z() / voxelSize)));
        }

        // 21-bit-per-axis world-space voxel key for MergeCandidates buckets.
        uint64_t VoxelKey(int32_t x, int32_t y, int32_t z) {
            return (uint64_t(uint32_t(x) & 0x1FFFFFu) << 42) |
                   (uint64_t(uint32_t(y) & 0x1FFFFFu) << 21) |
                   uint64_t(uint32_t(z) & 0x1FFFFFu);
        }
    } // namespace

    void AdvancedTSDF::Build(Engine::Core::Context &ctx,
                             float voxelSize,
                             float truncation,
                             uint32_t hashCapacity,
                             uint32_t maxPoints,
                             const Eigen::Vector3f &windowMinCorner) {
        m_ctx = &ctx;
        m_voxelSize = voxelSize;
        m_truncation = truncation;
        m_hashCapacity = hashCapacity;
        m_maxPoints = maxPoints;

        // Default (NaN corner) => centre the 512^3 window on the world origin at ANY voxelSize
        // (originVoxel = -256), fixing CompactDirectionalTSDF's voxelSize-0.1-only default.
        if (std::isnan(windowMinCorner.x()))
            m_originVoxel = Eigen::Vector3i::Constant(-256);
        else
            m_originVoxel = FloorToVoxel(windowMinCorner, voxelSize);

        m_hashBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_pointBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_normalBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_statBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_firstFrameBuffer = std::make_unique<Engine::Core::Buffer>(ctx);

        m_hashBuffer->Allocate(hashCapacity * sizeof(AdvDirEntry));
        m_pointBuffer->AllocateHostVisible(maxPoints * 3u * sizeof(float));
        m_normalBuffer->AllocateHostVisible(maxPoints * 3u * sizeof(float));
        // Host-visible so the fill count (written by the integrate shader's atomicAdd) can be read each
        // frame with a mapped read -- no staging/submit -- to drive auto-grow. Same pattern as the
        // compaction count buffer. On this UMA device shader atomics on host-visible memory are coherent.
        m_statBuffer->AllocateHostVisibleReadback(sizeof(uint32_t));
        m_firstFrameBuffer->Allocate(hashCapacity * sizeof(int32_t)); // per-slot first-fill frame

        m_kernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_kernel->Build("advanced_tsdf_integrate.comp.glsl")
                .Bind(0, *m_hashBuffer)
                .Bind(1, *m_pointBuffer)
                .Bind(2, *m_normalBuffer)
                .Bind(3, *m_statBuffer)
                .Bind(4, *m_firstFrameBuffer);

        // Compaction pass for DownloadEntries/CompactInto (built once, re-dispatched per download).
        // The (out, count) scratch is NOT allocated here: CompactInto() binds caller-provided scratch,
        // and standalone DownloadEntries() lazily allocates its own the first time it runs. This keeps
        // per-tile compaction memory off the GPU until a download actually needs it. The per-slot
        // first-fill buffer (binding 3) is bound here since it's owned by this tile.
        m_compactKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_compactKernel->Build("advanced_tsdf_compact.comp.glsl").Bind(3, *m_firstFrameBuffer);

        // Clear kernel: empties the hash on the GPU (one thread per slot). Replaces a per-tile 24 MB
        // host upload of EMPTY -- the dominant tile-creation cost -- with a ~0.1 ms compute dispatch.
        m_clearKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_clearKernel->Build("advanced_tsdf_clear.comp.glsl").Bind(0, *m_hashBuffer);

        // Rehash kernel: re-inserts occupied slots into a larger hash on auto-grow (buffers are bound
        // per-grow, since the old/new handles change each time). Compiled once here.
        m_rehashKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_rehashKernel->Build("advanced_tsdf_rehash.comp.glsl");

        Reset();
    }

    void AdvancedTSDF::Reset() {
        // Empty the hash on the GPU (one dispatch) rather than uploading a 24 MB "empty" buffer per
        // tile -- the host upload dominated tile creation, which spikes the integrate step whenever the
        // scan reaches new regions.
        struct ClearPC {
            uint32_t hashCapacity;
        };
        m_clearKernel->Args(ClearPC{m_hashCapacity});
        m_clearKernel->DispatchElements(m_hashCapacity); // synchronous
        *static_cast<uint32_t *>(m_statBuffer->MappedPtr()) = 0; // host-visible fill count
        m_statBuffer->FlushMapped(sizeof(uint32_t));
        // firstFrame is stamped on each slot's first fill, so empty slots' stale values never surface
        // (compaction only reads occupied slots) -- no explicit clear needed.
    }

    void AdvancedTSDF::Integrate(const std::vector<Eigen::Vector3f> &points,
                                 const std::vector<Eigen::Vector3f> &normals,
                                 const Eigen::Vector3f &cameraPos) {
        if (points.empty()) return;
        Engine::Compute::CommandBatch batch(*m_ctx);
        RecordIntegrate(points, normals, cameraPos, batch);
        batch.Submit();
    }

    void AdvancedTSDF::RecordIntegrate(const std::vector<Eigen::Vector3f> &points,
                                       const std::vector<Eigen::Vector3f> &normals,
                                       const Eigen::Vector3f &cameraPos,
                                       Engine::Compute::CommandBatch &batch) {
        if (points.empty()) return;
        // Clamp to the Build-time capacity: excess points are dropped (see IntegrateGPU for the
        // grow-instead-of-clamp real-time path).
        const uint32_t N = std::min({static_cast<uint32_t>(points.size()),
                                     static_cast<uint32_t>(normals.size()), m_maxPoints});
        if (N == 0) return;
        maybeGrow(); // keep the hash load bounded (constant integrate cost) before recording this frame
        recordUpload(points, normals, cameraPos, N, batch);
    }

    void AdvancedTSDF::RecordIntegrateGPU(const std::vector<Eigen::Vector3f> &points,
                                          const std::vector<Eigen::Vector3f> &normals,
                                          const Eigen::Vector3f &cameraPos,
                                          Engine::Compute::CommandBatch &batch) {
        if (points.empty()) return;
        // Upload the WHOLE frame -- grow the buffers (with slack) rather than clamp, so nothing drops.
        const uint32_t N =
                std::min(static_cast<uint32_t>(points.size()), static_cast<uint32_t>(normals.size()));
        if (N == 0) return;
        maybeGrow(); // keep the hash load bounded (constant integrate cost) before recording this frame
        ensureUploadCapacity(N);
        recordUpload(points, normals, cameraPos, N, batch);
    }

    void AdvancedTSDF::IntegrateGPU(const std::vector<Eigen::Vector3f> &points,
                                    const std::vector<Eigen::Vector3f> &normals,
                                    const Eigen::Vector3f &cameraPos) {
        if (points.empty() || !m_ctx) return;
        Engine::Compute::CommandBatch batch(*m_ctx);
        RecordIntegrateGPU(points, normals, cameraPos, batch);
        batch.Submit();
    }

    void AdvancedTSDF::ensureUploadCapacity(uint32_t n) {
        if (n <= m_maxPoints) return;
        // 1.5x slack so streaming frames of similar size don't reallocate every call. AllocateHostVisible
        // frees the old allocation, so the VkBuffer handles change -- re-bind the kernel to the new ones.
        const uint32_t grown = n + n / 2u;
        m_pointBuffer->AllocateHostVisible(grown * 3u * sizeof(float));
        m_normalBuffer->AllocateHostVisible(grown * 3u * sizeof(float));
        m_kernel->Bind(1, *m_pointBuffer).Bind(2, *m_normalBuffer);
        m_maxPoints = grown;
    }

    void AdvancedTSDF::recordUpload(const std::vector<Eigen::Vector3f> &points,
                                    const std::vector<Eigen::Vector3f> &normals,
                                    const Eigen::Vector3f &cameraPos, uint32_t n,
                                    Engine::Compute::CommandBatch &batch) {
        // Zero-copy upload: memcpy straight into the persistently mapped storage buffers (no staging,
        // no submit) — the dispatch is recorded into the caller's batch, not self-submitted.
        std::memcpy(m_pointBuffer->MappedPtr(), points.data(), n * 3u * sizeof(float));
        std::memcpy(m_normalBuffer->MappedPtr(), normals.data(), n * 3u * sizeof(float));
        m_pointBuffer->FlushMapped(n * 3u * sizeof(float));
        m_normalBuffer->FlushMapped(n * 3u * sizeof(float));

        // Bind this tile's OWN buffers (a prior RecordIntegrateShared may have left the shared ones).
        m_kernel->Bind(1, *m_pointBuffer).Bind(2, *m_normalBuffer);
        IntegratePC pc{
                n, m_hashCapacity, m_voxelSize, m_truncation,
                cameraPos.x(), cameraPos.y(), cameraPos.z(),
                m_quality.maxDirections, m_quality.dirExponent,
                m_quality.viewAngleWeight ? 1u : 0u,
                m_originVoxel.x(), m_originVoxel.y(), m_originVoxel.z(),
                uint32_t(m_pointToPlane ? 1u : 0u), m_confWeight, m_currentFrame};
        m_kernel->Args(pc);
        batch.DispatchElements(*m_kernel, n);
    }

    void AdvancedTSDF::RecordIntegrateShared(Engine::Core::Buffer &points, Engine::Core::Buffer &normals,
                                             uint32_t n, const Eigen::Vector3f &cameraPos,
                                             Engine::Compute::CommandBatch &batch) {
        if (n == 0 || !m_ctx) return;
        maybeGrow(); // bound the hash load (constant integrate cost) before recording this frame
        // Integrate from a SHARED whole-cloud buffer (uploaded once by the tiled coordinator, not
        // copied per tile). The shader's window filter keeps only the points inside THIS tile's window,
        // so the CPU never routes/copies points -- it just dispatches every tile over the same cloud.
        m_kernel->Bind(1, points).Bind(2, normals);
        IntegratePC pc{
                n, m_hashCapacity, m_voxelSize, m_truncation,
                cameraPos.x(), cameraPos.y(), cameraPos.z(),
                m_quality.maxDirections, m_quality.dirExponent,
                m_quality.viewAngleWeight ? 1u : 0u,
                m_originVoxel.x(), m_originVoxel.y(), m_originVoxel.z(),
                uint32_t(m_pointToPlane ? 1u : 0u), m_confWeight, m_currentFrame};
        m_kernel->Args(pc);
        batch.DispatchElements(*m_kernel, n);
    }

    uint32_t AdvancedTSDF::FilledCount() const {
        m_statBuffer->InvalidateMapped(sizeof(uint32_t));
        return *static_cast<const uint32_t *>(m_statBuffer->MappedPtr());
    }

    // Bound the hash load factor so probe chains -- and thus per-frame integrate cost -- stay ~constant
    // as the map accumulates, and so no insert overflows MAX_PROBE (which silently drops a voxel). Read
    // the GPU fill count; if the table is at least half full, double it and rehash. Only tiles that fill
    // grow, so total memory tracks real occupancy.
    void AdvancedTSDF::maybeGrow() {
        if (!m_hashBuffer || m_hashCapacity == 0) return;
        const uint32_t filled = FilledCount();
        if (uint64_t(filled) * 2u < m_hashCapacity) return; // load < 0.5 -> probe chains still short
        growHash(m_hashCapacity * 2u);
    }

    void AdvancedTSDF::growHash(uint32_t newCapacity) {
        if (newCapacity <= m_hashCapacity) return;

        // Allocate the larger hash + parallel first-fill buffer, empty the new hash (findOrInsert needs
        // EMPTY slots), then GPU-rehash every occupied old slot into it. Both dispatches self-submit
        // synchronously -- a rare, one-off cost paid only on the frame a tile crosses the load threshold.
        auto newHash = std::make_unique<Engine::Core::Buffer>(*m_ctx);
        auto newFirst = std::make_unique<Engine::Core::Buffer>(*m_ctx);
        newHash->Allocate(newCapacity * sizeof(AdvDirEntry));
        newFirst->Allocate(newCapacity * sizeof(int32_t));

        struct ClearPC {
            uint32_t hashCapacity;
        };
        m_clearKernel->Bind(0, *newHash).Args(ClearPC{newCapacity});
        m_clearKernel->DispatchElements(newCapacity); // synchronous: clear before rehash reads it

        struct RehashPC {
            uint32_t oldCapacity;
            uint32_t newCapacity;
        };
        m_rehashKernel->Bind(0, *m_hashBuffer)
                .Bind(1, *m_firstFrameBuffer)
                .Bind(2, *newHash)
                .Bind(3, *newFirst)
                .Args(RehashPC{m_hashCapacity, newCapacity});
        m_rehashKernel->DispatchElements(m_hashCapacity); // synchronous

        // Swap in the grown buffers and re-point every kernel that reads the hash / first-fill table.
        // The fill count is unchanged (a rehash moves entries, it does not add or drop any).
        m_hashBuffer = std::move(newHash);
        m_firstFrameBuffer = std::move(newFirst);
        m_hashCapacity = newCapacity;
        m_kernel->Bind(0, *m_hashBuffer).Bind(4, *m_firstFrameBuffer);
        m_compactKernel->Bind(3, *m_firstFrameBuffer);
        m_clearKernel->Bind(0, *m_hashBuffer);
        m_compactBuffer.reset(); // standalone-download scratch was hash-sized -> re-alloc on next use
        m_compactCountBuffer.reset();
    }

    void AdvancedTSDF::RecordCompact(Engine::Core::Buffer &out, Engine::Core::Buffer &count,
                                     Engine::Compute::CommandBatch &batch,
                                     const Eigen::Vector3i &coreMinWorld,
                                     const Eigen::Vector3i &coreMaxWorld) const {
        if (!m_ctx) return;
        // `out` holds decoded AdvancedEntry records; its capacity caps the append (overflow is reported
        // back via `count`, which the caller reads to grow + redo). Core bounds arrive in world voxel
        // coords; the kernel filters in LOCAL coords, so shift by this tile's origin.
        const uint32_t capacity = static_cast<uint32_t>(out.Size() / sizeof(AdvancedEntry));
        const Eigen::Vector3i lo = coreMinWorld - m_originVoxel;
        const Eigen::Vector3i hi = coreMaxWorld - m_originVoxel;

        m_compactKernel->Bind(0, *m_hashBuffer).Bind(1, out).Bind(2, count);
        m_compactKernel->Args(CompactPC{m_hashCapacity, capacity, m_voxelSize,
                                        m_originVoxel.x(), m_originVoxel.y(), m_originVoxel.z(),
                                        lo.x(), lo.y(), lo.z(), hi.x(), hi.y(), hi.z()});
        batch.DispatchElements(*m_compactKernel, m_hashCapacity); // recorded; caller submits once
    }

    std::vector<AdvancedEntry> AdvancedTSDF::DownloadEntries() const {
        if (!m_ctx) return {};
        // Standalone download: lazily allocate this tile's OWN scratch (hash-sized in AdvancedEntry
        // units, so the append can never overflow) and compact the WHOLE window (no core cropping) in
        // one self-submitted batch. A tiled coordinator instead shares one buffer across all tiles and
        // gives each its own core via RecordCompact -- one submit for the whole map.
        if (!m_compactBuffer) {
            m_compactBuffer = std::make_unique<Engine::Core::Buffer>(*m_ctx);
            m_compactCountBuffer = std::make_unique<Engine::Core::Buffer>(*m_ctx);
            m_compactBuffer->AllocateHostVisibleReadback(m_hashCapacity * sizeof(AdvancedEntry));
            m_compactCountBuffer->AllocateHostVisibleReadback(sizeof(uint32_t));
        }
        auto *countPtr = static_cast<uint32_t *>(m_compactCountBuffer->MappedPtr());
        *countPtr = 0;
        m_compactCountBuffer->FlushMapped(sizeof(uint32_t));

        Engine::Compute::CommandBatch batch(*m_ctx);
        const Eigen::Vector3i whole = Eigen::Vector3i::Constant(512); // append the full 512^3 window
        RecordCompact(*m_compactBuffer, *m_compactCountBuffer, batch, m_originVoxel,
                      m_originVoxel + whole);
        batch.Submit();

        m_compactCountBuffer->InvalidateMapped(sizeof(uint32_t));
        const uint32_t n = std::min(*countPtr, m_hashCapacity); // out is hash-sized -> never truncates
        std::vector<AdvancedEntry> result(n);
        if (n > 0) {
            m_compactBuffer->InvalidateMapped(n * sizeof(AdvancedEntry));
            std::memcpy(result.data(), m_compactBuffer->MappedPtr(), n * sizeof(AdvancedEntry));
        }
        return result;
    }

    OrientedPointCloud AdvancedTSDF::ExtractPointCloud(uint32_t maxCandidates, bool merge) const {
        OrientedPointCloud cloud;
        if (!m_ctx) return cloud;

        Engine::Core::Buffer candBuf(*m_ctx);
        Engine::Core::Buffer countBuf(*m_ctx);
        candBuf.Allocate(maxCandidates * 6u * sizeof(float)); // 6 floats/candidate: pos + normal
        countBuf.Allocate(sizeof(uint32_t));

        const uint32_t zero = 0;
        countBuf.Upload(&zero, sizeof(uint32_t));

        ExtractPC pc{m_voxelSize, m_hashCapacity, maxCandidates,
                     m_originVoxel.x(), m_originVoxel.y(), m_originVoxel.z(),
                     m_truncation, uint32_t(m_hermite ? 1u : 0u)};

        Engine::Core::ComputePipeline kernel(*m_ctx);
        kernel.Build("advanced_tsdf_extract.comp.glsl")
                .Bind(0, *m_hashBuffer)
                .Bind(1, candBuf)
                .Bind(2, countBuf)
                .Args(pc)
                .DispatchElements(m_hashCapacity);

        uint32_t count = 0;
        countBuf.Download(&count, sizeof(uint32_t));
        const uint32_t n = std::min(count, maxCandidates);
        if (n == 0) return cloud;

        std::vector<float> raw(n * 6u);
        candBuf.Download(raw.data(), n * 6u * sizeof(float));

        cloud.points.resize(n);
        cloud.normals.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            cloud.points[i] = Eigen::Vector3f(raw[i * 6u + 0u], raw[i * 6u + 1u], raw[i * 6u + 2u]);
            cloud.normals[i] = Eigen::Vector3f(raw[i * 6u + 3u], raw[i * 6u + 4u], raw[i * 6u + 5u]);
        }

        if (!merge) return cloud;
        return MergeCandidates(cloud.points, cloud.normals, m_voxelSize);
    }

    OrientedPointCloud AdvancedTSDF::MergeCandidates(const std::vector<Eigen::Vector3f> &points,
                                                     const std::vector<Eigen::Vector3f> &normals,
                                                     float voxelSize) {
        struct Cluster {
            Eigen::Vector3f posSum = Eigen::Vector3f::Zero();
            Eigen::Vector3f nSum = Eigen::Vector3f::Zero();
            int count = 0;
        };

        const float posThresh = 0.6f * voxelSize; // positionMergeThreshold
        const float cosThresh = 0.866f;           // normalMergeThreshold = 30 deg
        const float strongSplitCos = 0.5f;        // 60 deg -- never merge beyond this (corner)

        std::unordered_map<uint64_t, std::vector<Cluster>> buckets;
        const size_t nCand = std::min(points.size(), normals.size());
        for (size_t i = 0; i < nCand; ++i) {
            const Eigen::Vector3f &pos = points[i];
            const Eigen::Vector3f &nrm = normals[i];
            const int vx = int(std::floor(pos.x() / voxelSize));
            const int vy = int(std::floor(pos.y() / voxelSize));
            const int vz = int(std::floor(pos.z() / voxelSize));
            auto &clusters = buckets[VoxelKey(vx, vy, vz)];

            bool merged = false;
            for (auto &cl: clusters) {
                const Eigen::Vector3f mean = cl.posSum / float(cl.count);
                const Eigen::Vector3f meanN = cl.nSum.normalized();
                if (nrm.dot(meanN) < strongSplitCos) continue; // strong split: cannot merge
                if ((pos - mean).norm() < posThresh && nrm.dot(meanN) > cosThresh) {
                    cl.posSum += pos;
                    cl.nSum += nrm;
                    cl.count++;
                    merged = true;
                    break;
                }
            }
            if (!merged && clusters.size() < kMaxClustersPerVoxel)
                clusters.push_back({pos, nrm, 1});
        }

        OrientedPointCloud out;
        for (auto &bucket: buckets)
            for (auto &cl: bucket.second) {
                out.points.push_back(cl.posSum / float(cl.count));
                out.normals.push_back(cl.nSum.normalized());
            }
        return out;
    }

} // namespace Engine::Spatial

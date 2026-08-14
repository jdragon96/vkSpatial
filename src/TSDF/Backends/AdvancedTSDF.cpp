#include "TSDF/Backends/AdvancedTSDF.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace TSDF {

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
                             const Eigen::Vector3f &windowMinCorner,
                             const HashStrategy &hash) {
        m_ctx = &ctx;
        m_voxelSize = voxelSize;
        m_truncation = truncation;
        m_hashCapacity = hashCapacity;
        m_maxPoints = maxPoints;
        m_hash = &hash;

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
        m_insertFailureBuffer = std::make_unique<Engine::Core::Buffer>(ctx);

        m_hashBuffer->Allocate(hashCapacity * sizeof(AdvDirEntry));
        m_pointBuffer->AllocateHostVisible(maxPoints * 3u * sizeof(float));
        m_normalBuffer->AllocateHostVisible(maxPoints * 3u * sizeof(float));
        m_statBuffer->AllocateHostVisibleReadback(sizeof(uint32_t));
        m_firstFrameBuffer->Allocate(hashCapacity * sizeof(int32_t));
        m_insertFailureBuffer->AllocateHostVisibleReadback(sizeof(uint32_t));

        kernel_integratePoints = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        if (m_hash->macroName) kernel_integratePoints->Define(m_hash->macroName);
        kernel_integratePoints->Build("TSDF/Backends/AdvancedTSDF.integrate.comp.glsl")
                .Bind(0, *m_hashBuffer)
                .Bind(1, *m_pointBuffer)
                .Bind(2, *m_normalBuffer)
                .Bind(3, *m_statBuffer)
                .Bind(4, *m_firstFrameBuffer)
                .Bind(5, *m_insertFailureBuffer);

        kernel_compactTable = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        kernel_compactTable->Build("TSDF/Backends/AdvancedTSDF.compact.comp.glsl").Bind(3, *m_firstFrameBuffer);

        kernel_clearVoxel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        kernel_clearVoxel->Build("TSDF/Backends/AdvancedTSDF.clear.comp.glsl").Bind(0, *m_hashBuffer);

        kernel_rehashTable = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        kernel_rehashTable->Build("TSDF/Backends/AdvancedTSDF.rehash.comp.glsl");

        Reset();
    }

    void AdvancedTSDF::Reset() {
        // Empty the hash on the GPU (one dispatch) rather than uploading a 24 MB "empty" buffer per
        // tile -- the host upload dominated tile creation, which spikes the integrate step whenever the
        // scan reaches new regions.
        struct ClearPC {
            uint32_t hashCapacity;
        };
        kernel_clearVoxel->Args(ClearPC{m_hashCapacity});
        kernel_clearVoxel->DispatchElements(m_hashCapacity);     // synchronous
        *static_cast<uint32_t *>(m_statBuffer->MappedPtr()) = 0; // host-visible fill count
        m_statBuffer->MakeVisibleToGPU(sizeof(uint32_t));
        // firstFrame is stamped on each slot's first fill, so empty slots' stale values never surface
        // (compaction only reads occupied slots) -- no explicit clear needed.
        *static_cast<uint32_t *>(m_insertFailureBuffer->MappedPtr()) = 0;
        m_insertFailureBuffer->MakeVisibleToGPU(sizeof(uint32_t));
        m_growCount = 0; // otherwise a second scene reusing this instance would inherit the first's count
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
        const uint32_t N = std::min({static_cast<uint32_t>(points.size()),
                                     static_cast<uint32_t>(normals.size()),
                                     m_maxPoints});
        if (N == 0) return;
        maybeGrow();
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
        kernel_integratePoints->Bind(1, *m_pointBuffer).Bind(2, *m_normalBuffer);
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
        m_pointBuffer->MakeVisibleToGPU(n * 3u * sizeof(float));
        m_normalBuffer->MakeVisibleToGPU(n * 3u * sizeof(float));

        // Bind this tile's OWN buffers (a prior RecordIntegrateShared may have left the shared ones).
        kernel_integratePoints->Bind(1, *m_pointBuffer).Bind(2, *m_normalBuffer);
        IntegratePC pc{
                n, m_hashCapacity, m_voxelSize, m_truncation,
                cameraPos.x(), cameraPos.y(), cameraPos.z(),
                m_quality.maxDirections, m_quality.dirExponent,
                m_quality.viewAngleWeight ? 1u : 0u,
                m_originVoxel.x(), m_originVoxel.y(), m_originVoxel.z(),
                uint32_t(m_pointToPlane ? 1u : 0u), m_confWeight, m_currentFrame};
        kernel_integratePoints->Args(pc);
        batch.DispatchElements(*kernel_integratePoints, n);
    }

    void AdvancedTSDF::RecordIntegrateShared(Engine::Core::Buffer &points,
                                             Engine::Core::Buffer &normals,
                                             uint32_t n,
                                             const Eigen::Vector3f &cameraPos,
                                             Engine::Compute::CommandBatch &batch) {
        if (n == 0 || !m_ctx) return;
        maybeGrow();
        kernel_integratePoints->Bind(1, points).Bind(2, normals);
        IntegratePC pc{
                n,
                m_hashCapacity,
                m_voxelSize,
                m_truncation,
                cameraPos.x(),
                cameraPos.y(),
                cameraPos.z(),
                m_quality.maxDirections,
                m_quality.dirExponent,
                m_quality.viewAngleWeight ? 1u : 0u,
                m_originVoxel.x(),
                m_originVoxel.y(),
                m_originVoxel.z(),
                uint32_t(m_pointToPlane ? 1u : 0u),
                m_confWeight,
                m_currentFrame};
        kernel_integratePoints->Args(pc);
        batch.DispatchElements(*kernel_integratePoints, n);
    }

    uint32_t AdvancedTSDF::FilledCount() const {
        m_statBuffer->MakeVisibleToCPU(sizeof(uint32_t));
        return *static_cast<const uint32_t *>(m_statBuffer->MappedPtr());
    }

    uint32_t AdvancedTSDF::InsertFailureCount() const {
        if (!m_insertFailureBuffer) return 0;
        m_insertFailureBuffer->MakeVisibleToCPU(sizeof(uint32_t));
        return *static_cast<const uint32_t *>(m_insertFailureBuffer->MappedPtr());
    }

    void AdvancedTSDF::maybeGrow() {
        if (!m_hashBuffer || m_hashCapacity == 0) return;
        const uint32_t filled = FilledCount();
        // The threshold belongs to the hash strategy: linear probing collapses well before a
        // bucketed table does, so a shared constant would either waste memory or drop voxels.
        if (double(filled) < double(m_hashCapacity) * double(m_hash->loadFactorLimit)) return;
        growHash(m_hashCapacity * 2u);
    }

    void AdvancedTSDF::growHash(uint32_t newCapacity) {
        if (newCapacity <= m_hashCapacity) return;

        auto newHash = std::make_unique<Engine::Core::Buffer>(*m_ctx);
        auto newFirst = std::make_unique<Engine::Core::Buffer>(*m_ctx);
        newHash->Allocate(newCapacity * sizeof(AdvDirEntry));
        newFirst->Allocate(newCapacity * sizeof(int32_t));

        struct ClearPC {
            uint32_t hashCapacity;
        };
        kernel_clearVoxel->Bind(0, *newHash).Args(ClearPC{newCapacity});
        kernel_clearVoxel->DispatchElements(newCapacity); // synchronous: clear before rehash reads it

        struct RehashPC {
            uint32_t oldCapacity;
            uint32_t newCapacity;
        };
        kernel_rehashTable->Bind(0, *m_hashBuffer)
                .Bind(1, *m_firstFrameBuffer)
                .Bind(2, *newHash)
                .Bind(3, *newFirst)
                .Args(RehashPC{m_hashCapacity, newCapacity});
        kernel_rehashTable->DispatchElements(m_hashCapacity); // synchronous

        m_hashBuffer = std::move(newHash);
        m_firstFrameBuffer = std::move(newFirst);
        m_hashCapacity = newCapacity;
        // binding 5 (m_insertFailureBuffer) is untouched by a grow -- it never gets reallocated,
        // so it does not need re-binding here.
        kernel_integratePoints->Bind(0, *m_hashBuffer).Bind(4, *m_firstFrameBuffer);
        kernel_compactTable->Bind(3, *m_firstFrameBuffer);
        kernel_clearVoxel->Bind(0, *m_hashBuffer);
        m_compactBuffer.reset(); // standalone-download scratch was hash-sized -> re-alloc on next use
        m_compactCountBuffer.reset();
        ++m_growCount;
    }

    void AdvancedTSDF::RecordCompact(Engine::Core::Buffer &out,
                                     Engine::Core::Buffer &count,
                                     Engine::Compute::CommandBatch &batch,
                                     const Eigen::Vector3i &coreMinWorld,
                                     const Eigen::Vector3i &coreMaxWorld) const {
        if (!m_ctx) return;
        const uint32_t capacity = static_cast<uint32_t>(out.Size() / sizeof(AdvancedEntry));
        const Eigen::Vector3i lo = coreMinWorld - m_originVoxel;
        const Eigen::Vector3i hi = coreMaxWorld - m_originVoxel;

        kernel_compactTable->Bind(0, *m_hashBuffer).Bind(1, out).Bind(2, count);
        kernel_compactTable->Args(CompactPC{m_hashCapacity,
                                            capacity,
                                            m_voxelSize,
                                            m_originVoxel.x(), m_originVoxel.y(), m_originVoxel.z(),
                                            lo.x(), lo.y(), lo.z(),
                                            hi.x(), hi.y(), hi.z()});
        batch.DispatchElements(*kernel_compactTable, m_hashCapacity);
    }

    std::vector<AdvancedEntry> AdvancedTSDF::DownloadEntries() const {
        if (!m_ctx) return {};
        if (!m_compactBuffer) {
            m_compactBuffer = std::make_unique<Engine::Core::Buffer>(*m_ctx);
            m_compactCountBuffer = std::make_unique<Engine::Core::Buffer>(*m_ctx);
            m_compactBuffer->AllocateHostVisibleReadback(m_hashCapacity * sizeof(AdvancedEntry));
            m_compactCountBuffer->AllocateHostVisibleReadback(sizeof(uint32_t));
        }

        // 1. 채워진 복샐 개수 초기화
        auto *countPtr = static_cast<uint32_t *>(m_compactCountBuffer->MappedPtr());
        *countPtr = 0;
        m_compactCountBuffer->MakeVisibleToGPU(sizeof(uint32_t));

        // 2. 복셀 압축
        Engine::Compute::CommandBatch batch(*m_ctx);
        const Eigen::Vector3i whole = Eigen::Vector3i::Constant(512); // append the full 512^3 window
        RecordCompact(
                *m_compactBuffer,
                *m_compactCountBuffer,
                batch,
                m_originVoxel,
                m_originVoxel + whole);
        batch.Submit();

        // 3. Copy
        m_compactCountBuffer->MakeVisibleToCPU(sizeof(uint32_t));
        const uint32_t n = std::min(*countPtr, m_hashCapacity); // out is hash-sized -> never truncates
        std::vector<AdvancedEntry> result(n);
        if (n > 0) {
            m_compactBuffer->MakeVisibleToCPU(n * sizeof(AdvancedEntry));
            std::memcpy(result.data(), m_compactBuffer->MappedPtr(), n * sizeof(AdvancedEntry));
        }
        return result;
    }

    Engine::Core::OrientedPointCloud AdvancedTSDF::ExtractPointCloud(uint32_t maxCandidates, bool merge) const {
        Engine::Core::OrientedPointCloud cloud;
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
        if (m_hash->macroName) kernel.Define(m_hash->macroName);
        kernel.Build("TSDF/Backends/AdvancedTSDF.extract.comp.glsl")
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

    Engine::Core::OrientedPointCloud AdvancedTSDF::MergeCandidates(const std::vector<Eigen::Vector3f> &points,
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

        Engine::Core::OrientedPointCloud out;
        for (auto &bucket: buckets)
            for (auto &cl: bucket.second) {
                out.points.push_back(cl.posSum / float(cl.count));
                out.normals.push_back(cl.nSum.normalized());
            }
        return out;
    }

} // namespace TSDF

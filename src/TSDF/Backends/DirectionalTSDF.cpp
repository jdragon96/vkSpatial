#include "TSDF/Backends/DirectionalTSDF.h"

#include "Engine/Core/OneShotCommands.h"
#include "TSDF/Backends/DirectionalIntegrationQuality.h" // TopKDirections (shared direction selector)
#include "TSDF/Backends/Residency/IResidencyBackend.h" // MakeResidencyBackend (backend selection: Task 5)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace Engine::Spatial {

    namespace {
        struct IntegratePC {
            uint32_t numPoints;
            float voxelSize;
            float truncation;
            int32_t baseX;
            int32_t baseY;
            int32_t baseZ;
            float camX;
            float camY;
            float camZ;
            uint32_t maxDirections;
            uint32_t dirExponent;
            uint32_t viewAngleWeight;
            uint32_t pointToPlane;
        };

        struct ExtractPC {
            uint32_t numGroups;
            float voxelSize;
            uint32_t maxCandidates;
            int32_t baseX;
            int32_t baseY;
            int32_t baseZ;
            uint32_t mode;
            float truncation;
        };

        // Must match dominantAxis() in directional_tsdf_integrate.comp exactly.
        uint8_t dominantAxisOf(const Eigen::Vector3f &n) {
            const float ax = std::fabs(n.x()), ay = std::fabs(n.y()), az = std::fabs(n.z());
            if (ax >= ay && ax >= az) return n.x() >= 0.0f ? 0 : 1;
            if (ay >= ax && ay >= az) return n.y() >= 0.0f ? 2 : 3;
            return n.z() >= 0.0f ? 4 : 5;
        }

        uint64_t spatialKey(int32_t x, int32_t y, int32_t z) {
            return (uint64_t(uint32_t(x) & 0x1FFFFFu) << 42) |
                   (uint64_t(uint32_t(y) & 0x1FFFFFu) << 21) |
                   uint64_t(uint32_t(z) & 0x1FFFFFu);
        }
    } // namespace

    void DirectionalTSDF::Build(Engine::Core::Context &ctx,
                                float voxelSize,
                                float truncation,
                                uint32_t poolCapacity,
                                uint32_t maxPoints,
                                uint32_t maxCandidates,
                                ResidencyMode residency) {
        m_ctx = &ctx;
        m_voxelSize = voxelSize;
        m_truncation = truncation;
        m_maxPoints = maxPoints;
        m_maxCandidates = maxCandidates;

        // Backend selection delegated to the factory (Task 5): env override, then the
        // explicit `residency` arg, then (Auto) memory-topology probing. `residency`
        // defaults to Streaming so every existing caller (which passes nothing new here)
        // keeps today's characterized behavior on all platforms, including UMA.
        m_backend = MakeResidencyBackend(ctx, poolCapacity, residency);

        m_pointBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_candidateBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_candidateCounter = std::make_unique<Engine::Core::Buffer>(ctx);
        m_pointBuffer->Allocate(maxPoints * 6u * sizeof(float));
        m_candidateBuffer->Allocate(maxCandidates * sizeof(DirectionalCandidate));
        m_candidateCounter->Allocate(sizeof(uint32_t));

        // Recompute-group slot list for the extract kernel (binding 0). Core-owned: see
        // DirectionalTSDF.h for why this isn't the backend's internal register-list buffer.
        m_groupSlotListBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_groupSlotListBuffer->Allocate(poolCapacity * sizeof(uint32_t));

        // Persistent host-visible staging (Phase 5): allocated once, reused every frame.
        using Engine::Compute::StagingBuffer;
        const VkBufferUsageFlags kSrc = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        const VkBufferUsageFlags kDst = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        m_stagePoints = std::make_unique<StagingBuffer>(ctx, VkDeviceSize(maxPoints) * 6u * sizeof(float), kSrc);
        m_stageCandidates = std::make_unique<StagingBuffer>(ctx, VkDeviceSize(maxCandidates) * sizeof(DirectionalCandidate), kDst);
        m_stageGroupSlotList = std::make_unique<StagingBuffer>(ctx, VkDeviceSize(poolCapacity) * sizeof(uint32_t), kSrc);
        m_stageCandidateCount = std::make_unique<StagingBuffer>(ctx, sizeof(uint32_t), kDst);

        const VkDeviceSize indexGridBytes = VkDeviceSize(kIndexGridCells) * sizeof(uint32_t);
        const VkDeviceSize poolVoxelsBytes = VkDeviceSize(poolCapacity) * VkDeviceSize(kVoxelsPerGroup) * sizeof(GpuTsdfVoxel);
        const VkDeviceSize metaBytes = VkDeviceSize(poolCapacity) * sizeof(ActiveGroupMeta);

        m_integrateKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_integrateKernel->Build("directional_tsdf_integrate.comp")
                .Bind(0, *m_pointBuffer)
                .Bind(1, m_backend->IndexGridBuffer(), indexGridBytes)
                .Bind(2, m_backend->PoolVoxelBuffer(), poolVoxelsBytes)
                .Bind(3, m_backend->MetaBuffer(), metaBytes);

        m_extractKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_extractKernel->Build("directional_tsdf_extract.comp")
                .Bind(0, *m_groupSlotListBuffer) // recompute-group list (core-owned)
                .Bind(1, m_backend->MetaBuffer(), metaBytes)
                .Bind(2, m_backend->PoolVoxelBuffer(), poolVoxelsBytes)
                .Bind(3, m_backend->IndexGridBuffer(), indexGridBytes)
                .Bind(4, *m_candidateBuffer)
                .Bind(5, *m_candidateCounter);

        m_pointCloud.clear();
        m_stats = {};
        m_lastCounts = {};
    }

    void DirectionalTSDF::BeginFrame(const Eigen::Vector3f &aabbCenterHint) {
        if (!m_ctx)
            throw std::runtime_error("DirectionalTSDF: Build() must be called first");

        m_backend->BeginFrame(quantizeLocalBase(aabbCenterHint));

        const ResidencyStats fs = m_backend->FrameStats();
        m_stats = {};
        m_stats.residentCount = fs.residentCount;
        m_stats.writeBackCount = fs.writeBackCount;
        m_stats.h2dBytes = fs.h2dBytes;
        m_stats.d2hBytes = fs.d2hBytes;
        m_stats.overlapRatio = fs.overlapRatio;
        m_stats.gpuSubmits = fs.gpuSubmits;

        m_lastCounts.reusable = fs.reusableCount;
        m_lastCounts.cleanFree = fs.cleanFreeCount;
        m_lastCounts.writeBack = fs.writeBackCount;
    }

    void DirectionalTSDF::EnsureResident(const std::vector<DirectionalGroupKey> &required) {
        if (!m_ctx)
            throw std::runtime_error("DirectionalTSDF: Build() must be called first");

        m_backend->EnsureResident(required);

        const ResidencyStats fs = m_backend->FrameStats();
        m_stats.residentCount = fs.residentCount;
        m_stats.missingCount = fs.missingCount;
        m_stats.h2dBytes = fs.h2dBytes;
        m_stats.overlapRatio = fs.overlapRatio;
        m_stats.gpuSubmits = fs.gpuSubmits;
    }

    void DirectionalTSDF::Integrate(const std::vector<Eigen::Vector3f> &points,
                                    const std::vector<Eigen::Vector3f> &normals,
                                    const Eigen::Vector3f &cameraPos,
                                    const Eigen::Vector3f &aabbCenterHint) {
        if (points.size() != normals.size())
            throw std::runtime_error("DirectionalTSDF: points/normals size mismatch");
        if (points.empty()) return;

        using Clock = std::chrono::steady_clock;
        auto msBetween = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<float, std::milli>(b - a).count();
        };

        const auto t0 = Clock::now();
        BeginFrame(aabbCenterHint);
        const auto t1 = Clock::now();

        const uint32_t N = std::min(uint32_t(points.size()), m_maxPoints);

        // IntegrationWriteSet: per-sample conservative box over the truncation-band
        // ray segment (design doc §7/§13), keyed by the sample's dominant direction.
        std::unordered_set<DirectionalGroupKey, DirectionalGroupKeyHash> writeSet;
        for (uint32_t i = 0; i < N; ++i) {
            Eigen::Vector3f diff = points[i] - cameraPos;
            float depth = diff.norm();
            if (depth < 1e-6f) continue;
            Eigen::Vector3f dir = diff / depth;
            DirWeight dw[6];
            const int nd = TopKDirections(normals[i], m_quality, dw);
            const float band = m_truncation + m_voxelSize;
            Eigen::Vector3f a = points[i] - dir * band;
            Eigen::Vector3f b = points[i] + dir * band;
            Eigen::Vector3i vmin, vmax;
            for (int c = 0; c < 3; ++c) {
                const float lo = std::min(a[c], b[c]);
                const float hi = std::max(a[c], b[c]);
                vmin[c] = int(std::floor(lo / m_voxelSize)) - 1;
                vmax[c] = int(std::floor(hi / m_voxelSize)) + 1;
            }
            for (int gz = vmin.z() >> 3; gz <= (vmax.z() >> 3); ++gz)
                for (int gy = vmin.y() >> 3; gy <= (vmax.y() >> 3); ++gy)
                    for (int gx = vmin.x() >> 3; gx <= (vmax.x() >> 3); ++gx)
                        for (int di = 0; di < nd; ++di)
                            writeSet.insert({gx, gy, gz, dw[di].direction});
        }

        // ResidentRequiredSet = writeSet + 1-group halo (extraction neighbourhood, §7).
        std::vector<DirectionalGroupKey> required;
        {
            std::unordered_set<DirectionalGroupKey, DirectionalGroupKeyHash> requiredSet;
            for (const auto &k : writeSet)
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                            requiredSet.insert(
                                    {k.gx + dx, k.gy + dy, k.gz + dz, k.direction});
            required.assign(requiredSet.begin(), requiredSet.end());
        }
        // Stage point samples into the persistent staging buffer (CPU side).
        auto *ptStage = static_cast<float *>(m_stagePoints->Mapped());
        for (uint32_t i = 0; i < N; ++i) {
            ptStage[size_t(i) * 6 + 0] = points[i].x();
            ptStage[size_t(i) * 6 + 1] = points[i].y();
            ptStage[size_t(i) * 6 + 2] = points[i].z();
            ptStage[size_t(i) * 6 + 3] = normals[i].x();
            ptStage[size_t(i) * 6 + 4] = normals[i].y();
            ptStage[size_t(i) * 6 + 5] = normals[i].z();
        }

        // Batch 2: residency (upload missing + combined register) + point upload + integrate.
        // RecordResidency is folded into this SAME batch (rather than going through
        // IResidencyBackend::EnsureResident, which owns its own batch/submit) to preserve
        // the pre-refactor submit count (Phase 5 batching metric). RecordResidency updates
        // FrameStats().overlapRatio internally (backend-side), so the core just re-reads
        // FrameStats() below — it never calls into backend overlap-ratio bookkeeping directly.
        {
            Engine::Compute::CommandBatch batch(*m_ctx);
            m_backend->RecordResidency(required, batch);
            std::vector<VkBufferCopy> ptRegion(1);
            ptRegion[0] = {0, 0, VkDeviceSize(N) * 6u * sizeof(float)};
            batch.CopyBuffer(m_stagePoints->Handle(), m_pointBuffer->Handle(), ptRegion);
            batch.Barrier();
            const Eigen::Vector3i localBase = m_backend->LocalBase();
            IntegratePC ipc{N, m_voxelSize, m_truncation, localBase.x(), localBase.y(),
                            localBase.z(), cameraPos.x(), cameraPos.y(), cameraPos.z(),
                            m_quality.maxDirections, m_quality.dirExponent,
                            uint32_t(m_quality.viewAngleWeight ? 1 : 0),
                            uint32_t(m_pointToPlane ? 1u : 0u)};
            m_integrateKernel->Args(ipc);
            batch.DispatchElements(*m_integrateKernel, N);
            batch.Submit();
            ++m_stats.gpuSubmits;
        }
        {
            const ResidencyStats fs = m_backend->FrameStats();
            m_stats.residentCount = fs.residentCount;
            m_stats.missingCount = fs.missingCount;
            m_stats.h2dBytes = fs.h2dBytes;
            m_stats.d2hBytes = fs.d2hBytes;
            m_stats.overlapRatio = fs.overlapRatio;
        }
        const auto t3 = Clock::now();

        // recomputeMask is SPATIAL (design doc §14): re-extract every resident direction
        // layer at the written spatial locations, otherwise other-layer surface points
        // at those locations would be dropped by the old-point merge and never rebuilt.
        std::unordered_set<uint64_t> recomputeSpatial;
        for (const auto &k : writeSet)
            recomputeSpatial.insert(spatialKey(k.gx, k.gy, k.gz));

        std::vector<uint32_t> groupSlots;
        for (const auto &entry : m_backend->ResidentIndex())
            if (recomputeSpatial.count(
                        spatialKey(entry.first.gx, entry.first.gy, entry.first.gz)) > 0)
                groupSlots.push_back(entry.second);

        // Batch 3: extract + candidate readback (counter + full candidate buffer), one submit.
        uint32_t candidateCount = 0;
        if (!groupSlots.empty()) {
            auto *slotStage = static_cast<uint32_t *>(m_stageGroupSlotList->Mapped());
            std::memcpy(slotStage, groupSlots.data(), groupSlots.size() * sizeof(uint32_t));
            std::vector<VkBufferCopy> slotRegion(1);
            slotRegion[0] = {0, 0, VkDeviceSize(groupSlots.size()) * sizeof(uint32_t)};
            std::vector<VkBufferCopy> cntCopy(1);
            cntCopy[0] = {0, 0, sizeof(uint32_t)};
            std::vector<VkBufferCopy> candCopy(1);
            candCopy[0] = {0, 0, VkDeviceSize(m_maxCandidates) * sizeof(DirectionalCandidate)};

            const Eigen::Vector3i localBase = m_backend->LocalBase();
            ExtractPC epc{uint32_t(groupSlots.size()), m_voxelSize, m_maxCandidates,
                          localBase.x(), localBase.y(), localBase.z(),
                          m_extractMode, m_truncation};
            m_extractKernel->Args(epc);

            Engine::Compute::CommandBatch batch(*m_ctx);
            batch.CopyBuffer(m_stageGroupSlotList->Handle(), m_groupSlotListBuffer->Handle(), slotRegion);
            batch.FillBuffer(m_candidateCounter->Handle(), 0, VK_WHOLE_SIZE, 0u);
            batch.Barrier();
            batch.DispatchElements(*m_extractKernel, uint32_t(groupSlots.size()) * kVoxelsPerGroup);
            batch.Barrier();
            batch.CopyBuffer(m_candidateCounter->Handle(), m_stageCandidateCount->Handle(), cntCopy);
            batch.CopyBuffer(m_candidateBuffer->Handle(), m_stageCandidates->Handle(), candCopy);
            batch.Submit();
            ++m_stats.gpuSubmits;

            std::memcpy(&candidateCount, m_stageCandidateCount->Mapped(), sizeof(uint32_t));
            candidateCount = std::min(candidateCount, m_maxCandidates);
        }
        std::vector<DirectionalCandidate> candidates(candidateCount);
        if (candidateCount > 0)
            std::memcpy(candidates.data(), m_stageCandidates->Mapped(),
                        candidateCount * sizeof(DirectionalCandidate));
        const auto t4 = Clock::now();

        // Candidate merge + old-point replacement (§14/§15, invariant #9).
        std::vector<ExtractedPoint> fresh = mergeCandidates(candidates);
        m_pointCloud.erase(
                std::remove_if(m_pointCloud.begin(), m_pointCloud.end(),
                               [&](const ExtractedPoint &pt) {
                                   return recomputeSpatial.count(spatialKey(
                                                  pt.ownerGx, pt.ownerGy, pt.ownerGz)) > 0;
                               }),
                m_pointCloud.end());
        m_pointCloud.insert(m_pointCloud.end(), fresh.begin(), fresh.end());
        const auto t5 = Clock::now();

        m_backend->EndFrame();

        m_stats.beginFrameMs = msBetween(t0, t1);
        m_stats.ensureResidentMs = 0.0f;         // residency folded into the integrate batch
        m_stats.integrateMs = msBetween(t1, t3); // BeginFrame → end of Batch 2
        m_stats.extractMs = msBetween(t3, t4);
        m_stats.mergeMs = msBetween(t4, t5);
    }

    std::vector<ExtractedPoint> DirectionalTSDF::mergeCandidates(
            const std::vector<DirectionalCandidate> &candidates) const {
        struct Cluster {
            Eigen::Vector3f posSum = Eigen::Vector3f::Zero();
            Eigen::Vector3f nSum = Eigen::Vector3f::Zero();
            int count = 0;
            uint8_t dirMask = 0;
            int32_t gx = 0, gy = 0, gz = 0;
        };
        auto voxelKey = [](int x, int y, int z) {
            return (uint64_t(uint32_t(x) & 0x1FFFFFu) << 42) |
                   (uint64_t(uint32_t(y) & 0x1FFFFFu) << 21) |
                   uint64_t(uint32_t(z) & 0x1FFFFFu);
        };
        const float posThresh = 0.6f * m_voxelSize; // positionMergeThreshold (§15)
        const float cosThresh = 0.866f;             // normalMergeThreshold = 30° (§15)
        const float strongSplitCos = 0.5f;          // 60° — never merge beyond this (§7)

        std::unordered_map<uint64_t, std::vector<Cluster>> buckets;
        for (const auto &c : candidates) {
            Eigen::Vector3f pos(c.px, c.py, c.pz);
            Eigen::Vector3f nrm(c.nx, c.ny, c.nz);
            const int vx = int(std::floor(pos.x() / m_voxelSize));
            const int vy = int(std::floor(pos.y() / m_voxelSize));
            const int vz = int(std::floor(pos.z() / m_voxelSize));
            auto &clusters = buckets[voxelKey(vx, vy, vz)];
            bool merged = false;
            for (auto &cl : clusters) {
                const Eigen::Vector3f mean = cl.posSum / float(cl.count);
                const Eigen::Vector3f meanN = cl.nSum.normalized();
                if (nrm.dot(meanN) < strongSplitCos) continue; // strong split: cannot merge (§7)
                if ((pos - mean).norm() < posThresh && nrm.dot(meanN) > cosThresh) {
                    cl.posSum += pos;
                    cl.nSum += nrm;
                    cl.count++;
                    cl.dirMask |= uint8_t(1u << c.direction);
                    merged = true;
                    break;
                }
            }
            if (!merged && clusters.size() < kNumDirections)
                clusters.push_back({pos, nrm, 1, uint8_t(1u << c.direction),
                                    c.gx, c.gy, c.gz});
        }

        std::vector<ExtractedPoint> out;
        for (auto &bucket : buckets)
            for (auto &cl : bucket.second) {
                ExtractedPoint pt;
                pt.position = cl.posSum / float(cl.count);
                pt.normal = cl.nSum.normalized();
                pt.ownerGx = cl.gx;
                pt.ownerGy = cl.gy;
                pt.ownerGz = cl.gz;
                pt.dirMask = cl.dirMask;
                out.push_back(pt);
            }
        return out;
    }

    void DirectionalTSDF::ExportPointCloud(const std::string &path) const {
        std::ofstream f(path);
        if (!f.is_open())
            throw std::runtime_error("DirectionalTSDF::ExportPointCloud: cannot open " + path);

        f << "ply\nformat ascii 1.0\n"
          << "element vertex " << m_pointCloud.size() << "\n"
          << "property float x\nproperty float y\nproperty float z\n"
          << "property float nx\nproperty float ny\nproperty float nz\n"
          << "end_header\n";
        for (const auto &pt : m_pointCloud)
            f << pt.position.x() << ' ' << pt.position.y() << ' ' << pt.position.z() << ' '
              << pt.normal.x() << ' ' << pt.normal.y() << ' ' << pt.normal.z() << '\n';
    }

    Eigen::Vector3i DirectionalTSDF::quantizeLocalBase(const Eigen::Vector3f &center) const {
        const float g = GroupWorldSize();
        const int half = int(kLocalGroupGrid) / 2;
        return Eigen::Vector3i(int(std::floor(center.x() / g)) - half,
                               int(std::floor(center.y() / g)) - half,
                               int(std::floor(center.z() / g)) - half);
    }

} // namespace Engine::Spatial

#include "TSDF/Backends/Residency/StreamingResidencyBackend.h"

#include "Engine/Core/OneShotCommands.h"
#include "TSDF/Backends/DirectionalVoxelConvert.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace Engine::Spatial {

    namespace {
        constexpr uint32_t kGroupBytes = kVoxelsPerGroup * uint32_t(sizeof(GpuTsdfVoxel)); // 10240
        constexpr uint32_t kStageGroupCap = 4096; // max groups staged per H2D/D2H batch (chunked if exceeded)

        struct RegisterPC {
            uint32_t count;
            int32_t baseX;
            int32_t baseY;
            int32_t baseZ;
        };

        struct ClassifyPC {
            uint32_t poolCapacity;
            int32_t baseX;
            int32_t baseY;
            int32_t baseZ;
        };
    } // namespace

    void StreamingResidencyBackend::Build(Engine::Core::Context &ctx, uint32_t poolCapacity) {
        m_ctx = &ctx;
        m_poolCapacity = poolCapacity;

        m_indexGrid = std::make_unique<Engine::Core::Buffer>(ctx);
        m_poolVoxels = std::make_unique<Engine::Core::Buffer>(ctx);
        m_metaBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_slotListBuffer = std::make_unique<Engine::Core::Buffer>(ctx);

        m_indexGrid->Allocate(kIndexGridCells * sizeof(uint32_t));
        m_poolVoxels->Allocate(poolCapacity * kGroupBytes);
        m_metaBuffer->Allocate(poolCapacity * sizeof(ActiveGroupMeta));
        m_slotListBuffer->Allocate(poolCapacity * sizeof(uint32_t));

        m_reusableList = std::make_unique<Engine::Core::Buffer>(ctx);
        m_cleanFreeList = std::make_unique<Engine::Core::Buffer>(ctx);
        m_writeBackList = std::make_unique<Engine::Core::Buffer>(ctx);
        m_countsBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_reusableList->Allocate(poolCapacity * sizeof(uint32_t));
        m_cleanFreeList->Allocate(poolCapacity * sizeof(uint32_t));
        m_writeBackList->Allocate(poolCapacity * sizeof(uint32_t));
        m_countsBuffer->Allocate(3u * sizeof(uint32_t));

        // Persistent host-visible staging (Phase 5): allocated once, reused every frame.
        using Engine::Compute::StagingBuffer;
        const VkBufferUsageFlags kSrc = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        const VkBufferUsageFlags kDst = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const VkBufferUsageFlags kSrcDst = kSrc | kDst;
        m_stageCounts = std::make_unique<StagingBuffer>(ctx, 3u * sizeof(uint32_t), kDst);
        m_stageLists = std::make_unique<StagingBuffer>(ctx, VkDeviceSize(poolCapacity) * sizeof(uint32_t), kDst);
        m_stageCleanFree = std::make_unique<StagingBuffer>(ctx, VkDeviceSize(poolCapacity) * sizeof(uint32_t), kDst);
        m_stageGroups = std::make_unique<StagingBuffer>(ctx, VkDeviceSize(kStageGroupCap) * kGroupBytes, kSrcDst);
        m_stageMeta = std::make_unique<StagingBuffer>(ctx, VkDeviceSize(poolCapacity) * sizeof(ActiveGroupMeta), kSrcDst);
        m_stageSlotList = std::make_unique<StagingBuffer>(ctx, VkDeviceSize(poolCapacity) * sizeof(uint32_t), kSrc);

        m_registerKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_registerKernel->Build("directional_tsdf_register_reusable.comp")
                .Bind(0, *m_slotListBuffer)
                .Bind(1, *m_metaBuffer)
                .Bind(2, *m_indexGrid);

        m_classifyKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_classifyKernel->Build("directional_tsdf_classify.comp")
                .Bind(0, *m_metaBuffer)
                .Bind(1, *m_reusableList)
                .Bind(2, *m_cleanFreeList)
                .Bind(3, *m_writeBackList)
                .Bind(4, *m_countsBuffer);

        // classify reads every slot's meta, so never-used slots must read as invalid.
        VkBuffer meta = m_metaBuffer->Handle();
        Engine::Core::SubmitOneShot(ctx, Engine::Core::QueueRole::Compute,
                                    [&](VkCommandBuffer cmd) {
                                        vkCmdFillBuffer(cmd, meta, 0, VK_WHOLE_SIZE, 0u);
                                    });

        m_slotKeys.assign(poolCapacity, {});
        m_residentIndex.clear();
        m_freeSlots.clear();
        m_requiredThisFrame.clear();
        m_stats = {};

        fillIndexGridInvalid();
    }

    void StreamingResidencyBackend::BeginFrame(const Eigen::Vector3i &localBase) {
        if (!m_ctx)
            throw std::runtime_error("StreamingResidencyBackend: Build() must be called first");

        m_localBase = localBase;
        m_stats = {};
        m_requiredThisFrame.clear();

        // Batch 1: reset indexGrid + counts → classify → copy counts + reusable + cleanFree
        // lists back, all in a single submit.
        {
            ClassifyPC cpc{m_poolCapacity, m_localBase.x(), m_localBase.y(), m_localBase.z()};
            m_classifyKernel->Args(cpc);

            std::vector<VkBufferCopy> countsRegion(1);
            countsRegion[0] = {0, 0, 3u * sizeof(uint32_t)};
            std::vector<VkBufferCopy> listRegion(1);
            listRegion[0] = {0, 0, VkDeviceSize(m_poolCapacity) * sizeof(uint32_t)};

            Engine::Compute::CommandBatch batch(*m_ctx);
            batch.FillBuffer(m_indexGrid->Handle(), 0, VK_WHOLE_SIZE, kInvalidPoolIndex);
            batch.FillBuffer(m_countsBuffer->Handle(), 0, VK_WHOLE_SIZE, 0u);
            batch.Barrier();
            batch.DispatchElements(*m_classifyKernel, m_poolCapacity);
            batch.Barrier();
            batch.CopyBuffer(m_countsBuffer->Handle(), m_stageCounts->Handle(), countsRegion);
            batch.CopyBuffer(m_reusableList->Handle(), m_stageLists->Handle(), listRegion);
            batch.CopyBuffer(m_cleanFreeList->Handle(), m_stageCleanFree->Handle(), listRegion);
            batch.Submit();
            ++m_stats.gpuSubmits;
        }

        uint32_t counts[3];
        std::memcpy(counts, m_stageCounts->Mapped(), sizeof(counts));
        m_stats.reusableCount = counts[0];
        m_stats.cleanFreeCount = counts[1];

        std::vector<uint32_t> reusable(counts[0]);
        if (counts[0] > 0)
            std::memcpy(reusable.data(), m_stageLists->Mapped(), counts[0] * sizeof(uint32_t));

        std::vector<uint32_t> cleanFree(counts[1]);
        if (counts[1] > 0)
            std::memcpy(cleanFree.data(), m_stageCleanFree->Mapped(), counts[1] * sizeof(uint32_t));
        m_freeSlots.assign(cleanFree.begin(), cleanFree.end());

        // Write-back (design doc §12): dirty groups that left the window → host store,
        // clear their meta, free their slots. Chunked by kStageGroupCap; guarantees
        // invariant #5 because it completes synchronously before any slot reuse.
        m_stats.writeBackCount = counts[2];
        if (counts[2] > 0) {
            std::vector<uint32_t> writeBack(counts[2]);
            {
                std::vector<VkBufferCopy> listRegion(1);
                listRegion[0] = {0, 0, VkDeviceSize(m_poolCapacity) * sizeof(uint32_t)};
                Engine::Compute::CommandBatch batch(*m_ctx);
                batch.CopyBuffer(m_writeBackList->Handle(), m_stageLists->Handle(), listRegion);
                batch.Submit();
                ++m_stats.gpuSubmits;
            }
            std::memcpy(writeBack.data(), m_stageLists->Mapped(), counts[2] * sizeof(uint32_t));

            const uint32_t groupBytes = kVoxelsPerGroup * uint32_t(sizeof(GpuTsdfVoxel));
            std::vector<ActiveGroupMeta> clearMeta(kStageGroupCap); // zero = invalid/Free

            for (uint32_t base = 0; base < counts[2]; base += kStageGroupCap) {
                const uint32_t chunk = std::min(kStageGroupCap, counts[2] - base);
                std::vector<VkBufferCopy> voxRegions(chunk), metaRegions(chunk);
                for (uint32_t i = 0; i < chunk; ++i) {
                    const uint32_t slot = writeBack[base + i];
                    voxRegions[i] = {VkDeviceSize(slot) * groupBytes, VkDeviceSize(i) * groupBytes, groupBytes};
                    metaRegions[i] = {VkDeviceSize(i) * sizeof(ActiveGroupMeta),
                                      VkDeviceSize(slot) * sizeof(ActiveGroupMeta), sizeof(ActiveGroupMeta)};
                }
                std::memcpy(m_stageMeta->Mapped(), clearMeta.data(), chunk * sizeof(ActiveGroupMeta));

                Engine::Compute::CommandBatch batch(*m_ctx);
                batch.CopyBuffer(m_poolVoxels->Handle(), m_stageGroups->Handle(), voxRegions);
                batch.CopyBuffer(m_stageMeta->Handle(), m_metaBuffer->Handle(), metaRegions);
                batch.Submit();
                ++m_stats.gpuSubmits;

                const auto *raw = static_cast<const GpuTsdfVoxel *>(m_stageGroups->Mapped());
                for (uint32_t i = 0; i < chunk; ++i) {
                    const uint32_t slot = writeBack[base + i];
                    DirectionalHostStore::Group group{};
                    for (uint32_t v = 0; v < kVoxelsPerGroup; ++v)
                        group[v] = GpuVoxelToHost(raw[size_t(i) * kVoxelsPerGroup + v]);
                    m_hostStore.Put(m_slotKeys[slot], group);
                    m_freeSlots.push_back(slot);
                }
                m_stats.d2hBytes += chunk * groupBytes;
            }
        }

        // Reusable slots are re-registered into the indexGrid inside the EnsureResident /
        // Integrate batch (RecordResidency), together with the missing slots, so the
        // register dispatch runs once. Missing detection uses the CPU mirror below, not the
        // GPU indexGrid — invariant #6 holds at the mirror level.
        m_residentIndex.clear();
        for (uint32_t slot : reusable)
            m_residentIndex.emplace(m_slotKeys[slot], slot);
        m_stats.residentCount = uint32_t(m_residentIndex.size());
        m_reusableSlots.assign(reusable.begin(), reusable.end());
    }

    uint32_t StreamingResidencyBackend::RecordResidency(const std::vector<DirectionalGroupKey> &required,
                                                        Engine::Compute::CommandBatch &batch) {
        struct Pending {
            DirectionalGroupKey key;
            uint32_t slot;
        };
        std::vector<Pending> pending;
        pending.reserve(required.size());

        for (const auto &key : required) {
            const int lx = key.gx - m_localBase.x();
            const int ly = key.gy - m_localBase.y();
            const int lz = key.gz - m_localBase.z();
            if (lx < 0 || ly < 0 || lz < 0 ||
                lx >= int(kLocalGroupGrid) || ly >= int(kLocalGroupGrid) ||
                lz >= int(kLocalGroupGrid))
                throw std::runtime_error("StreamingResidencyBackend: required group is outside the local window");
            m_requiredThisFrame.insert(key);
            if (m_residentIndex.find(key) != m_residentIndex.end())
                continue;
            if (m_freeSlots.empty())
                throw std::runtime_error("StreamingResidencyBackend: active pool exhausted");
            const uint32_t slot = m_freeSlots.back();
            m_freeSlots.pop_back();
            m_residentIndex.emplace(key, slot);
            m_slotKeys[slot] = key;
            pending.push_back({key, slot});
        }
        m_stats.residentCount = uint32_t(m_residentIndex.size());

        if (pending.size() > kStageGroupCap)
            throw std::runtime_error("StreamingResidencyBackend: missing groups exceed staging cap (chunking TODO)");

        const uint32_t groupBytes = kVoxelsPerGroup * uint32_t(sizeof(GpuTsdfVoxel));
        auto *voxStage = static_cast<GpuTsdfVoxel *>(m_stageGroups->Mapped());
        auto *metaStage = static_cast<ActiveGroupMeta *>(m_stageMeta->Mapped());
        auto *slotStage = static_cast<uint32_t *>(m_stageSlotList->Mapped());

        // Combined register list: reusable slots (already have valid GPU meta) + missing
        // slots (meta uploaded below). One register dispatch handles both.
        uint32_t regCount = 0;
        for (uint32_t s : m_reusableSlots) slotStage[regCount++] = s;

        std::vector<VkBufferCopy> voxRegions(pending.size()), metaRegions(pending.size());
        for (size_t i = 0; i < pending.size(); ++i) {
            const auto &pk = pending[i];
            const auto &group = m_hostStore.GetOrCreate(pk.key);
            for (uint32_t v = 0; v < kVoxelsPerGroup; ++v) {
                voxStage[i * kVoxelsPerGroup + v] = HostVoxelToGpu(group[v]);
            }
            metaStage[i] = ActiveGroupMeta{pk.key.gx, pk.key.gy, pk.key.gz,
                                           PackMeta(pk.key.direction, SlotState::ResidentClean, false, true)};
            voxRegions[i] = {VkDeviceSize(i) * groupBytes, VkDeviceSize(pk.slot) * groupBytes, groupBytes};
            metaRegions[i] = {VkDeviceSize(i) * sizeof(ActiveGroupMeta),
                              VkDeviceSize(pk.slot) * sizeof(ActiveGroupMeta), sizeof(ActiveGroupMeta)};
            slotStage[regCount++] = pk.slot;
        }
        if (!pending.empty()) {
            batch.CopyBuffer(m_stageGroups->Handle(), m_poolVoxels->Handle(), voxRegions);
            batch.CopyBuffer(m_stageMeta->Handle(), m_metaBuffer->Handle(), metaRegions);
            m_stats.h2dBytes += uint32_t(pending.size()) * groupBytes;
        }
        m_stats.missingCount += uint32_t(pending.size());

        if (regCount > 0) {
            std::vector<VkBufferCopy> slotRegion(1);
            slotRegion[0] = {0, 0, VkDeviceSize(regCount) * sizeof(uint32_t)};
            batch.CopyBuffer(m_stageSlotList->Handle(), m_slotListBuffer->Handle(), slotRegion);
            batch.Barrier();
            RegisterPC rpc{regCount, m_localBase.x(), m_localBase.y(), m_localBase.z()};
            m_registerKernel->Args(rpc).Bind(0, *m_slotListBuffer);
            batch.DispatchElements(*m_registerKernel, regCount);
        }
        // Overlap ratio depends only on m_requiredThisFrame/m_stats.missingCount, both
        // finalized above; computing it here (rather than requiring a separate external
        // call) keeps FrameStats().overlapRatio correct for every RecordResidency caller,
        // including DirectionalTSDF::Integrate's combined batch which never calls
        // EnsureResident().
        updateOverlapRatio();
        return regCount;
    }

    void StreamingResidencyBackend::EnsureResident(const std::vector<DirectionalGroupKey> &required) {
        if (!m_ctx)
            throw std::runtime_error("StreamingResidencyBackend: Build() must be called first");
        Engine::Compute::CommandBatch batch(*m_ctx);
        const uint32_t reg = RecordResidency(required, batch);
        if (reg > 0) {
            batch.Submit();
            ++m_stats.gpuSubmits;
        }
    }

    void StreamingResidencyBackend::EndFrame() {
        // Write-back is already folded into BeginFrame's classify pass (this repo's Phase 4
        // dirty write-back/eviction, done up front on window shift) — see RecordResidency
        // and the write-back block in BeginFrame. Nothing left to do at end-of-frame.
    }

    void StreamingResidencyBackend::updateOverlapRatio() {
        const size_t requiredUnique = m_requiredThisFrame.size();
        m_stats.overlapRatio =
                requiredUnique == 0
                        ? 0.0f
                        : 1.0f - float(m_stats.missingCount) / float(requiredUnique);
    }

    std::vector<uint32_t> StreamingResidencyBackend::DebugDownloadIndexGrid() {
        std::vector<uint32_t> grid(kIndexGridCells);
        m_indexGrid->Download(grid.data(), kIndexGridCells * sizeof(uint32_t));
        return grid;
    }

    uint32_t StreamingResidencyBackend::DebugQueryPoolIndex(const DirectionalGroupKey &key) {
        const int lx = key.gx - m_localBase.x();
        const int ly = key.gy - m_localBase.y();
        const int lz = key.gz - m_localBase.z();
        if (lx < 0 || ly < 0 || lz < 0 ||
            lx >= int(kLocalGroupGrid) || ly >= int(kLocalGroupGrid) ||
            lz >= int(kLocalGroupGrid))
            return kInvalidPoolIndex;

        auto grid = DebugDownloadIndexGrid();
        return grid[IndexGridOffset(uint32_t(lx), uint32_t(ly), uint32_t(lz), key.direction)];
    }

    DirectionalHostStore::Group
    StreamingResidencyBackend::DebugDownloadGroupVoxels(const DirectionalGroupKey &key) {
        auto it = m_residentIndex.find(key);
        if (it == m_residentIndex.end())
            throw std::runtime_error("StreamingResidencyBackend: group is not resident");
        const uint32_t slot = it->second;

        Engine::Core::Buffer staging(*m_ctx);
        staging.Allocate(kGroupBytes);
        VkBuffer src = m_poolVoxels->Handle();
        VkBuffer dst = staging.Handle();
        Engine::Core::SubmitOneShot(*m_ctx, Engine::Core::QueueRole::Compute,
                                    [&](VkCommandBuffer cmd) {
                                        VkBufferCopy region{};
                                        region.srcOffset = VkDeviceSize(slot) * kGroupBytes;
                                        region.dstOffset = 0;
                                        region.size = kGroupBytes;
                                        vkCmdCopyBuffer(cmd, src, dst, 1, &region);
                                    });

        std::vector<GpuTsdfVoxel> raw(kVoxelsPerGroup);
        staging.Download(raw.data(), kGroupBytes);

        DirectionalHostStore::Group group{};
        for (uint32_t v = 0; v < kVoxelsPerGroup; ++v)
            group[v] = GpuVoxelToHost(raw[v]);
        return group;
    }

    void StreamingResidencyBackend::fillIndexGridInvalid() {
        VkBuffer grid = m_indexGrid->Handle();
        Engine::Core::SubmitOneShot(*m_ctx, Engine::Core::QueueRole::Compute,
                                    [&](VkCommandBuffer cmd) {
                                        vkCmdFillBuffer(cmd, grid, 0, VK_WHOLE_SIZE,
                                                        kInvalidPoolIndex);
                                    });
    }

} // namespace Engine::Spatial

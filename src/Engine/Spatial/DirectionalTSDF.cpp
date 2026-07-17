#include "Engine/Spatial/DirectionalTSDF.h"

#include "Engine/Core/OneShotCommands.h"

#include <cmath>
#include <stdexcept>

namespace Engine::Spatial {

    namespace {
        constexpr uint32_t kGroupBytes = kVoxelsPerGroup * uint32_t(sizeof(GpuTsdfVoxel)); // 4096

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

    void DirectionalTSDF::Build(Engine::Core::Context &ctx,
                                float voxelSize,
                                float truncation,
                                uint32_t poolCapacity) {
        m_ctx = &ctx;
        m_voxelSize = voxelSize;
        m_truncation = truncation;
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
        m_lastCounts = {};

        fillIndexGridInvalid();
    }

    void DirectionalTSDF::BeginFrame(const Eigen::Vector3f &aabbCenterHint) {
        if (!m_ctx)
            throw std::runtime_error("DirectionalTSDF: Build() must be called first");

        m_localBase = quantizeLocalBase(aabbCenterHint);
        m_stats = {};
        m_requiredThisFrame.clear();

        fillIndexGridInvalid();

        // 1. Classify every pool slot against the new local base (design doc §9).
        const uint32_t zeros[3] = {0, 0, 0};
        m_countsBuffer->Upload(zeros, sizeof(zeros));
        ClassifyPC cpc{m_poolCapacity, m_localBase.x(), m_localBase.y(), m_localBase.z()};
        m_classifyKernel->Args(cpc).DispatchElements(m_poolCapacity);

        // 2. Download the three lists (synchronous; the pool is small).
        uint32_t counts[3] = {0, 0, 0};
        m_countsBuffer->Download(counts, sizeof(counts));
        m_lastCounts = {counts[0], counts[1], counts[2]};

        std::vector<uint32_t> reusable(counts[0]);
        if (counts[0] > 0)
            m_reusableList->Download(reusable.data(), counts[0] * sizeof(uint32_t));

        std::vector<uint32_t> cleanFree(counts[1]);
        if (counts[1] > 0)
            m_cleanFreeList->Download(cleanFree.data(), counts[1] * sizeof(uint32_t));
        m_freeSlots.assign(cleanFree.begin(), cleanFree.end());

        // WriteBackList is produced for Phase 4; nothing marks dirty until integration.
        m_stats.writeBackCount = counts[2];

        // 3. Re-register reusable slots into the freshly-reset indexGrid. Invariant #6:
        //    this must complete before any missing-group decision is made.
        if (counts[0] > 0) {
            m_registerKernel->Bind(0, *m_reusableList);
            RegisterPC rpc{counts[0], m_localBase.x(), m_localBase.y(), m_localBase.z()};
            m_registerKernel->Args(rpc).DispatchElements(counts[0]);
        }

        // 4. Rebuild the CPU resident mirror from the reusable set.
        m_residentIndex.clear();
        for (uint32_t slot : reusable)
            m_residentIndex.emplace(m_slotKeys[slot], slot);
        m_stats.residentCount = uint32_t(m_residentIndex.size());
    }

    void DirectionalTSDF::EnsureResident(const std::vector<DirectionalGroupKey> &required) {
        if (!m_ctx)
            throw std::runtime_error("DirectionalTSDF: Build() must be called first");

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
                throw std::runtime_error(
                        "DirectionalTSDF: required group is outside the local window");

            m_requiredThisFrame.insert(key);

            if (m_residentIndex.find(key) != m_residentIndex.end())
                continue;
            if (m_freeSlots.empty())
                throw std::runtime_error("DirectionalTSDF: active pool exhausted");

            const uint32_t slot = m_freeSlots.back();
            m_freeSlots.pop_back();
            m_residentIndex.emplace(key, slot);
            m_slotKeys[slot] = key;
            pending.push_back({key, slot});
        }
        m_stats.residentCount = uint32_t(m_residentIndex.size());
        if (pending.empty()) {
            updateOverlapRatio();
            return;
        }

        // Encode host groups into the GPU fixed-point format and build their meta entries.
        std::vector<GpuTsdfVoxel> voxelData(pending.size() * kVoxelsPerGroup);
        std::vector<ActiveGroupMeta> metaData(pending.size());
        for (size_t i = 0; i < pending.size(); ++i) {
            const auto &key = pending[i].key;
            const auto &group = m_hostStore.GetOrCreate(key);
            for (uint32_t v = 0; v < kVoxelsPerGroup; ++v) {
                const HostTsdfVoxel &h = group[v];
                GpuTsdfVoxel &g = voxelData[i * kVoxelsPerGroup + v];
                g.sumW = uint32_t(std::lround(double(h.weight) * kTsdfFixedScale));
                g.sumDW = int32_t(
                        std::lround(double(h.value) * double(h.weight) * kTsdfFixedScale));
            }
            metaData[i] = ActiveGroupMeta{key.gx, key.gy, key.gz,
                                          PackMeta(key.direction, SlotState::ResidentClean,
                                                   /*dirty=*/false, /*valid=*/true)};
        }

        // Voxel payloads → scattered pool slots: staging buffer + one multi-region copy.
        {
            const uint32_t bytes = uint32_t(voxelData.size() * sizeof(GpuTsdfVoxel));
            Engine::Core::Buffer staging(*m_ctx);
            staging.Allocate(bytes);
            staging.Upload(voxelData.data(), bytes);

            std::vector<VkBufferCopy> regions(pending.size());
            for (size_t i = 0; i < pending.size(); ++i) {
                regions[i].srcOffset = VkDeviceSize(i) * kGroupBytes;
                regions[i].dstOffset = VkDeviceSize(pending[i].slot) * kGroupBytes;
                regions[i].size = kGroupBytes;
            }
            VkBuffer src = staging.Handle();
            VkBuffer dst = m_poolVoxels->Handle();
            Engine::Core::SubmitOneShot(*m_ctx, Engine::Core::QueueRole::Compute,
                                        [&](VkCommandBuffer cmd) {
                                            vkCmdCopyBuffer(cmd, src, dst,
                                                            uint32_t(regions.size()),
                                                            regions.data());
                                        });
            m_stats.h2dBytes += bytes;
        }

        // Meta entries → scattered meta slots: same staging + multi-region pattern.
        {
            const uint32_t bytes = uint32_t(metaData.size() * sizeof(ActiveGroupMeta));
            Engine::Core::Buffer staging(*m_ctx);
            staging.Allocate(bytes);
            staging.Upload(metaData.data(), bytes);

            std::vector<VkBufferCopy> regions(pending.size());
            for (size_t i = 0; i < pending.size(); ++i) {
                regions[i].srcOffset = VkDeviceSize(i) * sizeof(ActiveGroupMeta);
                regions[i].dstOffset = VkDeviceSize(pending[i].slot) * sizeof(ActiveGroupMeta);
                regions[i].size = sizeof(ActiveGroupMeta);
            }
            VkBuffer src = staging.Handle();
            VkBuffer dst = m_metaBuffer->Handle();
            Engine::Core::SubmitOneShot(*m_ctx, Engine::Core::QueueRole::Compute,
                                        [&](VkCommandBuffer cmd) {
                                            vkCmdCopyBuffer(cmd, src, dst,
                                                            uint32_t(regions.size()),
                                                            regions.data());
                                        });
        }

        // Register the freshly-filled slots in the indexGrid.
        std::vector<uint32_t> slots(pending.size());
        for (size_t i = 0; i < pending.size(); ++i)
            slots[i] = pending[i].slot;
        m_slotListBuffer->Upload(slots.data(), uint32_t(slots.size() * sizeof(uint32_t)));

        m_registerKernel->Bind(0, *m_slotListBuffer);
        RegisterPC pc{uint32_t(slots.size()), m_localBase.x(), m_localBase.y(),
                      m_localBase.z()};
        m_registerKernel->Args(pc).DispatchElements(uint32_t(slots.size()));

        m_stats.missingCount += uint32_t(pending.size());
        updateOverlapRatio();
    }

    void DirectionalTSDF::updateOverlapRatio() {
        const size_t requiredUnique = m_requiredThisFrame.size();
        m_stats.overlapRatio =
                requiredUnique == 0
                        ? 0.0f
                        : 1.0f - float(m_stats.missingCount) / float(requiredUnique);
    }

    std::vector<uint32_t> DirectionalTSDF::DebugDownloadIndexGrid() {
        std::vector<uint32_t> grid(kIndexGridCells);
        m_indexGrid->Download(grid.data(), kIndexGridCells * sizeof(uint32_t));
        return grid;
    }

    uint32_t DirectionalTSDF::DebugQueryPoolIndex(const DirectionalGroupKey &key) {
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
    DirectionalTSDF::DebugDownloadGroupVoxels(const DirectionalGroupKey &key) {
        auto it = m_residentIndex.find(key);
        if (it == m_residentIndex.end())
            throw std::runtime_error("DirectionalTSDF: group is not resident");
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
        for (uint32_t v = 0; v < kVoxelsPerGroup; ++v) {
            group[v].weight = float(raw[v].sumW) / float(kTsdfFixedScale);
            group[v].value = raw[v].sumW > 0
                                     ? float(double(raw[v].sumDW) / double(raw[v].sumW))
                                     : 0.0f;
        }
        return group;
    }

    void DirectionalTSDF::fillIndexGridInvalid() {
        VkBuffer grid = m_indexGrid->Handle();
        Engine::Core::SubmitOneShot(*m_ctx, Engine::Core::QueueRole::Compute,
                                    [&](VkCommandBuffer cmd) {
                                        vkCmdFillBuffer(cmd, grid, 0, VK_WHOLE_SIZE,
                                                        kInvalidPoolIndex);
                                    });
    }

    Eigen::Vector3i DirectionalTSDF::quantizeLocalBase(const Eigen::Vector3f &center) const {
        const float g = GroupWorldSize();
        const int half = int(kLocalGroupGrid) / 2;
        return Eigen::Vector3i(int(std::floor(center.x() / g)) - half,
                               int(std::floor(center.y() / g)) - half,
                               int(std::floor(center.z() / g)) - half);
    }

} // namespace Engine::Spatial

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

        m_registerKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_registerKernel->Build("directional_tsdf_register_reusable.comp")
                .Bind(0, *m_slotListBuffer)
                .Bind(1, *m_metaBuffer)
                .Bind(2, *m_indexGrid);

        m_slotKeys.assign(poolCapacity, {});
        m_residentIndex.clear();
        m_nextFreeSlot = 0;
        m_stats = {};

        fillIndexGridInvalid();
    }

    void DirectionalTSDF::BeginFrame(const Eigen::Vector3f &aabbCenterHint) {
        if (!m_ctx)
            throw std::runtime_error("DirectionalTSDF: Build() must be called first");

        m_localBase = quantizeLocalBase(aabbCenterHint);

        // Phase 1: full-reload semantics — every frame starts from an empty pool.
        // Phase 2 replaces this with device-side classification + reusable re-registration.
        m_residentIndex.clear();
        m_slotKeys.assign(m_poolCapacity, {});
        m_nextFreeSlot = 0;
        m_stats = {};

        fillIndexGridInvalid();
    }

    void DirectionalTSDF::EnsureResident(const std::vector<DirectionalGroupKey> &required) {
        (void) required;
        throw std::runtime_error("DirectionalTSDF: EnsureResident not implemented yet");
    }

    std::vector<uint32_t> DirectionalTSDF::DebugDownloadIndexGrid() {
        std::vector<uint32_t> grid(kIndexGridCells);
        m_indexGrid->Download(grid.data(), kIndexGridCells * sizeof(uint32_t));
        return grid;
    }

    uint32_t DirectionalTSDF::DebugQueryPoolIndex(const DirectionalGroupKey &key) {
        (void) key;
        throw std::runtime_error("DirectionalTSDF: DebugQueryPoolIndex not implemented yet");
    }

    DirectionalHostStore::Group
    DirectionalTSDF::DebugDownloadGroupVoxels(const DirectionalGroupKey &key) {
        (void) key;
        throw std::runtime_error("DirectionalTSDF: DebugDownloadGroupVoxels not implemented yet");
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

#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalHostStore.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"

#include <Eigen/Core>
#include <memory>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Spatial {

    // Residency-side counters the core aggregates into DirectionalTSDF::Stats.
    struct ResidencyStats {
        uint32_t residentCount = 0;
        uint32_t missingCount = 0;
        uint32_t writeBackCount = 0;
        uint32_t h2dBytes = 0;
        uint32_t d2hBytes = 0;
        float overlapRatio = 0.0f;
        uint32_t reusableCount = 0;   // classify: resident & inside new window
        uint32_t cleanFreeCount = 0;  // classify: free or clean-evicted slots
        uint32_t gpuSubmits = 0;      // queue submissions the backend made this frame
    };

    // Abstracts "where a directional group lives and how it becomes device-addressable".
    // Streaming (discrete/PCIe) and Unified (UMA/zero-copy) implement this identically to
    // the core, which never issues vkCmdCopyBuffer/vkCmdFillBuffer/allocations itself.
    class IResidencyBackend {
    public:
        virtual ~IResidencyBackend() = default;

        // Reset the indexGrid for `localBase`, classify the pool, re-register reusable slots.
        virtual void BeginFrame(const Eigen::Vector3i &localBase) = 0;
        // Make `required` (write-set + halo) resident. Streaming: upload missing. Unified: no-op.
        virtual void EnsureResident(const std::vector<DirectionalGroupKey> &required) = 0;
        // End-of-frame cleanup. Streaming: dirty write-back + eviction. Unified: no-op (or cold compact).
        virtual void EndFrame() = 0;

        // Fold this frame's residency work into an existing command batch (Phase-5 batched path).
        // Streaming: stage missing groups + record copies/register into `batch`; returns slots recorded.
        // Unified: CPU slot bookkeeping only, records nothing to `batch`; returns slots touched.
        virtual uint32_t RecordResidency(const std::vector<DirectionalGroupKey> &required,
                                         Engine::Compute::CommandBatch &batch) = 0;
        // Enumerate resident (key -> pool slot) so the core can pick recompute slots for extraction.
        virtual const std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> &
        ResidentIndex() const = 0;

        // GPU buffers the integrate/extract kernels bind (owned by the backend).
        virtual VkBuffer IndexGridBuffer() const = 0;
        virtual VkBuffer PoolVoxelBuffer() const = 0;
        virtual VkBuffer MetaBuffer() const = 0;
        virtual uint32_t PoolCapacity() const = 0;
        virtual Eigen::Vector3i LocalBase() const = 0;

        // Capability flags (core uses these to gate quality bonuses in later plans).
        virtual bool IsUnified() const = 0;
        virtual bool SupportsZeroCopyCpuAccess() const = 0;

        // Authoritative group store (streaming: the map; unified: a view over the pool).
        // Used by demos to seed geometry and by debug downloads.
        virtual DirectionalHostStore &HostStore() = 0;

        // Residency counters for the frame just processed.
        virtual ResidencyStats FrameStats() const = 0;

        // Synchronous debug/test downloads (not for per-frame use).
        virtual std::vector<uint32_t> DebugDownloadIndexGrid() = 0;
        virtual uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &key) = 0;
        virtual DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &key) = 0;
    };

    // Explicit backend choice. Auto probes memory topology (UMA→Unified, else Streaming).
    enum class ResidencyMode { Auto, Streaming, Unified };

    // Builds the requested backend. `Auto` chooses UnifiedResidencyBackend when the device
    // exposes a DEVICE_LOCAL|HOST_VISIBLE heap large enough for the model, else Streaming.
    // The env var VKLBVH_RESIDENCY=streaming|unified overrides `mode` at runtime (escape hatch;
    // defined in ResidencyBackendFactory.cpp, Task 5).
    std::unique_ptr<IResidencyBackend>
    MakeResidencyBackend(Engine::Core::Context &ctx, uint32_t poolCapacity,
                         ResidencyMode mode = ResidencyMode::Auto);

} // namespace Engine::Spatial

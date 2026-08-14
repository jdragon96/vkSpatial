#include "TSDF/Backends/Residency/IResidencyBackend.h"
#include "TSDF/Backends/Residency/StreamingResidencyBackend.h"
#include "TSDF/Backends/Residency/UnifiedResidencyBackend.h"

#include <cstdlib>
#include <cstring>

namespace TSDF {

    static bool hasUnifiedHeap(VkPhysicalDevice dev, VkDeviceSize needed) {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(dev, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            bool devLocal = f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            bool hostVis = f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            if (devLocal && hostVis &&
                mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size >= needed)
                return true;
        }
        return false;
    }

    std::unique_ptr<IResidencyBackend>
    MakeResidencyBackend(Engine::Core::Context &ctx, uint32_t poolCapacity, ResidencyMode mode) {
        const char *ov = std::getenv("VKLBVH_RESIDENCY");
        VkDeviceSize needed = VkDeviceSize(sizeof(GpuTsdfVoxel)) * kVoxelsPerGroup * poolCapacity +
                              VkDeviceSize(sizeof(uint32_t)) * kIndexGridCells +
                              VkDeviceSize(sizeof(ActiveGroupMeta)) * poolCapacity;

        bool useUnified;
        if (ov && std::strcmp(ov, "unified") == 0)         useUnified = true;   // env escape hatch wins
        else if (ov && std::strcmp(ov, "streaming") == 0)  useUnified = false;
        else if (mode == ResidencyMode::Unified)           useUnified = true;   // explicit request
        else if (mode == ResidencyMode::Streaming)         useUnified = false;
        else useUnified = hasUnifiedHeap(ctx.physicalDevice, needed);           // Auto: probe topology

        if (useUnified) {
            auto b = std::make_unique<UnifiedResidencyBackend>();
            b->Build(ctx, poolCapacity);
            return b;
        }
        auto b = std::make_unique<StreamingResidencyBackend>();
        b->Build(ctx, poolCapacity);
        return b;
    }

} // namespace TSDF

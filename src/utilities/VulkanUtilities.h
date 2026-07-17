#pragma once

#include <optional>
#include <vulkan/vulkan.h>

class VulkanUtilities {
public:
    static std::optional<uint32_t> FindMemoryType(
            VkPhysicalDevice physicalDevice,
            uint32_t typeFilter,
            VkMemoryPropertyFlags properties) {

        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1u << i)) &&
                (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }

        return std::nullopt;
    }
};
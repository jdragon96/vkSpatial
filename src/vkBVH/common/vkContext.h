#pragma once

#include <vulkan/vulkan.h>

class VkContext {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeFamily = 0;
    VkCommandPool cmdPool = VK_NULL_HANDLE;

    void init();
    void shutdown();

private:
    void createInstance();
    void pickPhysicalDevice();
    void createDevice();
    void createCommandPool();
};

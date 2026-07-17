#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <functional>
#include <vector>

namespace Engine::Core {

    class Context {
    public:
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;

        VkQueue computeQueue = VK_NULL_HANDLE;
        uint32_t computeFamily = 0;
        VkCommandPool cmdPool = VK_NULL_HANDLE;

        // Populated only when enablePresent=true.
        VkQueue graphicsQueue = VK_NULL_HANDLE;
        uint32_t graphicsFamily = 0;
        VkCommandPool graphicsCmdPool = VK_NULL_HANDLE;
        VkSurfaceKHR surface = VK_NULL_HANDLE;

        VmaAllocator allocator = VK_NULL_HANDLE;

        explicit Context(bool enablePresent = false,
                         const std::vector<const char *> &extraInstanceExtensions = {},
                         const std::function<VkSurfaceKHR(VkInstance)> &surfaceFactory = nullptr);
        ~Context();

        Context(const Context &) = delete;
        Context &operator=(const Context &) = delete;

    private:
        bool m_presentEnabled = false;

        void createCommandPools();
        void createAllocator();
        void cleanup();
    };

} // namespace Engine::Core

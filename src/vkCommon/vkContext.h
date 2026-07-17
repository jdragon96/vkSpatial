#pragma once

#include <vulkan/vulkan.h>

#include <functional>
#include <vector>

namespace vkCommon {

    class VkContext {
    public:
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue computeQueue = VK_NULL_HANDLE;
        uint32_t computeFamily = 0;
        VkCommandPool cmdPool = VK_NULL_HANDLE;

        // enablePresent=true일 때만 채워짐 (창 렌더링용 그래픽스/프레젠트 큐 + surface)
        VkQueue graphicsQueue = VK_NULL_HANDLE;
        uint32_t graphicsFamily = 0;
        VkSurfaceKHR surface = VK_NULL_HANDLE;

        void init(bool enablePresent = false,
                  const std::vector<const char *> &extraInstanceExtensions = {},
                  const std::function<VkSurfaceKHR(VkInstance)> &surfaceFactory = nullptr);
        void shutdown();

    private:
        void createCommandPool();

        bool m_presentEnabled = false;
    };

} // namespace vkCommon

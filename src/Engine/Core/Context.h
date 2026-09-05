#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <functional>
#include <mutex>
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

        // Vulkan requires EXTERNAL SYNCHRONISATION of a VkCommandPool and a VkQueue, and that
        // covers recording too -- vkBeginCommandBuffer and every vkCmd* on a buffer from the pool,
        // not just allocation. This Context hands out one compute pool and one compute queue, and
        // the reconstruction pipeline already drives them from two threads (RegistrationThread
        // through CommandBatch, IntegrationThread through TSDF's ComputePipeline dispatches). A
        // third arrives with the GPU depth front end.
        //
        // Everything funnels through two places -- SubmitOneShot and CommandBatch -- so one lock
        // taken there covers the whole surface.
        //
        // Costs nothing worth having: compute is ONE queue, so the GPU serialises this work
        // regardless. What the lock serialises is command RECORDING, which is CPU-cheap.
        //
        // Recursive because the nesting is legitimate: a caller may Upload a buffer, which submits
        // its own one-shot, while a CommandBatch of its own is open and holding the lock.
        std::recursive_mutex submissionMutex;

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

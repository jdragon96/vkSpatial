#include "vkContext.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace vkCommon {

    void VkContext::init(bool enablePresent,
                          const std::vector<const char *> &extraInstanceExtensions,
                          const std::function<VkSurfaceKHR(VkInstance)> &surfaceFactory) {
        m_presentEnabled = enablePresent;

        createInstance(enablePresent, extraInstanceExtensions);

        if (enablePresent) {
            if (!surfaceFactory)
                throw std::runtime_error(
                        "VkContext: enablePresent requires a surfaceFactory");
            surface = surfaceFactory(instance);
            if (surface == VK_NULL_HANDLE)
                throw std::runtime_error(
                        "VkContext: surfaceFactory returned VK_NULL_HANDLE");
        }

        pickPhysicalDevice();
        createDevice(enablePresent);
        createCommandPool();
    }

    void VkContext::shutdown() {
        if (cmdPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, cmdPool, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
        cmdPool = VK_NULL_HANDLE;
        device = VK_NULL_HANDLE;
        surface = VK_NULL_HANDLE;
        instance = VK_NULL_HANDLE;
    }

    void VkContext::createInstance(
            bool enablePresent,
            const std::vector<const char *> &extraInstanceExtensions) {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.apiVersion = VK_API_VERSION_1_3;

        std::vector<const char *> exts;
#ifdef __APPLE__
        exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif
        if (enablePresent)
            exts.insert(exts.end(),
                        extraInstanceExtensions.begin(),
                        extraInstanceExtensions.end());

        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &appInfo;
        ici.enabledExtensionCount = static_cast<uint32_t>(exts.size());
        ici.ppEnabledExtensionNames = exts.data();
#ifdef __APPLE__
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS)
            throw std::runtime_error("VkContext: failed to create VkInstance");
    }

    void VkContext::pickPhysicalDevice() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (count == 0)
            throw std::runtime_error("VkContext: no Vulkan-capable device found");

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        physDevice = devices[0];

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physDevice, &props);
        std::cout << "[VkContext] Device: " << props.deviceName << "\n";
    }

    void VkContext::createDevice(bool enablePresent) {
        uint32_t qc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qc, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qc);
        vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qc, qprops.data());

        auto supportsPresent = [&](uint32_t family) -> bool {
            if (!enablePresent || surface == VK_NULL_HANDLE) return false;
            VkBool32 supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physDevice, family, surface, &supported);
            return supported == VK_TRUE;
        };

        bool foundCompute = false;
        bool foundGraphics = false;

        if (enablePresent) {
            // 우선 compute+graphics+present를 모두 지원하는 단일 큐 패밀리를 찾는다
            // (Apple/MoltenVK 등 통합 큐 패밀리 환경에서의 일반적인 경우).
            for (uint32_t i = 0; i < qc; i++) {
                const bool hasCompute = (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
                const bool hasGraphics = (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
                if (hasCompute && hasGraphics && supportsPresent(i)) {
                    computeFamily = i;
                    graphicsFamily = i;
                    foundCompute = true;
                    foundGraphics = true;
                    break;
                }
            }
        }

        if (!foundCompute || (enablePresent && !foundGraphics)) {
            for (uint32_t i = 0; i < qc; i++) {
                if (!foundCompute && (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    computeFamily = i;
                    foundCompute = true;
                }
                if (enablePresent && !foundGraphics &&
                    (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 &&
                    supportsPresent(i)) {
                    graphicsFamily = i;
                    foundGraphics = true;
                }
            }
        }

        if (!foundCompute)
            throw std::runtime_error("VkContext: no compute queue family found");
        if (enablePresent && !foundGraphics)
            throw std::runtime_error("VkContext: no graphics+present queue family found");

        const float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;

        VkDeviceQueueCreateInfo computeQci{};
        computeQci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        computeQci.queueFamilyIndex = computeFamily;
        computeQci.queueCount = 1;
        computeQci.pQueuePriorities = &priority;
        qcis.push_back(computeQci);

        if (enablePresent && graphicsFamily != computeFamily) {
            VkDeviceQueueCreateInfo graphicsQci{};
            graphicsQci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            graphicsQci.queueFamilyIndex = graphicsFamily;
            graphicsQci.queueCount = 1;
            graphicsQci.pQueuePriorities = &priority;
            qcis.push_back(graphicsQci);
        }

        std::vector<const char *> devExts;
#ifdef __APPLE__
        devExts.push_back("VK_KHR_portability_subset");
#endif
        if (enablePresent)
            devExts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

        VkPhysicalDeviceVulkan13Features features13{};
        features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        features13.dynamicRendering = VK_TRUE;

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
        dci.pQueueCreateInfos = qcis.data();
        dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
        dci.ppEnabledExtensionNames = devExts.data();
        if (enablePresent)
            dci.pNext = &features13;

        if (vkCreateDevice(physDevice, &dci, nullptr, &device) != VK_SUCCESS)
            throw std::runtime_error("VkContext: failed to create VkDevice");

        vkGetDeviceQueue(device, computeFamily, 0, &computeQueue);
        if (enablePresent)
            vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
    }

    void VkContext::createCommandPool() {
        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = computeFamily;

        if (vkCreateCommandPool(device, &pci, nullptr, &cmdPool) != VK_SUCCESS)
            throw std::runtime_error("VkContext: failed to create VkCommandPool");
    }

} // namespace vkCommon

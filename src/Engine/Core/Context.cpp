#include "Engine/Core/Context.h"

#include "VkBootstrap.h"

#include <iostream>
#include <stdexcept>

namespace Engine::Core {

    Context::Context(bool enablePresent,
                     const std::vector<const char *> &extraInstanceExtensions,
                     const std::function<VkSurfaceKHR(VkInstance)> &surfaceFactory) {
        try {
            m_presentEnabled = enablePresent;

            vkb::InstanceBuilder instanceBuilder;
            instanceBuilder.set_app_name("vkbvh")
                           .require_api_version(1, 3, 0);

            if (enablePresent)
                for (const char *ext : extraInstanceExtensions)
                    instanceBuilder.enable_extension(ext);

            // This vk-bootstrap version has no InstanceBuilder::enable_extension_if_present().
            // Its InstanceBuilder::build() already auto-detects and enables
            // VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME (and sets
            // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR) internally when the extension is
            // supported, so no explicit call is needed here (verified against the fetched
            // vk-bootstrap source and against vkCommon::VkContext's identical, already-working
            // implementation).

            auto instRet = instanceBuilder.build();
            if (!instRet)
                throw std::runtime_error(
                        "Context: failed to create instance: " + instRet.error().message());
            vkb::Instance vkbInstance = instRet.value();
            instance = vkbInstance.instance;

            if (enablePresent) {
                if (!surfaceFactory)
                    throw std::runtime_error("Context: enablePresent requires a surfaceFactory");
                surface = surfaceFactory(instance);
                if (surface == VK_NULL_HANDLE)
                    throw std::runtime_error("Context: surfaceFactory returned VK_NULL_HANDLE");
            }

            vkb::PhysicalDeviceSelector selector(vkbInstance, surface);
            selector.set_minimum_version(1, 3)
                    .require_present(enablePresent);

            VkPhysicalDeviceVulkan13Features features13{};
            features13.dynamicRendering = VK_TRUE;
            if (enablePresent) {
                selector.add_required_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
                selector.set_required_features_13(features13);
            }

            auto physRet = selector.select();
            if (!physRet)
                throw std::runtime_error(
                        "Context: failed to select physical device: " + physRet.error().message());
            vkb::PhysicalDevice vkbPhysDevice = physRet.value();
            physicalDevice = vkbPhysDevice.physical_device;

            std::cout << "[Engine::Core::Context] Device: " << vkbPhysDevice.properties.deviceName << "\n";

            vkb::DeviceBuilder deviceBuilder(vkbPhysDevice);
            auto devRet = deviceBuilder.build();
            if (!devRet)
                throw std::runtime_error(
                        "Context: failed to create device: " + devRet.error().message());
            vkb::Device vkbDevice = devRet.value();
            device = vkbDevice.device;

            auto computeQueueRet = vkbDevice.get_queue(vkb::QueueType::compute);
            auto computeFamilyRet = vkbDevice.get_queue_index(vkb::QueueType::compute);
            if (!computeQueueRet || !computeFamilyRet)
                throw std::runtime_error("Context: no compute queue available");
            computeQueue = computeQueueRet.value();
            computeFamily = computeFamilyRet.value();

            if (enablePresent) {
                auto presentQueueRet = vkbDevice.get_queue(vkb::QueueType::present);
                auto presentFamilyRet = vkbDevice.get_queue_index(vkb::QueueType::present);
                if (!presentQueueRet || !presentFamilyRet)
                    throw std::runtime_error("Context: no present-capable graphics queue available");
                uint32_t presentFamilyIndex = presentFamilyRet.value();
                if (presentFamilyIndex >= vkbDevice.queue_families.size() ||
                    !(vkbDevice.queue_families[presentFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT))
                    throw std::runtime_error(
                            "Context: present-capable queue family does not support graphics");
                graphicsQueue = presentQueueRet.value();
                graphicsFamily = presentFamilyIndex;
            }

            createCommandPools();
            createAllocator();
        } catch (...) {
            cleanup();
            throw;
        }
    }

    Context::~Context() {
        cleanup();
    }

    void Context::cleanup() {
        if (allocator != VK_NULL_HANDLE) vmaDestroyAllocator(allocator);
        if (graphicsCmdPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, graphicsCmdPool, nullptr);
        if (cmdPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, cmdPool, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    }

    void Context::createCommandPools() {
        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = computeFamily;
        if (vkCreateCommandPool(device, &pci, nullptr, &cmdPool) != VK_SUCCESS)
            throw std::runtime_error("Context: failed to create compute VkCommandPool");

        if (m_presentEnabled) {
            pci.queueFamilyIndex = graphicsFamily;
            if (vkCreateCommandPool(device, &pci, nullptr, &graphicsCmdPool) != VK_SUCCESS)
                throw std::runtime_error("Context: failed to create graphics VkCommandPool");
        }
    }

    void Context::createAllocator() {
        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.instance = instance;
        allocatorInfo.physicalDevice = physicalDevice;
        allocatorInfo.device = device;
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

        if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS)
            throw std::runtime_error("Context: failed to create VmaAllocator");
    }

} // namespace Engine::Core

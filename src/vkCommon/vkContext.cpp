#include "vkContext.h"
#include "VkBootstrap.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace vkCommon {

    void VkContext::init(bool enablePresent,
                          const std::vector<const char *> &extraInstanceExtensions,
                          const std::function<VkSurfaceKHR(VkInstance)> &surfaceFactory) {
        m_presentEnabled = enablePresent;

        vkb::InstanceBuilder instanceBuilder;
        instanceBuilder.set_app_name("vkbvh")
                       .require_api_version(1, 3, 0);

        if (enablePresent)
            for (const char *ext : extraInstanceExtensions)
                instanceBuilder.enable_extension(ext);

        // Note: this vk-bootstrap version has no InstanceBuilder::enable_extension_if_present().
        // Its InstanceBuilder::build() already auto-detects and enables
        // VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME (and sets
        // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR) internally when the extension is
        // supported, so no explicit call is needed here (see VkBootstrap.cpp, InstanceBuilder::build()).

        auto instRet = instanceBuilder.build();
        if (!instRet)
            throw std::runtime_error(
                    "VkContext: failed to create instance: " + instRet.error().message());
        vkb::Instance vkbInstance = instRet.value();
        instance = vkbInstance.instance;

        if (enablePresent) {
            if (!surfaceFactory)
                throw std::runtime_error(
                        "VkContext: enablePresent requires a surfaceFactory");
            surface = surfaceFactory(instance);
            if (surface == VK_NULL_HANDLE)
                throw std::runtime_error(
                        "VkContext: surfaceFactory returned VK_NULL_HANDLE");
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
                    "VkContext: failed to select physical device: " + physRet.error().message());
        vkb::PhysicalDevice vkbPhysDevice = physRet.value();
        physDevice = vkbPhysDevice.physical_device;

        std::cout << "[VkContext] Device: " << vkbPhysDevice.properties.deviceName << "\n";

        vkb::DeviceBuilder deviceBuilder(vkbPhysDevice);
        auto devRet = deviceBuilder.build();
        if (!devRet)
            throw std::runtime_error(
                    "VkContext: failed to create device: " + devRet.error().message());
        vkb::Device vkbDevice = devRet.value();
        device = vkbDevice.device;

        auto computeQueueRet = vkbDevice.get_queue(vkb::QueueType::compute);
        auto computeFamilyRet = vkbDevice.get_queue_index(vkb::QueueType::compute);
        if (!computeQueueRet || !computeFamilyRet)
            throw std::runtime_error("VkContext: no compute queue available");
        computeQueue = computeQueueRet.value();
        computeFamily = computeFamilyRet.value();

        if (enablePresent) {
            auto presentQueueRet = vkbDevice.get_queue(vkb::QueueType::present);
            auto presentFamilyRet = vkbDevice.get_queue_index(vkb::QueueType::present);
            if (!presentQueueRet || !presentFamilyRet)
                throw std::runtime_error("VkContext: no present-capable graphics queue available");
            graphicsQueue = presentQueueRet.value();
            graphicsFamily = presentFamilyRet.value();
        }

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

    void VkContext::createCommandPool() {
        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = computeFamily;

        if (vkCreateCommandPool(device, &pci, nullptr, &cmdPool) != VK_SUCCESS)
            throw std::runtime_error("VkContext: failed to create VkCommandPool");
    }

} // namespace vkCommon

#include "vkContext.h"

#include <iostream>
#include <stdexcept>
#include <vector>

void VkContext::init() {
    createInstance();
    pickPhysicalDevice();
    createDevice();
    createCommandPool();
}

void VkContext::shutdown() {
    if (cmdPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, cmdPool, nullptr);
    if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    cmdPool = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
}

void VkContext::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char *> exts;
#ifdef __APPLE__
    exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

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

void VkContext::createDevice() {
    uint32_t qc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qc, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qc);
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qc, qprops.data());

    bool found = false;
    for (uint32_t i = 0; i < qc; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeFamily = i;
            found = true;
            break;
        }
    }
    if (!found)
        throw std::runtime_error("VkContext: no compute queue family found");

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = computeFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    std::vector<const char *> devExts;
#ifdef __APPLE__
    devExts.push_back("VK_KHR_portability_subset");
#endif

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    dci.ppEnabledExtensionNames = devExts.data();

    if (vkCreateDevice(physDevice, &dci, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("VkContext: failed to create VkDevice");

    vkGetDeviceQueue(device, computeFamily, 0, &computeQueue);
}

void VkContext::createCommandPool() {
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = computeFamily;

    if (vkCreateCommandPool(device, &pci, nullptr, &cmdPool) != VK_SUCCESS)
        throw std::runtime_error("VkContext: failed to create VkCommandPool");
}

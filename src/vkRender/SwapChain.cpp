#include "vkRender/SwapChain.h"

#include <algorithm>
#include <stdexcept>

namespace vkRender {
    namespace {

        VkSurfaceFormatKHR ChooseSurfaceFormat(
                const std::vector<VkSurfaceFormatKHR> &formats,
                VkFormat preferredFormat) {
            for (const auto &format: formats) {
                if (format.format == preferredFormat &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                    return format;
            }

            for (const auto &format: formats) {
                if (format.format == preferredFormat)
                    return format;
            }

            for (const auto &format: formats) {
                if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                    return format;
            }

            return formats.front();
        }

        VkPresentModeKHR ChoosePresentMode(
                const std::vector<VkPresentModeKHR> &presentModes,
                VkPresentModeKHR preferredMode) {
            for (VkPresentModeKHR mode: presentModes) {
                if (mode == preferredMode)
                    return mode;
            }
            return VK_PRESENT_MODE_FIFO_KHR;
        }

        VkExtent2D ChooseExtent(const VkSurfaceCapabilitiesKHR &caps,
                                uint32_t requestedWidth,
                                uint32_t requestedHeight) {
            if (caps.currentExtent.width != UINT32_MAX)
                return caps.currentExtent;

            VkExtent2D extent{};
            extent.width = std::max(1u, requestedWidth);
            extent.height = std::max(1u, requestedHeight);
            extent.width = std::clamp(extent.width,
                                      caps.minImageExtent.width,
                                      caps.maxImageExtent.width);
            extent.height = std::clamp(extent.height,
                                       caps.minImageExtent.height,
                                       caps.maxImageExtent.height);
            return extent;
        }

    } // namespace

    SwapChain::SwapChain(vkCommon::VkContext *context,
                         const SwapChainDescriptor &descriptor)
        : m_context(context), m_descriptor(descriptor) {
        if (!m_context)
            throw std::runtime_error("SwapChain requires a valid VkContext");
        if (m_context->surface == VK_NULL_HANDLE)
            throw std::runtime_error("SwapChain requires VkContext::surface");
        if (m_context->graphicsQueue == VK_NULL_HANDLE)
            throw std::runtime_error("SwapChain requires VkContext::graphicsQueue");

        Create(VK_NULL_HANDLE);
    }

    SwapChain::~SwapChain() {
        Destroy();
    }

    void SwapChain::Recreate(uint32_t width, uint32_t height) {
        if (width != 0) m_descriptor.width = width;
        if (height != 0) m_descriptor.height = height;

        VkSwapchainKHR oldSwapchain = m_handle;
        DestroyViews();
        m_images.clear();
        m_handle = VK_NULL_HANDLE;

        Create(oldSwapchain);

        if (oldSwapchain != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(m_context->device, oldSwapchain, nullptr);
    }

    VkResult SwapChain::AcquireNextImage(VkSemaphore signalSemaphore,
                                         VkFence signalFence,
                                         uint32_t *imageIndex,
                                         uint64_t timeout) {
        return vkAcquireNextImageKHR(m_context->device, m_handle, timeout,
                                     signalSemaphore, signalFence, imageIndex);
    }

    VkResult SwapChain::Present(uint32_t imageIndex, VkSemaphore waitSemaphore) {
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = waitSemaphore != VK_NULL_HANDLE ? 1u : 0u;
        presentInfo.pWaitSemaphores = waitSemaphore != VK_NULL_HANDLE ? &waitSemaphore : nullptr;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &m_handle;
        presentInfo.pImageIndices = &imageIndex;
        return vkQueuePresentKHR(m_context->graphicsQueue, &presentInfo);
    }

    bool SwapChain::RequiresRedBlueSwap() const {
        return m_format == VK_FORMAT_B8G8R8A8_UNORM ||
               m_format == VK_FORMAT_B8G8R8A8_SRGB;
    }

    void SwapChain::Create(VkSwapchainKHR oldSwapchain) {
        VkSurfaceCapabilitiesKHR caps{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_context->physDevice,
                                                  m_context->surface, &caps);

        if ((m_descriptor.imageUsage & caps.supportedUsageFlags) != m_descriptor.imageUsage)
            throw std::runtime_error("SwapChain image usage is not supported by surface");

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_context->physDevice,
                                             m_context->surface,
                                             &formatCount, nullptr);

        if (formatCount == 0)
            throw std::runtime_error("SwapChain surface has no supported formats");
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_context->physDevice,
                                             m_context->surface,
                                             &formatCount, formats.data());

        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(m_context->physDevice,
                                                  m_context->surface,
                                                  &presentModeCount, nullptr);
        if (presentModeCount == 0)
            throw std::runtime_error("SwapChain surface has no present modes");
        std::vector<VkPresentModeKHR> presentModes(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(m_context->physDevice,
                                                  m_context->surface,
                                                  &presentModeCount,
                                                  presentModes.data());

        const VkSurfaceFormatKHR surfaceFormat =
                ChooseSurfaceFormat(formats, m_descriptor.preferredFormat);
        const VkPresentModeKHR presentMode =
                ChoosePresentMode(presentModes, m_descriptor.presentMode);
        const VkExtent2D extent =
                ChooseExtent(caps, m_descriptor.width, m_descriptor.height);

        uint32_t imageCount = caps.minImageCount + 1;
        if (caps.maxImageCount > 0)
            imageCount = std::min(imageCount, caps.maxImageCount);

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = m_context->surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = m_descriptor.imageUsage;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform = caps.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = oldSwapchain;

        if (vkCreateSwapchainKHR(m_context->device, &createInfo,
                                 nullptr, &m_handle) != VK_SUCCESS)
            throw std::runtime_error("SwapChain failed to create VkSwapchainKHR");

        m_format = surfaceFormat.format;
        m_extent = extent;
        m_descriptor.width = extent.width;
        m_descriptor.height = extent.height;

        uint32_t actualImageCount = 0;
        vkGetSwapchainImagesKHR(m_context->device, m_handle,
                                &actualImageCount, nullptr);
        m_images.resize(actualImageCount);
        vkGetSwapchainImagesKHR(
                m_context->device,
                m_handle,
                &actualImageCount,
                m_images.data());

        m_imageViews.resize(actualImageCount);
        for (uint32_t i = 0; i < actualImageCount; ++i) {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_images[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = m_format;
            viewInfo.components = {
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY};
            viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            if (vkCreateImageView(m_context->device, &viewInfo,
                                  nullptr, &m_imageViews[i]) != VK_SUCCESS)
                throw std::runtime_error("SwapChain failed to create image view");
        }
    }

    void SwapChain::DestroyViews() {
        for (VkImageView view: m_imageViews)
            vkDestroyImageView(m_context->device, view, nullptr);
        m_imageViews.clear();
    }

    void SwapChain::Destroy() {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            return;

        DestroyViews();
        m_images.clear();
        if (m_handle != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(m_context->device, m_handle, nullptr);
        m_handle = VK_NULL_HANDLE;
    }

} // namespace vkRender

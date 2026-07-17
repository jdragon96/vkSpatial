#pragma once

#include "Engine/Render/RenderTypes.h"

#include "Engine/Core/Context.h"

#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Render {

    class SwapChain {
    public:
        using UniquePtr = std::unique_ptr<SwapChain>;

        explicit SwapChain(Engine::Core::Context &context,
                           const SwapChainDescriptor &descriptor = {});
        ~SwapChain();

        SwapChain(const SwapChain &) = delete;
        SwapChain &operator=(const SwapChain &) = delete;

        void Recreate(uint32_t width = 0, uint32_t height = 0);

        VkResult AcquireNextImage(VkSemaphore signalSemaphore,
                                  VkFence signalFence,
                                  uint32_t *imageIndex,
                                  uint64_t timeout = UINT64_MAX);
        VkResult Present(uint32_t imageIndex, VkSemaphore waitSemaphore);

        VkSwapchainKHR Handle() const { return m_handle; }
        VkFormat Format() const { return m_format; }
        VkExtent2D Extent() const { return m_extent; }
        uint32_t ImageCount() const { return static_cast<uint32_t>(m_images.size()); }
        VkImage Image(uint32_t index) const { return m_images[index]; }
        VkImageView ImageView(uint32_t index) const { return m_imageViews[index]; }
        bool RequiresRedBlueSwap() const;

    private:
        Engine::Core::Context *m_context = nullptr;
        SwapChainDescriptor m_descriptor;
        VkSwapchainKHR m_handle = VK_NULL_HANDLE;
        VkFormat m_format = VK_FORMAT_UNDEFINED;
        VkExtent2D m_extent{};
        std::vector<VkImage> m_images;
        std::vector<VkImageView> m_imageViews;

        void Create(VkSwapchainKHR oldSwapchain);
        void DestroyViews();
        void Destroy();
    };

} // namespace Engine::Render

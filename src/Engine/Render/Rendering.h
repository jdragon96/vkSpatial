#pragma once

#include "Engine/Render/RenderAttachments.h"
#include "Engine/Render/RenderTypes.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Render {

    struct RenderingDescriptor {
        VkRect2D renderArea{};
        uint32_t layerCount = 1;
        std::vector<VkRenderingAttachmentInfo> colorAttachments;
        bool hasDepthAttachment = false;
        VkRenderingAttachmentInfo depthAttachment{};
        bool setViewport = true;
        bool setScissor = true;
        VkViewport viewport{};
        VkRect2D scissor{};

        RenderingDescriptor() = default;
        explicit RenderingDescriptor(VkExtent2D extent);

        static RenderingDescriptor ColorDepth(VkExtent2D extent,
                                              VkImageView colorView,
                                              VkImageView depthView,
                                              const ClearOptions &clear);

        RenderingDescriptor &SetExtent(VkExtent2D extent);
        RenderingDescriptor &SetRenderArea(VkRect2D area);
        RenderingDescriptor &AddColorAttachment(const VkRenderingAttachmentInfo &attachment);
        RenderingDescriptor &SetDepthAttachment(const VkRenderingAttachmentInfo &attachment);
        RenderingDescriptor &SetViewport(VkViewport value);
        RenderingDescriptor &SetScissor(VkRect2D value);
        RenderingDescriptor &UseDefaultViewportAndScissor(VkExtent2D extent);
    };

    class RenderingScope {
    public:
        RenderingScope(VkCommandBuffer commandBuffer,
                       const RenderingDescriptor &descriptor);
        ~RenderingScope();

        RenderingScope(const RenderingScope &) = delete;
        RenderingScope &operator=(const RenderingScope &) = delete;

        void End();

    private:
        VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
        bool m_active = false;
    };

} // namespace Engine::Render

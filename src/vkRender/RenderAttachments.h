#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

namespace vkRender {

    class ColorAttachment {
    public:
        explicit ColorAttachment(
                VkImageView imageView,
                VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            m_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            m_info.imageView = imageView;
            m_info.imageLayout = layout;
            m_info.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            m_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }

        ColorAttachment &Clear(const float color[4], bool enabled = true) {
            m_info.loadOp = enabled ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                    : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            m_clear.color = {{
                    color[0],
                    color[1],
                    color[2],
                    color[3],
            }};
            return *this;
        }

        ColorAttachment &Load() {
            m_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            return *this;
        }

        ColorAttachment &DontCareLoad() {
            m_info.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            return *this;
        }

        ColorAttachment &Store(bool enabled = true) {
            m_info.storeOp = enabled ? VK_ATTACHMENT_STORE_OP_STORE
                                     : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            return *this;
        }

        VkRenderingAttachmentInfo Build() const {
            VkRenderingAttachmentInfo info = m_info;
            info.clearValue = m_clear;
            return info;
        }

    private:
        VkRenderingAttachmentInfo m_info{};
        VkClearValue m_clear{};
    };

    class DepthAttachment {
    public:
        explicit DepthAttachment(VkImageView imageView,
                                 VkImageLayout layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
            m_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            m_info.imageView = imageView;
            m_info.imageLayout = layout;
            m_info.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            m_info.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        }

        DepthAttachment &Clear(float depth,
                               bool enabled = true,
                               uint32_t stencil = 0) {
            m_info.loadOp = enabled ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                    : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            m_clear.depthStencil = {depth, stencil};
            return *this;
        }

        DepthAttachment &Load() {
            m_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            return *this;
        }

        DepthAttachment &Store(bool enabled = true) {
            m_info.storeOp = enabled ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            return *this;
        }

        VkRenderingAttachmentInfo Build() const {
            VkRenderingAttachmentInfo info = m_info;
            info.clearValue = m_clear;
            return info;
        }

    private:
        VkRenderingAttachmentInfo m_info{};
        VkClearValue m_clear{};
    };

} // namespace vkRender

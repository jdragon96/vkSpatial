#include "vkRender/Rendering.h"

#include <stdexcept>

namespace vkRender {

    RenderingDescriptor::RenderingDescriptor(VkExtent2D extent) {
        SetExtent(extent);
    }

    RenderingDescriptor RenderingDescriptor::ColorDepth(VkExtent2D extent,
                                                        VkImageView colorView,
                                                        VkImageView depthView,
                                                        const ClearOptions &clear) {
        RenderingDescriptor descriptor(extent);
        descriptor
                .AddColorAttachment(
                        ColorAttachment(colorView)
                                .Clear(clear.color, clear.clearColor)
                                .Build())
                .SetDepthAttachment(
                        DepthAttachment(depthView)
                                .Clear(clear.depth, clear.clearDepth, clear.stencil)
                                .Build());
        return descriptor;
    }

    RenderingDescriptor &RenderingDescriptor::SetExtent(VkExtent2D extent) {
        renderArea = {{0, 0}, extent};
        return UseDefaultViewportAndScissor(extent);
    }

    RenderingDescriptor &RenderingDescriptor::SetRenderArea(VkRect2D area) {
        renderArea = area;
        return *this;
    }

    RenderingDescriptor &RenderingDescriptor::AddColorAttachment(
            const VkRenderingAttachmentInfo &attachment) {
        colorAttachments.push_back(attachment);
        return *this;
    }

    RenderingDescriptor &RenderingDescriptor::SetDepthAttachment(
            const VkRenderingAttachmentInfo &attachment) {
        depthAttachment = attachment;
        hasDepthAttachment = true;
        return *this;
    }

    RenderingDescriptor &RenderingDescriptor::SetViewport(VkViewport value) {
        viewport = value;
        setViewport = true;
        return *this;
    }

    RenderingDescriptor &RenderingDescriptor::SetScissor(VkRect2D value) {
        scissor = value;
        setScissor = true;
        return *this;
    }

    RenderingDescriptor &RenderingDescriptor::UseDefaultViewportAndScissor(
            VkExtent2D extent) {
        viewport = {
                0.0f,
                0.0f,
                static_cast<float>(extent.width),
                static_cast<float>(extent.height),
                0.0f,
                1.0f,
        };
        scissor = {{0, 0}, extent};
        setViewport = true;
        setScissor = true;
        return *this;
    }

    RenderingScope::RenderingScope(VkCommandBuffer commandBuffer,
                                   const RenderingDescriptor &descriptor)
        : m_commandBuffer(commandBuffer) {
        if (m_commandBuffer == VK_NULL_HANDLE)
            throw std::runtime_error("RenderingScope requires a valid VkCommandBuffer");
        if (descriptor.renderArea.extent.width == 0 ||
            descriptor.renderArea.extent.height == 0)
            throw std::runtime_error("RenderingScope requires a non-empty render area");

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = descriptor.renderArea;
        renderingInfo.layerCount = descriptor.layerCount;
        renderingInfo.colorAttachmentCount =
                static_cast<uint32_t>(descriptor.colorAttachments.size());
        renderingInfo.pColorAttachments =
                descriptor.colorAttachments.empty()
                        ? nullptr
                        : descriptor.colorAttachments.data();
        renderingInfo.pDepthAttachment =
                descriptor.hasDepthAttachment ? &descriptor.depthAttachment : nullptr;

        vkCmdBeginRendering(m_commandBuffer, &renderingInfo);
        m_active = true;

        if (descriptor.setViewport)
            vkCmdSetViewport(m_commandBuffer, 0, 1, &descriptor.viewport);
        if (descriptor.setScissor)
            vkCmdSetScissor(m_commandBuffer, 0, 1, &descriptor.scissor);
    }

    RenderingScope::~RenderingScope() {
        End();
    }

    void RenderingScope::End() {
        if (!m_active)
            return;
        vkCmdEndRendering(m_commandBuffer);
        m_active = false;
    }

} // namespace vkRender

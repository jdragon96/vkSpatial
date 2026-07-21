#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Render/RenderGraph.h"

#include <cstdint>
#include <vulkan/vulkan.h>

struct GLFWwindow;

// Draws a Dear ImGui control panel on top of whatever earlier passes rendered into the
// swapchain image (Task 2 of the TSDF viewer plan: proves the ImGui<->Vulkan dynamic-
// rendering integration works; Task 3 fills in real UI + interactivity).
//
// Must be added LAST to the RenderGraph: Execute() opens its RenderingScope with
// VK_ATTACHMENT_LOAD_OP_LOAD (via ColorAttachment::Load()), not CLEAR, so it composites on
// top of PointCloudPass's output instead of erasing it. See Execute() for details.
class ImGuiPass : public Engine::Render::RenderPass {
public:
    // `colorFormat`/`imageCount` should come from the app's SwapChain (Format()/ImageCount()).
    ImGuiPass(Engine::Core::Context &context, GLFWwindow *window, VkFormat colorFormat,
              uint32_t imageCount);
    ~ImGuiPass() override;

    ImGuiPass(const ImGuiPass &) = delete;
    ImGuiPass &operator=(const ImGuiPass &) = delete;

    const char *Name() const override { return "ImGuiPass"; }
    void Execute(Engine::Render::RenderContext &ctx) override;

    // Task-2 stats hook: shown in the panel as "points: N". Task 3 will replace/extend this
    // with the full viewer UI + shared interactive state.
    void SetPointCount(uint32_t count) { m_pointCount = count; }

private:
    Engine::Core::Context &m_context;
    // Stored so the pointer handed to ImGui_ImplVulkan_InitInfo::PipelineRenderingCreateInfo
    // (which the backend keeps a copy of and dereferences on every pipeline (re)build)
    // stays valid for the lifetime of this pass.
    VkFormat m_colorFormat;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    uint32_t m_pointCount = 0;
};

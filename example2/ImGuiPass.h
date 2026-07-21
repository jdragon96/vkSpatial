#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Render/RenderGraph.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vulkan/vulkan.h>

struct GLFWwindow;

// Draws a caller-supplied Dear ImGui panel on top of whatever earlier passes rendered into
// the swapchain image. The panel content itself (windows, controls, stats) is fully owned by
// the app via SetUi()'s drawUi callback; this pass only drives the ImGui frame lifecycle.
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

    // Sets the per-frame panel builder, invoked between ImGui::NewFrame() and ImGui::Render()
    // inside Execute(). Must be called before the first Execute(); anything captured by
    // drawUi must outlive this pass.
    void SetUi(std::function<void()> drawUi) { m_drawUi = std::move(drawUi); }

private:
    Engine::Core::Context &m_context;
    // Stored so the pointer handed to ImGui_ImplVulkan_InitInfo::PipelineRenderingCreateInfo
    // (which the backend keeps a copy of and dereferences on every pipeline (re)build)
    // stays valid for the lifetime of this pass.
    VkFormat m_colorFormat;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    std::function<void()> m_drawUi;
};

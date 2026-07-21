#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Render/RenderGraph.h"

#include <cstddef>
#include <cstdint>
#include <vulkan/vulkan.h>

struct GLFWwindow;
class PointCloudPass;

// Shared interactive state for the TSDF viewer, owned by the app (tsdf_viewer.cpp) and
// referenced by ImGuiPass. The UI mutates it; the app's between-frames rebuild reads it and
// writes back the stats fields. `dirty` requests a full TSDF rebuild (scene/quality change);
// the show* flags drive cheap PointCloudPass::SetVisible toggles with no rebuild.
struct ViewerState {
    int scene = 0;         // 0 = plane, 1 = interproximal
    int maxDirections = 1; // K in IntegrationQuality (1..2)
    bool viewAngle = false;
    bool showInput = true, showExtracted = true, showSliceP = true, showSliceN = true;
    int extractColor = 0; // 0 = by direction bitmask, 1 = flat green
    bool dirty = true;    // request rebuild (starts true so the first frame builds)

    // Stats, filled by the rebuild and displayed in the panel.
    size_t nInput = 0, nExtracted = 0;
    float extractedZMean = 0.0f; // mean |z| of extracted points
    float crossP = 0.0f;         // +Z-layer (dir 4) central-column zero-crossing z
    float crossN = 0.0f;         // -Z-layer (dir 5) central-column zero-crossing z
};

// Draws the Dear ImGui control panel on top of whatever earlier passes rendered into the
// swapchain image. Owns the full viewer UI (scene / quality / layer-toggle controls + stats
// readout) and drives interactivity through a shared ViewerState + PointCloudPass reference.
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

    // Wire the panel to the shared state and the point-cloud pass it toggles. Must be called
    // before the first Execute(). Both references must outlive this pass.
    void SetViewer(ViewerState *state, PointCloudPass *pass) {
        m_state = state;
        m_pass = pass;
    }

private:
    Engine::Core::Context &m_context;
    // Stored so the pointer handed to ImGui_ImplVulkan_InitInfo::PipelineRenderingCreateInfo
    // (which the backend keeps a copy of and dereferences on every pipeline (re)build)
    // stays valid for the lifetime of this pass.
    VkFormat m_colorFormat;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    ViewerState *m_state = nullptr;
    PointCloudPass *m_pass = nullptr;
};

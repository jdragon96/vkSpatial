#include "ImGuiPass.h"

#include "Engine/Render/RenderAttachments.h"
#include "Engine/Render/Rendering.h"
#include "Engine/Render/SwapChain.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <array>
#include <stdexcept>

namespace {

    // Per imgui_impl_vulkan.h's note above ImGui_ImplVulkan_InitInfo: the pool must have
    // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT and a handful of combined-image-
    // sampler descriptors (1 for the font atlas + 1 per ImGui_ImplVulkan_AddTexture call,
    // none of which this pass makes yet). 1000 sets matches the brief / common imgui usage.
    VkDescriptorPool CreateImGuiDescriptorPool(VkDevice device) {
        constexpr uint32_t kMaxSets = 1000;
        std::array<VkDescriptorPoolSize, 1> poolSizes{{
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxSets},
        }};

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = kMaxSets;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();

        VkDescriptorPool pool = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS)
            throw std::runtime_error("ImGuiPass failed to create descriptor pool");
        return pool;
    }

} // namespace

ImGuiPass::ImGuiPass(Engine::Core::Context &context, GLFWwindow *window, VkFormat colorFormat,
                     uint32_t imageCount)
    : m_context(context), m_colorFormat(colorFormat) {
    m_descriptorPool = CreateImGuiDescriptorPool(context.device);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, true);

    // Vendored imgui is 1.91.9 (lib/imgui/imgui.h IMGUI_VERSION). Its
    // backends/imgui_impl_vulkan.h ImGui_ImplVulkan_InitInfo has UseDynamicRendering +
    // PipelineRenderingCreateInfo fields (no VkRenderPass needed when the former is true;
    // RenderPass is explicitly documented "ignored if using dynamic rendering").
    const uint32_t clampedImageCount = imageCount >= 2 ? imageCount : 2;

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3; // matches Context.cpp's VmaAllocatorCreateInfo
    initInfo.Instance = context.instance;
    initInfo.PhysicalDevice = context.physicalDevice;
    initInfo.Device = context.device;
    initInfo.QueueFamily = context.graphicsFamily;
    initInfo.Queue = context.graphicsQueue;
    initInfo.DescriptorPool = m_descriptorPool;
    initInfo.RenderPass = VK_NULL_HANDLE; // ignored: UseDynamicRendering=true below
    initInfo.MinImageCount = clampedImageCount;
    initInfo.ImageCount = clampedImageCount;
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &m_colorFormat;
    initInfo.MinAllocationSize = 1024 * 1024; // per header comment: avoids zealous best-practices validation warnings

    if (!ImGui_ImplVulkan_Init(&initInfo))
        throw std::runtime_error("ImGuiPass failed to initialize ImGui Vulkan backend");

    // Font atlas upload: this imgui version auto-creates the fonts texture the first time
    // ImGui_ImplVulkan_NewFrame() runs (see imgui_impl_vulkan.cpp: NewFrame checks
    // `!bd->FontTexture.DescriptorSet` and calls ImGui_ImplVulkan_CreateFontsTexture()
    // itself, which allocates its own one-time command pool/buffer and submits it
    // internally). No manual command-buffer upload/submit is needed here, unlike older
    // (~pre-1.90) imgui versions.
}

ImGuiPass::~ImGuiPass() {
    vkDeviceWaitIdle(m_context.device);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (m_descriptorPool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(m_context.device, m_descriptorPool, nullptr);
}

void ImGuiPass::Execute(Engine::Render::RenderContext &ctx) {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("TSDF Viewer");
    ImGui::Text("points: %u", m_pointCount);
    ImGui::Text("frame: %llu", static_cast<unsigned long long>(ctx.frame.frameIndex));
    ImGui::End();

    ImGui::Render();

    // Clear-vs-load (see class comment): this pass runs last in the RenderGraph, after
    // PointCloudPass already wrote the points into this same swapchain image this frame.
    // Building the RenderingDescriptor by hand (instead of RenderingDescriptor::ColorDepth,
    // which always clears per ClearOptions) and using ColorAttachment::Load() sets
    // VK_ATTACHMENT_LOAD_OP_LOAD, so vkCmdBeginRendering preserves the existing image
    // contents and ImGui's draw data composites on top instead of blanking the points.
    const VkExtent2D extent = ctx.swapChain->Extent();
    Engine::Render::RenderingDescriptor descriptor(extent);
    descriptor.AddColorAttachment(
            Engine::Render::ColorAttachment(ctx.swapChain->ImageView(ctx.imageIndex))
                    .Load()
                    .Build());
    // No depth attachment: ImGui doesn't depth-test, and RenderingDescriptor defaults to
    // hasDepthAttachment=false, so vkCmdBeginRendering gets pDepthAttachment=nullptr.

    Engine::Render::RenderingScope scope(ctx.commandBuffer, descriptor);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), ctx.commandBuffer);
}

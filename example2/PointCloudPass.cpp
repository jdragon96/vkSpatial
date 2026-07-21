#include "PointCloudPass.h"

#include "Engine/Core/Image.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/Rendering.h"
#include "Engine/Render/SwapChain.h"
#include "Engine/Render/View.h"

#include "utilities/Math.h"

#include <cstddef>
#include <stdexcept>

namespace {

    struct PushConstants {
        vkMath::Mat4 mvp;
        float pointSize;
    };

} // namespace

PointCloudPass::PointCloudPass(Engine::Core::Context &context, VkFormat colorFormat,
                               const std::string &shaderDir)
    : m_pipeline(context) {
    for (int i = 0; i < kMaxSets; ++i)
        m_sets[static_cast<size_t>(i)] = std::make_unique<PointSet>(context);

    Engine::Render::GraphicsPipelineDescriptor descriptor;
    descriptor.VertexShader(shaderDir + "/pointcloud.vert.spv")
            .FragmentShader(shaderDir + "/pointcloud.frag.spv")
            .VertexBinding<PointVertex>()
            .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PointVertex, pos))
            .VertexAttribute(1, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(PointVertex, rgba))
            .ColorTarget(colorFormat)
            .DepthTarget(VK_FORMAT_D32_SFLOAT)
            .PushConstant<PushConstants>(VK_SHADER_STAGE_VERTEX_BIT);
    descriptor.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    m_pipeline.Build(descriptor);
}

void PointCloudPass::SetPointSet(int id, const std::vector<PointVertex> &vertices) {
    if (id < 0 || id >= kMaxSets)
        throw std::out_of_range("PointCloudPass::SetPointSet: id out of range");

    PointSet &set = *m_sets[static_cast<size_t>(id)];
    if (vertices.empty()) {
        set.count = 0;
        return;
    }

    const uint32_t bytes = static_cast<uint32_t>(vertices.size() * sizeof(PointVertex));
    set.buffer.Allocate(bytes);
    set.buffer.Upload(vertices.data(), bytes, Engine::Core::QueueRole::Graphics);
    set.count = static_cast<uint32_t>(vertices.size());
}

void PointCloudPass::SetVisible(int id, bool visible) {
    if (id < 0 || id >= kMaxSets)
        throw std::out_of_range("PointCloudPass::SetVisible: id out of range");
    m_sets[static_cast<size_t>(id)]->visible = visible;
}

void PointCloudPass::Execute(Engine::Render::RenderContext &ctx) {
    Engine::Render::ClearOptions clear{};
    clear.color[0] = 0.02f;
    clear.color[1] = 0.02f;
    clear.color[2] = 0.025f;
    clear.color[3] = 1.0f;

    const VkExtent2D extent = ctx.swapChain->Extent();
    auto renderingDescriptor = Engine::Render::RenderingDescriptor::ColorDepth(
            extent,
            ctx.swapChain->ImageView(ctx.imageIndex),
            ctx.depthImage->View(),
            clear);

    Engine::Render::RenderingScope scope(ctx.commandBuffer, renderingDescriptor);

    m_pipeline.Bind(ctx.commandBuffer);

    Engine::Render::Camera *camera = ctx.view->GetCamera();
    PushConstants push{camera->GetProjectionMatrix() * camera->GetViewMatrix(), m_pointSize};
    m_pipeline.PushConstants(ctx.commandBuffer, VK_SHADER_STAGE_VERTEX_BIT, push);

    for (const std::unique_ptr<PointSet> &setPtr : m_sets) {
        const PointSet &set = *setPtr;
        if (!set.visible || set.count == 0)
            continue;
        VkBuffer handle = set.buffer.Handle();
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &handle, &offset);
        vkCmdDraw(ctx.commandBuffer, set.count, 1, 0, 0);
    }
}

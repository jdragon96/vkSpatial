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
    : m_pointPipeline(context), m_linePipeline(context) {
    for (int i = 0; i < kMaxSets; ++i)
        m_sets[static_cast<size_t>(i)] = std::make_unique<VertexSet>(context);

    // Both topologies share the vertex format, the shaders and the push constants -- only the
    // input assembly differs, so the descriptor is built once and rebuilt with one field changed.
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
    m_pointPipeline.Build(descriptor);

    descriptor.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    m_linePipeline.Build(descriptor);
}

void PointCloudPass::SetPointSet(int id, const std::vector<PointVertex> &vertices) {
    upload(id, vertices, VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
}

void PointCloudPass::SetLineSet(int id, const std::vector<PointVertex> &vertices) {
    if (vertices.size() % 2 != 0)
        throw std::invalid_argument("PointCloudPass::SetLineSet: a line list needs an even vertex "
                                    "count; the last segment has no end point");
    upload(id, vertices, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
}

void PointCloudPass::upload(int id, const std::vector<PointVertex> &vertices,
                            VkPrimitiveTopology topology) {
    if (id < 0 || id >= kMaxSets)
        throw std::out_of_range("PointCloudPass::upload: id out of range");

    VertexSet &set = *m_sets[static_cast<size_t>(id)];
    set.topology = topology;
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

    Engine::Render::Camera *camera = ctx.view->GetCamera();
    const PushConstants push{camera->GetProjectionMatrix() * camera->GetViewMatrix(), m_pointSize};

    // Bound lazily and only when the topology actually changes: the sets are usually grouped, so
    // the common case is one pipeline bind for the whole pass rather than one per set.
    Engine::Render::GraphicsPipeline *bound = nullptr;

    for (const std::unique_ptr<VertexSet> &setPtr : m_sets) {
        const VertexSet &set = *setPtr;
        if (!set.visible || set.count == 0)
            continue;

        Engine::Render::GraphicsPipeline *wanted =
                set.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ? &m_linePipeline : &m_pointPipeline;
        if (wanted != bound) {
            wanted->Bind(ctx.commandBuffer);
            wanted->PushConstants(ctx.commandBuffer, VK_SHADER_STAGE_VERTEX_BIT, push);
            bound = wanted;
        }

        VkBuffer handle = set.buffer.Handle();
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &handle, &offset);
        vkCmdDraw(ctx.commandBuffer, set.count, 1, 0, 0);
    }
}

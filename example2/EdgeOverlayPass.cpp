#include "EdgeOverlayPass.h"

#include "Engine/Render/Camera.h"
#include "Engine/Render/Rendering.h"
#include "Engine/Render/SwapChain.h"
#include "Engine/Render/View.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

    // Camera transform only -- unlike IsosurfaceMeshPass::TransformPushConstants, there is no
    // separate `model` matrix (world-space lighting isn't needed, see EdgeOverlay.frag.glsl) and
    // m_edgeObject's model is always identity (edge endpoints are already world-space, taken
    // straight from SurfaceMesh::vertices -- see isosurface_viewer.cpp's BuildEdgeOverlayGroup).
    struct TransformPushConstants {
        vkMath::Mat4 modelViewProjection = vkMath::Mat4::Identity();
    };

} // namespace

EdgeOverlayPass::EdgeOverlayPass(Engine::Core::Context &context,
                                 VkFormat colorFormat,
                                 const std::string &shaderDirectory)
    : m_context(context),
      m_pipeline(context) {
    BuildPipeline(colorFormat, shaderDirectory);
}

void EdgeOverlayPass::BuildPipeline(VkFormat colorFormat, const std::string &shaderDirectory) {
    Engine::Render::GraphicsPipelineDescriptor descriptor;
    descriptor.VertexShader(shaderDirectory + "/EdgeOverlay.vert.glsl.spv")
            .FragmentShader(shaderDirectory + "/EdgeOverlay.frag.glsl.spv")
            .VertexBinding<EdgeVertex>()
            .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(EdgeVertex, position))
            .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(EdgeVertex, color))
            .ColorTarget(colorFormat)
            .PushConstant<TransformPushConstants>(VK_SHADER_STAGE_VERTEX_BIT);
    // LINE_LIST, and deliberately NO DepthTarget() call: the descriptor's depthFormat stays
    // VK_FORMAT_UNDEFINED and depthTestEnable/depthWriteEnable stay false, so this pipeline never
    // tests or writes depth at all -- see the class comment on why flagged edges must stay visible
    // even behind the mesh, and Execute() below (no depth attachment bound at draw time either).
    descriptor.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    m_pipeline.Build(descriptor);
}

void EdgeOverlayPass::SetEdges(const std::vector<EdgeOverlayGroup> &groups) {
    std::size_t segmentCount = 0;
    for (const EdgeOverlayGroup &group : groups)
        segmentCount += group.segments.size();

    std::vector<EdgeVertex> vertices;
    vertices.reserve(segmentCount * 2);
    for (const EdgeOverlayGroup &group : groups) {
        const vkMath::Vec3 color(group.color.x(), group.color.y(), group.color.z());
        for (const std::pair<Eigen::Vector3f, Eigen::Vector3f> &segment : group.segments) {
            vertices.push_back(EdgeVertex(
                    vkMath::Vec3(segment.first.x(), segment.first.y(), segment.first.z()), color));
            vertices.push_back(EdgeVertex(
                    vkMath::Vec3(segment.second.x(), segment.second.y(), segment.second.z()), color));
        }
    }

    // No shared-vertex welding needed (unlike IsosurfaceMeshPass, which indexes into
    // SurfaceMesh::triangles): every segment already owns two distinct vertices and LINE_LIST
    // draws consecutive pairs (0,1), (2,3), ... -- so an identity index buffer is exactly right.
    std::vector<Engine::Render::Object<EdgeVertex>::Index> indices(vertices.size());
    for (std::size_t vertexIndex = 0; vertexIndex < indices.size(); ++vertexIndex)
        indices[vertexIndex] = static_cast<Engine::Render::Object<EdgeVertex>::Index>(vertexIndex);

    m_edgeObject.SetGeometry(std::move(vertices), std::move(indices));
    // Guard the realloc: an empty overlay (e.g. a fully closed, edge-manifold mesh -- no
    // non-manifold or boundary edges at all) must not attempt Object::Upload, which throws on
    // empty geometry -- Execute() below checks Empty() again and simply draws nothing that frame.
    if (!m_edgeObject.Empty())
        m_edgeObject.Upload(m_context);
}

void EdgeOverlayPass::Execute(Engine::Render::RenderContext &ctx) {
    if (!ctx.swapChain || !ctx.view || !ctx.view->GetCamera())
        throw std::runtime_error("EdgeOverlayPass requires swapchain, view and camera");

    if (!m_visible || m_edgeObject.Empty() || !m_edgeObject.Uploaded())
        return; // toggled off, or nothing flagged this rebuild (e.g. a closed manifold sphere)

    // LOAD (not clear) color, and NO depth attachment at all: this pass runs after
    // IsosurfaceMeshPass already wrote the shaded mesh (and depth) into this same swapchain image
    // this frame, and must composite on top of it without erasing it or depth-testing against it
    // -- the same LOAD_OP_LOAD idiom ImGuiPass uses to draw its panel over the mesh, see that
    // class's Execute(). Omitting the depth attachment entirely (rather than loading it) matches
    // how BuildPipeline built this pipeline with no depth format declared.
    const VkExtent2D extent = ctx.swapChain->Extent();
    Engine::Render::RenderingDescriptor renderingDescriptor(extent);
    renderingDescriptor.AddColorAttachment(
            Engine::Render::ColorAttachment(ctx.swapChain->ImageView(ctx.imageIndex))
                    .Load()
                    .Build());
    Engine::Render::RenderingScope scope(ctx.commandBuffer, renderingDescriptor);

    m_pipeline.Bind(ctx.commandBuffer);

    const Engine::Render::Camera *camera = ctx.view->GetCamera();
    TransformPushConstants transform{};
    transform.modelViewProjection =
            camera->GetProjectionMatrix() * camera->GetViewMatrix() * m_edgeObject.Model();
    m_pipeline.PushConstants(ctx.commandBuffer, VK_SHADER_STAGE_VERTEX_BIT, transform);

    m_edgeObject.Render(ctx.commandBuffer);
}

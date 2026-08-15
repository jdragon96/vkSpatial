#include "IsosurfaceMeshPass.h"

#include "Engine/Core/Image.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/Rendering.h"
#include "Engine/Render/SwapChain.h"
#include "Engine/Render/View.h"

#include "utilities/Math.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

    // position/normal/color -- see Engine/Render/Object.h.
    using MeshVertex = Engine::Render::Vertex;

    struct TransformPushConstants {
        vkMath::Mat4 modelViewProjection = vkMath::Mat4::Identity();
        vkMath::Mat4 model = vkMath::Mat4::Identity();
    };

    // Layout mirrors BlinnPhong.cpp's BlinnPhongConstants: every vec3 is followed by a float so
    // each vec3+float pair fills exactly one 16-byte push-constant slot (GLSL's vec3 alignment),
    // matching std140/push-constant packing with no manual padding needed between pairs.
    struct LightingPushConstants {
        vkMath::Vec3 lightPosition;
        float shininess = 48.0f;
        vkMath::Vec3 lightColor{1.0f, 0.97f, 0.90f};
        float specularStrength = 0.35f;
        vkMath::Vec3 cameraPosition;
        float padding = 0.0f;
    };

    // Fixed world-space light: a debug viewer benefits from stable, non-animated lighting so any
    // shading difference between rebuilds comes from the MESH changing, not the light moving.
    const vkMath::Vec3 kWorldLightPosition{3.0f, 4.0f, 3.5f};

} // namespace

IsosurfaceMeshPass::IsosurfaceMeshPass(Engine::Core::Context &context,
                                       VkFormat colorFormat,
                                       const std::string &shaderDirectory)
    : m_context(context),
      m_solidPipeline(context),
      m_wireframePipeline(context) {
    BuildPipeline(m_solidPipeline, VK_POLYGON_MODE_FILL, colorFormat, shaderDirectory);
    BuildPipeline(m_wireframePipeline, VK_POLYGON_MODE_LINE, colorFormat, shaderDirectory);
}

void IsosurfaceMeshPass::BuildPipeline(Engine::Render::GraphicsPipeline &pipeline,
                                       VkPolygonMode polygonMode,
                                       VkFormat colorFormat,
                                       const std::string &shaderDirectory) {
    Engine::Render::GraphicsPipelineDescriptor descriptor;
    descriptor.VertexShader(shaderDirectory + "/IsosurfaceMesh.vert.glsl.spv")
            .FragmentShader(shaderDirectory + "/IsosurfaceMesh.frag.glsl.spv")
            .VertexBinding<MeshVertex>()
            .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, position))
            .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, normal))
            .VertexAttribute(2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, color))
            .ColorTarget(colorFormat)
            .DepthTarget(VK_FORMAT_D32_SFLOAT)
            .PushConstant<TransformPushConstants>(VK_SHADER_STAGE_VERTEX_BIT)
            .PushConstant<LightingPushConstants>(VK_SHADER_STAGE_FRAGMENT_BIT);
    // VK_POLYGON_MODE_LINE for the wireframe variant: MoltenVK supports line polygon mode, so no
    // extra device feature gating is done here (see the class comment / viewer header comment
    // for the fallback note if that assumption ever turns out wrong on some runtime).
    descriptor.polygonMode = polygonMode;
    pipeline.Build(descriptor);
}

void IsosurfaceMeshPass::SetMesh(const Mesh::SurfaceMesh &mesh,
                                 const Eigen::Vector3f &color) {
    std::vector<MeshVertex> vertices;
    vertices.reserve(mesh.vertices.size());
    for (std::size_t vertexIndex = 0; vertexIndex < mesh.vertices.size(); ++vertexIndex) {
        const Eigen::Vector3f &position = mesh.vertices[vertexIndex];
        const Eigen::Vector3f normal =
                vertexIndex < mesh.normals.size() ? mesh.normals[vertexIndex] : Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        vertices.push_back(MeshVertex(
                vkMath::Vec3(position.x(), position.y(), position.z()),
                vkMath::Vec3(normal.x(), normal.y(), normal.z()),
                vkMath::Vec3(color.x(), color.y(), color.z())));
    }

    std::vector<Engine::Render::Object<>::Index> indices;
    indices.reserve(mesh.triangles.size() * 3);
    for (const Eigen::Vector3i &triangle : mesh.triangles) {
        indices.push_back(static_cast<Engine::Render::Object<>::Index>(triangle.x()));
        indices.push_back(static_cast<Engine::Render::Object<>::Index>(triangle.y()));
        indices.push_back(static_cast<Engine::Render::Object<>::Index>(triangle.z()));
    }

    m_meshObject.SetGeometry(std::move(vertices), std::move(indices));
    // Guard the realloc: an empty extraction (e.g. an all-outside narrow band) must not attempt
    // Object::Upload, which throws on empty geometry -- Execute() below checks Empty() again and
    // simply draws nothing for that frame.
    if (!m_meshObject.Empty())
        m_meshObject.Upload(m_context);
}

void IsosurfaceMeshPass::Execute(Engine::Render::RenderContext &ctx) {
    if (!m_visible) return;
    if (!ctx.swapChain || !ctx.depthImage || !ctx.view || !ctx.view->GetCamera())
        throw std::runtime_error("IsosurfaceMeshPass requires swapchain, depth image, view and camera");

    const VkExtent2D extent = ctx.swapChain->Extent();
    Engine::Render::RenderingDescriptor renderingDescriptor(extent);
    if (m_clears) {
        Engine::Render::ClearOptions clear{};
        clear.color[0] = 0.030f;
        clear.color[1] = 0.033f;
        clear.color[2] = 0.038f;
        clear.color[3] = 1.0f;
        renderingDescriptor = Engine::Render::RenderingDescriptor::ColorDepth(
                extent, ctx.swapChain->ImageView(ctx.imageIndex), ctx.depthImage->View(), clear);
    } else {
        // Draw OVER whatever ran before (the point cloud), depth loaded so the mesh occludes and is
        // occluded correctly instead of overwriting the points.
        renderingDescriptor.AddColorAttachment(
                Engine::Render::ColorAttachment(ctx.swapChain->ImageView(ctx.imageIndex))
                        .Load()
                        .Build());
        renderingDescriptor.SetDepthAttachment(
                Engine::Render::DepthAttachment(ctx.depthImage->View()).Load().Build());
    }
    Engine::Render::RenderingScope scope(ctx.commandBuffer, renderingDescriptor);

    if (m_meshObject.Empty() || !m_meshObject.Uploaded())
        return; // nothing extracted yet (e.g. a narrow band that produced zero triangles)

    Engine::Render::GraphicsPipeline &pipeline = m_wireframe ? m_wireframePipeline : m_solidPipeline;
    pipeline.Bind(ctx.commandBuffer);

    const Engine::Render::Camera *camera = ctx.view->GetCamera();
    const vkMath::Mat4 viewProjection = camera->GetProjectionMatrix() * camera->GetViewMatrix();

    TransformPushConstants transform{};
    transform.model = m_meshObject.Model();
    transform.modelViewProjection = viewProjection * m_meshObject.Model();
    pipeline.PushConstants(ctx.commandBuffer, VK_SHADER_STAGE_VERTEX_BIT, transform);

    LightingPushConstants lighting{};
    lighting.lightPosition = kWorldLightPosition; // already world-space, no view transform needed
    lighting.cameraPosition = camera->GetEye();   // GetEye() is also world-space -- see .frag.glsl
    pipeline.PushConstants(ctx.commandBuffer, VK_SHADER_STAGE_FRAGMENT_BIT, lighting);

    m_meshObject.Render(ctx.commandBuffer);
}

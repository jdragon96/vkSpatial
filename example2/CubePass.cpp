#include "CubePass.h"

#include "Engine/Core/Image.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/Rendering.h"
#include "Engine/Render/SwapChain.h"
#include "Engine/Render/View.h"

#include "utilities/Math.h"
#include "utilities/SimpleResource.h"

#include <cstddef>

namespace {

    struct Vertex {
        float position[3];
        float color[3];
    };

    struct PushConstants {
        vkMath::Mat4 mvp;
    };

} // namespace

CubePass::CubePass(Engine::Core::Context &context, VkFormat colorFormat, const std::string &shaderDir)
    : m_vertexBuffer(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT),
      m_indexBuffer(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT),
      m_pipeline(context) {
    Primitives cube = SimpleResource::CreateCube(1.0f);
    std::vector<Vertex> vertices;
    vertices.reserve(cube.vertices.size());
    for (size_t i = 0; i < cube.vertices.size(); ++i) {
        const Eigen::Vector3f color =
                i < cube.colors.size() ? cube.colors[i] : Eigen::Vector3f(1.0f, 1.0f, 1.0f);
        vertices.push_back({
                {cube.vertices[i].x(), cube.vertices[i].y(), cube.vertices[i].z()},
                {color.x(), color.y(), color.z()},
        });
    }
    m_indexCount = static_cast<uint32_t>(cube.indices.size());

    const uint32_t vertexBytes = static_cast<uint32_t>(vertices.size() * sizeof(Vertex));
    m_vertexBuffer.Allocate(vertexBytes);
    m_vertexBuffer.Upload(vertices.data(), vertexBytes, Engine::Core::QueueRole::Graphics);

    const uint32_t indexBytes = static_cast<uint32_t>(cube.indices.size() * sizeof(uint32_t));
    m_indexBuffer.Allocate(indexBytes);
    m_indexBuffer.Upload(cube.indices.data(), indexBytes, Engine::Core::QueueRole::Graphics);

    Engine::Render::GraphicsPipelineDescriptor descriptor;
    descriptor.VertexShader(shaderDir + "/cube.vert.spv")
            .FragmentShader(shaderDir + "/cube.frag.spv")
            .VertexBinding<Vertex>()
            .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position))
            .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
            .ColorTarget(colorFormat)
            .DepthTarget(VK_FORMAT_D32_SFLOAT)
            .PushConstant<PushConstants>(VK_SHADER_STAGE_VERTEX_BIT);
    m_pipeline.Build(descriptor);
}

void CubePass::Execute(Engine::Render::RenderContext &ctx) {
    Engine::Render::ClearOptions clear{};
    clear.color[0] = 0.025f;
    clear.color[1] = 0.027f;
    clear.color[2] = 0.032f;
    clear.color[3] = 1.0f;

    const VkExtent2D extent = ctx.swapChain->Extent();
    auto renderingDescriptor = Engine::Render::RenderingDescriptor::ColorDepth(
            extent, ctx.swapChain->ImageView(ctx.imageIndex), ctx.depthImage->View(), clear);

    Engine::Render::RenderingScope scope(ctx.commandBuffer, renderingDescriptor);

    m_pipeline.Bind(ctx.commandBuffer);
    VkBuffer vertexHandle = m_vertexBuffer.Handle();
    VkDeviceSize vertexOffset = 0;
    vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &vertexHandle, &vertexOffset);
    vkCmdBindIndexBuffer(ctx.commandBuffer, m_indexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

    Engine::Render::Camera *camera = ctx.view->GetCamera();
    PushConstants push{camera->GetProjectionMatrix() * camera->GetViewMatrix()};
    m_pipeline.PushConstants(ctx.commandBuffer, VK_SHADER_STAGE_VERTEX_BIT, push);
    vkCmdDrawIndexed(ctx.commandBuffer, m_indexCount, 1, 0, 0, 0);
}

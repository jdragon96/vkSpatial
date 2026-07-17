#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Render/GraphicsPipeline.h"
#include "Engine/Render/RenderGraph.h"

#include <cstdint>
#include <string>

class CubePass : public Engine::Render::RenderPass {
public:
    CubePass(Engine::Core::Context &context, VkFormat colorFormat, const std::string &shaderDir);

    const char *Name() const override { return "CubePass"; }
    void Execute(Engine::Render::RenderContext &ctx) override;

private:
    Engine::Core::Buffer m_vertexBuffer;
    Engine::Core::Buffer m_indexBuffer;
    Engine::Render::GraphicsPipeline m_pipeline;
    uint32_t m_indexCount = 0;
};

#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Render/GraphicsPipeline.h"
#include "Engine/Render/RenderGraph.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// GPU-side vertex layout for point-cloud rendering: position + packed 8-bit RGBA color.
struct PointVertex {
    float pos[3];
    uint8_t rgba[4];
};

// Renders up to kMaxSets independently-visible named point sets (e.g. TSDF input samples,
// extracted surface points, ...) as VK_PRIMITIVE_TOPOLOGY_POINT_LIST. Mirrors CubePass's
// pass/pipeline structure; see CubePass.{h,cpp}.
class PointCloudPass : public Engine::Render::RenderPass {
public:
    static constexpr int kMaxSets = 4;

    PointCloudPass(Engine::Core::Context &context, VkFormat colorFormat, const std::string &shaderDir);

    const char *Name() const override { return "PointCloudPass"; }
    void Execute(Engine::Render::RenderContext &ctx) override;

    // (Re)uploads `vertices` as the contents of named point set `id` (0..kMaxSets-1).
    // Passing an empty vector hides the set (draw count becomes 0) without deallocating.
    void SetPointSet(int id, const std::vector<PointVertex> &vertices);

    // Toggles whether point set `id` is drawn. All sets are visible by default.
    void SetVisible(int id, bool visible);

    // Point size in pixels, shared by all sets, applied via push constant each frame.
    void SetPointSize(float pixels) { m_pointSize = pixels; }

private:
    struct PointSet {
        explicit PointSet(Engine::Core::Context &context)
            : buffer(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {}

        Engine::Core::Buffer buffer;
        uint32_t count = 0;
        bool visible = true;
    };

    // Engine::Core::Buffer is move/copy-disabled, so PointSet can't live in a std::vector
    // (libc++'s vector::reserve requires Cpp17MoveInsertable even when no reallocation
    // ever actually happens). unique_ptr sidesteps that: each slot is heap-allocated once
    // in the constructor and never relocated.
    std::array<std::unique_ptr<PointSet>, kMaxSets> m_sets;
    Engine::Render::GraphicsPipeline m_pipeline;
    float m_pointSize = 4.0f;
};

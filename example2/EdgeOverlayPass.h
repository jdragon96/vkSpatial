#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Render/GraphicsPipeline.h"
#include "Engine/Render/Object.h"
#include "Engine/Render/RenderGraph.h"

#include "utilities/Math.h"

#include <Eigen/Core>
#include <string>
#include <utility>
#include <vector>

// One highlighted group of world-space line segments sharing a single color -- e.g. every
// non-manifold edge in red, or every boundary edge in yellow (see isosurface_viewer.cpp's
// rebuild(), which builds one EdgeOverlayGroup per
// Engine::Spatial::Extraction::ConnectivityReport category via BuildEdgeOverlayGroup).
struct EdgeOverlayGroup {
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> segments;
    Eigen::Vector3f color;
};

// Draws a DYNAMIC (re-uploadable) VK_PRIMITIVE_TOPOLOGY_LINE_LIST overlay on top of whatever
// IsosurfaceMeshPass rendered this frame -- the mesh-connectivity diagnostic for
// isosurface_viewer.cpp: non-manifold edges and boundary edges highlighted in distinct colors so
// an extractor defect (crack, T-junction, bowtie vertex) is something you SEE, not something you
// have to infer from a wireframe gap. Structurally mirrors IsosurfaceMeshPass (RenderPass +
// Engine::Render::Object vertex/index buffer upload, camera push-constant, geometry re-uploaded
// on demand instead of fixed at construction), with two differences:
//   - the vertex format carries a per-vertex color, baked in by SetEdges from each group's
//     uniform color, so several differently-colored groups (non-manifold red + boundary yellow)
//     draw from one pass in one SetEdges call instead of needing one pass instance per color;
//   - the pipeline has depth-testing DISABLED (see BuildPipeline) so flagged edges stay visible
//     even where they sit behind the mesh surface from the current view angle -- an edge you
//     can't see is a defect you can't see, which defeats the purpose of the overlay.
//
// SetEdges() re-allocates GPU buffers; same between-frames-only calling convention as
// IsosurfaceMeshPass::SetMesh() -- see that class's comment.
class EdgeOverlayPass : public Engine::Render::RenderPass {
public:
    EdgeOverlayPass(Engine::Core::Context &context,
                     VkFormat colorFormat,
                     const std::string &shaderDirectory);

    const char *Name() const override { return "EdgeOverlayPass"; }
    void Execute(Engine::Render::RenderContext &ctx) override;

    // Replaces all previously uploaded edge geometry with the concatenation of `groups` (both
    // vertices of every segment in a group take that group's color). An empty `groups` (or one
    // whose groups all have empty segments) uploads nothing, and Execute() then draws nothing --
    // same empty guard as IsosurfaceMeshPass::SetMesh. Must be called between frames -- see the
    // class comment above.
    void SetEdges(const std::vector<EdgeOverlayGroup> &groups);

    // Toggles whether Execute() draws anything this frame -- a cheap flag flip, no rebuild.
    void SetVisible(bool visible) { m_visible = visible; }
    bool Visible() const { return m_visible; }

private:
    // position/color -- no normal (unlit shader, see EdgeOverlay.frag.glsl) unlike
    // Engine::Render::Vertex, which IsosurfaceMeshPass uses for its lit mesh.
    struct EdgeVertex {
        float position[3] = {};
        float color[3] = {};

        EdgeVertex() = default;

        EdgeVertex(const vkMath::Vec3 &p, const vkMath::Vec3 &c)
            : position{p.x(), p.y(), p.z()}, color{c.x(), c.y(), c.z()} {}
    };

    Engine::Core::Context &m_context;
    Engine::Render::Object<EdgeVertex> m_edgeObject;
    Engine::Render::GraphicsPipeline m_pipeline;
    bool m_visible = true;

    void BuildPipeline(VkFormat colorFormat, const std::string &shaderDirectory);
};

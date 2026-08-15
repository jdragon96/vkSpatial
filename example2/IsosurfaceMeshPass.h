#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Render/GraphicsPipeline.h"
#include "Engine/Render/Object.h"
#include "Engine/Render/RenderGraph.h"
#include "Mesh/SurfaceMesh.h"

#include <Eigen/Core>
#include <string>

// Draws a DYNAMIC (re-uploadable) lit indexed triangle mesh: the live output of whichever
// isosurface extractor the viewer currently has selected. Structurally adapted from CubePass
// (RenderPass + Engine::Render::Object vertex/index buffer upload) but the geometry is
// re-uploaded on demand (SetMesh) instead of fixed at construction, shaded with the mesh's own
// per-vertex normals (Blinn-Phong, same technique as BlinnPhong.cpp) so surface shape and
// creases read clearly, and exposes a wireframe toggle so extractor artifacts -- cracks,
// non-manifold edges -- are visible.
//
// SetMesh() re-allocates GPU buffers; the caller must ensure no in-flight frame still references
// the previous buffers. isosurface_viewer.cpp's render loop waits for the device to go idle
// before calling SetMesh, the same guard tsdf_feature_compare.cpp uses before its own
// between-frames rebuild().
class IsosurfaceMeshPass : public Engine::Render::RenderPass {
public:
    IsosurfaceMeshPass(Engine::Core::Context &context,
                        VkFormat colorFormat,
                        const std::string &shaderDirectory);

    const char *Name() const override { return "IsosurfaceMeshPass"; }
    void Execute(Engine::Render::RenderContext &ctx) override;

    // Converts the SurfaceMesh to render vertices (position + normal from the mesh, color = the
    // caller-supplied debug tint) and indices from mesh.triangles, then re-uploads the
    // vertex/index buffers. Must be called between frames -- see the class comment above.
    void SetMesh(const Mesh::SurfaceMesh &mesh, const Eigen::Vector3f &color);

    // Selects which of the two pipelines built up front Execute() binds: VK_POLYGON_MODE_FILL
    // (default) or VK_POLYGON_MODE_LINE. A cheap flag flip -- no rebuild, no pipeline creation.
    void SetWireframe(bool wireframe) { m_wireframe = wireframe; }
    bool Wireframe() const { return m_wireframe; }

private:
    Engine::Core::Context &m_context;
    Engine::Render::Object<> m_meshObject;
    Engine::Render::GraphicsPipeline m_solidPipeline;
    Engine::Render::GraphicsPipeline m_wireframePipeline;
    bool m_wireframe = false;

    void BuildPipeline(Engine::Render::GraphicsPipeline &pipeline,
                        VkPolygonMode polygonMode,
                        VkFormat colorFormat,
                        const std::string &shaderDirectory);
};

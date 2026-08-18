#pragma once

#include "EdgeOverlayPass.h"    // EdgeOverlayPass
#include "IsosurfaceMeshPass.h" // IsosurfaceMeshPass
#include "Mesh/MeshConnectivity.h" // Mesh::ConnectivityReport
#include "VoxelFillDebug.h" // voxdbg::ColorMode

#include "Pipeline/Render/RenderStrategy.h" // Pipeline::IRenderStrategy
#include "Pipeline/Types.h"                 // Frame, ModelSnapshot

#include "utilities/StageProfiler.h" // util::StageProfiler

#include <Eigen/Core>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Engine::Core {
    class Context;
}
class PointCloudPass; // example2 (global namespace)
class ImGuiPass;      // example2

class VoxelFillRenderStrategy : public Pipeline::IRenderStrategy {
public:
    struct Opts {
        bool submap = true, pointToPlane = true, confidence = true, hermite = false, downsample = false;
    };

    struct Params {
        std::shared_ptr<const std::vector<Pipeline::Frame>> frames; // input clouds (display)
        float voxel = 0.5f;
        float trunc = 1.5f;
        int nFrames = 0;
        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        float extent = 1.0f;
        float wThresh = 0.0f;                        // initial weight threshold
        std::string trackerName;                     // shown in the UI
        Pipeline::MapConfig map;                     // shown in the UI (configuration panel)
        double intervalMs = 0.0;                     // acquisition pacing
        bool loop = false;
        Opts opts;                                   // initial option-toggle state
        // Rebuild the pipeline with the edited opts + map config and replay from frame 0.
        std::function<void(const Opts &, const Pipeline::MapConfig &)> onRebuild;
    };

    explicit VoxelFillRenderStrategy(Params params);

    void Build(Pipeline::Pipeline &pipe, Engine::Render::Application &app,
               Engine::Render::RenderGraph &graph) override;
    void CameraFit(Eigen::Vector3f &center, float &extent) const override;
    void OnModel(std::shared_ptr<const Pipeline::ModelSnapshot> snap) override;
    void OnFrame() override;

private:
    void refresh(); // rebuild the point sets from m_snap + m_state
    void pushEdgeOverlay();
    void applyVisibility(); // push m_renderMode + per-layer flags onto the passes
    void extractMesh();
    void drawStatsPanel(Pipeline::Pipeline &pipe);
    void drawUi(Pipeline::Pipeline &pipe); // ImGui panel (called during the ImGui pass)

    Params m_p;
    Engine::Core::Context *m_ctx = nullptr;
    PointCloudPass *m_pc = nullptr;        // owned by the render graph
    IsosurfaceMeshPass *m_mesh = nullptr;  // owned by the render graph
    EdgeOverlayPass *m_edges = nullptr;    // owned by the render graph

    // Mesh diagnostics from the last extraction. Boundary edges are the holes; non-manifold edges
    // and bowtie vertices are topology defects the extractor produced.
    Mesh::ConnectivityReport m_connectivity;
    Mesh::SurfaceMesh m_lastMesh;
    bool m_showBoundaryEdges = true;
    bool m_showNonManifoldEdges = true;

    // Extraction runs on the CPU over the whole map, so it is on demand (a button), never per
    // frame. m_meshDirty just tells the UI the shown mesh is older than the map.
    // Top-level choice: the reconstructed point cloud, the extracted triangle mesh, or both
    // (the mesh pass loads the framebuffer rather than clearing it, so they compose).
    enum class RenderMode { Points, Mesh, Both };
    RenderMode m_renderMode = RenderMode::Points;

    std::string m_extractorName = "mc";
    bool m_meshWireframe = false;
    bool m_meshDirty = true;
    std::size_t m_meshTriangles = 0;
    double m_meshExtractMs = 0.0;
    bool m_pendingExtract = false;
    ImGuiPass *m_imgui = nullptr;   // owned by the render graph

    struct State {
        voxdbg::ColorMode mode = voxdbg::ColorMode::TsdfSign;
        float wThresh = 0.0f;
        bool hideBelow = false;
        bool showOccupied = true, showNew = true, showInput = true, showCamera = true;
        bool showWindowBox = true, showAllocBox = true, showSubmapBox = true;
        float wMax = 1.0f;
        bool dirty = false;
    } m_state;

    Opts m_opts;                   // live option-toggle state (edited by the UI)
    Pipeline::MapConfig m_map;     // live map config (edited by the UI while paused)
    bool m_configDirty = false;    // config edited but not applied
    bool m_pendingRebuild = false; // a toggle flipped -> rebuild the pipeline on the next frame

    std::shared_ptr<const Pipeline::ModelSnapshot> m_snap;
    util::StageProfiler m_prof;       // render-thread stage times (waitIdle/buildSets/upload)
    util::StageProfiler m_workerProf; // worker stage times accumulated per snapshot (avg over frames)
};

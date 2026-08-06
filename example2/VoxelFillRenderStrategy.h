#pragma once

#include "VoxelFillDebug.h" // voxdbg::ColorMode

#include "Engine/Pipeline/Render/RenderStrategy.h" // Engine::Pipeline::IRenderStrategy
#include "Engine/Pipeline/Types.h"                 // Frame, ModelSnapshot

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

// Concrete render strategy for the voxel-fill debugger: draws the TSDF's occupied voxels (coloured by
// tsdf / weight / fill-frame / direction), the "new this frame" highlight, the input cloud, a camera
// marker, and the tile / alloc / submap boxes, with an ImGui control panel. Owns its PointCloudPass +
// ImGuiPass. RenderThread owns the window/camera/loop and drives Build/OnModel/OnFrame; swapping this
// for another IRenderStrategy renders the same pipeline a different way.
class VoxelFillRenderStrategy : public Engine::Pipeline::IRenderStrategy {
public:
    // Per-stage integration options the UI can flip at runtime. Flipping any rebuilds the pipeline
    // (Pipeline::Reconfigure) and replays from frame 0 -- the only coherent way to change an
    // accumulating map. The app supplies `onRebuild` to apply a new Opts to its pipeline.
    struct Opts {
        bool submap = true, pointToPlane = true, confidence = true, hermite = false, downsample = false;
    };

    struct Params {
        std::shared_ptr<const std::vector<Engine::Pipeline::Frame>> frames; // input clouds (display)
        float voxel = 0.5f;
        float trunc = 1.5f;
        int nFrames = 0;
        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        float extent = 1.0f;
        float wThresh = 0.0f;  // initial weight threshold
        std::string trackerName; // shown in the UI
        Opts opts;             // initial option-toggle state
        std::function<void(const Opts &)> onRebuild; // rebuild the pipeline with new opts (may be null)
    };

    explicit VoxelFillRenderStrategy(Params params);

    void Build(Engine::Pipeline::Pipeline &pipe, Engine::Render::Application &app,
               Engine::Render::RenderGraph &graph) override;
    void CameraFit(Eigen::Vector3f &center, float &extent) const override;
    void OnModel(std::shared_ptr<const Engine::Pipeline::ModelSnapshot> snap) override;
    void OnFrame() override;

private:
    void refresh();                                // rebuild the point sets from m_snap + m_state
    void drawUi(Engine::Pipeline::Pipeline &pipe); // ImGui panel (called during the ImGui pass)

    Params m_p;
    Engine::Core::Context *m_ctx = nullptr;
    PointCloudPass *m_pc = nullptr; // owned by the render graph
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
    bool m_pendingRebuild = false; // a toggle flipped -> rebuild the pipeline on the next frame

    std::shared_ptr<const Engine::Pipeline::ModelSnapshot> m_snap;
    util::StageProfiler m_prof;       // render-thread stage times (waitIdle/buildSets/upload)
    util::StageProfiler m_workerProf; // worker stage times accumulated per snapshot (avg over frames)
};

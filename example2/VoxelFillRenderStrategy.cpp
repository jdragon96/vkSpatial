#include "VoxelFillRenderStrategy.h"

#include "ImGuiPass.h"
#include "PointCloudPass.h"

#include "Engine/Core/Context.h"
#include "Engine/Pipeline/Pipeline.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/GlfwWindow.h"

#include "imgui.h"

#include <algorithm>
#include <cstddef>
#include <utility>

using Eigen::Vector3f;
using Engine::Spatial::AdvancedEntry;
namespace ep = Engine::Pipeline;

namespace {

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Point-set builders. PointCloudPass draws points only, so boxes are sampled as point wireframes
    // and the camera marker as a short point trail.
    ///////////////////////////////////////////////////////////////////////////////////////////

    void pushCloud(std::vector<PointVertex> &out, const std::vector<Vector3f> &pts,
                   uint8_t r, uint8_t g, uint8_t b) {
        for (const auto &p: pts) out.push_back({{p.x(), p.y(), p.z()}, {r, g, b, 255}});
    }

    // A small red camera marker: the eye plus a short segment toward the origin (view direction).
    std::vector<PointVertex> cameraMarker(const Vector3f &eye) {
        std::vector<PointVertex> v;
        for (int i = 0; i < 30; ++i) {
            const float t = float(i) / 30.0f * 0.15f;
            const Vector3f p = eye * (1.0f - t);
            v.push_back({{p.x(), p.y(), p.z()}, {255, 40, 40, 255}});
        }
        return v;
    }

    // Sample points along the 12 edges of an AABB so the box reads as a wireframe in the point renderer.
    std::vector<PointVertex> boxEdges(const Vector3f &mn, const Vector3f &mx, float voxel,
                                      uint8_t r, uint8_t g, uint8_t b) {
        std::vector<PointVertex> v;
        const Vector3f c[8] = {{mn.x(), mn.y(), mn.z()}, {mx.x(), mn.y(), mn.z()}, {mx.x(), mx.y(), mn.z()}, {mn.x(), mx.y(), mn.z()}, {mn.x(), mn.y(), mx.z()}, {mx.x(), mn.y(), mx.z()}, {mx.x(), mx.y(), mx.z()}, {mn.x(), mx.y(), mx.z()}};
        static const int E[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        const int per = std::clamp(int((mx - mn).maxCoeff() / std::max(1e-6f, voxel)), 24, 512);
        for (const auto &e: E)
            for (int i = 0; i <= per; ++i) {
                const float t = float(i) / float(per);
                const Vector3f p = c[e[0]] + t * (c[e[1]] - c[e[0]]);
                v.push_back({{p.x(), p.y(), p.z()}, {r, g, b, 255}});
            }
        return v;
    }

    // Build the "occupied" (coloured by mode; below-threshold dimmed grey or skipped) and "new this
    // frame" highlight point sets from downloaded voxel entries. First-seen frame is carried per entry
    // (AdvancedEntry::firstFrame, GPU-stamped), so the fill-frame colour and "new this frame" come
    // straight off the entries -- no parallel isNew/firstFrame arrays.
    void buildVoxelSets(const std::vector<AdvancedEntry> &entries, voxdbg::ColorMode mode, float trunc,
                        float wMax, float wThresh, bool hideBelow, int currentFrame, int nFrames,
                        std::vector<PointVertex> &occupied, std::vector<PointVertex> &newThis) {
        occupied.clear();
        newThis.clear();
        for (const AdvancedEntry &e: entries) {
            const bool below = voxdbg::belowThreshold(e.weight, wThresh);
            if (below && hideBelow) continue;
            voxdbg::Rgba c;
            switch (mode) {
                case voxdbg::ColorMode::TsdfSign:
                    c = voxdbg::tsdfColor(e.tsdf, trunc);
                    break;
                case voxdbg::ColorMode::Weight:
                    c = voxdbg::weightColor(e.weight, wMax);
                    break;
                case voxdbg::ColorMode::FillFrame:
                    c = voxdbg::fillFrameColor(e.firstFrame, nFrames);
                    break;
                case voxdbg::ColorMode::Direction:
                    c = voxdbg::directionColor(uint8_t(e.direction));
                    break;
            }
            if (below) c = {80, 80, 80, 255}; // extraction drop-out, dimmed
            occupied.push_back({{e.center.x(), e.center.y(), e.center.z()}, {c[0], c[1], c[2], c[3]}});
            if (e.firstFrame == currentFrame)
                newThis.push_back({{e.center.x(), e.center.y(), e.center.z()}, {255, 240, 40, 255}});
        }
    }

} // namespace

VoxelFillRenderStrategy::VoxelFillRenderStrategy(Params params) : m_p(std::move(params)) {
    m_state.wThresh = m_p.wThresh;
    m_opts = m_p.opts;
}

void VoxelFillRenderStrategy::Build(ep::Pipeline &pipe, Engine::Render::Application &app,
                                    Engine::Render::RenderGraph &graph) {
    m_ctx = &app.GetContext();

    const std::string shaderDir = VOXDBG_SHADER_DIR;
    auto pcOwned = std::make_unique<PointCloudPass>(*m_ctx, app.GetSwapChain().Format(), shaderDir);
    m_pc = pcOwned.get();
    graph.AddPass(std::move(pcOwned));

    auto &glfwWindow = static_cast<Engine::Render::GlfwWindow &>(app.GetWindow());
    auto imguiOwned = std::make_unique<ImGuiPass>(*m_ctx, glfwWindow.Handle(),
                                                  app.GetSwapChain().Format(),
                                                  app.GetSwapChain().ImageCount());
    m_imgui = imguiOwned.get();
    graph.AddPass(std::move(imguiOwned));

    m_pc->SetPointSize(3.0f);
    m_imgui->SetUi([this, &pipe]() { drawUi(pipe); });
}

void VoxelFillRenderStrategy::CameraFit(Eigen::Vector3f &center, float &extent) const {
    center = m_p.center;
    extent = m_p.extent;
}

void VoxelFillRenderStrategy::OnModel(std::shared_ptr<const ep::ModelSnapshot> snap) {
    m_snap = std::move(snap);
    m_state.wMax = 1.0f;
    for (const auto &e: m_snap->entries) m_state.wMax = std::max(m_state.wMax, e.weight);
    m_state.wThresh = std::min(m_state.wThresh, m_state.wMax);
    m_state.dirty = true;

    // Accumulate this frame's worker-stage times so the UI can show a running average per stage.
    m_workerProf.Add("integrate", m_snap->integrateMs);
    m_workerProf.Add("download", m_snap->downloadMs);
    m_workerProf.Add("tracker", m_snap->trackerMs);
}

void VoxelFillRenderStrategy::OnFrame() {
    // A UI option toggle rebuilds the pipeline and replays from frame 0. Do it here (between frames,
    // before rendering) rather than inside the ImGui pass, so no render frame is in flight. The fresh
    // map starts empty, so clear the reconstructed layers + drop the stale snapshot immediately.
    if (m_pendingRebuild) {
        m_pendingRebuild = false;
        if (m_p.onRebuild) {
            vkDeviceWaitIdle(m_ctx->device); // no frame in flight -> safe to reupload point sets
            m_p.onRebuild(m_opts);
            m_snap.reset();
            m_pc->SetPointSet(0, {}); // occupied voxels
            m_pc->SetPointSet(1, {}); // new-this-frame voxels
        }
    }
    if (m_state.dirty && m_snap) {
        refresh();
        m_state.dirty = false;
    }
}

// Rebuild the render point-sets from the immutable snapshot. SetPointSet reallocates PointCloudPass
// buffers; the prior frame's command buffer may still be in flight, so wait once here first.
void VoxelFillRenderStrategy::refresh() {
    {
        util::ScopedStageTimer t(m_prof, "waitIdle");
        vkDeviceWaitIdle(m_ctx->device);
    }
    const ep::ModelSnapshot &s = *m_snap;
    std::vector<PointVertex> occ, nw;
    {
        util::ScopedStageTimer t(m_prof, "buildSets");
        buildVoxelSets(s.entries, m_state.mode, m_p.trunc, m_state.wMax, m_state.wThresh,
                       m_state.hideBelow, s.processedFrame, m_p.nFrames, occ, nw);
    }
    util::ScopedStageTimer t(m_prof, "upload"); // CPU box gen + all 7 SetPointSet uploads
    m_pc->SetPointSet(0, occ);
    m_pc->SetPointSet(1, nw);

    if (!m_p.frames->empty()) {
        const int pf = std::clamp(s.processedFrame, 0, int(m_p.frames->size()) - 1);
        std::vector<PointVertex> in;
        pushCloud(in, (*m_p.frames)[pf].pts, 100, 110, 120);
        m_pc->SetPointSet(2, in);
        m_pc->SetPointSet(3, cameraMarker((*m_p.frames)[pf].cam));
    }

    std::vector<PointVertex> tileBoxes;
    for (const auto &b: s.baseCoreBoxes) {
        const std::vector<PointVertex> e = boxEdges(b.first, b.second, m_p.voxel, 40, 220, 220);
        tileBoxes.insert(tileBoxes.end(), e.begin(), e.end());
    }
    m_pc->SetPointSet(4, tileBoxes);
    if (s.hasAlloc)
        m_pc->SetPointSet(5, boxEdges(s.allocMin, s.allocMax, m_p.voxel, 255, 160, 40));
    else
        m_pc->SetPointSet(5, {});

    std::vector<PointVertex> submapBoxes;
    for (const auto &b: s.denseBlockBoxes) {
        const std::vector<PointVertex> e = boxEdges(b.first, b.second, m_p.voxel, 230, 60, 230);
        submapBoxes.insert(submapBoxes.end(), e.begin(), e.end());
    }
    m_pc->SetPointSet(6, submapBoxes);
}

void VoxelFillRenderStrategy::drawUi(ep::Pipeline &pipe) {
    ImGui::Begin("Voxel Fill Debug");
    // Play / pause the acquisition (mapping + render keep running regardless).
    if (pipe.IsPaused()) {
        if (ImGui::Button("Play")) pipe.SetPaused(false);
    } else {
        if (ImGui::Button("Pause")) pipe.SetPaused(true);
    }
    ImGui::SameLine();
    ImGui::Text("%s", pipe.IsPaused() ? "(paused)" : "(playing)");
    ImGui::Text("frame %d / %d (source: %s)", pipe.ProcessedFrame() + 1, m_p.nFrames,
                m_p.trackerName.c_str());
    ImGui::SeparatorText("Color mode");
    int m = int(m_state.mode);
    bool cm = false;
    cm |= ImGui::RadioButton("tsdf", &m, 0);
    ImGui::SameLine();
    cm |= ImGui::RadioButton("weight", &m, 1);
    ImGui::SameLine();
    cm |= ImGui::RadioButton("fill-frame", &m, 2);
    ImGui::SameLine();
    cm |= ImGui::RadioButton("direction", &m, 3);
    if (cm) {
        m_state.mode = voxdbg::ColorMode(m);
        m_state.dirty = true;
    }
    if (ImGui::SliderFloat("weight thresh", &m_state.wThresh, 0.0f, m_state.wMax, "%.3f"))
        m_state.dirty = true;
    if (ImGui::Checkbox("hide below thresh", &m_state.hideBelow)) m_state.dirty = true;
    ImGui::SeparatorText("Layers");
    if (ImGui::Checkbox("occupied", &m_state.showOccupied)) m_pc->SetVisible(0, m_state.showOccupied);
    if (ImGui::Checkbox("new this frame", &m_state.showNew)) m_pc->SetVisible(1, m_state.showNew);
    if (ImGui::Checkbox("input", &m_state.showInput)) m_pc->SetVisible(2, m_state.showInput);
    if (ImGui::Checkbox("camera", &m_state.showCamera)) m_pc->SetVisible(3, m_state.showCamera);
    if (ImGui::Checkbox("base tile windows", &m_state.showWindowBox))
        m_pc->SetVisible(4, m_state.showWindowBox);
    if (ImGui::Checkbox("allocated box", &m_state.showAllocBox))
        m_pc->SetVisible(5, m_state.showAllocBox);
    if (ImGui::Checkbox("submap regions (detail)", &m_state.showSubmapBox))
        m_pc->SetVisible(6, m_state.showSubmapBox);

    // Per-stage option toggles. Flipping any checkbox rebuilds the pipeline with the new config and
    // replays from frame 0 (handled in OnFrame via m_p.onRebuild). Only shown when a rebuild hook exists.
    if (m_p.onRebuild) {
        ImGui::SeparatorText("Pipeline options (rebuild + replay)");
        bool changed = false;
        changed |= ImGui::Checkbox("submap (detail)", &m_opts.submap);
        changed |= ImGui::Checkbox("downsample (voxel/2)", &m_opts.downsample);
        changed |= ImGui::Checkbox("point-to-plane", &m_opts.pointToPlane);
        changed |= ImGui::Checkbox("confidence (A1)", &m_opts.confidence);
        changed |= ImGui::Checkbox("hermite (A2)", &m_opts.hermite);
        if (changed) m_pendingRebuild = true;
    }
    ImGui::SeparatorText("Stats");
    const ep::PipelineStats ps = pipe.GetStats();
    ImGui::Text("tracker: %s   integrated frames: %d", m_p.trackerName.c_str(), ps.processedFrame + 1);
    ImGui::Text("queues: capture %zu, track %zu (dropped %zu)", ps.captureDepth, ps.trackDepth,
                ps.trackDropped);

    // Per-pipeline-STAGE average cost: each worker's mean per-frame time + its processed count. A
    // count that keeps rising while playing = that thread is alive (drawn green); grey = idle/not run.
    ImGui::SeparatorText("Pipeline stages (avg ms / frames)");
    auto stageLine = [](const char *name, double avgMs, unsigned long long count) {
        const ImVec4 col = count > 0 ? ImVec4(0.40f, 0.90f, 0.45f, 1.0f)  // alive
                                     : ImVec4(0.60f, 0.60f, 0.60f, 1.0f); // idle
        ImGui::TextColored(col, "%-14s %8.2f / %llu", name, avgMs, count);
    };
    stageLine("Reconstruction", ps.acquireMsAvg, (unsigned long long) ps.acquiredFrames);
    stageLine("ICP", ps.alignMsAvg, (unsigned long long) ps.alignedFrames);
    stageLine("Integration", ps.integrateMsAvg, (unsigned long long) ps.integratedFrames);
    if (m_snap) {
        ImGui::Text("occupied voxels: %zu", m_snap->entries.size());
        std::size_t below = 0;
        for (const auto &e: m_snap->entries)
            if (voxdbg::belowThreshold(e.weight, m_state.wThresh)) ++below;
        ImGui::Text("below thresh: %zu", below);
        ImGui::Text("base/detail tiles: %u / %u, dense blocks: %u", m_snap->baseTiles,
                    m_snap->detailTiles, m_snap->denseBlocks);
        if (m_snap->hasAlloc) {
            const Vector3f sz = m_snap->allocMax - m_snap->allocMin;
            ImGui::Text("allocated box: %.2f x %.2f x %.2f", sz.x(), sz.y(), sz.z());
        }
        ImGui::SeparatorText("Worker stage times (avg / last ms)");
        for (const char *s: {"integrate", "download", "tracker"})
            ImGui::Text("%-10s %8.2f / %8.2f", s, m_workerProf.AvgMs(s), m_workerProf.LastMs(s));
        const double workerAvg = m_workerProf.AvgMs("integrate") + m_workerProf.AvgMs("download") +
                                 m_workerProf.AvgMs("tracker");
        ImGui::Text("%-10s %8.2f", "map total", workerAvg); // avg per-frame Integration-stage cost
    } else {
        ImGui::Text("(waiting for first snapshot...)");
    }
    ImGui::SeparatorText("Render stage times (avg / last ms)");
    for (const char *s: {"waitIdle", "buildSets", "upload"})
        ImGui::Text("%-10s %8.2f / %8.2f", s, m_prof.AvgMs(s), m_prof.LastMs(s));
    ImGui::End();
}

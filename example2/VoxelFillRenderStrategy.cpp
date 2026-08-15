#include "VoxelFillRenderStrategy.h"

#include "ImGuiPass.h"
#include "Mesh/ExtractorRegistry.h"
#include "Mesh/VoxelField.h"
#include "PointCloudPass.h"

#include "Engine/Core/Context.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/GlfwWindow.h"
#include "Pipeline/Pipeline.h"

#include "imgui.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <utility>

using Eigen::Vector3f;

namespace ep = Pipeline;

namespace {

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Point-set builders. PointCloudPass draws points only, so boxes are sampled as point wireframes
    // and the camera marker as a short point trail.
    ///////////////////////////////////////////////////////////////////////////////////////////

    void pushCloud(std::vector<PointVertex> &out, const std::vector<Vector3f> &pts,
                   uint8_t r, uint8_t g, uint8_t b) {
        for (const auto &p: pts) out.push_back({{p.x(), p.y(), p.z()}, {r, g, b, 255}});
    }

    std::vector<PointVertex> cameraMarker(const Vector3f &eye) {
        std::vector<PointVertex> v;
        for (int i = 0; i < 30; ++i) {
            const float t = float(i) / 30.0f * 0.15f;
            const Vector3f p = eye * (1.0f - t);
            v.push_back({{p.x(), p.y(), p.z()}, {255, 40, 40, 255}});
        }
        return v;
    }

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

    void buildVoxelSets(const std::vector<TSDFVoxel> &entries, voxdbg::ColorMode mode, float trunc,
                        float wMax, float wThresh, bool hideBelow, int currentFrame, int nFrames,
                        std::vector<PointVertex> &occupied, std::vector<PointVertex> &newThis) {
        occupied.clear();
        newThis.clear();
        for (const TSDFVoxel &e: entries) {
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
    m_map = m_p.map;
}

void VoxelFillRenderStrategy::Build(ep::Pipeline &pipe, Engine::Render::Application &app,
                                    Engine::Render::RenderGraph &graph) {
    m_ctx = &app.GetContext();

    const std::string shaderDir = VOXDBG_SHADER_DIR;
    auto pcOwned = std::make_unique<PointCloudPass>(*m_ctx, app.GetSwapChain().Format(), shaderDir);
    m_pc = pcOwned.get();
    graph.AddPass(std::move(pcOwned));

    // Triangle mesh of the extracted iso-surface. Added before ImGuiPass so the UI draws on top.
    auto meshOwned = std::make_unique<IsosurfaceMeshPass>(*m_ctx, app.GetSwapChain().Format(),
                                                          VOXDBG_SHADER_DIR);
    m_mesh = meshOwned.get();
    m_mesh->SetClears(false); // PointCloudPass already cleared and drew

    graph.AddPass(std::move(meshOwned));

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
    m_meshDirty = true;

    // Accumulate this frame's worker-stage times so the UI can show a running average per stage.
    m_workerProf.Add("integrate", m_snap->integrateMs);
    m_workerProf.Add("download", m_snap->downloadMs);
    m_workerProf.Add("tracker", m_snap->trackerMs);
}

void VoxelFillRenderStrategy::OnFrame() {
    if (m_pendingRebuild) {
        m_pendingRebuild = false;
        if (m_p.onRebuild) {
            // no frame in flight -> safe to reupload point sets
            vkDeviceWaitIdle(m_ctx->device);
            m_p.onRebuild(m_opts, m_map);
            // The render side sizes its box outlines off the voxel, so it follows the edit.
            m_p.voxel = m_map.baseVoxel;
            m_p.trunc = m_map.truncation;
            m_snap.reset();
            m_pc->SetPointSet(0, {}); // occupied voxels
            m_pc->SetPointSet(1, {}); // new-this-frame voxels
        }
    }
    if (m_pendingExtract) {
        m_pendingExtract = false;
        extractMesh();
    }
    if (m_state.dirty && m_snap) {
        refresh();
        m_state.dirty = false;
    }
}

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

// Marching-Cubes family extraction over the whole downloaded map. CPU, seconds on a large scan --
// hence the explicit button rather than a per-frame refresh.
// Single place that decides what is drawn: the mode gates whole families, the per-layer flags
// pick within the point cloud. Anything that changes either calls this rather than poking a pass.
void VoxelFillRenderStrategy::applyVisibility() {
    const bool points = m_renderMode != RenderMode::Mesh;
    if (m_pc) {
        m_pc->SetVisible(0, points && m_state.showOccupied);
        m_pc->SetVisible(1, points && m_state.showNew);
        m_pc->SetVisible(2, points && m_state.showInput);
        m_pc->SetVisible(3, points && m_state.showCamera);
        m_pc->SetVisible(4, points && m_state.showWindowBox);
        m_pc->SetVisible(5, points && m_state.showAllocBox);
        m_pc->SetVisible(6, points && m_state.showSubmapBox);
    }
    // Nothing extracted yet -> nothing to show, whatever the mode says.
    if (m_mesh) m_mesh->SetVisible(m_renderMode != RenderMode::Points && m_meshTriangles > 0);
}

void VoxelFillRenderStrategy::extractMesh() {
    if (!m_snap || !m_mesh) return;

    const auto start = std::chrono::steady_clock::now();
    const Mesh::VoxelField field = Mesh::FromVoxels(m_snap->entries, m_p.voxel);

    Mesh::ExtractorRegistry registry = Mesh::ExtractorRegistry::Default();
    std::unique_ptr<Mesh::IsoSurfaceExtractor> extractor = registry.Create(m_extractorName);
    if (!extractor) return;

    Mesh::ExtractParams params;
    params.isoLevel = 0.0f;
    const Mesh::SurfaceMesh mesh = extractor->Extract(field, params);

    vkDeviceWaitIdle(m_ctx->device); // SetMesh reallocates buffers a frame may still reference
    m_mesh->SetMesh(mesh, Eigen::Vector3f(0.85f, 0.85f, 0.90f));
    m_meshTriangles = mesh.triangles.size();
    m_meshExtractMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    m_meshDirty = false;
    if (m_renderMode == RenderMode::Points) m_renderMode = RenderMode::Both;
    applyVisibility();
}

void VoxelFillRenderStrategy::drawStatsPanel(ep::Pipeline &pipe) {
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    const float margin = 10.0f;
    const float width = 380.0f;
    ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x - width - margin,
                             viewport->WorkPos.y + margin},
                            ImGuiCond_Always);
    ImGui::SetNextWindowSize({width, viewport->WorkSize.y - 2.0f * margin}, ImGuiCond_Always);
    ImGui::Begin("Pipeline Stats", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

    // Editable while paused, locked while playing: every field here forces a full rebuild and a
    // replay from frame 0, so changing one mid-scan would silently discard the map being watched.
    if (ImGui::CollapsingHeader("Configuration", ImGuiTreeNodeFlags_DefaultOpen)) {
        const bool locked = !pipe.IsPaused();
        if (locked) ImGui::TextDisabled("(pause to edit)");

        ImGui::BeginDisabled(locked);
        bool edited = false;

        // Typed, not dragged: these are exact values an experiment is defined by (voxel 0.05, hash
        // 1<<19), and a drag can never land on them. Commit on Enter so a half-typed field never
        // reaches the config.
        const ImGuiInputTextFlags commit = ImGuiInputTextFlags_EnterReturnsTrue;

        // A third of ImGui's default field width: these hold short numbers, and the narrow field
        // leaves room for the derived value printed beside it.
        ImGui::PushItemWidth(ImGui::CalcItemWidth() / 3.0f);

        if (ImGui::InputFloat("base voxel (m)", &m_map.baseVoxel, 0.0f, 0.0f, "%.4f", commit)) {
            m_map.baseVoxel = std::max(1e-4f, m_map.baseVoxel);
            edited = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("detail %.4f", m_map.baseVoxel * 0.5f);

        // The integrate splat cost scales ~(truncation/voxel)^3, so the ratio is what is actually
        // tuned -- typed in voxels, stored in metres.
        float truncationVoxels = m_map.baseVoxel > 0.0f ? m_map.truncation / m_map.baseVoxel : 2.0f;
        if (ImGui::InputFloat("truncation (voxels)", &truncationVoxels, 0.0f, 0.0f, "%.4f", commit)) {
            m_map.truncation = std::max(0.1f, truncationVoxels) * m_map.baseVoxel;
            edited = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%.4f m", m_map.truncation);

        int hashSlots = int(m_map.tileHash);
        if (ImGui::InputInt("hash slots/window", &hashSlots, 0, 0, commit)) {
            m_map.tileHash = uint32_t(std::max(1024, hashSlots));
            edited = true;
        }

        int maxPoints = int(m_map.maxPoints);
        if (ImGui::InputInt("max points/frame", &maxPoints, 0, 0, commit)) {
            m_map.maxPoints = uint32_t(std::max(1024, maxPoints));
            edited = true;
        }

        // The classifier packs a cell coordinate within its block into 6 bits per axis, so wider
        // than 32 aliases distinct cells onto one key.
        if (ImGui::InputInt("splitter block (vox)", &m_map.blockVoxels, 0, 0, commit)) {
            m_map.blockVoxels = std::clamp(m_map.blockVoxels, 1, 32);
            edited = true;
        }

        int directions = int(m_map.maxDirections);
        if (ImGui::InputInt("max directions", &directions, 0, 0, commit)) {
            m_map.maxDirections = uint32_t(std::clamp(directions, 1, 6));
            edited = true;
        }

        int exponent = int(m_map.directionExponent);
        if (ImGui::InputInt("direction exponent", &exponent, 0, 0, commit)) {
            m_map.directionExponent = uint32_t(std::clamp(exponent, 1, 8));
            edited = true;
        }

        ImGui::PopItemWidth();

        edited |= ImGui::Checkbox("view-angle weight", &m_map.viewAngleWeight);
        ImGui::SameLine();
        edited |= ImGui::Checkbox("probe stats", &m_map.probeStats);

        if (edited) m_configDirty = true;

        if (m_configDirty) {
            ImGui::TextColored(ImVec4(1.0f, 0.80f, 0.35f, 1.0f), "edited -- apply to rebuild");
            if (ImGui::Button("Apply & replay")) {
                m_configDirty = false;
                m_pendingRebuild = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Revert")) {
                m_map = m_p.map;
                m_configDirty = false;
            }
        }
        ImGui::EndDisabled();

        ImGui::Separator();
        ImGui::Text("window: 512 vox  (%.2f m)", m_map.baseVoxel * 512.0f);
        ImGui::Text("toggles: submap %s, p2p %s, conf %.2f, hermite %s", m_opts.submap ? "on" : "off",
                    m_opts.pointToPlane ? "on" : "off", m_opts.confidence ? m_map.confidence : 0.0f,
                    m_opts.hermite ? "on" : "off");
        ImGui::Text("source: %d frames, interval %.0f ms%s", m_p.nFrames, m_p.intervalMs,
                    m_p.loop ? ", loop" : "");
        ImGui::Text("tracker: %s", m_p.trackerName.c_str());
    }

    const ep::PipelineStats ps = pipe.GetStats();

    // A stage that has processed something is alive (green); one that has not is idle (grey).
    auto stageHeader = [](const char *name, unsigned long long frames, double avgMs) {
        const ImVec4 color = frames > 0 ? ImVec4(0.40f, 0.90f, 0.45f, 1.0f)
                                        : ImVec4(0.60f, 0.60f, 0.60f, 1.0f);
        ImGui::SeparatorText(name);
        ImGui::TextColored(color, "%llu frames   %.2f ms avg", frames, avgMs);
    };

    stageHeader("Reconstruction (acquire)", (unsigned long long) ps.acquiredFrames, ps.acquireMsAvg);
    ImGui::Text("queue depth: %zu", ps.captureDepth);

    stageHeader("Track (ICP)", (unsigned long long) ps.alignedFrames, ps.alignMsAvg);
    ImGui::Text("tracker: %s", m_p.trackerName.c_str());
    ImGui::Text("rmse (avg): %.4f", ps.trackerRmseAvg);
    ImGui::Text("queue depth: %zu   dropped: %zu", ps.trackDepth, ps.trackDropped);

    stageHeader("Map (TSDF integrate)", (unsigned long long) ps.integratedFrames, ps.integrateMsAvg);
    for (const char *stage: {"integrate", "download", "tracker"})
        ImGui::Text("%-10s %7.2f / %7.2f ms  (avg/last)", stage, m_workerProf.AvgMs(stage),
                    m_workerProf.LastMs(stage));

    if (!m_snap) {
        ImGui::TextDisabled("waiting for the first integrated frame (press Play)");
    } else {
        const TSDFBackendStats &map = m_snap->map;
        const double megabyte = 1024.0 * 1024.0;

        ImGui::SeparatorText("Map contents");
        std::size_t below = 0;
        for (const auto &e: m_snap->entries)
            if (voxdbg::belowThreshold(e.weight, m_state.wThresh)) ++below;
        ImGui::Text("occupied: %zu voxels   below thresh: %zu", m_snap->entries.size(), below);
        ImGui::Text("windows: %u  (base %u, detail %u)", map.tableCount, m_snap->baseTiles,
                    m_snap->detailTiles);
        ImGui::Text("dense blocks: %u", m_snap->denseBlocks);
        if (m_snap->hasAlloc) {
            const Vector3f size = m_snap->allocMax - m_snap->allocMin;
            ImGui::Text("allocated box: %.2f x %.2f x %.2f", size.x(), size.y(), size.z());
        }

        ImGui::SeparatorText("Map memory / hash");
        ImGui::Text("memory: %.1f MB", double(map.deviceMemoryBytes) / megabyte);
        ImGui::Text("  %.2f MB/window   %.1f B/voxel",
                    map.tableCount ? double(map.deviceMemoryBytes) / megabyte / map.tableCount : 0.0,
                    map.filledCount ? double(map.deviceMemoryBytes) / double(map.filledCount) : 0.0);
        ImGui::Text("hash: %llu / %llu slots", (unsigned long long) map.filledCount,
                    (unsigned long long) map.hashCapacity);
        ImGui::Text("  load %.3f   grows %llu", map.LoadFactor(),
                    (unsigned long long) map.growCount);

        if (map.probeQueryCount > 0) {
            const double alpha = map.LoadFactor();
            const double predicted = alpha < 1.0 ? 0.5 * (1.0 + 1.0 / (1.0 - alpha)) : 0.0;
            ImGui::Text("probes: %.2f slots/lookup", map.AverageProbes());
            ImGui::Text("  worst %u   linear-probe theory %.2f", map.probeSlotMax, predicted);
        } else {
            ImGui::TextDisabled("probes: start with --probe-stats to measure");
        }

        if (map.insertFailureCount > 0)
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f), "DROPPED %llu observation(s)",
                               (unsigned long long) map.insertFailureCount);
        if (m_snap->windowLimitRefusals > 0)
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f),
                               "REFUSED %u point(s) (window ceiling)", m_snap->windowLimitRefusals);
    }

    ImGui::SeparatorText("Render");
    const ImGuiIO &io = ImGui::GetIO();
    ImGui::Text("%.1f fps   %.2f ms/frame", io.Framerate, 1000.0f / io.Framerate);

    ImGui::End();
}

void VoxelFillRenderStrategy::drawUi(ep::Pipeline &pipe) {
    // Anchored to the window's top-left, mirroring the stats panel on the right.
    {
        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        const float margin = 10.0f;
        ImGui::SetNextWindowPos({viewport->WorkPos.x + margin, viewport->WorkPos.y + margin},
                                ImGuiCond_Always);
        ImGui::SetNextWindowSize({360.0f, viewport->WorkSize.y - 2.0f * margin}, ImGuiCond_Always);
    }
    ImGui::Begin("Voxel Fill Debug", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
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
    ImGui::SeparatorText("Render");
    {
        int mode = int(m_renderMode);
        bool changed = ImGui::RadioButton("point cloud", &mode, 0);
        ImGui::SameLine();
        changed |= ImGui::RadioButton("mesh", &mode, 1);
        ImGui::SameLine();
        changed |= ImGui::RadioButton("both", &mode, 2);
        if (changed) {
            m_renderMode = RenderMode(mode);
            applyVisibility();
        }
        if (m_renderMode != RenderMode::Points && m_meshTriangles == 0)
            ImGui::TextDisabled("no mesh yet -- extract below");
    }

    ImGui::SeparatorText("Iso-surface mesh");
    {
        static const char *kExtractors[] = {"mc", "mc33", "mtet", "emc", "dc", "dmc", "cms"};
        int current = 0;
        for (int i = 0; i < IM_ARRAYSIZE(kExtractors); ++i)
            if (m_extractorName == kExtractors[i]) current = i;
        ImGui::PushItemWidth(ImGui::CalcItemWidth() / 3.0f);
        if (ImGui::Combo("extractor", &current, kExtractors, IM_ARRAYSIZE(kExtractors))) {
            m_extractorName = kExtractors[current];
            m_meshDirty = true;
        }
        ImGui::PopItemWidth();

        // Extraction walks the whole map on the CPU, so it never runs per frame.
        if (ImGui::Button("Extract mesh")) m_pendingExtract = true;
        ImGui::SameLine();
        if (ImGui::Checkbox("wireframe", &m_meshWireframe)) m_mesh->SetWireframe(m_meshWireframe);

        if (m_meshTriangles > 0)
            ImGui::Text("%zu triangles   %.0f ms", m_meshTriangles, m_meshExtractMs);
        if (m_meshDirty && m_meshTriangles > 0)
            ImGui::TextColored(ImVec4(1.0f, 0.80f, 0.35f, 1.0f), "map changed -- re-extract");
    }

    ImGui::SeparatorText("Layers");
    if (ImGui::Checkbox("occupied", &m_state.showOccupied)) applyVisibility();
    if (ImGui::Checkbox("new this frame", &m_state.showNew)) applyVisibility();
    if (ImGui::Checkbox("input", &m_state.showInput)) applyVisibility();
    if (ImGui::Checkbox("camera", &m_state.showCamera)) applyVisibility();
    if (ImGui::Checkbox("base tile windows", &m_state.showWindowBox))
        applyVisibility();
    if (ImGui::Checkbox("allocated box", &m_state.showAllocBox))
        applyVisibility();
    if (ImGui::Checkbox("submap regions (detail)", &m_state.showSubmapBox))
        applyVisibility();

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
    ImGui::End();

    drawStatsPanel(pipe);
}

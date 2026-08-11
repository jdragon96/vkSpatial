// Isosurface (Marching-Cubes family) debug viewer.
//
// Extracts a surface with a SELECTABLE isosurface extractor (mc/mc33/mtet/emc/dc/dmc/cms; see
// Engine::Spatial::Extraction::ExtractorRegistry) over a selectable input shape -- an analytic
// sphere/box/torus, or (with --dir) a folder of pre-registered frame_*.ply scans integrated into
// an Engine::Spatial::AdvancedTSDF -- and renders the resulting triangle mesh so the differences
// between algorithms (sharp-feature preservation, cracks, topology, rounding) are visible.
// Switching the extractor / shape / cellSize / featureAngle live re-extracts and re-renders;
// wireframe toggles instantly with no re-extraction.
//
// Run:
//   ./build/example2/isosurface_viewer                              analytic sphere/box/torus
//   ./build/example2/isosurface_viewer --extractor dc                start on a given extractor
//   ./build/example2/isosurface_viewer --dir scan_out --voxel 0.5    also offers the "scan" shape
//
// Args:
//   --dir <folder>      optional; folder of frame_*.ply to also offer as the "scan" shape
//   --voxel <float>     cellSize for BOTH analytic and scan fields (default 0.05)
//   --extractor <name>  initial extractor: mc|mc33|mtet|emc|dc|dmc|cms (default mc)
//
// Controls: left-drag orbits, scroll zooms (same trackball as tsdf_feature_compare.cpp /
// BlinnPhong.cpp); Tab cycles the extractor; Escape closes the window.
//
// What to look for: at a coarse cellSize, box/torus should visibly show emc/dc/cms preserving
// sharp edges while mc rounds them off, and dc's ambiguous-face cracks should appear as gaps in
// the wireframe view.

#include "IsosurfaceMeshPass.h"
#include "ImGuiPass.h"

#include "Engine/Core/Context.h"
#include "Engine/Pipeline/Reconstruction/FrameLoader.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"
#include "Engine/Spatial/AdvancedTSDF.h"
#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Spatial/Extraction/VoxelField.h"

#include "utilities/Math.h"

#include "imgui.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef ISOSURFACE_VIEWER_SHADER_DIR
#define ISOSURFACE_VIEWER_SHADER_DIR "."
#endif

namespace fs = std::filesystem;
namespace ep = Engine::Pipeline;
using Eigen::Vector3f;
using Engine::Spatial::Extraction::ExtractorRegistry;
using Engine::Spatial::Extraction::ExtractParams;
using Engine::Spatial::Extraction::IsoSurfaceExtractor;
using Engine::Spatial::Extraction::SurfaceMesh;
using Engine::Spatial::Extraction::VoxelField;

namespace {

    /// ---------------------------------------------------------------------------------------
    /// Mesh-topology predicates -- copied from test/isosurface_test_util.h (IsEdgeManifold /
    /// IsWatertight, ~10 lines) so this viewer has no dependency on the test target. Keep in
    /// sync with that file if the definition there ever changes.
    /// ---------------------------------------------------------------------------------------

    std::map<std::pair<int, int>, int> ComputeEdgeIncidenceCounts(const SurfaceMesh &mesh) {
        std::map<std::pair<int, int>, int> counts;
        auto addEdge = [&](int a, int b) { counts[{std::min(a, b), std::max(a, b)}]++; };
        for (const Eigen::Vector3i &triangle : mesh.triangles) {
            addEdge(triangle[0], triangle[1]);
            addEdge(triangle[1], triangle[2]);
            addEdge(triangle[2], triangle[0]);
        }
        return counts;
    }

    bool IsEdgeManifold(const SurfaceMesh &mesh) {
        for (const auto &edgeCount : ComputeEdgeIncidenceCounts(mesh))
            if (edgeCount.second > 2) return false;
        return true;
    }

    bool IsWatertight(const SurfaceMesh &mesh) {
        const std::map<std::pair<int, int>, int> counts = ComputeEdgeIncidenceCounts(mesh);
        if (counts.empty()) return false;
        for (const auto &edgeCount : counts)
            if (edgeCount.second != 2) return false;
        return true;
    }

    /// ---------------------------------------------------------------------------------------
    /// Analytic VoxelField builders -- mirror SphereField/BoxField in test/isosurface_test_util.h
    /// (same FromImplicit adapter, same narrow-band VoxelField), plus a new torus. Kept in the
    /// viewer TU (not the shared test header) per the design: these are demo shapes, not fixtures.
    /// ---------------------------------------------------------------------------------------

    constexpr float kSphereRadius = 1.0f;
    const Eigen::Vector3f kBoxHalfExtents(1.0f, 0.65f, 0.85f); // non-uniform: shows distinct edges
    constexpr float kTorusMajorRadius = 1.0f;
    constexpr float kTorusMinorRadius = 0.35f;

    // Half the grid's cell span along one axis: enough cells to cover the shape's world-space
    // half-extent at the given cellSize, plus a few cells of margin.
    int ComputeHalfCellCount(float worldHalfExtent, float cellSize) {
        return static_cast<int>(std::ceil(worldHalfExtent / std::max(cellSize, 1e-4f))) + 3;
    }

    // Central-difference gradient of `valueFunction` at `point`, normalized. Shared by
    // BuildBoxField/BuildTorusField (both lack a convenient closed-form gradient).
    Eigen::Vector3f CentralDifferenceGradient(const std::function<float(const Eigen::Vector3f &)> &valueFunction,
                                              const Eigen::Vector3f &point) {
        const float epsilon = 1e-3f;
        Eigen::Vector3f gradient;
        gradient.x() = valueFunction(point + Eigen::Vector3f(epsilon, 0.0f, 0.0f)) -
                       valueFunction(point - Eigen::Vector3f(epsilon, 0.0f, 0.0f));
        gradient.y() = valueFunction(point + Eigen::Vector3f(0.0f, epsilon, 0.0f)) -
                       valueFunction(point - Eigen::Vector3f(0.0f, epsilon, 0.0f));
        gradient.z() = valueFunction(point + Eigen::Vector3f(0.0f, 0.0f, epsilon)) -
                       valueFunction(point - Eigen::Vector3f(0.0f, 0.0f, epsilon));
        const float gradientNorm = gradient.norm();
        return gradientNorm > 1e-6f ? Eigen::Vector3f(gradient / gradientNorm) : Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    }

    // Sphere signed field (value = |p| - radius), exact gradient = p/|p|.
    VoxelField BuildSphereField(float cellSize) {
        const int halfCellCount = ComputeHalfCellCount(kSphereRadius, cellSize);
        const std::array<int, 3> minCoord{-halfCellCount, -halfCellCount, -halfCellCount};
        const std::array<int, 3> maxCoord{halfCellCount, halfCellCount, halfCellCount};

        std::function<float(const Eigen::Vector3f &)> valueFunction = [](const Eigen::Vector3f &point) {
            return point.norm() - kSphereRadius;
        };
        std::function<Eigen::Vector3f(const Eigen::Vector3f &)> gradientFunction =
                [](const Eigen::Vector3f &point) -> Eigen::Vector3f {
            const float pointNorm = point.norm();
            return pointNorm > 1e-6f ? Eigen::Vector3f(point / pointNorm) : Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        };
        return Engine::Spatial::Extraction::FromImplicit(minCoord, maxCoord, cellSize, valueFunction, &gradientFunction);
    }

    // Axis-aligned box SDF centred at the origin, non-uniform half-extents so its edges and
    // corners are visually distinct. Central-difference gradient (sharp at edges/corners).
    VoxelField BuildBoxField(float cellSize) {
        const int halfCellCount = ComputeHalfCellCount(kBoxHalfExtents.maxCoeff(), cellSize);
        const std::array<int, 3> minCoord{-halfCellCount, -halfCellCount, -halfCellCount};
        const std::array<int, 3> maxCoord{halfCellCount, halfCellCount, halfCellCount};

        std::function<float(const Eigen::Vector3f &)> valueFunction = [](const Eigen::Vector3f &point) {
            const Eigen::Vector3f distanceOutsideBox = point.cwiseAbs() - kBoxHalfExtents;
            const Eigen::Vector3f clampedPositive = distanceOutsideBox.cwiseMax(0.0f);
            return clampedPositive.norm() +
                   std::min(std::max(distanceOutsideBox.x(), std::max(distanceOutsideBox.y(), distanceOutsideBox.z())),
                            0.0f);
        };
        std::function<Eigen::Vector3f(const Eigen::Vector3f &)> gradientFunction =
                [&valueFunction](const Eigen::Vector3f &point) {
            return CentralDifferenceGradient(valueFunction, point);
        };
        return Engine::Spatial::Extraction::FromImplicit(minCoord, maxCoord, cellSize, valueFunction, &gradientFunction);
    }

    // Torus SDF, ring axis = Y (the ring lies in the XZ plane): value = distance to the ring
    // circle (radius kTorusMajorRadius) minus the tube radius kTorusMinorRadius. The thin tube is
    // exactly the kind of thin/curved feature that separates the extractors' handling of
    // low-sample-density regions. Central-difference gradient.
    VoxelField BuildTorusField(float cellSize) {
        const int halfCellCount = ComputeHalfCellCount(kTorusMajorRadius + kTorusMinorRadius, cellSize);
        const std::array<int, 3> minCoord{-halfCellCount, -halfCellCount, -halfCellCount};
        const std::array<int, 3> maxCoord{halfCellCount, halfCellCount, halfCellCount};

        std::function<float(const Eigen::Vector3f &)> valueFunction = [](const Eigen::Vector3f &point) {
            const float radialDistanceFromRingAxis =
                    std::sqrt(point.x() * point.x() + point.z() * point.z()) - kTorusMajorRadius;
            return std::sqrt(radialDistanceFromRingAxis * radialDistanceFromRingAxis + point.y() * point.y()) -
                   kTorusMinorRadius;
        };
        std::function<Eigen::Vector3f(const Eigen::Vector3f &)> gradientFunction =
                [&valueFunction](const Eigen::Vector3f &point) {
            return CentralDifferenceGradient(valueFunction, point);
        };
        return Engine::Spatial::Extraction::FromImplicit(minCoord, maxCoord, cellSize, valueFunction, &gradientFunction);
    }

    /// ---------------------------------------------------------------------------------------
    /// Optional scan input (--dir): folder listing + AdvancedTSDF integration.
    /// ---------------------------------------------------------------------------------------

    uint32_t NextPowerOfTwo(uint32_t value) {
        if (value <= 1) return 1;
        --value;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        return value + 1;
    }

    // Sorted frame_*.ply paths from `directory` (skips ground_truth_*); mirrors
    // voxel_fill_debugger.cpp's collectFramePaths. Directory listing only -- the clouds
    // themselves are read by Engine::Pipeline::LoadFrames.
    std::vector<std::string> CollectFramePaths(const std::string &directory) {
        std::vector<std::string> paths;
        for (const auto &entry : fs::directory_iterator(directory)) {
            if (!entry.is_regular_file()) continue;
            const std::string name = entry.path().filename().string();
            if (name.rfind("ground_truth", 0) == 0) continue;
            if (entry.path().extension() == ".ply" && name.rfind("frame_", 0) == 0)
                paths.push_back(entry.path().string());
        }
        std::sort(paths.begin(), paths.end());
        return paths;
    }

    // Frames + bounds loaded once up front (folder IO); the AdvancedTSDF integrate itself is
    // redone by BuildScanField on every rebuild (cellSize/extractor/featureAngle can all change
    // without touching this).
    struct ScanInput {
        std::shared_ptr<std::vector<ep::Frame>> frames;
        ep::FrameBounds bounds;
        bool available = false;
    };

    // Fits an AdvancedTSDF's single movable 512^3-voxel window to the scan's bounding box, then
    // integrates every frame at IDENTITY pose (the folder is already registered into one
    // consistent frame -- see CollectFramePaths/LoadFrames above) and downloads its entries.
    // Mirrors tsdf_folder_eval.cpp's window-sizing: the default ORIGIN-CENTRED window silently
    // drops any voxel outside it, which an off-origin or large real scan hits immediately.
    VoxelField BuildScanField(Engine::Core::Context &context,
                              const std::vector<ep::Frame> &frames,
                              const ep::FrameBounds &bounds,
                              float cellSize) {
        const float truncation = cellSize * 3.0f;
        const Eigen::Vector3f span = bounds.max - bounds.min;
        const float maxSpanAxis = span.maxCoeff();
        const int marginVoxels = static_cast<int>(std::ceil(truncation / cellSize)) + 2;
        const long axisVoxelCount = static_cast<long>(std::ceil(maxSpanAxis / cellSize)) + 2L * marginVoxels;
        if (axisVoxelCount > 512)
            throw std::runtime_error(
                    "scan span too large for a single AdvancedTSDF window at this cellSize -- raise the cellSize slider");

        const Eigen::Vector3f windowMinCorner =
                bounds.min - static_cast<float>(marginVoxels) * Eigen::Vector3f::Constant(cellSize);
        const double surfaceAreaProxy =
                2.0 * double(span.x() * span.y() + span.y() * span.z() + span.z() * span.x());
        const double shellDepthVoxels = 2.0 * double(truncation) / double(cellSize);
        const double estimatedEntryCount =
                (surfaceAreaProxy / (double(cellSize) * double(cellSize))) * shellDepthVoxels * 1.5;
        const uint32_t hashCapacity = std::max(
                1u << 20, NextPowerOfTwo(static_cast<uint32_t>(std::min(estimatedEntryCount * 2.0, double(1u << 24)))));
        const uint32_t maxPointsPerFrame =
                NextPowerOfTwo(static_cast<uint32_t>(std::max<std::size_t>(bounds.maxFramePoints, 1u << 15)));

        Engine::Spatial::AdvancedTSDF tsdf;
        tsdf.Build(context, cellSize, truncation, hashCapacity, maxPointsPerFrame, windowMinCorner);
        tsdf.SetIntegrationQuality({3, 4, true});
        for (const ep::Frame &frame : frames)
            tsdf.Integrate(frame.pts, frame.nrm, frame.cam); // identity pose: folder is pre-registered

        const std::vector<Engine::Spatial::AdvancedEntry> entries = tsdf.DownloadEntries();
        return Engine::Spatial::Extraction::FromAdvancedEntries(entries, cellSize);
    }

    /// ---------------------------------------------------------------------------------------
    /// Viewer state + rebuild + render-graph plumbing.
    /// ---------------------------------------------------------------------------------------

    constexpr int kExtractorCount = 7;
    constexpr const char *kExtractorNames[kExtractorCount] = {"mc", "mc33", "mtet", "emc", "dc", "dmc", "cms"};
    constexpr const char *kExtractorComboItems = "mc\0mc33\0mtet\0emc\0dc\0dmc\0cms\0\0";

    // Per-extractor debug tint: shading from the mesh's own normals carries the real shape
    // detail, this just makes screenshots / manual runs easy to identify at a glance.
    constexpr float kExtractorTints[kExtractorCount][3] = {
            {0.55f, 0.68f, 0.92f}, // mc
            {0.52f, 0.85f, 0.62f}, // mc33
            {0.90f, 0.62f, 0.40f}, // mtet
            {0.85f, 0.55f, 0.85f}, // emc
            {0.90f, 0.85f, 0.35f}, // dc
            {0.40f, 0.85f, 0.85f}, // dmc
            {0.90f, 0.45f, 0.45f}, // cms
    };

    enum class Shape : int { Sphere = 0, Box = 1, Torus = 2, Scan = 3 };
    constexpr const char *kShapeComboItemsAnalyticOnly = "sphere\0box\0torus\0\0";
    constexpr const char *kShapeComboItemsWithScan = "sphere\0box\0torus\0scan\0\0";

    struct ViewerState {
        int extractorIndex = 0;
        int shapeIndex = 0; // Shape enum
        float cellSize = 0.05f;
        float featureAngleCosine = 0.9f;
        bool wireframe = false;
        bool dirty = true; // starts true so the first loop iteration performs the initial build

        // Stats filled by RebuildMesh(), shown read-only in the panel.
        std::size_t vertexCount = 0;
        std::size_t triangleCount = 0;
        bool edgeManifold = false;
        bool watertight = false;
    };

    // Builds the selected input VoxelField, runs the selected extractor over it, uploads the
    // result into meshPass, and refreshes the stats block. Called from the render loop whenever
    // state.dirty is set (never from inside a RenderPass::Execute -- SetMesh reallocates GPU
    // buffers, see IsosurfaceMeshPass.h). On a SHAPE change, also re-centres/re-distances the
    // camera onto the new mesh's bounding box (preserving the user's trackball orientation) so
    // wildly different scales -- unit-sized analytic shapes vs a metre-scale scan -- both land
    // on screen without the user having to hunt for the mesh.
    void RebuildMesh(ViewerState &state,
                     Engine::Core::Context &context,
                     const ExtractorRegistry &registry,
                     const ScanInput &scanInput,
                     bool shapeChanged,
                     IsosurfaceMeshPass &meshPass,
                     Engine::Render::Camera &camera) {
        const Shape shape = static_cast<Shape>(state.shapeIndex);
        VoxelField field;
        switch (shape) {
            case Shape::Sphere: field = BuildSphereField(state.cellSize); break;
            case Shape::Box:    field = BuildBoxField(state.cellSize); break;
            case Shape::Torus:  field = BuildTorusField(state.cellSize); break;
            case Shape::Scan:
                if (!scanInput.available)
                    throw std::runtime_error("'scan' shape selected but no --dir was given");
                field = BuildScanField(context, *scanInput.frames, scanInput.bounds, state.cellSize);
                break;
        }

        const std::string extractorName = kExtractorNames[state.extractorIndex];
        std::unique_ptr<IsoSurfaceExtractor> extractor = registry.Create(extractorName);
        if (!extractor)
            throw std::runtime_error("unknown extractor: " + extractorName);

        ExtractParams params;
        params.featureAngleCosineThreshold = state.featureAngleCosine;
        const SurfaceMesh mesh = extractor->Extract(field, params);

        state.vertexCount = mesh.vertices.size();
        state.triangleCount = mesh.triangles.size();
        state.edgeManifold = IsEdgeManifold(mesh);
        state.watertight = IsWatertight(mesh);

        const float *tint = kExtractorTints[state.extractorIndex];
        meshPass.SetMesh(mesh, Eigen::Vector3f(tint[0], tint[1], tint[2]));

        if (shapeChanged && !mesh.vertices.empty()) {
            Eigen::Vector3f boundsMin = mesh.vertices.front(), boundsMax = mesh.vertices.front();
            for (const Eigen::Vector3f &vertex : mesh.vertices) {
                boundsMin = boundsMin.cwiseMin(vertex);
                boundsMax = boundsMax.cwiseMax(vertex);
            }
            const Eigen::Vector3f center = 0.5f * (boundsMin + boundsMax);
            const float radius = std::max(0.05f, 0.5f * (boundsMax - boundsMin).norm());
            camera.SetTarget(vkMath::Vec3(center.x(), center.y(), center.z()));
            camera.SetDistance(std::clamp(radius * 2.6f, 0.1f, 200.0f));
        }

        state.dirty = false;

        std::printf("[rebuild] extractor=%s shape=%d cellSize=%.4f featureAngleCosine=%.3f | "
                    "vertices=%zu triangles=%zu edgeManifold=%s watertight=%s\n",
                    extractorName.c_str(), state.shapeIndex, state.cellSize, state.featureAngleCosine,
                    state.vertexCount, state.triangleCount, state.edgeManifold ? "Y" : "N",
                    state.watertight ? "Y" : "N");
    }

} // namespace

int main(int argc, char **argv) {
    try {
        std::string scanDirectoryArgument;
        float cellSizeArgument = 0.05f;
        std::string extractorArgument = "mc";
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--dir" && i + 1 < argc)
                scanDirectoryArgument = argv[++i];
            else if (argument == "--voxel" && i + 1 < argc)
                cellSizeArgument = std::stof(argv[++i]);
            else if (argument == "--extractor" && i + 1 < argc)
                extractorArgument = argv[++i];
        }

        const ExtractorRegistry registry = ExtractorRegistry::Default();
        if (!registry.Has(extractorArgument)) {
            std::cerr << "unknown --extractor '" << extractorArgument << "'; falling back to 'mc'\n";
            extractorArgument = "mc";
        }
        int initialExtractorIndex = 0;
        for (int i = 0; i < kExtractorCount; ++i)
            if (extractorArgument == kExtractorNames[i]) { initialExtractorIndex = i; break; }

        // ---- Optional scan input: folder IO happens once, up front; the AdvancedTSDF integrate
        // itself happens lazily inside RebuildMesh, only when the "scan" shape is selected. ----
        ScanInput scanInput;
        if (!scanDirectoryArgument.empty()) {
            const std::vector<std::string> framePaths = CollectFramePaths(scanDirectoryArgument);
            if (framePaths.empty())
                throw std::runtime_error("no frame_*.ply found in " + scanDirectoryArgument);
            scanInput.frames = std::make_shared<std::vector<ep::Frame>>(ep::LoadFrames(framePaths));
            if (scanInput.frames->empty())
                throw std::runtime_error("no usable frames (need per-point normals) in " + scanDirectoryArgument);
            scanInput.bounds = ep::ComputeBounds(*scanInput.frames);
            scanInput.available = true;
            std::printf("scan dir  : %s (%zu frames, extent %.4f)\n", scanDirectoryArgument.c_str(),
                        scanInput.frames->size(), scanInput.bounds.Extent());
        }

        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {1280, 800, "Isosurface (Marching-Cubes) Debug Viewer"};
        Engine::Render::Application app(descriptor);

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D swapChainExtent = app.GetSwapChain().Extent();
        const float aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
        camera.SetPerspective(55.0f * 3.14159265f / 180.0f, aspect, 0.02f, 500.0f);
        camera.SetOrbit({0.0f, 0.0f, 0.0f}, 4.0f);

        ViewerState state;
        state.extractorIndex = initialExtractorIndex;
        state.cellSize = cellSizeArgument;

        const std::string shaderDirectory = ISOSURFACE_VIEWER_SHADER_DIR;
        Engine::Render::RenderGraph graph;

        auto meshPassOwned = std::make_unique<IsosurfaceMeshPass>(
                app.GetContext(), app.GetSwapChain().Format(), shaderDirectory);
        IsosurfaceMeshPass *meshPass = meshPassOwned.get();
        graph.AddPass(std::move(meshPassOwned));

        // Window backend is fixed to GLFW (ApplicationDescriptor default, not overridden above),
        // so the base Window& is always actually a GlfwWindow -- safe to downcast to reach
        // Handle(), which ImGui's GLFW backend needs (same cast tsdf_feature_compare.cpp makes).
        auto &glfwWindow = static_cast<Engine::Render::GlfwWindow &>(app.GetWindow());
        auto imGuiPassOwned = std::make_unique<ImGuiPass>(
                app.GetContext(), glfwWindow.Handle(), app.GetSwapChain().Format(), app.GetSwapChain().ImageCount());
        ImGuiPass *imGuiPass = imGuiPassOwned.get();
        // Added last: ImGuiPass's RenderingScope LOADS (does not clear) the swapchain image, so
        // it must run after IsosurfaceMeshPass within the same frame to draw the panel over the
        // mesh instead of wiping it out.
        graph.AddPass(std::move(imGuiPassOwned));

        int previousShapeIndex = -1; // forces the very first rebuild to also frame the camera

        imGuiPass->SetUi([&]() {
            ImGui::Begin("Isosurface Viewer");

            ImGui::SeparatorText("Extractor (Tab also cycles)");
            if (ImGui::Combo("extractor", &state.extractorIndex, kExtractorComboItems))
                state.dirty = true;

            ImGui::SeparatorText("Shape");
            const char *shapeItems = scanInput.available ? kShapeComboItemsWithScan : kShapeComboItemsAnalyticOnly;
            if (ImGui::Combo("shape", &state.shapeIndex, shapeItems))
                state.dirty = true;

            ImGui::SeparatorText("Parameters");
            if (ImGui::SliderFloat("cellSize", &state.cellSize, 0.02f, 0.6f, "%.3f"))
                state.dirty = true;
            if (ImGui::SliderFloat("featureAngle (cosine)", &state.featureAngleCosine, 0.0f, 1.0f, "%.3f"))
                state.dirty = true;

            ImGui::SeparatorText("Display");
            if (ImGui::Checkbox("wireframe", &state.wireframe))
                meshPass->SetWireframe(state.wireframe); // display-only flag flip, no rebuild

            ImGui::SeparatorText("Stats (read-only)");
            ImGui::Text("vertices      : %zu", state.vertexCount);
            ImGui::Text("triangles     : %zu", state.triangleCount);
            ImGui::Text("edge-manifold : %s", state.edgeManifold ? "Y" : "N");
            ImGui::Text("watertight    : %s", state.watertight ? "Y" : "N");

            ImGui::Spacing();
            if (ImGui::Button("Re-run"))
                state.dirty = true;

            ImGui::End();
        });

        app.GetView().SetScene(&scene);
        app.GetView().SetCamera(&camera);
        app.GetView().SetRenderGraph(&graph);

        Engine::Render::MouseListenerGroup trackball(app.GetWindow().Mouse());
        trackball.Add(Engine::Render::MouseEventType::Drag, [&](Engine::Render::MouseEvent &e) {
            if (e.button != Engine::Render::MouseButton::Left)
                return;
            const VkExtent2D size = app.GetWindow().FramebufferSize();
            if (size.width == 0 || size.height == 0)
                return;

            if (!camera.IsTrackballDragging()) {
                camera.BeginTrackballDrag(e.x, e.y, static_cast<int>(size.width), static_cast<int>(size.height));
                return;
            }
            camera.DragTrackball(e.x, e.y, static_cast<int>(size.width), static_cast<int>(size.height));
            e.handled = true;
        });
        trackball.Add(Engine::Render::MouseEventType::ButtonUp, [&](Engine::Render::MouseEvent &e) {
            if (e.button == Engine::Render::MouseButton::Left)
                camera.EndTrackballDrag();
        });
        trackball.Add(Engine::Render::MouseEventType::Scroll, [&](Engine::Render::MouseEvent &e) {
            camera.SetDistance(std::clamp(
                    camera.GetDistance() * std::exp(static_cast<float>(-e.scrollY) * 0.08f), 0.05f, 200.0f));
            e.handled = true;
        });

        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape) {
                app.GetWindow().RequestClose();
            } else if (e.keyCode == Engine::Render::KeyCode::Tab) {
                // Cycles the extractor, wrap-around -- mirrors tsdf_feature_compare.cpp's Tab
                // A/B toggle, generalized from 2 choices to kExtractorCount.
                state.extractorIndex = (state.extractorIndex + 1) % kExtractorCount;
                state.dirty = true;
            }
        });

        // Manual render loop (mirrors tsdf_feature_compare.cpp) so the mesh can be rebuilt
        // BETWEEN frames: RebuildMesh's Extract()/SetMesh() reallocates GPU buffers, which must
        // never happen inside a RenderPass::Execute (mid graphics command buffer). The UI only
        // sets state.dirty; the device-idle wait below guards the reallocation against the
        // single frame in flight, exactly like tsdf_feature_compare.cpp's own loop.
        Engine::Core::Context &context = app.GetContext();
        try {
            while (!app.GetWindow().ShouldClose()) {
                app.GetWindow().PollEvents();

                const VkExtent2D framebufferSize = app.GetWindow().FramebufferSize();
                if (framebufferSize.width == 0 || framebufferSize.height == 0)
                    continue;

                if (state.dirty) {
                    vkDeviceWaitIdle(context.device);
                    const bool shapeChanged = state.shapeIndex != previousShapeIndex;
                    RebuildMesh(state, context, registry, scanInput, shapeChanged, *meshPass, camera);
                    previousShapeIndex = state.shapeIndex;
                }

                if (!app.GetRenderer().BeginFrame(framebufferSize.width, framebufferSize.height))
                    continue;
                app.GetRenderer().Render(app.GetView());
                app.GetRenderer().EndFrame();
            }
        } catch (...) {
            vkDeviceWaitIdle(context.device);
            throw;
        }
        vkDeviceWaitIdle(context.device);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

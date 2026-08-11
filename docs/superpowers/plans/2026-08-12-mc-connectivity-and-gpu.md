# MC Connectivity + GPU Extractors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A half-edge mesh-connectivity facility (non-manifold/boundary detection) + its viewer visualization, and GPU-compute implementations of `mc`/`mc33`/`mtet` — all keeping the existing `SurfaceMesh` as the single output type.

**Architecture:** Add `Engine::Spatial::Extraction::MeshConnectivity` (half-edge build + `AnalyzeConnectivity`) consumed by both tests and the viewer overlay; add `GpuExtractorRegistry` + three `Gpu*Extractor` classes that upload a `VoxelField`, run a per-cube compute shader emitting raw triangles via an int atomic counter, then CPU-weld via `core::WeldAndComputeNormals` — producing a `SurfaceMesh` that matches the CPU extractor (the consistency gate).

**Tech Stack:** C++17, Eigen, Vulkan/MoltenVK compute (int atomics only, no float atomics), GoogleTest, ImGui (viewer).

## Global Constraints

- Keep `SurfaceMesh` (no rename); every extractor CPU+GPU returns it. GPU extractors implement `IsoSurfaceExtractor` (`SurfaceMesh Extract(const VoxelField&, const ExtractParams&) const`), constructed with a `Engine::Core::Context&`.
- MoltenVK has NO GPU float atomics: the compute emit reserves output slots with an **int `atomicAdd`** counter and writes to distinct slots — never a float accumulation. Measure GPU perf only in `build-rel`.
- Full descriptive names, no abbreviations; GLSL uses the repo's `///`-banner/Allman/numbered-step style (see `feedback_glsl_style`); transcribed tables keep published names + a provenance comment.
- Reuse: `MarchingCubesCore` (`WeldAndComputeNormals`/`VertexInterpolate`/`kEdgeCornerPairs`/tables), `voxel_common.glsl`/`MarchingCubesTables.h` tables, the compute pattern of `src/shader/voxel_tsdf_mc.comp` + `src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.cpp` (upload/dispatch/readback), the viewer passes. Do not reinvent.
- Do NOT touch the user's uncommitted WIP: `src/Engine/Spatial/AdvancedTSDF.cpp/.h`, `src/shader/advanced_tsdf_*.comp.glsl`, `src/Engine/Pipeline/Registration/{RegistrationThread,GpuIcpTracker,GpuPointToPlaneIcp,PointToPlaneIcpTracker}.*`, `src/Engine/Pipeline/Integration/IntegrationThread.cpp`, `src/Engine/Spatial/{SubmapAdvancedTSDF,TiledAdvancedTSDF,TiledDirectionalTSDF}.h`, `lib/SPIRV-Reflect`. Each task `git add`s ONLY its own files — never `git add .`/`-A`.
- New `.cpp`/shader needs a CMake reconfigure: `cmake -S . -B build -DGTest_DIR=/opt/homebrew/lib/cmake/GTest -Dgflags_DIR=/opt/homebrew/lib/cmake/gflags -Dglog_DIR=/opt/homebrew/lib/cmake/glog` then build the target. NEVER `rm -rf build`. Commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do NOT push.
- Baseline suite is green (255 total / 254 pass / 1 skip). It must stay green.

## Reference (read before starting)

- `src/Engine/Spatial/Extraction/` — `SurfaceMesh.h`, `IsoSurfaceExtractor.h` (`ExtractParams`), `VoxelField.h` (`Sample`/`Gradient`/`CellSize`/`OccupiedCoords`, coords `std::array<int,3>`), `MarchingCubesCore.h` (`WeldAndComputeNormals`, `VertexInterpolate`, `kEdgeCornerPairs`, `CandidateBases`, `RawTriangle`), `ExtractorRegistry.h`, `MarchingCubesExtractor.cpp`/`MarchingCubes33Extractor.cpp`/`MarchingTetrahedraExtractor.cpp` (the CPU twins to match).
- `src/shader/voxel_tsdf_mc.comp` + `src/shader/voxel_common.glsl` — GPU MC structure + the `edgeTable`/`triTable`/`CORNER` tables in GLSL.
- `src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.{h,cpp}` — the upload/allocate/bind/dispatch/readback compute pattern + int fixed-point/atomic usage on MoltenVK; `TrackerRegistry.cpp` for the registry `Default()` shape.
- `src/Engine/Spatial/Extraction/MarchingCubes33Tables.h` — MC33 tables (transcribe to GLSL for Task 4).
- `test/isosurface_test_util.h` — analytic `SphereField`/`BoxField`; `Engine::Eval::NearestNeighbourRMSE`.
- `example2/isosurface_viewer.cpp` + `example2/IsosurfaceMeshPass.{h,cpp}` + `example2/Shaders/IsosurfaceMesh.{vert,frag}.glsl` — the viewer to extend (Task 2).

## File Structure

```
src/Engine/Spatial/Extraction/
  MeshConnectivity.h / .cpp             half-edge + AnalyzeConnectivity (Task 1)
  GpuExtractorRegistry.h / .cpp         Context-aware registry (Task 3)
  GpuIsoSurfaceExtractorCommon.h / .cpp shared upload/dispatch/readback + CPU weld (Task 3)
  GpuMarchingCubesExtractor.h / .cpp    "mc-gpu"   (Task 3)
  GpuMarchingCubes33Extractor.h / .cpp  "mc33-gpu" (Task 4)
  GpuMarchingTetrahedraExtractor.h/.cpp "mtet-gpu" (Task 5)
src/shader/
  extract_mc.comp                       (Task 3)
  extract_mc33.comp                     (Task 4)
  extract_mtet.comp                     (Task 5)
test/
  test_mesh_connectivity.cpp            (Task 1)
  test_gpu_extractors.cpp               (Tasks 3-5 append)
example2/  (Task 2 extends the viewer + its pass)
```

---

### Task 1: Mesh connectivity facility

**Files:** Create `src/Engine/Spatial/Extraction/MeshConnectivity.{h,cpp}`, `test/test_mesh_connectivity.cpp`.

**Interfaces (Produces — Task 2 + others consume):**
```cpp
namespace Engine::Spatial::Extraction {
    struct HalfEdge { int origin; int face; int next; int twin; }; // twin=-1 if boundary/non-manifold
    struct HalfEdgeMesh { std::vector<HalfEdge> halfEdges; std::vector<int> vertexHalfEdge; };
    HalfEdgeMesh BuildHalfEdgeMesh(const SurfaceMesh& mesh);
    struct ConnectivityReport {
        std::vector<std::pair<int,int>> boundaryEdges;    // (v0<v1), 1 incident triangle
        std::vector<std::pair<int,int>> nonManifoldEdges; // > 2 incident triangles
        std::vector<int> nonManifoldVertices;             // bowtie
        int degenerateTriangles = 0;
        bool IsEdgeManifold() const { return nonManifoldEdges.empty(); }
        bool IsClosed() const { return boundaryEdges.empty() && nonManifoldEdges.empty(); }
    };
    ConnectivityReport AnalyzeConnectivity(const SurfaceMesh& mesh);
}
```

- [ ] **Step 1: Write the failing tests** `test/test_mesh_connectivity.cpp`

```cpp
#include "Engine/Spatial/Extraction/MeshConnectivity.h"
#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial::Extraction;

TEST(MeshConnectivity, SphereIsClosedManifold) {
    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh mesh = ExtractorRegistry::Default().Create("mc")->Extract(field, ExtractParams{});
    ConnectivityReport r = AnalyzeConnectivity(mesh);
    EXPECT_TRUE(r.IsEdgeManifold());
    EXPECT_TRUE(r.IsClosed());
    EXPECT_TRUE(r.boundaryEdges.empty());
    EXPECT_TRUE(r.nonManifoldVertices.empty());
    EXPECT_EQ(r.degenerateTriangles, 0);
}

TEST(MeshConnectivity, DetectsNonManifoldEdge) {
    // 3 triangles all sharing edge (0,1) -> that edge has 3 incident faces.
    SurfaceMesh m;
    m.vertices = {{0,0,0},{1,0,0},{0,1,0},{0,-1,0},{0,0,1}};
    m.normals.assign(5, Eigen::Vector3f(0,0,1));
    m.triangles = {{0,1,2},{0,1,3},{0,1,4}};
    ConnectivityReport r = AnalyzeConnectivity(m);
    EXPECT_FALSE(r.IsEdgeManifold());
    ASSERT_EQ(r.nonManifoldEdges.size(), 1u);
    EXPECT_EQ(r.nonManifoldEdges[0], std::make_pair(0,1));
}

TEST(MeshConnectivity, DetectsBoundaryEdges) {
    SurfaceMesh m;
    m.vertices = {{0,0,0},{1,0,0},{0,1,0}};
    m.normals.assign(3, Eigen::Vector3f(0,0,1));
    m.triangles = {{0,1,2}};
    ConnectivityReport r = AnalyzeConnectivity(m);
    EXPECT_EQ(r.boundaryEdges.size(), 3u);
    EXPECT_FALSE(r.IsClosed());
    EXPECT_TRUE(r.IsEdgeManifold()); // boundary != non-manifold
}

TEST(MeshConnectivity, DetectsBowtieVertex) {
    // Two triangles sharing ONLY vertex 0.
    SurfaceMesh m;
    m.vertices = {{0,0,0},{1,0,0},{1,1,0},{-1,0,0},{-1,1,0}};
    m.normals.assign(5, Eigen::Vector3f(0,0,1));
    m.triangles = {{0,1,2},{0,3,4}};
    ConnectivityReport r = AnalyzeConnectivity(m);
    ASSERT_FALSE(r.nonManifoldVertices.empty());
    EXPECT_NE(std::find(r.nonManifoldVertices.begin(), r.nonManifoldVertices.end(), 0),
              r.nonManifoldVertices.end());
}
```

- [ ] **Step 2: Run to verify it fails** — `cmake --build build --target vkspatial_tests -j` → FAIL to compile (`MeshConnectivity.h` absent).
- [ ] **Step 3: Implement `MeshConnectivity.{h,cpp}`** — build directed half-edges (3/triangle, `next` = CCW within face), group by undirected key `(min,max)` to count incidence → boundary/manifold/non-manifold + set `twin` only for the unique 2-incidence pair; bowtie via one-ring twin-walk splitting into ≥2 fans; degenerate = repeated index or `|(b-a)x(c-a)| < 1e-12`.
- [ ] **Step 4: Reconfigure, build, run** — `... --gtest_filter='MeshConnectivity.*'` PASS; full suite green.
- [ ] **Step 5: Commit** (`git add` the 2 files only).

---

### Task 2: Viewer connectivity visualization

**Files:** Modify `example2/isosurface_viewer.cpp`; Create `example2/EdgeOverlayPass.{h,cpp}` + `example2/Shaders/EdgeOverlay.{vert,frag}.glsl` (a `LINE_LIST` pass); Modify `example2/CMakeLists.txt` (add the pass sources + shaders to the `isosurface_viewer` target).

**Interfaces:** Consumes Task 1 `AnalyzeConnectivity`/`ConnectivityReport`.

**Approach:** `EdgeOverlayPass` mirrors `IsosurfaceMeshPass` but draws `VK_PRIMITIVE_TOPOLOGY_LINE_LIST` with a per-vertex color; `SetEdges(std::vector<std::pair<Eigen::Vector3f,Eigen::Vector3f>> lines, color)` uploads segments. In the viewer's `rebuild()`, after extraction run `AnalyzeConnectivity(mesh)`; build line lists from `mesh.vertices[edge.first/second]` for `nonManifoldEdges` (red 1,0,0) and `boundaryEdges` (yellow 1,1,0); call `overlayPass.SetEdges(...)`. ImGui: `Checkbox("show non-manifold/boundary edges", &state.showEdges)` (toggles the pass visible) + add `nonManifoldEdges/boundaryEdges/nonManifoldVertices` counts to the stats block. Draw the overlay after the mesh with a small depth bias (or depth-test disabled) so flagged edges are always visible.

- [ ] **Step 1: Write `EdgeOverlayPass.{h,cpp}` + its shaders** (line-list pipeline; camera push-constant like `IsosurfaceMeshPass`).
- [ ] **Step 2: Wire into the viewer** — `rebuild()` computes connectivity + uploads edges; ImGui toggle + counts.
- [ ] **Step 3: CMake** — add the pass sources + `add_compiled_shaders` entries to `isosurface_viewer`.
- [ ] **Step 4: Reconfigure + build** — `cmake --build build --target isosurface_viewer -j` links; `Isosurface.*`/`MeshConnectivity.*` tests still green (framework unchanged). (Visual output needs a display — document in the header comment that switching to `dc` on a box lights the ambiguous-face crack red.)
- [ ] **Step 5: Commit** (viewer + pass + shaders + CMake edit only).

---

### Task 3: GPU `mc` extractor + `GpuExtractorRegistry` + consistency test

**Files:** Create `src/Engine/Spatial/Extraction/GpuExtractorRegistry.{h,cpp}`, `GpuIsoSurfaceExtractorCommon.{h,cpp}`, `GpuMarchingCubesExtractor.{h,cpp}`, `src/shader/extract_mc.comp`, `test/test_gpu_extractors.cpp`.

**Interfaces (Produces — Tasks 4-5 consume):**
```cpp
namespace Engine::Spatial::Extraction {
    class GpuExtractorRegistry {
    public:
        using Factory = std::function<std::unique_ptr<IsoSurfaceExtractor>(Engine::Core::Context&)>;
        void Register(const std::string& name, Factory);
        std::unique_ptr<IsoSurfaceExtractor> Create(const std::string& name, Engine::Core::Context&) const;
        bool Has(const std::string& name) const;
        static GpuExtractorRegistry Default(); // "mc-gpu","mc33-gpu","mtet-gpu"
    private:
        std::unordered_map<std::string, Factory> m_factories;
    };
    // GpuIsoSurfaceExtractorCommon: shared helpers Tasks 4-5 reuse --
    //   UploadField(field) -> GPU buffers (candidate cube bases + a coord->value lookup);
    //   ReadbackRawTriangles(...) -> std::vector<core::RawTriangle>;
    //   the emit uses an int atomicAdd slot counter (buffer 0 = counter, buffer 1 = vertex slots).
}
```

**Approach:** `GpuMarchingCubesExtractor(Context&)`; `Extract` (a) builds candidate cube bases (`core::CandidateBases(field.OccupiedCoords())`) and uploads them + a coord→value lookup (simplest first cut: a dense value grid over the field's integer bounding box, uploaded as a flat buffer + origin/dims push-constants; document this); (b) dispatches `extract_mc.comp` one invocation per candidate cube — sample 8 corners, `cubeIndex`, walk `edgeTable`/`triTable` (from `voxel_common.glsl`), `VertexInterpolate` the edge crossings, reserve `3*triCount` vertex slots via `atomicAdd` on an int counter, write the raw triangle vertices (world positions) to those slots (SAME winding swap as the CPU core: `(ev0,ev2,ev1)`); (c) reads back the raw triangles and runs `core::WeldAndComputeNormals(raw, params.weldFraction*field.CellSize())` → `SurfaceMesh`. Register `"mc-gpu"`.

- [ ] **Step 1: Write the failing test** `test/test_gpu_extractors.cpp`

```cpp
#include "Engine/Spatial/Extraction/GpuExtractorRegistry.h"
#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Spatial/Extraction/MeshConnectivity.h"
#include "Engine/Core/Context.h"
#include "Engine/Eval/RmseMetrics.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial::Extraction;

TEST(GpuExtractors, McGpuMatchesCpuOnSphere) {
    Engine::Core::Context ctx;
    auto gpuReg = GpuExtractorRegistry::Default();
    ASSERT_TRUE(gpuReg.Has("mc-gpu"));
    ASSERT_NE(gpuReg.Create("mc-gpu", ctx), nullptr);
    EXPECT_EQ(gpuReg.Create("nonexistent", ctx), nullptr);

    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh cpu = ExtractorRegistry::Default().Create("mc")->Extract(field, ExtractParams{});
    SurfaceMesh gpu = gpuReg.Create("mc-gpu", ctx)->Extract(field, ExtractParams{});

    ASSERT_GT(gpu.triangles.size(), 100u);
    // Same tables + same weld -> vertex sets coincide to fp tolerance, both directions.
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(gpu.vertices, cpu.vertices), 1e-4f);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(cpu.vertices, gpu.vertices), 1e-4f);
    // And the GPU mesh is itself closed+manifold like the CPU one.
    ConnectivityReport r = AnalyzeConnectivity(gpu);
    EXPECT_TRUE(r.IsEdgeManifold());
    EXPECT_TRUE(r.IsClosed());
}
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`GpuExtractorRegistry.h` absent).
- [ ] **Step 3: Implement** the registry + common upload/readback + `GpuMarchingCubesExtractor` + `extract_mc.comp` (reconfigure so the new `.comp` compiles; follow how `voxel_tsdf_mc.comp` is built/loaded).
- [ ] **Step 4: Reconfigure/build/run** — `... --gtest_filter='GpuExtractors.McGpu*'` PASS; full suite green.
- [ ] **Step 5: Commit** (registry + common + mc extractor + shader + test — its own files only).

---

### Task 4: GPU `mc33` extractor

**Files:** Create `src/Engine/Spatial/Extraction/GpuMarchingCubes33Extractor.{h,cpp}`, `src/shader/extract_mc33.comp`; Modify `GpuExtractorRegistry.cpp` (register `"mc33-gpu"`); Modify `test/test_gpu_extractors.cpp` (append).

**Interfaces:** Consumes Task 3 common helpers; the CPU `mc33` twin.

**Approach:** Like Task 3 but `extract_mc33.comp` transcribes the MC33 case/subcase tables from `MarchingCubes33Tables.h` into GLSL (with a provenance comment) and runs the asymptotic-decider face test + interior test per cube (mirror `MarchingCubes33Extractor.cpp`'s logic), emitting to the same int-atomic vertex buffer; CPU-weld identically. Reuse Task 3's `GpuIsoSurfaceExtractorCommon` upload/readback verbatim.

- [ ] **Step 1: Append the failing test** to `test_gpu_extractors.cpp`

```cpp
TEST(GpuExtractors, Mc33GpuMatchesCpuOnSphere) {
    Engine::Core::Context ctx;
    ASSERT_TRUE(GpuExtractorRegistry::Default().Has("mc33-gpu"));
    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh cpu = ExtractorRegistry::Default().Create("mc33")->Extract(field, ExtractParams{});
    SurfaceMesh gpu = GpuExtractorRegistry::Default().Create("mc33-gpu", ctx)->Extract(field, ExtractParams{});
    ASSERT_GT(gpu.triangles.size(), 100u);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(gpu.vertices, cpu.vertices), 1e-4f);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(cpu.vertices, gpu.vertices), 1e-4f);
    EXPECT_TRUE(AnalyzeConnectivity(gpu).IsEdgeManifold());
}
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`Has("mc33-gpu")` false).
- [ ] **Step 3: Implement** `extract_mc33.comp` (transcribe tables + subcase logic) + the extractor; register `"mc33-gpu"`.
- [ ] **Step 4: Reconfigure/build/run** — `GpuExtractors.Mc33Gpu*` PASS; full suite green.
- [ ] **Step 5: Commit** (its own files only).

---

### Task 5: GPU `mtet` extractor

**Files:** Create `src/Engine/Spatial/Extraction/GpuMarchingTetrahedraExtractor.{h,cpp}`, `src/shader/extract_mtet.comp`; Modify `GpuExtractorRegistry.cpp` (register `"mtet-gpu"`); Modify `test/test_gpu_extractors.cpp` (append).

**Interfaces:** Consumes Task 3 common helpers; the CPU `mtet` twin.

**Approach:** Like Task 3 but `extract_mtet.comp` splits each cube into the 6 tetrahedra sharing the 0-6 diagonal (`{0,5,1,6},{0,1,2,6},{0,2,3,6},{0,3,7,6},{0,7,4,6},{0,4,5,6}`, mirroring `MarchingTetrahedraExtractor.cpp`) and does per-tetra 1-/2-triangle emission on the cut edges, into the int-atomic vertex buffer; CPU-weld with the SAME `kWeldDistanceDivisor` the CPU `mtet` uses (read it from `MarchingTetrahedraExtractor.cpp`) so the GPU/CPU weld matches. Reuse Task 3's common helpers.

- [ ] **Step 1: Append the failing test** to `test_gpu_extractors.cpp`

```cpp
TEST(GpuExtractors, MtetGpuMatchesCpuOnSphere) {
    Engine::Core::Context ctx;
    ASSERT_TRUE(GpuExtractorRegistry::Default().Has("mtet-gpu"));
    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh cpu = ExtractorRegistry::Default().Create("mtet")->Extract(field, ExtractParams{});
    SurfaceMesh gpu = GpuExtractorRegistry::Default().Create("mtet-gpu", ctx)->Extract(field, ExtractParams{});
    ASSERT_GT(gpu.triangles.size(), 100u);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(gpu.vertices, cpu.vertices), 1e-4f);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(cpu.vertices, gpu.vertices), 1e-4f);
    ConnectivityReport r = AnalyzeConnectivity(gpu);
    EXPECT_TRUE(r.IsEdgeManifold());
    EXPECT_TRUE(r.IsClosed());
}
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`Has("mtet-gpu")` false).
- [ ] **Step 3: Implement** `extract_mtet.comp` + the extractor; register `"mtet-gpu"`.
- [ ] **Step 4: Reconfigure/build/run** — `GpuExtractors.MtetGpu*` PASS; full suite green.
- [ ] **Step 5: Commit** (its own files only).

---

## Self-review notes

- **Spec coverage:** #1 unified output = every extractor returns `SurfaceMesh` (Tasks 3-5 GPU return it, gated equal to CPU); #2 connectivity (Task 1) + viz (Task 2); #3 GPU mc/mc33/mtet (Tasks 3-5). Dual/feature GPU + rename correctly out of scope.
- **Type consistency:** `SurfaceMesh`, `ConnectivityReport`/`AnalyzeConnectivity`/`HalfEdgeMesh`, `GpuExtractorRegistry` (Create takes `Context&`, nullptr on unknown), `GpuIsoSurfaceExtractorCommon` helpers, extractor names `mc-gpu`/`mc33-gpu`/`mtet-gpu` — used identically across tasks. GPU extractors reuse `core::WeldAndComputeNormals` so their `SurfaceMesh` matches the CPU weld.
- **Consistency gate:** each GPU task asserts `NearestNeighbourRMSE(gpu,cpu) < 1e-4` both directions + the GPU mesh is manifold/closed (via Task 1) — the analogue of AdaptiveVoxelGrid's CPU≡GPU MC test.
- **GPU shader tables (Tasks 3-4):** the plan cannot inline the full GLSL table transcriptions; it names the source (`voxel_common.glsl` for mc, `MarchingCubes33Tables.h` for mc33) + the winding-swap/atomic-emit contract, and the CPU-equality test is the correctness gate.

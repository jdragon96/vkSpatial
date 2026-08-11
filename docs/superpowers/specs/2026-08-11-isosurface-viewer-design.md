# Isosurface (Marching-Cubes) Debug Viewer — Design

## Goal

An interactive `example2/isosurface_viewer` that extracts a surface with a **selectable** isosurface
extractor (`mc`/`mc33`/`mtet`/`emc`/`dc`/`dmc`/`cms`) and **renders the resulting triangle mesh** so the
differences between algorithms (sharp-feature preservation, cracks, topology, rounding) are visible.
Switching the extractor (and the input shape / parameters) live re-extracts and re-renders.

## Approach

Mirror `example2/tsdf_feature_compare.cpp`'s structure — a `RenderGraph` with a mesh pass + an
`ImGuiPass` whose `SetUi(...)` callback drives shared state — and reuse the lit-mesh drawing pattern of
`example2/CubePass.cpp` / `BlinnPhong.cpp` (`Engine::Render::Object<Vertex>` with position/normal/color,
uploaded to vertex+index buffers, drawn with a camera + a light). Single view (not A/B): one mesh, one
switcher.

## Components

### 1. `IsosurfaceMeshPass` (`example2/IsosurfaceMeshPass.{h,cpp}`)
A `RenderPass` that draws a dynamic indexed triangle mesh, adapted from `CubePass`:
- `void SetMesh(const Engine::Spatial::Extraction::SurfaceMesh& mesh, const Eigen::Vector3f& color)` —
  converts the SurfaceMesh to `std::vector<Object<>::Vertex>` (position = vertex, normal = mesh normal,
  color = the debug color) + indices from `mesh.triangles`, and re-uploads the vertex/index buffers
  (guard the realloc like the existing dynamic-upload passes — do not upload mid-frame without the
  device idle/appropriate sync the engine's Buffer::Upload already provides).
- `void SetWireframe(bool)` — toggles between a solid (`VK_POLYGON_MODE_FILL`) and wireframe
  (`VK_POLYGON_MODE_LINE`) pipeline; build BOTH pipelines up front and select in `Execute` (MoltenVK
  supports line polygon mode; if a runtime line-width/feature issue arises, fall back to drawing the
  solid pipeline and document it).
- Lit shading: reuse CubePass's camera/light push-constant + its `.vert`/`.frag` (Blinn-Phong-ish); the
  mesh normals drive the lighting so surface shape + creases read clearly. Shaders may be reused from
  CubePass/BlinnPhong or a small dedicated pair compiled the same way (see `example2/CMakeLists.txt`'s
  `compile_shaders`/`glslc` block).

### 2. Analytic `VoxelField` builders (in the viewer TU)
`sphere`, `box`, `torus` signed fields with exact/central-difference gradients (box + torus expose sharp
edges / thin features that separate the algorithms). Build a `Engine::Spatial::Extraction::VoxelField`
over an integer grid using the `FromImplicit(minCoord, maxCoord, cellSize, valueFn, &gradientFn)` adapter
(same one `test/isosurface_test_util.h` uses — mirror `SphereField`/`BoxField` there; add a `torus`).

### 3. Optional scan input (`--dir <folder>`)
When `--dir` is given, load `frame_*.ply` with `Engine::Pipeline::LoadFrames` (see
`voxel_fill_debugger.cpp`), integrate them into an `Engine::Spatial::AdvancedTSDF` (identity poses — the
folders are pre-registered), `DownloadEntries()`, and build a `VoxelField` via
`Extraction::FromAdvancedEntries(entries, cellSize)`. This is a selectable input shape ("scan") in the UI
alongside the analytic ones. If `--dir` is absent, only the analytic shapes are offered.

### 4. Viewer main (`example2/isosurface_viewer.cpp`)
- Args: `--dir <folder>` (optional), `--voxel v` (cellSize for analytic + scan), `--extractor name`
  (initial, default `mc`).
- Shared `ViewerState { int shape; int extractor; float cellSize; float featureAngleCosine; bool
  wireframe; bool dirty; }`.
- `rebuild()`: build the selected input `VoxelField` → `ExtractorRegistry::Default().Create(name)` →
  `Extract(field, params)` → `meshPass.SetMesh(mesh, color)`, and cache stats (vertex/triangle count +
  `isEdgeManifold`/`isWatertight` computed inline — reuse the predicates from
  `test/isosurface_test_util.h`; if including a test header from example2 is awkward, copy the two tiny
  edge-count predicates into the viewer, they are ~10 lines).
- `ImGuiPass::SetUi`: `Combo("extractor", "mc\0mc33\0mtet\0emc\0dc\0dmc\0cms\0")`, `Combo("shape", ...)`
  (analytic shapes + "scan" when `--dir` present), `SliderFloat("cellSize", ...)`,
  `SliderFloat("featureAngle", ...)`, `Checkbox("wireframe", ...)`, and a read-only stats block:
  vertices, triangles, edge-manifold (Y/N), watertight (Y/N). Any change sets `state.dirty`; the render
  loop calls `rebuild()` when dirty (extractor/shape/cellSize/featureAngle changes rebuild the mesh;
  wireframe only calls `meshPass.SetWireframe`). Keyboard: cycle extractor with a key (e.g. Tab), like
  the feature-compare viewer.

## CMake

Add to `example2/CMakeLists.txt`: `add_executable(isosurface_viewer isosurface_viewer.cpp
IsosurfaceMeshPass.cpp)`; `target_link_libraries(isosurface_viewer PRIVATE Engine::Render Engine::Spatial
Engine::Core imgui)` (mirror `tsdf_feature_compare` / `voxel_fill_debugger`'s render+imgui link line).
Compile the pass's shaders via the same `compile_shaders`/glslc mechanism the other render examples use.

## Constraints

- Full descriptive names, no abbreviations. Reuse existing render passes/objects; do not reinvent the
  render graph.
- Do NOT touch the user's uncommitted WIP (`AdvancedTSDF.cpp/.h`, `advanced_tsdf_*.comp.glsl`,
  ICP registration files, `Submap/Tiled*.h`, `lib/SPIRV-Reflect`) or the uncommitted `icp_quality_diag`
  diagnostic. `git add` ONLY the new viewer files (+ its shaders + the `example2/CMakeLists.txt` edit).
- Build target `isosurface_viewer` (needs a reconfigure — GLOB/explicit target add). Never `rm -rf build`.
- Commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do NOT push.

## Acceptance / Testing

- It is a GUI example (needs a display to view) — the automated gate is: **it compiles and links**
  (`cmake --build build --target isosurface_viewer`), the existing `Isosurface.*` extraction tests stay
  green (the viewer reuses that framework, changes none of it), and the full suite stays green.
- Manual: `./build/example2/isosurface_viewer` (analytic) and `--dir scan_out --voxel 0.5` (scan) opens a
  window; switching extractor/shape/wireframe re-renders; box/torus visibly show emc/dc/cms preserving
  edges vs mc rounding, and dc's ambiguous-face cracks appear in wireframe. Document the run commands in
  a header comment.

## Non-goals

- No A/B side-by-side (single view + switcher, per the request "switch and inspect").
- No GPU extractor (the extractors are CPU; only the RENDER is GPU).
- No new isosurface algorithm — this only VISUALIZES the existing 7.

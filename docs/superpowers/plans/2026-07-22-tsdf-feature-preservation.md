# Sharp/Round Feature Preservation Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** An in-app A/B viewer + numeric report comparing how `SimpleTSDF` (averaged single field → rounds sharp corners) vs `DirectionalTSDF` (direction layers → preserves them) reconstruct synthetic cube/cylinder shapes, with per-region (flat/curved/edge) ground-truth error metrics.

**Architecture:** New app `example2/tsdf_feature_compare` reusing the viewer stack (`Application`/`Camera`/trackball/`RenderGraph`/`PointCloudPass`/`ImGuiPass`). Analytic shape fixtures provide multi-view oriented samples (for integration), exact nearest-surface distance (for error), and region classification. The `--dump` metric is the correctness oracle. `ImGuiPass` is generalized to a `drawUi` callback so both apps share it.

**Tech Stack:** C++17, Vulkan (dynamic rendering, MoltenVK), GLFW, Dear ImGui (`lib/imgui`, vendored 1.91.9), Eigen. glslc for shaders.

Spec: `docs/superpowers/specs/2026-07-22-tsdf-feature-preservation-design.md`.

## Global Constraints

- All new code lives in `example2/`. Do NOT change `Engine::Spatial`/`Engine::Core`/TSDF cores. `ImGuiPass`/`PointCloudPass` (already in `example2/`) may change.
- Mirror existing patterns: `example2/tsdf_viewer.cpp` (manual render loop with between-frames rebuild, `--frames N` self-exit), `example2/CubePass.cpp`/`PointCloudPass` (pass/pipeline), `example2/tsdf_slice_debug.cpp` & `tsdf_fixtures.h` (fixture/color idioms). Reuse `PointCloudPass` and the `Shaders/pointcloud.{vert,frag}` shaders as-is.
- **DirectionalTSDF preserving preset:** `IntegrationQuality{ maxDirections=3, dirExponent=4, viewAngleWeight=true }` (fields from `src/Engine/Spatial/DirectionalIntegrationQuality.h`). SimpleTSDF and DirectionalTSDF use the SAME `voxelSize`/`truncation` for fairness (default `voxelSize=0.1`, `truncation=0.3`).
- **Known large-N GPU non-determinism** (memory `project_engine_core_large_n_bug`): keep EACH per-view `Integrate` call's sample count ≲ 800 (well under ~1000). Multiple views accumulate in the volume; that's fine. Also respect `SimpleTSDF` Build `maxPoints` (default 4096) per call.
- GUI has no unit-test oracle, BUT the per-region error metric IS a numeric oracle. Gate = builds + `--frames N` launches clean + `--dump` numbers hold the hypothesis. `--frames N` (default 0 = run until closed) self-exits after N frames. A pre-existing `VUID-vkQueueSubmit-pSignalSemaphores-00067` (shared by cube_render2) is OUT OF SCOPE. No unit tests this round (per user).
- Build env (memory `project_repo_build_worktree_gotchas`): configure `VULKAN_SDK=/usr/local cmake -S . -B build -DGTest_DIR=/opt/homebrew/lib/cmake/GTest -Dgflags_DIR=/opt/homebrew/lib/cmake/gflags`; build `VULKAN_SDK=/usr/local cmake --build build --parallel --target <t>`; re-run configure after adding a new example target (each is explicit via `add_spatial_example`). macOS has no `timeout`; use `--frames`.
- Commit after each task with a passing build (+ launch-smoke / `--dump` where applicable).

## File Structure

**New:** `example2/shape_fixtures.h` (analytic cube/cylinder: sampler + distance + region + color helpers), `example2/tsdf_feature_compare.cpp` (app: headless metric in T2, interactive GUI in T3).
**Modified:** `example2/ImGuiPass.{h,cpp}` (generalize to `drawUi` callback), `example2/tsdf_viewer.cpp` (migrate to callback), `example2/CMakeLists.txt` (new target, shader reuse, links).
**Reused unchanged:** `example2/PointCloudPass.{h,cpp}`, `example2/Shaders/pointcloud.{vert,frag}`.

---

## Task 1: Generalize `ImGuiPass` to a `drawUi` callback + migrate `tsdf_viewer`

**Goal:** Decouple `ImGuiPass` from the tsdf_viewer-specific `ViewerState`/UI so both apps can share it, with tsdf_viewer regression-verified unchanged.

**Files:** modify `example2/ImGuiPass.h`, `example2/ImGuiPass.cpp`, `example2/tsdf_viewer.cpp`.

**Interfaces produced (used by T3):**
- `ImGuiPass(Engine::Core::Context&, GLFWwindow*, VkFormat colorFormat, uint32_t imageCount)` — unchanged ctor.
- `void ImGuiPass::SetUi(std::function<void()> drawUi)` — sets the per-frame panel builder; must be called before first `Execute()`; the callback runs between `ImGui::NewFrame()` and `ImGui::Render()`.

- [ ] **Step 1: Strip app-specific state from `ImGuiPass.h`.** Remove the `struct ViewerState { ... }` definition, the `SetViewer(ViewerState*, PointCloudPass*)` method, the `class PointCloudPass;` forward decl, and the `ViewerState *m_state`/`PointCloudPass *m_pass` members. Add `#include <functional>`. Add public `void SetUi(std::function<void()> drawUi) { m_drawUi = std::move(drawUi); }` and private member `std::function<void()> m_drawUi;`. Update the class doc-comment to say it renders a caller-supplied ImGui panel (drop the "owns the full viewer UI" wording).
- [ ] **Step 2: Update `ImGuiPass.cpp::Execute`.** Keep the `ImGui_ImplVulkan_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();` prologue and the load-not-clear `RenderingScope` + `ImGui_ImplVulkan_RenderDrawData(...)` epilogue exactly as-is. Replace the hardcoded panel body (the `ImGui::Begin("TSDF Viewer") ... ImGui::End()` block that reads `m_state`/`m_pass`) with: `if (m_drawUi) m_drawUi();`. Remove any now-unused includes tied to the old UI.
- [ ] **Step 3: Move `ViewerState` into `tsdf_viewer.cpp`.** Paste the `struct ViewerState { ... }` (verbatim from the old `ImGuiPass.h`, including the stats fields) near the top of `tsdf_viewer.cpp` (anonymous namespace). Remove `imguiPass.SetViewer(&state, pointCloudPass)`; instead call `imguiPass.SetUi([&state, pointCloudPass]() { /* panel */ });` where the lambda body is the exact panel that used to live in `ImGuiPass.cpp::Execute` (scene combo, maxDirections slider, viewAngle checkbox, extractColor combo, the 4 show* checkboxes calling `pointCloudPass->SetVisible(...)`, the stats `ImGui::Text`, and the "Re-integrate" button — all reading/writing `state`). Keep the between-frames rebuild logic and `--frames`/`--scene`/`--maxdir`/`--viewangle` args untouched.
- [ ] **Step 4: Build.** `VULKAN_SDK=/usr/local cmake --build build --parallel --target tsdf_viewer`. Expected: clean (reconfigure first only if you touched CMake — you didn't).
- [ ] **Step 5: Regression launch-smoke + numeric check.** Run:
  - `VULKAN_SDK=/usr/local ./build/example2/tsdf_viewer --frames 120 ; echo exit=$?` → exit=0.
  - `VULKAN_SDK=/usr/local ./build/example2/tsdf_viewer --scene interproximal --frames 60` → prints a `[rebuild] scene=interproximal ... crossP(dir4)=-0.2004... crossN(dir5)=0.2004...` line. Confirm the numbers are unchanged vs before the refactor (dir4 ≈ -0.2005, dir5 ≈ +0.2005). This proves the refactor didn't alter behavior.
- [ ] **Step 6: Commit.** `git commit -am "refactor(example2): generalize ImGuiPass to a drawUi callback; migrate tsdf_viewer"`.

---

## Task 2: `shape_fixtures.h` + headless `tsdf_feature_compare --dump` (correctness milestone)

**Goal:** Analytic cube/cylinder fixtures + the compute-and-measure pipeline, verified purely by the `--dump` metric (no GUI yet): Directional beats Simple on edge error, ties on flat/curved.

**Files:** create `example2/shape_fixtures.h`, create `example2/tsdf_feature_compare.cpp` (headless), modify `example2/CMakeLists.txt`.

**Interfaces produced (used by T3):**
- `namespace fixtures { enum class Shape { Cube, Cylinder }; enum class Region { Flat, Curved, Edge }; struct View { Eigen::Vector3f camPos; std::vector<Eigen::Vector3f> points, normals; }; std::vector<View> SampleViews(Shape, float voxelSize); float NearestDistance(Shape, const Eigen::Vector3f& p); Region ClassifyRegion(Shape, const Eigen::Vector3f& p, float voxelSize); }`
- app CLI: `--dump`, `--shape cube|cylinder`, `--voxel <f>`.

- [ ] **Step 1: Write `shape_fixtures.h` geometry.** Header-only, `namespace fixtures`. Shapes centered at origin, sized to the voxel scale: **Cube** half-extent `H = 1.5` (3mm side, `voxelSize=0.1`); **Cylinder** axis = +Z, radius `R = 1.5`, half-height `HZ = 1.5`.
  - `float NearestDistance(Shape s, const Eigen::Vector3f& p)`:
    - Cube: exact box distance `d = ||max(|p|-H, 0)|| + min(max(|p|.x,|p|.y,|p|.z) - H, 0)` (standard signed box SDF); return `std::abs(d)`.
    - Cylinder: radial `dr = hypot(p.x,p.y) - R`, axial `dz = |p.z| - HZ`; outside-distance `= hypot(max(dr,0),max(dz,0))`, inside-distance `= min(max(dr,dz),0)`; return `std::abs(outside + inside)`.
  - `Region ClassifyRegion(Shape s, const Eigen::Vector3f& p, float voxelSize)`: `band = 2*voxelSize`.
    - Cube: `Edge` if at least two of `{|p.x|,|p.y|,|p.z|}` are within `band` of `H` (near an edge/corner); else `Flat`.
    - Cylinder: let `r = hypot(p.x,p.y)`; near a rim if `|r-R| <= band && ||p.z|-HZ| <= band` → `Edge`; else if `||p.z|-HZ| <= band` (on a cap plane) → `Flat`; else `Curved` (side wall).
- [ ] **Step 2: Write `SampleViews`.** Choose `V = 8` virtual viewpoints on a sphere of radius `5` around the origin (e.g., the 8 cube-corner directions `(±1,±1,±1)/√3 * 5`). For each view `camPos`, generate candidate surface samples on a grid and keep only FRONT-FACING ones (`normal.dot((camPos - p).normalized()) > 0.1`), budget-capped to ≲ 800 per view (coarsen the grid step to hit the budget):
  - Cube: sample each of the 6 faces on a `step ≈ 0.15` grid over `[-H,H]²`, with the outward axis-aligned normal; front-facing filter drops back faces. (~ up to ~3 faces × ~20×20 visible per view → coarsen step so per-view kept ≲ 800.)
  - Cylinder: sample the side wall (`θ` × `z` grid, normal = radial outward) and the two caps (`x,y` grid, normal = ±Z); front-facing filter per view. Coarsen to per-view ≲ 800.
  - Return the per-view `points`/`normals`. (These feed BOTH TSDFs: Simple uses `points`+`camPos`; Directional uses `points`+`normals`+`camPos`.)
- [ ] **Step 3: Write the headless app `tsdf_feature_compare.cpp`.** Parse `--shape` (default cube), `--voxel` (default 0.1), `--dump`. Build the shape's views. Run both TSDFs:
  - `Engine::Spatial::SimpleTSDF simple; simple.Build(ctx, voxel, 0.3f); for (auto& v : views) simple.Integrate(v.points, v.camPos); auto simpleCloud = simple.ExtractPointCloud();` → `OrientedPointCloud{points,normals}`.
  - `Engine::Spatial::DirectionalTSDF dir; dir.Build(ctx, voxel, 0.3f); dir.SetIntegrationQuality({3,4,true}); for (auto& v : views) dir.Integrate(v.points, v.normals, v.camPos, Eigen::Vector3f::Zero()); const auto& dirCloud = dir.PointCloud();` → `vector<ExtractedPoint>` (use `.position`).
  - Metric: a helper `void accumulate(Shape, voxel, const std::vector<Eigen::Vector3f>& pts, perRegion mean/max accumulators)` computing `err=NearestDistance`, `region=ClassifyRegion` per point. Compute for both clouds.
  - In `--dump`: print a table per shape: `region | Simple mean/max | Dir mean/max` (mm), plus `nInput` (total sampled), `nSimple`/`nDir` extracted counts. THEN `return 0` (no window in T2).
- [ ] **Step 4: CMake — add the headless target.** In `example2/CMakeLists.txt`, add `add_spatial_example(tsdf_feature_compare tsdf_feature_compare.cpp)` (links `Engine::Spatial`; `SimpleTSDF`/`DirectionalTSDF` need `Engine::Core` transitively — add `target_link_libraries(tsdf_feature_compare PRIVATE Engine::Core)` if the link fails). Reconfigure.
- [ ] **Step 5: Build + `--dump` verification (the oracle).** Build `--target tsdf_feature_compare`. Run:
  - `VULKAN_SDK=/usr/local ./build/example2/tsdf_feature_compare --shape cube --dump`
  - `VULKAN_SDK=/usr/local ./build/example2/tsdf_feature_compare --shape cylinder --dump`
  Confirm in BOTH: `edge` region `Dir.mean < Simple.mean` (the hypothesis — Directional preserves the corner), and `flat`/`curved` regions roughly comparable (Dir not materially worse). Paste both tables in the report. If the hypothesis fails, there is a bug in sampling / integration / classification — fix before DONE.
- [ ] **Step 6: Commit.** `git commit -am "feat(example2): analytic cube/cylinder fixtures + headless Simple-vs-Directional error metric"`.

---

## Task 3: Interactive `tsdf_feature_compare` viewer — A/B toggle, color modes, panel

**Goal:** Add the windowed viewer around the T2 pipeline: A/B toggle between Simple/Directional extracted clouds (same camera), error/region color modes, ImGui panel with the metric table, between-frames rebuild.

**Files:** modify `example2/tsdf_feature_compare.cpp`, `example2/CMakeLists.txt`; create nothing new (reuse `PointCloudPass`, `ImGuiPass`, `Shaders/pointcloud.*`). Add color helpers to `example2/shape_fixtures.h`.

**Consumes:** T1 `ImGuiPass::SetUi`; T2 `fixtures::*` and the metric helpers; `PointCloudPass` (`SetPointSet(int,const std::vector<PointVertex>&)`, `SetVisible(int,bool)`, `struct PointVertex{float pos[3]; uint8_t rgba[4];}`).

- [ ] **Step 1: Color helpers in `shape_fixtures.h`.** Add `inline Rgb errorColor(float err, float maxErr)` (sequential blue→red: `t=clamp(err/maxErr,0,1)`, e.g. `{uint8_t(255*t), uint8_t(60), uint8_t(255*(1-t))}`) and `inline Rgb regionColor(Region r)` (`Flat`={200,200,200}, `Curved`={80,160,230}, `Edge`={240,80,80}). Reuse the `Rgb` type from `tsdf_fixtures.h` (include it) or define a local `Rgb{uint8_t r,g,b;}` if that pulls in render deps — keep this header render-free (T2 links no render). Normalize error with `maxErr = 3*voxelSize`.
- [ ] **Step 2: `CompareState` + rebuild.** In `tsdf_feature_compare.cpp` add `struct CompareState { int shape=0; int method=1; /*0=Simple,1=Directional*/ int colorMode=0; /*0=error,1=region*/ bool showInput=false; float voxel=0.1f; bool dirty=true; /* metrics per region for both methods; nInput,nSimple,nDir */ };`. Write `rebuild(CompareState&, Context&, PointCloudPass&)`: regenerate views, run both TSDFs (as T2), compute metrics, then build 3 point sets via `SetPointSet`: set 0 = INPUT (all view points, white); set 1 = SIMPLE extracted (each point colored by `colorMode`: `errorColor(NearestDistance,3*voxel)` or `regionColor(ClassifyRegion)`); set 2 = DIRECTIONAL extracted (same coloring). Apply visibility: `SetVisible(0, showInput)`, `SetVisible(1, method==0)`, `SetVisible(2, method==1)`. Clear `dirty`. Print the same `[rebuild]` metric line to stdout.
- [ ] **Step 3: Windowed main + manual loop.** Mirror `tsdf_viewer.cpp`'s main: `Application app(...)`, `Camera`, trackball `MouseListenerGroup`, `RenderGraph` with `PointCloudPass` then `ImGuiPass` (last). Downcast `static_cast<Engine::Render::GlfwWindow&>(app.GetWindow())` for `Handle()`. Manual loop: `PollEvents → if state.dirty { vkDeviceWaitIdle; rebuild(...); } → BeginFrame → graph.Execute → EndFrame`, honoring `--frames`. ESC closes. Keep `--dump` (compute + print + return before the window) and `--shape`/`--voxel`.
- [ ] **Step 4: Panel via `SetUi` + Tab toggle.** `imguiPass.SetUi([&state, pointCloudPass]() { ImGui::Begin("Feature Compare"); scene/method/colorMode controls; metric table; End(); });`:
  - `ImGui::Combo("shape", &state.shape, "cube\0cylinder\0\0")` → sets `dirty`.
  - method radio `ImGui::RadioButton("Simple", &state.method, 0); ImGui::SameLine(); ImGui::RadioButton("Directional", &state.method, 1);` → on change call `pointCloudPass->SetVisible(1, method==0)`/`SetVisible(2, method==1)` immediately (no rebuild).
  - `ImGui::Combo("color", &state.colorMode, "error\0region\0\0")` → sets `dirty` (recolor needs rebuild).
  - `ImGui::Checkbox("show input", &state.showInput)` → `SetVisible(0, showInput)` immediately.
  - `ImGui::Text` metric table (per-region Simple vs Dir mean/max); `ImGui::Button("Re-run")` → `dirty=true`.
  - Also in the main loop, poll the `Tab` key (via the existing key listener path used for ESC) to flip `state.method` and update visibility (A/B toggle) without a rebuild.
- [ ] **Step 5: CMake — promote target to windowed.** Change the target to include the render sources and links, mirroring how `tsdf_viewer` is declared: `add_spatial_example(tsdf_feature_compare tsdf_feature_compare.cpp PointCloudPass.cpp)`, `target_link_libraries(tsdf_feature_compare PRIVATE Engine::Render Engine::Core)`, and `add_compiled_shaders(tsdf_feature_compare FEATURE_SHADER_DIR Shaders/pointcloud.vert Shaders/pointcloud.frag)` (reuses the existing shaders). Read `FEATURE_SHADER_DIR` in the app for the shader path. Reconfigure.
- [ ] **Step 6: Build + verify.** Build `--target tsdf_feature_compare`. Then:
  - `VULKAN_SDK=/usr/local ./build/example2/tsdf_feature_compare --shape cube --frames 120 ; echo exit=$?` → exit=0, no NEW validation errors.
  - `--shape cylinder --frames 120` → exit=0.
  - `--shape cube --dump` and `--shape cylinder --dump` → tables still hold the hypothesis (unchanged from T2).
  - Also re-run `tsdf_viewer --scene interproximal --frames 60` once to confirm the shared `ImGuiPass` still works there.
  Paste results in the report.
- [ ] **Step 7: Commit.** `git commit -am "feat(example2): interactive Simple-vs-Directional feature-preservation viewer (A/B, error/region color, metrics)"`.

---

## Out of scope (follow-on)
Noise/outlier injection slider; real scan data; side-by-side dual viewport; error histogram; sphere/wedge shapes; marching-cubes for Directional; unit tests.

## Self-Review Notes
- Spec coverage: fixtures+metric → T2; A/B viewer+color+panel → T3; ImGuiPass generalization → T1. Verification-by-`--dump`-oracle honored (T2 S5, T3 S6).
- Risk hotspots inline: per-view sample budget ≲800 for the large-N bug (T2 S2); front-facing filter must let the corner be seen from both sides to reproduce rounding (T2 S2); `shape_fixtures.h` must stay render-free so T2 links no `Engine::Render` (T3 S1); ImGuiPass generalization must not regress tsdf_viewer (T1 S5, re-checked T3 S6).
- Type consistency: `fixtures::Shape/Region/View`, `CompareState`, `PointVertex`, `IntegrationQuality{3,4,true}`, `ImGuiPass::SetUi(std::function<void()>)` used consistently across tasks.

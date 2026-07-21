# TSDF Interactive Viewer (Vulkan + ImGui) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** An in-app windowed viewer (`example2/tsdf_viewer`) that renders DirectionalTSDF's input / extracted / SDF-slice point clouds with an ImGui control panel and trackball, on synthetic fixtures, to visually diagnose integrate/extract.

**Architecture:** Reuse `Engine::Render` (`Application`/`Camera`/`RenderGraph`/trackball, dynamic-rendering `GraphicsPipeline`) — the `cube_render.cpp`/`CubePass` pattern. New: `PointCloudPass` (colored points), `ImGuiPass` (vendored `lib/imgui` glfw+vulkan backends), the viewer app. Point data comes from `DirectionalTSDF::PointCloud()` + `DebugDownloadGroupVoxels()`.

**Tech Stack:** C++17, Vulkan (dynamic rendering, MoltenVK), GLFW, ImGui (`lib/imgui`), Eigen. glslc for shaders.

## Global Constraints

- Everything lives in `example2/` (a demo app) except a 1-line accessor added to `src/Engine/Render/GlfwWindow.h` and possibly a `lib/CMakeLists.txt` imgui target. Do NOT change `Engine::Spatial` or the TSDF core.
- Mirror `example2/CubePass.{h,cpp}` for pass/pipeline structure. MVP push constant type is `vkMath::Mat4` (what `Camera::GetProjectionMatrix()/GetViewMatrix()` return), NOT Eigen.
- Reuse the fixture + slice-sampling logic from `example2/tsdf_slice_debug.cpp` (plane/interproximal generation, `sdfColor`, `dirColor`, `floorDiv8`, `valueAt` via `DebugDownloadGroupVoxels`). Factor shared bits into a small header (`example2/tsdf_fixtures.h`) or copy — do not re-derive the math.
- **GUI has no unit-test oracle.** The gate for each visual task is: **builds AND launches without Vulkan validation errors AND runs ≥ a few frames without crashing.** Add a `--frames N` arg (default 0 = run until closed) so the smoke check can auto-exit: `./build/example2/tsdf_viewer --frames 120` must exit 0 with no `VUID`/validation error on stderr. Numeric correctness is cross-checked against `tsdf_slice_debug` (already verified), not re-derived.
- Build env (memory `project-repo-build-worktree-gotchas`): `VULKAN_SDK=/usr/local`, `example2/ShadowMap.cpp` stub present, `-DGTest_DIR=... -Dgflags_DIR=...` on configure (Ceres in the tree). Build only the needed target; device Apple M4 Max.
- Commit after each task with a passing build (+ launch-smoke where applicable).

## File Structure

**New:** `example2/PointCloudPass.{h,cpp}`, `example2/ImGuiPass.{h,cpp}`, `example2/tsdf_viewer.cpp`, `example2/tsdf_fixtures.h`, `example2/Shaders/pointcloud.vert`, `example2/Shaders/pointcloud.frag`.
**Modified:** `src/Engine/Render/GlfwWindow.h` (add `GLFWwindow* Handle()`), `example2/CMakeLists.txt` (target + imgui link + shader compile), possibly `lib/CMakeLists.txt` (imgui target).

---

## Task 1: `PointCloudPass` + shaders + minimal viewer (points on screen)

**Goal:** a window that renders a static colored point cloud (the plane fixture's input points) with trackball — the first visual milestone. No ImGui yet.

**Files:** create `example2/PointCloudPass.{h,cpp}`, `example2/Shaders/pointcloud.vert`/`.frag`, `example2/tsdf_fixtures.h`, `example2/tsdf_viewer.cpp`; modify `src/Engine/Render/GlfwWindow.h`, `example2/CMakeLists.txt`.

- [ ] **Step 1: Add `GLFWwindow* Handle()` to `GlfwWindow.h`** (public accessor returning `m_window`; needed by ImGui in Task 2 but add now).
- [ ] **Step 2: Shaders** — `pointcloud.vert`: `layout(push_constant) uniform PC { mat4 mvp; float pointSize; };` in `vec3 inPos; in vec4 inColor;` → `gl_Position = mvp*vec4(inPos,1); gl_PointSize = pointSize; out vec4 vColor=inColor;`. `pointcloud.frag`: round-point mask via `gl_PointCoord` (discard outside radius 0.5), `out vColor`.
- [ ] **Step 3: `PointCloudPass`** — mirror `CubePass`: `struct PointVertex { float pos[3]; uint8_t rgba[4]; };` pipeline `topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST`, `.VertexBinding<PointVertex>().VertexAttribute(0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(pos)).VertexAttribute(1,0,VK_FORMAT_R8G8B8A8_UNORM,offsetof(rgba)).ColorTarget(colorFormat).DepthTarget(VK_FORMAT_D32_SFLOAT).PushConstant<PC>(VERTEX)`. Hold up to 4 named sets, each `{Engine::Core::Buffer buf; uint32_t count; bool visible;}`. `void SetPointSet(int id, const std::vector<PointVertex>&)` (allocate+upload, `QueueRole::Graphics`); `void SetVisible(int,bool)`. `Execute`: `RenderingScope` (like CubePass), `Bind`, `PushConstants({mvp, pointSize})`, for each visible set: bind its vertex buffer + `vkCmdDraw(count,1,0,0)`.
- [ ] **Step 4: `tsdf_fixtures.h`** — factor from `tsdf_slice_debug.cpp`: plane/interproximal point+normal generation, `sdfColor`, `dirColor`, `floorDiv8`, and a `valueAt(tsdf, cache, dir, voxelSize, p, occupied)` helper.
- [ ] **Step 5: minimal `tsdf_viewer.cpp`** — like `cube_render.cpp`: `Application`, `Camera`, `RenderGraph` with a `PointCloudPass`. Build a `DirectionalTSDF`, integrate the plane fixture, build the INPUT point set (white) → `pass.SetPointSet`. Trackball (copy cube_render's mouse handlers), key ESC to close. `--frames N` arg → after N rendered frames call `window.RequestClose()`. `app.Run()`.
- [ ] **Step 6: CMake** — `add_spatial_example(tsdf_viewer tsdf_viewer.cpp PointCloudPass.cpp)`; `add_compiled_shaders(tsdf_viewer VIEWER_SHADER_DIR Shaders/pointcloud.vert Shaders/pointcloud.frag)`; link `Engine::Render` (the helper links Engine::Spatial; add Render). Reconfigure.
- [ ] **Step 7: Build + launch-smoke** — `cmake --build build --target tsdf_viewer`; `./build/example2/tsdf_viewer --frames 120` exits 0, no validation error. (A window may appear briefly.) Commit `feat(example2): PointCloudPass + minimal TSDF point viewer`.

---

## Task 2: `ImGuiPass` — control panel renders

**Goal:** an ImGui panel drawn over the points (start with a stats/hello window), proving the ImGui-Vulkan integration.

**Files:** create `example2/ImGuiPass.{h,cpp}`; modify `example2/CMakeLists.txt` (link imgui), maybe `lib/CMakeLists.txt`.

- [ ] **Step 1: imgui link** — `lib/CMakeLists.txt` ALREADY defines an `imgui` STATIC target (core `imgui*.cpp` + `backends/imgui_impl_glfw.cpp` + `imgui_impl_vulkan.cpp`, `target_include_directories PUBLIC imgui + imgui/backends`, `target_link_libraries PUBLIC Vulkan::Vulkan glfw`). So this is done — just add `target_link_libraries(tsdf_viewer PRIVATE imgui)` in `example2/CMakeLists.txt`. Include paths come transitively from the `imgui` target (`#include "imgui.h"`, `"backends/imgui_impl_glfw.h"`, `"backends/imgui_impl_vulkan.h"` — verify exact include prefix against the target's include dirs).
- [ ] **Step 2: `ImGuiPass`** — ctor takes `Context&`, `GLFWwindow*`, `VkFormat colorFormat`, `uint32_t imageCount`. Init: `IMGUI_CHECKVERSION(); CreateContext(); ImGui_ImplGlfw_InitForVulkan(window, true);` create a `VkDescriptorPool` (combined-image-sampler, ~1000); fill `ImGui_ImplVulkan_InitInfo{ Instance, PhysicalDevice, Device, QueueFamily=graphicsFamily, Queue=graphicsQueue, DescriptorPool, MinImageCount/ImageCount, MSAASamples=VK_SAMPLE_COUNT_1_BIT, UseDynamicRendering=true, PipelineRenderingCreateInfo{ colorAttachmentCount=1, pColorAttachmentFormats=&colorFormat } }`; `ImGui_ImplVulkan_Init(&info)`. Dtor: `DeviceWaitIdle`, `ImGui_ImplVulkan_Shutdown`, `ImGui_ImplGlfw_Shutdown`, `DestroyContext`, destroy pool.
- [ ] **Step 3: Execute** — `ImGui_ImplVulkan_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();` build UI (Task-2: a simple `ImGui::Begin("TSDF Viewer")` + text); `ImGui::Render();` open a `RenderingScope` with **load (no clear)** on the same swap image/depth, then `ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), ctx.commandBuffer)`. (Verify the engine's `RenderingDescriptor`/`ClearOptions` supports a load-only variant; if it always clears, ImGui must draw inside the PointCloudPass's scope instead — reconcile so points+UI both show.)
- [ ] **Step 4: wire into viewer** — add `ImGuiPass` as the LAST pass in the render graph; pass `window.Handle()`, `swapChain.Format()`, `swapChain.ImageCount()`.
- [ ] **Step 5: Build + launch-smoke** — `--frames 120` exits 0, no validation error, (window shows points + an ImGui panel). Commit `feat(example2): ImGuiPass (vendored imgui glfw+vulkan, dynamic rendering)`.

---

## Task 3: Full viewer — fixtures, all point sets, controls, stats

**Goal:** the complete interactive viewer: scene + quality + layer toggles rebuild the TSDF and its point sets; stats readout.

**Files:** modify `example2/tsdf_viewer.cpp`, `example2/ImGuiPass.{h,cpp}` (UI + shared state).

- [ ] **Step 1: shared `ViewerState`** — `struct ViewerState { int scene=0; int maxDirections=1; bool viewAngle=false; bool showInput=true, showExtracted=true, showSliceP=true, showSliceN=true; int extractColor=0; bool dirty=true; // stats: size_t nInput,nExtracted; float zMean; float crossP,crossN; };` shared by reference between the app and `ImGuiPass`.
- [ ] **Step 2: `rebuild(state, tsdf, pass)`** — generate fixture (plane or interproximal via `tsdf_fixtures.h`), `tsdf.Build`+`SetIntegrationQuality({maxDirections,4,viewAngle})`+`Integrate` (interproximal = two passes), build point sets: INPUT (white), EXTRACTED (from `PointCloud()`, color by `dirMask` or green), SLICE+Z / SLICE−Z (iterate voxel centers on y=0 plane, `valueAt` per dir, `sdfColor`). `pass.SetPointSet(...)` for each; set `pass.SetVisible` from the show* flags. Fill stats (counts, extracted |z| mean, per-layer zero-crossing). Clear `dirty`.
- [ ] **Step 3: UI in `ImGuiPass`** — `ImGui::Combo` scene [plane, interproximal]; `ImGui::SliderInt`/radio maxDirections [1,2]; `ImGui::Checkbox` viewAngle; checkboxes showInput/Extracted/SliceP/SliceN; combo extractColor [direction, green]; `ImGui::Text` stats; `ImGui::Button("Re-integrate")`. Any change sets `state.dirty=true` (and toggles call `pass.SetVisible` immediately, no rebuild needed).
- [ ] **Step 4: app loop** — before `app.Run()` or via a per-frame hook, if `state.dirty` call `rebuild(...)`. (If `Application::Run` has no per-frame callback, add a minimal frame hook or drive the loop manually like `cube_render`.) Reconcile visibility toggles (cheap) vs rebuild (scene/quality).
- [ ] **Step 5: Build + launch-smoke + numeric cross-check** — `--frames 120` clean; run interproximal scene and confirm the printed/stat zero-crossings match `tsdf_slice_debug` (+Z ≈ −0.2, −Z ≈ +0.2). Commit `feat(example2): full interactive TSDF viewer (scenes, quality, layer toggles, stats)`.

---

## Out of scope (follow-on)
Real scanData/chair loading; movable slice-plane slider; screenshot save; single-vs-multi side-by-side; marching-cubes mesh.

## Self-Review Notes
- Spec coverage: PointCloudPass → T1; ImGuiPass → T2; scenes/quality/toggles/stats → T3; verification-by-launch honored via `--frames N` smoke each task.
- Risk hotspots called out inline: (a) RenderingScope clear-vs-load so points+UI coexist (T2 S3), (b) imgui link target (T2 S1), (c) Application per-frame hook for rebuild (T3 S4), (d) gl_PointSize on MoltenVK (T1 — fallback: instanced quads). Implementers must resolve these against the real API, not assume.
- No TSDF-core changes; only a 1-line GlfwWindow accessor outside example2.

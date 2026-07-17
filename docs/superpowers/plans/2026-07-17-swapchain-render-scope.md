# SwapchainColorDepthScope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small RAII helper, `vkRender::SwapchainColorDepthScope`, that wraps the swapchain-color-image + depth-image transition/render/transition-back boilerplate that `example/cube_render.cpp`'s `CubePass::Execute()` currently hand-rolls inline, then refactor `CubePass::Execute()` to use it.

**Architecture:** `SwapchainColorDepthScope` lives next to `RenderingScope`/`RenderingDescriptor` in `src/vkRender/Rendering.h`/`.cpp` (same file — tightly related concepts). Its constructor transitions the swapchain color image (`UNDEFINED → COLOR_ATTACHMENT_OPTIMAL`) and a caller-owned depth `Image` (`→ DEPTH_ATTACHMENT_OPTIMAL`) *before* opening a `RenderingScope` — fixing a latent ordering bug in the current `CubePass` code, where the `RenderingScope` opens before those transitions. Its destructor closes the `RenderingScope` and transitions the color image back to `PRESENT_SRC_KHR`. `CubePass::Execute()` then shrinks to: prepare resources → open the scope → draw → (RAII cleanup).

**Tech Stack:** C++17, Vulkan 1.3 (dynamic rendering), existing `vkRender` module conventions.

## Global Constraints

- Only these 3 files change: `src/vkRender/Rendering.h`, `src/vkRender/Rendering.cpp`, `example/cube_render.cpp`. Nothing else — specifically, `example/realtime_shadow.cpp` is explicitly out of scope for this work (its `RecordScenePass` has extra branching around the screenshot-capture path that doesn't cleanly fit this new type; adopting it there is a separate future task).
- `RenderingScope`'s and `RenderingDescriptor`'s existing public signatures and behavior do not change — `SwapchainColorDepthScope` is a new type built on top of them, not a modification to them.
- The pipeline-stage/access-mask values used in the new type's transitions must be exactly the values already used in `CubePass::Execute()` today (`example/cube_render.cpp:73-100`) — do not invent different values; these already match `RealtimeShadowPass::RecordScenePass`'s equivalent barriers byte-for-byte, so they're a proven-correct pair.
- The color/depth transitions must execute before `RenderingScope`'s constructor runs (before `vkCmdBeginRendering`) — this is the specific bug fix motivating this change. Enforce this via the `BeginTransitionsAndBuildDescriptor` static-helper-in-initializer-list pattern (below), not via reordering statements in a constructor body (member initialization order makes that not work — see Task 1).
- No CMake changes — all 3 files already exist and are already covered by their targets' `GLOB_RECURSE`/explicit source lists.
- Follow the existing header-coupling convention seen in `src/vkRender/RenderGraph.h:14-15` (forward-declares `SwapChain`/`View` rather than including their full headers, since they're only used as pointer/reference parameters): forward-declare `class Image;` in `Rendering.h` rather than `#include "vkRender/Image.h"` there; the full include only goes in `Rendering.cpp`.

---

### Task 1: Add `SwapchainColorDepthScope` to `vkRender`

**Files:**
- Modify: `src/vkRender/Rendering.h`
- Modify: `src/vkRender/Rendering.cpp`

**Interfaces:**
- Consumes: `vkRender::RenderingDescriptor::ColorDepth(...)` and `vkRender::RenderingScope` (both already exist in this same file, unchanged), `vkRender::Image::TransitionLayout` (instance method, for the depth target) and the static `vkRender::Image::TransitionLayout` overload (for the raw swapchain `VkImage`) from `src/vkRender/Image.h`.
- Produces: `vkRender::SwapchainColorDepthScope`, a new public type with this exact interface, for Task 2 to consume:
  ```cpp
  class SwapchainColorDepthScope {
  public:
      SwapchainColorDepthScope(VkCommandBuffer commandBuffer,
                               VkImage swapImage,
                               VkImageView swapImageView,
                               Image &depthTarget,
                               VkExtent2D extent,
                               const ClearOptions &clear);
      ~SwapchainColorDepthScope();

      SwapchainColorDepthScope(const SwapchainColorDepthScope &) = delete;
      SwapchainColorDepthScope &operator=(const SwapchainColorDepthScope &) = delete;

  private:
      static RenderingDescriptor BeginTransitionsAndBuildDescriptor(
              VkCommandBuffer commandBuffer,
              VkImage swapImage,
              VkImageView swapImageView,
              Image &depthTarget,
              VkExtent2D extent,
              const ClearOptions &clear);

      VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
      VkImage m_swapImage = VK_NULL_HANDLE;
      RenderingScope m_scope;
  };
  ```

- [ ] **Step 1: Add the forward declaration and class declaration to `Rendering.h`**

In `src/vkRender/Rendering.h`, add `class Image;` forward declaration after the `namespace vkRender {` line (before `struct RenderingDescriptor`), and add the `SwapchainColorDepthScope` class declaration (exact code above) after the closing brace of `class RenderingScope { ... };` and before `} // namespace vkRender`.

The full resulting file should read:
```cpp
#pragma once

#include "vkRender/RenderAttachments.h"
#include "vkRender/RenderTypes.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace vkRender {

    class Image;

    struct RenderingDescriptor {
        VkRect2D renderArea{};
        uint32_t layerCount = 1;
        std::vector<VkRenderingAttachmentInfo> colorAttachments;
        bool hasDepthAttachment = false;
        VkRenderingAttachmentInfo depthAttachment{};
        bool setViewport = true;
        bool setScissor = true;
        VkViewport viewport{};
        VkRect2D scissor{};

        RenderingDescriptor() = default;
        explicit RenderingDescriptor(VkExtent2D extent);

        static RenderingDescriptor ColorDepth(VkExtent2D extent,
                                              VkImageView colorView,
                                              VkImageView depthView,
                                              const ClearOptions &clear);

        RenderingDescriptor &SetExtent(VkExtent2D extent);
        RenderingDescriptor &SetRenderArea(VkRect2D area);
        RenderingDescriptor &AddColorAttachment(const VkRenderingAttachmentInfo &attachment);
        RenderingDescriptor &SetDepthAttachment(const VkRenderingAttachmentInfo &attachment);
        RenderingDescriptor &SetViewport(VkViewport value);
        RenderingDescriptor &SetScissor(VkRect2D value);
        RenderingDescriptor &UseDefaultViewportAndScissor(VkExtent2D extent);
    };

    class RenderingScope {
    public:
        RenderingScope(VkCommandBuffer commandBuffer,
                       const RenderingDescriptor &descriptor);
        ~RenderingScope();

        RenderingScope(const RenderingScope &) = delete;
        RenderingScope &operator=(const RenderingScope &) = delete;

        void End();

    private:
        VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
        bool m_active = false;
    };

    // RAII helper for the common "swapchain color image + depth image, one full-screen
    // dynamic-rendering pass" case: transitions both images to their attachment-optimal
    // layouts, opens a RenderingScope, and on destruction closes it and transitions the
    // color image back to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR. The color/depth transitions are
    // guaranteed to execute before the underlying RenderingScope opens (before
    // vkCmdBeginRendering), which a hand-written "open scope, then transition" ordering can
    // get backwards.
    class SwapchainColorDepthScope {
    public:
        SwapchainColorDepthScope(VkCommandBuffer commandBuffer,
                                 VkImage swapImage,
                                 VkImageView swapImageView,
                                 Image &depthTarget,
                                 VkExtent2D extent,
                                 const ClearOptions &clear);
        ~SwapchainColorDepthScope();

        SwapchainColorDepthScope(const SwapchainColorDepthScope &) = delete;
        SwapchainColorDepthScope &operator=(const SwapchainColorDepthScope &) = delete;

    private:
        static RenderingDescriptor BeginTransitionsAndBuildDescriptor(
                VkCommandBuffer commandBuffer,
                VkImage swapImage,
                VkImageView swapImageView,
                Image &depthTarget,
                VkExtent2D extent,
                const ClearOptions &clear);

        VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
        VkImage m_swapImage = VK_NULL_HANDLE;
        RenderingScope m_scope;
    };

} // namespace vkRender
```

- [ ] **Step 2: Implement `SwapchainColorDepthScope` in `Rendering.cpp`**

In `src/vkRender/Rendering.cpp`, add `#include "vkRender/Image.h"` to the includes at the top (after `#include "vkRender/Rendering.h"`), and add this implementation at the end of the file, before the closing `} // namespace vkRender`:

```cpp
    RenderingDescriptor SwapchainColorDepthScope::BeginTransitionsAndBuildDescriptor(
            VkCommandBuffer commandBuffer,
            VkImage swapImage,
            VkImageView swapImageView,
            Image &depthTarget,
            VkExtent2D extent,
            const ClearOptions &clear) {
        Image::TransitionLayout(commandBuffer,
                                swapImage,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                0,
                                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

        depthTarget.TransitionLayout(commandBuffer,
                                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                                     0,
                                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        return RenderingDescriptor::ColorDepth(extent, swapImageView, depthTarget.View(), clear);
    }

    SwapchainColorDepthScope::SwapchainColorDepthScope(VkCommandBuffer commandBuffer,
                                                        VkImage swapImage,
                                                        VkImageView swapImageView,
                                                        Image &depthTarget,
                                                        VkExtent2D extent,
                                                        const ClearOptions &clear)
        : m_commandBuffer(commandBuffer),
          m_swapImage(swapImage),
          m_scope(commandBuffer,
                  BeginTransitionsAndBuildDescriptor(
                          commandBuffer, swapImage, swapImageView, depthTarget, extent, clear)) {
    }

    SwapchainColorDepthScope::~SwapchainColorDepthScope() {
        m_scope.End();
        Image::TransitionLayout(m_commandBuffer,
                                m_swapImage,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                0);
    }
```

This is why the member initialization order matters and why the plan calls out the pattern explicitly: `m_scope`'s initializer calls `BeginTransitionsAndBuildDescriptor(...)` as a function argument, which is evaluated (running both `Image::TransitionLayout` calls) *before* `RenderingScope`'s own constructor (which calls `vkCmdBeginRendering`) runs on the returned descriptor. Declaration order in the class (`m_commandBuffer`, `m_swapImage`, `m_scope` — matching the initializer list order here) means `m_commandBuffer`/`m_swapImage` are already set by the time `m_scope`'s initializer evaluates, so using `commandBuffer`/`swapImage` (the constructor parameters, not the not-yet-relevant member names) in that call is correct and clearer.

- [ ] **Step 3: Build**

Run:
```bash
cmake --build build --target vkRender --parallel
```
Expected: builds with no errors. There's no existing automated test that exercises `vkRender` types directly (they require a live Vulkan device + swapchain to do anything meaningful) — this project's established convention (see `docs/superpowers/specs/2026-07-17-swapchain-render-scope-design.md`'s own verification section) is to verify `vkRender` changes by building and running an example, which Task 2 does. A clean build of the `vkRender` static library is the correct and sufficient check for this task in isolation.

- [ ] **Step 4: Run the existing test suite as a regression check**

Run:
```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
Expected: same pass count as before this change (44/45, with the pre-existing, unrelated `WideBVHTest.RadiusMatchesCpuReference` as the only failure — this project's tests don't exercise `vkRender` at all, so this step is confirming you haven't broken the build for anything else, not that your new code is behaviorally correct).

- [ ] **Step 5: Commit**

```bash
git add src/vkRender/Rendering.h src/vkRender/Rendering.cpp
git commit -m "Add SwapchainColorDepthScope RAII helper to vkRender"
```

---

### Task 2: Refactor `CubePass::Execute()` to use `SwapchainColorDepthScope`

**Files:**
- Modify: `example/cube_render.cpp`

**Interfaces:**
- Consumes: `vkRender::SwapchainColorDepthScope` from Task 1 (constructor signature above).

- [ ] **Step 1: Replace the body of `CubePass::Execute()`**

In `example/cube_render.cpp`, replace the entire `Execute` method (currently lines 53-101) with:

```cpp
        void Execute(vkRender::RenderContext &renderContext) override {
            if (!renderContext.swapChain)
                throw std::runtime_error("CubePass requires an active swapchain");

            const VkExtent2D extent = renderContext.swapChain->Extent();
            EnsureDepthResources(extent);
            EnsurePipeline(renderContext.swapChain->Format());

            VkCommandBuffer cmd = renderContext.commandBuffer;
            VkImage swapImage = renderContext.swapChain->Image(renderContext.imageIndex);
            const vkRender::ClearOptions clear =
                    renderContext.view ? renderContext.view->GetClearOptions() : vkRender::ClearOptions{};

            vkRender::SwapchainColorDepthScope scope(
                    cmd,
                    swapImage,
                    renderContext.swapChain->ImageView(renderContext.imageIndex),
                    *m_depthTarget,
                    extent,
                    clear);
            DrawObjects(cmd, extent);
        }
```

This is a straight behavioral port: same resource-preparation calls, same `DrawObjects` call, but the manual `RenderingDescriptor`/`RenderingScope`/four `TransitionLayout` calls (old lines 64-100) are replaced by constructing `scope`, whose constructor performs the color+depth transitions and opens the render scope (in the now-correct order — transitions before `vkCmdBeginRendering`), and whose destructor (running when `Execute()` returns and `scope` goes out of scope) closes the render scope and transitions the color image back to `PRESENT_SRC_KHR`.

- [ ] **Step 2: Build**

Run:
```bash
cmake --build build --target cube_render --parallel
```
Expected: builds with no errors.

- [ ] **Step 3: Run `cube_render` and visually confirm it renders identically to before this change**

Run:
```bash
./build/example/cube_render
```
Expected: a window opens showing the same rotating cube on the same dark background as before this refactor (color `(0.025, 0.027, 0.032)`), with correct depth testing (the cube's faces occlude each other correctly, not drawn in the wrong order). Press Enter once to trigger the screenshot capture and confirm a `screenshots/screenshot_*.png` file is created without error — this exercises the color image actually being in `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` when `Capture::CaptureToPNG` runs, which only holds if the new destructor's final transition ran correctly. Then resize the window (drag an edge/corner) and confirm it keeps rendering without crashing or validation errors — this exercises `EnsureDepthResources`/`EnsurePipeline`'s recreation path feeding into the new `SwapchainColorDepthScope` with a changed extent/depth image on the next frame. This path is unrelated to what Task 1/2 changed, but it's a cheap regression check since you already have the window open. Close the window when confirmed.

- [ ] **Step 4: Run the full test suite as a regression check**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: same 44/45 baseline as Task 1 Step 4 (unrelated to this change, confirms nothing else broke).

- [ ] **Step 5: Commit**

```bash
git add example/cube_render.cpp
git commit -m "Refactor CubePass::Execute to use SwapchainColorDepthScope"
```

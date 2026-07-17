# vkRender Architecture

`vkRender`는 `vkSpatial`과 같은 깊이의 렌더링 계층이다. 목표는 Vulkan 세부 구현을 숨기고, Google Filament처럼 `Engine`, `Renderer`, `SwapChain`, `View`, `Scene`, `Camera` 중심으로 사용할 수 있는 API를 제공하는 것이다.

## Design Goals

- App code should describe intent: scene, camera, view, render.
- Vulkan objects stay behind small RAII wrappers.
- `vkSpatial` compute/ray-trace results can be presented without copying back to CPU.
- Render paths can grow from buffer-present to raster, PBR, shadows, post-process, and RenderGraph passes.
- Window-system code stays outside `vkRender`; `vkRender` consumes the `VkSurfaceKHR` and queues already created by `vkCommon::VkContext`.

## Layering

```text
Application / GLFW / platform
    |
    v
vkCommon::VkContext
    - VkInstance, VkSurfaceKHR, VkDevice
    - compute queue, graphics/present queue
    |
    v
vkRender::Engine
    - factory and lifetime boundary
    |
    +-- vkRender::SwapChain
    +-- vkRender::Renderer
    +-- vkRender::View
    +-- vkRender::Scene
    +-- vkRender::Camera
    +-- vkRender::RenderGraph
    +-- vkRender::MouseInput
```

## Filament-style Object Model

| Filament concept        | vkRender concept                      | Role                                                            |
| ----------------------- | ------------------------------------- | --------------------------------------------------------------- |
| `Engine`                | `vkRender::Engine`                    | Creates API objects from a Vulkan context.                      |
| `SwapChain`             | `vkRender::SwapChain`                 | Owns presentable images and image views.                        |
| `Renderer`              | `vkRender::Renderer`                  | Owns per-frame command buffer, semaphores, fence, present flow. |
| `View`                  | `vkRender::View`                      | Binds scene, camera, viewport, clear options, render graph.     |
| `Scene`                 | `vkRender::Scene`                     | Holds renderable entity membership.                             |
| `Camera`                | `vkRender::Camera`                    | Stores view/projection state.                                   |
| input dispatcher        | `vkRender::MouseInput`                | Converts platform callbacks into reusable mouse events.          |
| `Material`/`Renderable` | `Resources.h` descriptors and handles | Placeholder for the next resource registry layer.               |

## First Milestone API

The first usable path is GPU buffer presentation. This matches the current `native_raytrace` flow: `vkSpatial::vkWideBVH::TraceRays()` writes an RGBA8 pixel buffer, then `vkRender::Renderer` copies it into the acquired swapchain image and presents.

```cpp
#include "vkRender/vkRender.h"

vkCommon::VkContext ctx;
ctx.init(true, glfwExtensions, surfaceFactory);

auto engine = vkRender::Engine::Create(&ctx);
auto swapChain = engine->CreateSwapChain({width, height});
auto renderer = engine->CreateRenderer();

while (running) {
    if (!renderer->BeginFrame(*swapChain)) {
        vkDeviceWaitIdle(ctx.device);
        swapChain->Recreate(width, height);
        renderer->ClearSwapChainRecreateFlag();
        continue;
    }

    // compute/ray tracing pass writes outputPixelBuffer here
    renderer->CopyBufferToSwapChain(outputPixelBuffer);
    renderer->EndFrame();

    if (renderer->NeedsSwapChainRecreate()) {
        vkDeviceWaitIdle(ctx.device);
        swapChain->Recreate(width, height);
        renderer->ClearSwapChainRecreateFlag();
    }
}

renderer.reset();
swapChain.reset();
engine.reset();
ctx.shutdown();
```

If the swapchain format is BGRA, pass `swapChain->RequiresRedBlueSwap()` into shaders that pack RGBA bytes.

## Planned Render Pipeline

`GraphicsPipeline` is the first concrete raster pipeline abstraction. It lets an
example declare vertex/fragment shaders, vertex input, color/depth targets, and
push constants without manually assembling every Vulkan pipeline structure. See
`docs/GRAPHICS_PIPELINE_BUILD.md` for the Build flow and Mermaid diagrams.

`Image` is the matching image-resource abstraction. It owns `VkImage`,
`VkDeviceMemory`, and `VkImageView` together, so depth/color/shadow targets can
be created and destroyed as one RAII object.

1. Resource registry
   - Mesh, texture, material, sampler, uniform/storage buffers.
   - Stable handles, explicit destroy, debug labels.

2. Scene and renderables
   - Entity to transform, mesh, material, bounds.
   - CPU culling first, GPU culling later.

3. RenderGraph
   - Pass nodes declare inputs, outputs, layouts, and queue needs.
   - Built-in passes: depth prepass, shadow, opaque, transparent, post-process, present.

4. Material system
   - Filament-inspired defaults: base color, metallic, roughness, normal, emissive.
   - Shader variants selected by material features and render pass.

5. Integration with `vkSpatial`
   - BVH-backed ray queries for picking, shadows, and ray-traced debug views.
   - Shared GPU buffers should stay in device memory.

## Mouse Input

`MouseInput` is the platform-independent mouse event hub. The application feeds
GLFW, SDL, or native window callbacks into `OnMouseMove()`, `OnButton()`,
`OnScroll()`, `OnCursorEnter()`, and `OnCursorLeave()`. Render tools then attach
listeners to `Move`, `ButtonDown`, `ButtonUp`, `DragBegin`, `Drag`, `DragEnd`,
`Scroll`, `Enter`, `Leave`, or `Any`.

Use `MouseListenerGroup` when a camera controller, picking tool, gizmo, or UI
mode owns several listener registrations. The group unregisters all of its
listeners when it is destroyed, so interaction modes can be enabled and disabled
without leaking callbacks.

See `docs/VK_RENDER.md` for the full structure and mouse event flow.

## Current Files

- `Engine.*`: factory and top-level API entry.
- `SwapChain.*`: Vulkan swapchain creation, recreation, image views, present.
- `Renderer.*`: frame begin/end, synchronization, buffer-to-swapchain presentation.
- `GraphicsPipeline.*`: declarative graphics pipeline descriptor and `VkPipeline`
  lifetime wrapper.
- `Image.*`: `VkImage`, `VkDeviceMemory`, and `VkImageView` lifetime wrapper.
- `RenderAttachments.h`: small color/depth dynamic rendering attachment builders.
- `MouseInput.*`: platform-independent mouse event dispatch and listener groups.
- `View.h`: scene/camera/render graph binding.
- `Scene.h`: simple entity membership.
- `Camera.h`: Vulkan-style camera matrices.
- `RenderGraph.*`: ordered render-pass execution scaffold.
- `Resources.h`: future mesh/material/texture handle descriptors.

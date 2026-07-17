# Engine::Render Orchestration Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the deferred orchestration layer to `Engine::Render` — `Scene`/`Camera`/`View`, `RenderPass`/`RenderGraph`, `Renderer` (auto-resize frame loop owner), `MouseInput`/`KeyInput`, and a `Window`/`Application` layer that hides GLFW and the `while` loop — then rewrite `example2/cube_render.cpp` against it so its `main()` has no Vulkan and no GLFW visible.

**Architecture:** See `docs/superpowers/specs/2026-07-17-engine-render-orchestration-design.md` for the full design. Summary: `Scene`/`Camera`/`View` are pure data-binding types; `RenderGraph` executes an ordered list of `RenderPass`; `Renderer` owns frame sync + a default depth image and auto-handles resize; `Window` (abstract, `WindowBackend`-selected, `GlfwWindow` the only implementation) wraps the windowing backend; `Application` owns `Window`+`Context`+`SwapChain`+`Renderer`+`View` and its `Run()` absorbs the frame loop entirely.

**Tech Stack:** C++17, Vulkan 1.3 (dynamic rendering), GLFW (wrapped, not exposed), Eigen (via `utilities/Math.h`), GoogleTest.

## Global Constraints

- All new code lives under `src/Engine/Render/` (namespace `Engine::Render`) except `CubePass.h`/`.cpp`, which live under `example2/` since they are example-specific, not reusable engine code.
- Every class that owns GPU/window resources takes references (`Engine::Core::Context&`, `SwapChain&`) — never raw pointers or stored copies — matching the established `Engine::Core`/`Engine::Render` convention.
- `Camera` uses `vkMath::Mat4`/`vkMath::Vec3` (`utilities/Math.h`, Eigen-backed) — reuse `vkMath::Perspective`/`Orthographic`/`LookAt` rather than reimplementing matrix math.
- `KeyEvent.keyCode` is `Engine::Render::KeyCode` (a small enum defined in `KeyInput.h`), not a raw platform int.
- Only these existing files may be modified: `src/Engine/CMakeLists.txt`, `example2/CMakeLists.txt`, `example2/cube_render.cpp`, `test/CMakeLists.txt`. Everything else in this plan is a new file. `src/vkRender/*` must be untouched (it is the prior-art reference, not something this plan changes).
- `Renderer::BeginFrame`/`EndFrame` tolerate `VK_SUBOPTIMAL_KHR`/`VK_ERROR_OUT_OF_DATE_KHR` from `Present()` without recreating immediately — recreation happens exclusively in `BeginFrame`, which always has a fresh width/height from the caller. This is a deliberate simplification over the old `vkRender::Renderer`'s flag-based approach (see spec's Prior Art section, departure #2).
- Full test suite baseline before this plan: 50/51 passing via `ctest --test-dir build --output-on-failure`, the one pre-existing failure being `WideBVHTest.RadiusMatchesCpuReference` (unrelated, do not attempt to fix it).

---

### Task 1: Scene + Camera + View

**Files:**
- Create: `src/Engine/Render/Scene.h`
- Create: `src/Engine/Render/Camera.h`
- Create: `src/Engine/Render/View.h`
- Create: `test/test_engineRender.cpp`
- Modify: `src/Engine/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Produces: `Engine::Render::Entity` (= `uint32_t`), `Engine::Render::Scene`, `Engine::Render::Camera` (with `Camera::Projection` enum), `Engine::Render::View`. Later tasks (`RenderGraph`, `Renderer`, `CubePass`) consume `View::GetCamera()`/`GetRenderGraph()` and `Camera::GetViewMatrix()`/`GetProjectionMatrix()`.
- Consumes: `vkMath::Mat4`/`Vec3`/`Perspective`/`Orthographic`/`LookAt` from `src/utilities/Math.h` (already exists, unchanged). `Engine::Render::Viewport`/`ClearOptions` from `src/Engine/Render/RenderTypes.h` (already exists, unchanged).

- [ ] **Step 1: Create `src/Engine/Render/Scene.h`**

```cpp
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Render {

    using Entity = uint32_t;

    class Scene {
    public:
        using UniquePtr = std::unique_ptr<Scene>;

        Entity CreateEntity() {
            const Entity entity = m_nextEntity++;
            m_entities.push_back(entity);
            return entity;
        }

        void AddEntity(Entity entity) {
            if (!Contains(entity))
                m_entities.push_back(entity);
        }

        void Remove(Entity entity) {
            m_entities.erase(std::remove(m_entities.begin(), m_entities.end(), entity),
                             m_entities.end());
        }

        void Clear() { m_entities.clear(); }

        bool Contains(Entity entity) const {
            return std::find(m_entities.begin(), m_entities.end(), entity) != m_entities.end();
        }

        const std::vector<Entity> &Entities() const { return m_entities; }

    private:
        Entity m_nextEntity = 1;
        std::vector<Entity> m_entities;
    };

} // namespace Engine::Render
```

This is a byte-for-byte port of `src/vkRender/Scene.h` aside from the namespace.

- [ ] **Step 2: Create `src/Engine/Render/Camera.h`**

```cpp
#pragma once

#include "utilities/Math.h"

#include <memory>

namespace Engine::Render {

    class Camera {
    public:
        using UniquePtr = std::unique_ptr<Camera>;

        enum class Projection {
            Perspective,
            Orthographic,
        };

        void SetPerspective(float fovYRadians, float aspect, float nearPlane, float farPlane) {
            m_projectionType = Projection::Perspective;
            m_nearPlane = nearPlane;
            m_farPlane = farPlane;
            m_projection = vkMath::Perspective(fovYRadians, aspect, nearPlane, farPlane);
        }

        void SetOrthographic(float left, float right, float bottom, float top,
                             float nearPlane, float farPlane) {
            m_projectionType = Projection::Orthographic;
            m_nearPlane = nearPlane;
            m_farPlane = farPlane;
            m_projection = vkMath::Orthographic(left, right, bottom, top, nearPlane, farPlane);
        }

        void LookAt(vkMath::Vec3 eye, vkMath::Vec3 target, vkMath::Vec3 up = {0.0f, 1.0f, 0.0f}) {
            m_eye = eye;
            m_target = target;
            m_up = up;
            m_view = vkMath::LookAt(eye, target, up);
        }

        Projection GetProjectionType() const { return m_projectionType; }
        float GetNearPlane() const { return m_nearPlane; }
        float GetFarPlane() const { return m_farPlane; }
        vkMath::Vec3 GetEye() const { return m_eye; }
        vkMath::Vec3 GetTarget() const { return m_target; }
        const vkMath::Mat4 &GetViewMatrix() const { return m_view; }
        const vkMath::Mat4 &GetProjectionMatrix() const { return m_projection; }

    private:
        Projection m_projectionType = Projection::Perspective;
        float m_nearPlane = 0.01f;
        float m_farPlane = 1000.0f;
        vkMath::Vec3 m_eye = {0.0f, 0.0f, 1.0f};
        vkMath::Vec3 m_target = {0.0f, 0.0f, 0.0f};
        vkMath::Vec3 m_up = {0.0f, 1.0f, 0.0f};
        vkMath::Mat4 m_view = vkMath::Mat4::Identity();
        vkMath::Mat4 m_projection = vkMath::Mat4::Identity();
    };

} // namespace Engine::Render
```

Unlike `src/vkRender/Camera.h`, this does not hand-roll matrix assembly — it calls the existing `vkMath::Perspective`/`Orthographic`/`LookAt` free functions.

- [ ] **Step 3: Create `src/Engine/Render/View.h`**

```cpp
#pragma once

#include "Engine/Render/RenderTypes.h"

#include <memory>

namespace Engine::Render {

    class Camera;
    class RenderGraph;
    class Scene;

    class View {
    public:
        using UniquePtr = std::unique_ptr<View>;

        void SetScene(Scene *scene) { m_scene = scene; }
        void SetCamera(Camera *camera) { m_camera = camera; }
        void SetRenderGraph(RenderGraph *renderGraph) { m_renderGraph = renderGraph; }
        void SetViewport(Viewport viewport) { m_viewport = viewport; }
        void SetClearOptions(ClearOptions clearOptions) { m_clearOptions = clearOptions; }

        Scene *GetScene() const { return m_scene; }
        Camera *GetCamera() const { return m_camera; }
        RenderGraph *GetRenderGraph() const { return m_renderGraph; }
        Viewport GetViewport() const { return m_viewport; }
        ClearOptions GetClearOptions() const { return m_clearOptions; }

    private:
        Scene *m_scene = nullptr;
        Camera *m_camera = nullptr;
        RenderGraph *m_renderGraph = nullptr;
        Viewport m_viewport;
        ClearOptions m_clearOptions;
    };

} // namespace Engine::Render
```

Byte-for-byte port of `src/vkRender/View.h` aside from the namespace. `RenderGraph` is only forward-declared here (used as a pointer), so this compiles before `RenderGraph.h` exists (Task 2).

- [ ] **Step 4: Create `test/test_engineRender.cpp` with Scene/Camera/View tests**

```cpp
#include <gtest/gtest.h>

#include "Engine/Render/Camera.h"
#include "Engine/Render/Scene.h"
#include "Engine/Render/View.h"

using namespace Engine::Render;

TEST(SceneTest, CreateEntityAssignsDistinctIds) {
    Scene scene;
    Entity a = scene.CreateEntity();
    Entity b = scene.CreateEntity();
    EXPECT_NE(a, b);
    EXPECT_TRUE(scene.Contains(a));
    EXPECT_TRUE(scene.Contains(b));
    EXPECT_EQ(scene.Entities().size(), 2u);
}

TEST(SceneTest, RemoveEntityDropsIt) {
    Scene scene;
    Entity a = scene.CreateEntity();
    scene.Remove(a);
    EXPECT_FALSE(scene.Contains(a));
}

TEST(CameraTest, PerspectiveProjectionMatchesVkMath) {
    Camera camera;
    const float fov = 60.0f * 3.14159265f / 180.0f;
    camera.SetPerspective(fov, 16.0f / 9.0f, 0.1f, 100.0f);
    const vkMath::Mat4 expected = vkMath::Perspective(fov, 16.0f / 9.0f, 0.1f, 100.0f);
    EXPECT_TRUE(camera.GetProjectionMatrix().isApprox(expected));
    EXPECT_EQ(camera.GetProjectionType(), Camera::Projection::Perspective);
}

TEST(CameraTest, LookAtMatchesVkMath) {
    Camera camera;
    const vkMath::Vec3 eye(0.0f, 0.0f, 5.0f);
    const vkMath::Vec3 target(0.0f, 0.0f, 0.0f);
    camera.LookAt(eye, target);
    const vkMath::Mat4 expected = vkMath::LookAt(eye, target, {0.0f, 1.0f, 0.0f});
    EXPECT_TRUE(camera.GetViewMatrix().isApprox(expected));
    EXPECT_EQ(camera.GetEye(), eye);
}

TEST(ViewTest, BindsSceneCameraAndGraph) {
    Scene scene;
    Camera camera;
    View view;
    view.SetScene(&scene);
    view.SetCamera(&camera);
    EXPECT_EQ(view.GetScene(), &scene);
    EXPECT_EQ(view.GetCamera(), &camera);
    EXPECT_EQ(view.GetRenderGraph(), nullptr);
}
```

- [ ] **Step 5: Add Eigen include path to `EngineRender` and link it in the test target**

In `src/Engine/CMakeLists.txt`, find the `EngineRender` block's `target_include_directories` and add the Eigen path to `PUBLIC` (this is the only edit in this task — do not touch the `glfw` link yet, that is Task 5):

```cmake
target_include_directories(EngineRender
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")
```

In `test/CMakeLists.txt`, add `Engine::Render` to `target_link_libraries(vkspatial_tests PRIVATE ...)` (alongside the existing `vkSpatial::vkSpatial`, `Engine::Core`, `GTest::gtest`, `GTest::gtest_main`):

```cmake
target_link_libraries(vkspatial_tests
        PRIVATE
        vkSpatial::vkSpatial
        Engine::Core
        Engine::Render
        GTest::gtest
        GTest::gtest_main)
```

- [ ] **Step 6: Build and run the new tests**

```bash
cmake -S . -B build
cmake --build build --target vkspatial_tests --parallel
ctest --test-dir build --output-on-failure -R "SceneTest|CameraTest|ViewTest"
```

Expected: builds clean, all new tests pass (7 tests: 2 Scene + 2 Camera + 1 View... count them as they appear).

- [ ] **Step 7: Commit**

```bash
git add src/Engine/Render/Scene.h src/Engine/Render/Camera.h src/Engine/Render/View.h test/test_engineRender.cpp src/Engine/CMakeLists.txt test/CMakeLists.txt
git commit -m "Add Engine::Render Scene/Camera/View"
```

---

### Task 2: RenderPass + RenderGraph

**Files:**
- Create: `src/Engine/Render/RenderGraph.h`
- Create: `src/Engine/Render/RenderGraph.cpp`
- Modify: `test/test_engineRender.cpp` (append tests)

**Interfaces:**
- Consumes: `Engine::Core::Context` (forward-declared), `Engine::Core::Image` (forward-declared), `Engine::Render::SwapChain` (forward-declared), `Engine::Render::View` (from Task 1), `Engine::Render::FrameInfo` (from `RenderTypes.h`, already exists).
- Produces: `Engine::Render::RenderContext`, `Engine::Render::RenderPass` (abstract), `Engine::Render::RenderGraph`. Consumed by Task 3 (`Renderer`) and Task 7 (`CubePass`).

- [ ] **Step 1: Create `src/Engine/Render/RenderGraph.h`**

```cpp
#pragma once

#include "Engine/Render/RenderTypes.h"

#include <cstddef>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Core {
    class Context;
    class Image;
} // namespace Engine::Core

namespace Engine::Render {

    class SwapChain;
    class View;

    struct RenderContext {
        Engine::Core::Context *context = nullptr;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        SwapChain *swapChain = nullptr;
        Engine::Core::Image *depthImage = nullptr;
        View *view = nullptr;
        uint32_t imageIndex = 0;
        FrameInfo frame;
    };

    class RenderPass {
    public:
        virtual ~RenderPass() = default;

        virtual const char *Name() const = 0;
        virtual void Execute(RenderContext &context) = 0;
    };

    class RenderGraph {
    public:
        using UniquePtr = std::unique_ptr<RenderGraph>;

        RenderGraph &AddPass(std::unique_ptr<RenderPass> pass);
        void Clear();
        void Execute(RenderContext &context) const;
        bool Empty() const { return m_passes.empty(); }
        size_t PassCount() const { return m_passes.size(); }

    private:
        std::vector<std::unique_ptr<RenderPass>> m_passes;
    };

} // namespace Engine::Render
```

This matches `src/vkRender/RenderGraph.h` with one addition: `RenderContext::depthImage` (the old version had no depth concept).

- [ ] **Step 2: Create `src/Engine/Render/RenderGraph.cpp`**

```cpp
#include "Engine/Render/RenderGraph.h"

#include <stdexcept>

namespace Engine::Render {

    RenderGraph &RenderGraph::AddPass(std::unique_ptr<RenderPass> pass) {
        if (!pass)
            throw std::runtime_error("RenderGraph::AddPass received null pass");
        m_passes.push_back(std::move(pass));
        return *this;
    }

    void RenderGraph::Clear() {
        m_passes.clear();
    }

    void RenderGraph::Execute(RenderContext &context) const {
        for (const auto &pass: m_passes)
            pass->Execute(context);
    }

} // namespace Engine::Render
```

Byte-for-byte port of `src/vkRender/RenderGraph.cpp` aside from the namespace.

- [ ] **Step 3: Append RenderGraph tests to `test/test_engineRender.cpp`**

Add this include near the top (alongside the existing includes):

```cpp
#include "Engine/Render/RenderGraph.h"

#include <memory>
#include <vector>
```

Add these tests at the end of the file:

```cpp
namespace {
    class RecordingPass : public RenderPass {
    public:
        RecordingPass(std::vector<int> &order, int id) : m_order(order), m_id(id) {}
        const char *Name() const override { return "RecordingPass"; }
        void Execute(RenderContext &) override { m_order.push_back(m_id); }

    private:
        std::vector<int> &m_order;
        int m_id;
    };
} // namespace

TEST(RenderGraphTest, ExecutesPassesInAddedOrder) {
    std::vector<int> order;
    RenderGraph graph;
    graph.AddPass(std::make_unique<RecordingPass>(order, 1));
    graph.AddPass(std::make_unique<RecordingPass>(order, 2));
    EXPECT_FALSE(graph.Empty());
    EXPECT_EQ(graph.PassCount(), 2u);

    RenderContext ctx{};
    graph.Execute(ctx);
    EXPECT_EQ(order, (std::vector<int>{1, 2}));
}

TEST(RenderGraphTest, AddNullPassThrows) {
    RenderGraph graph;
    EXPECT_THROW(graph.AddPass(nullptr), std::runtime_error);
}
```

- [ ] **Step 4: Build and run**

```bash
cmake --build build --target vkspatial_tests --parallel
ctest --test-dir build --output-on-failure -R "RenderGraphTest"
```

Expected: 2/2 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Render/RenderGraph.h src/Engine/Render/RenderGraph.cpp test/test_engineRender.cpp
git commit -m "Add Engine::Render RenderPass/RenderGraph"
```

---

### Task 3: Renderer

**Files:**
- Create: `src/Engine/Render/Renderer.h`
- Create: `src/Engine/Render/Renderer.cpp`

**Interfaces:**
- Consumes: `Engine::Core::Context` (`.device`, `.graphicsQueue`, `.graphicsFamily`), `Engine::Core::Image`/`ImageDescriptor::Depth2D` (from `Engine::Core`, already exists), `Engine::Render::SwapChain` (`.Extent()`, `.Recreate()`, `.AcquireNextImage()`, `.Present()`, `.Image()`), `Engine::Core::Image::TransitionLayout` (static overload, already exists), `Engine::Render::View`/`RenderGraph`/`RenderContext` (Tasks 1-2).
- Produces: `Engine::Render::Renderer` with `BeginFrame(width, height)`/`Render(view)`/`EndFrame()`. Consumed by Task 6 (`Application`).

This task cannot be unit-tested without a real window/surface (constructing a `SwapChain` requires `Context(enablePresent=true)` with a real `VkSurfaceKHR`, which needs an actual window). Verification for this task is "does it build," with functional verification deferred to Task 7's visual run — consistent with the design spec's Testing Strategy section.

- [ ] **Step 1: Create `src/Engine/Render/Renderer.h`**

```cpp
#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Core/Image.h"
#include "Engine/Render/RenderGraph.h"
#include "Engine/Render/SwapChain.h"
#include "Engine/Render/View.h"

#include <vulkan/vulkan.h>

namespace Engine::Render {

    class Renderer {
    public:
        explicit Renderer(Engine::Core::Context &context,
                          SwapChain &swapChain,
                          VkFormat depthFormat = VK_FORMAT_D32_SFLOAT);
        ~Renderer();

        Renderer(const Renderer &) = delete;
        Renderer &operator=(const Renderer &) = delete;

        // Takes the app's current framebuffer size every frame. If it differs from the
        // swapchain's extent, recreates the swapchain and depth image and returns false
        // (skip this frame) rather than proceeding. If AcquireNextImage reports
        // OUT_OF_DATE, does the same. The caller never calls SwapChain::Recreate() itself.
        bool BeginFrame(uint32_t width, uint32_t height);

        // No-op if view.GetRenderGraph() is null or empty. Otherwise transitions the
        // swapchain color image and the owned depth image to ATTACHMENT_OPTIMAL, builds a
        // RenderContext, and calls graph->Execute(context).
        void Render(View &view);

        // Transitions the color image back to PRESENT_SRC, ends and submits the command
        // buffer, and presents. Tolerates OUT_OF_DATE/SUBOPTIMAL from Present without
        // recreating (the next BeginFrame call handles that, since it always has a fresh
        // width/height).
        void EndFrame();

    private:
        Engine::Core::Context *m_context = nullptr;
        SwapChain *m_swapChain = nullptr;
        VkFormat m_depthFormat;
        Engine::Core::Image m_depthImage;

        VkCommandPool m_commandPool = VK_NULL_HANDLE;
        VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
        VkSemaphore m_imageAvailable = VK_NULL_HANDLE;
        VkSemaphore m_renderFinished = VK_NULL_HANDLE;
        VkFence m_inFlightFence = VK_NULL_HANDLE;

        uint32_t m_imageIndex = 0;
        bool m_frameActive = false;
        FrameInfo m_frameInfo;

        void RecreateDepthImage();
        void TransitionForRendering();
        void cleanup();
    };

} // namespace Engine::Render
```

- [ ] **Step 2: Create `src/Engine/Render/Renderer.cpp`**

```cpp
#include "Engine/Render/Renderer.h"

#include <stdexcept>

namespace Engine::Render {

    Renderer::Renderer(Engine::Core::Context &context, SwapChain &swapChain, VkFormat depthFormat)
        : m_context(&context),
          m_swapChain(&swapChain),
          m_depthFormat(depthFormat),
          m_depthImage(context) {
        if (context.graphicsQueue == VK_NULL_HANDLE)
            throw std::runtime_error("Renderer requires Context constructed with enablePresent=true");

        try {
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex = context.graphicsFamily;
            if (vkCreateCommandPool(context.device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS)
                throw std::runtime_error("Renderer failed to create command pool");

            VkCommandBufferAllocateInfo cmdAllocInfo{};
            cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdAllocInfo.commandPool = m_commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(context.device, &cmdAllocInfo, &m_commandBuffer) != VK_SUCCESS)
                throw std::runtime_error("Renderer failed to allocate command buffer");

            VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            if (vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &m_imageAvailable) != VK_SUCCESS ||
                vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &m_renderFinished) != VK_SUCCESS)
                throw std::runtime_error("Renderer failed to create semaphores");

            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            if (vkCreateFence(context.device, &fenceInfo, nullptr, &m_inFlightFence) != VK_SUCCESS)
                throw std::runtime_error("Renderer failed to create fence");

            m_depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(swapChain.Extent(), depthFormat));
        } catch (...) {
            cleanup();
            throw;
        }
    }

    Renderer::~Renderer() {
        cleanup();
    }

    void Renderer::cleanup() {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            return;

        vkDeviceWaitIdle(m_context->device);

        if (m_inFlightFence != VK_NULL_HANDLE)
            vkDestroyFence(m_context->device, m_inFlightFence, nullptr);
        if (m_renderFinished != VK_NULL_HANDLE)
            vkDestroySemaphore(m_context->device, m_renderFinished, nullptr);
        if (m_imageAvailable != VK_NULL_HANDLE)
            vkDestroySemaphore(m_context->device, m_imageAvailable, nullptr);
        if (m_commandPool != VK_NULL_HANDLE)
            vkDestroyCommandPool(m_context->device, m_commandPool, nullptr);

        m_inFlightFence = VK_NULL_HANDLE;
        m_renderFinished = VK_NULL_HANDLE;
        m_imageAvailable = VK_NULL_HANDLE;
        m_commandPool = VK_NULL_HANDLE;
        m_commandBuffer = VK_NULL_HANDLE;
    }

    void Renderer::RecreateDepthImage() {
        m_depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(m_swapChain->Extent(), m_depthFormat));
    }

    bool Renderer::BeginFrame(uint32_t width, uint32_t height) {
        if (m_frameActive)
            throw std::runtime_error("Renderer::BeginFrame called while frame is active");

        const VkExtent2D currentExtent = m_swapChain->Extent();
        if (width != currentExtent.width || height != currentExtent.height) {
            vkDeviceWaitIdle(m_context->device);
            m_swapChain->Recreate(width, height);
            RecreateDepthImage();
            return false;
        }

        vkWaitForFences(m_context->device, 1, &m_inFlightFence, VK_TRUE, UINT64_MAX);

        const VkResult acquireResult =
                m_swapChain->AcquireNextImage(m_imageAvailable, VK_NULL_HANDLE, &m_imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            vkDeviceWaitIdle(m_context->device);
            m_swapChain->Recreate(width, height);
            RecreateDepthImage();
            return false;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Renderer failed to acquire swapchain image");

        vkResetFences(m_context->device, 1, &m_inFlightFence);
        vkResetCommandBuffer(m_commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_commandBuffer, &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to begin command buffer");

        m_frameActive = true;
        return true;
    }

    void Renderer::TransitionForRendering() {
        VkImage swapImage = m_swapChain->Image(m_imageIndex);
        Engine::Core::Image::TransitionLayout(
                m_commandBuffer, swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        m_depthImage.TransitionLayout(
                m_commandBuffer, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    }

    void Renderer::Render(View &view) {
        if (!m_frameActive)
            throw std::runtime_error("Renderer::Render called outside BeginFrame/EndFrame");

        RenderGraph *graph = view.GetRenderGraph();
        if (!graph || graph->Empty())
            return;

        TransitionForRendering();

        RenderContext renderContext{};
        renderContext.context = m_context;
        renderContext.commandBuffer = m_commandBuffer;
        renderContext.swapChain = m_swapChain;
        renderContext.depthImage = &m_depthImage;
        renderContext.view = &view;
        renderContext.imageIndex = m_imageIndex;
        renderContext.frame = m_frameInfo;
        graph->Execute(renderContext);
    }

    void Renderer::EndFrame() {
        if (!m_frameActive)
            throw std::runtime_error("Renderer::EndFrame called without BeginFrame");

        VkImage swapImage = m_swapChain->Image(m_imageIndex);
        Engine::Core::Image::TransitionLayout(
                m_commandBuffer, swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0);

        if (vkEndCommandBuffer(m_commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to end command buffer");

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &m_imageAvailable;
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &m_renderFinished;

        if (vkQueueSubmit(m_context->graphicsQueue, 1, &submitInfo, m_inFlightFence) != VK_SUCCESS)
            throw std::runtime_error("Renderer failed to submit frame");

        const VkResult presentResult = m_swapChain->Present(m_imageIndex, m_renderFinished);
        if (presentResult != VK_SUCCESS &&
            presentResult != VK_SUBOPTIMAL_KHR &&
            presentResult != VK_ERROR_OUT_OF_DATE_KHR) {
            throw std::runtime_error("Renderer failed to present frame");
        }

        m_frameActive = false;
        ++m_frameInfo.frameIndex;
    }

} // namespace Engine::Render
```

Note the constructor's `try { ... } catch (...) { cleanup(); throw; }` wrapping — this matches the established leak-safety idiom from `Engine::Core::Context`'s constructor (if any Vulkan object creation fails partway through, already-created objects are torn down before rethrowing).

- [ ] **Step 3: Build**

```bash
cmake -S . -B build
cmake --build build --target EngineRender --parallel
```

Expected: builds with no errors. This only proves compilation — `Renderer` has no window to actually exercise yet (that's Task 7).

- [ ] **Step 4: Commit**

```bash
git add src/Engine/Render/Renderer.h src/Engine/Render/Renderer.cpp
git commit -m "Add Engine::Render Renderer (auto-resize frame loop owner)"
```

---

### Task 4: MouseInput + KeyInput

**Files:**
- Create: `src/Engine/Render/MouseInput.h`
- Create: `src/Engine/Render/MouseInput.cpp`
- Create: `src/Engine/Render/KeyInput.h`
- Create: `src/Engine/Render/KeyInput.cpp`
- Modify: `test/test_engineRender.cpp` (append tests)

**Interfaces:**
- Produces: `Engine::Render::MouseButton`/`MouseEventType`/`MouseModifierBits`/`MouseEvent`/`MouseInput`/`MouseListenerGroup`, `Engine::Render::KeyEventType`/`KeyCode`/`KeyEvent`/`KeyInput`/`KeyListenerGroup`. No dependency on Vulkan or GLFW. Consumed by Task 5 (`Window`/`GlfwWindow`) and Task 7 (`example2` trackball/escape-key bindings).

- [ ] **Step 1: Create `src/Engine/Render/MouseInput.h`**

```cpp
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace Engine::Render {

    enum class MouseButton : uint32_t {
        None = 0,
        Left = 1,
        Right = 2,
        Middle = 3,
        Button4 = 4,
        Button5 = 5,
        Other = 255,
    };

    enum class MouseEventType : uint32_t {
        Any = 0,
        Move,
        ButtonDown,
        ButtonUp,
        DragBegin,
        Drag,
        DragEnd,
        Scroll,
        Enter,
        Leave,
    };

    enum MouseModifierBits : uint32_t {
        MouseModifierShift = 1u << 0u,
        MouseModifierControl = 1u << 1u,
        MouseModifierAlt = 1u << 2u,
        MouseModifierSuper = 1u << 3u,
    };

    using MouseModifierFlags = uint32_t;
    using MouseButtonFlags = uint32_t;

    struct MouseEvent {
        MouseEventType type = MouseEventType::Move;
        MouseButton button = MouseButton::None;
        double x = 0.0;
        double y = 0.0;
        double previousX = 0.0;
        double previousY = 0.0;
        double deltaX = 0.0;
        double deltaY = 0.0;
        double scrollX = 0.0;
        double scrollY = 0.0;
        double dragStartX = 0.0;
        double dragStartY = 0.0;
        double timestampSeconds = 0.0;
        MouseModifierFlags modifiers = 0;
        MouseButtonFlags pressedButtons = 0;
        bool handled = false;

        bool IsDown(MouseButton queryButton) const;
        bool HasModifier(MouseModifierBits modifier) const;
    };

    class MouseInput {
    public:
        using ListenerId = uint64_t;
        using Callback = std::function<void(MouseEvent &)>;

        ListenerId AddListener(MouseEventType type,
                               Callback callback,
                               int priority = 0);
        bool RemoveListener(ListenerId id);
        void ClearListeners();

        void OnMouseMove(double x,
                         double y,
                         MouseModifierFlags modifiers = 0,
                         double timestampSeconds = 0.0);
        void OnButton(MouseButton button,
                      bool pressed,
                      double x,
                      double y,
                      MouseModifierFlags modifiers = 0,
                      double timestampSeconds = 0.0);
        void OnScroll(double x,
                      double y,
                      double scrollX,
                      double scrollY,
                      MouseModifierFlags modifiers = 0,
                      double timestampSeconds = 0.0);
        void OnCursorEnter(double x,
                           double y,
                           MouseModifierFlags modifiers = 0,
                           double timestampSeconds = 0.0);
        void OnCursorLeave(double x,
                           double y,
                           MouseModifierFlags modifiers = 0,
                           double timestampSeconds = 0.0);

        bool IsButtonDown(MouseButton button) const;
        MouseButtonFlags PressedButtons() const { return m_pressedButtons; }
        double X() const { return m_x; }
        double Y() const { return m_y; }
        bool HasPosition() const { return m_hasPosition; }

        static MouseButtonFlags ButtonMask(MouseButton button);

    private:
        struct Listener {
            ListenerId id = 0;
            MouseEventType type = MouseEventType::Any;
            int priority = 0;
            Callback callback;
        };

        struct ButtonState {
            bool down = false;
            bool dragging = false;
            double dragStartX = 0.0;
            double dragStartY = 0.0;
        };

        std::unordered_map<ListenerId, Listener> m_listeners;
        std::array<ButtonState, 5> m_buttons{};
        ListenerId m_nextListenerId = 1;
        double m_x = 0.0;
        double m_y = 0.0;
        bool m_hasPosition = false;
        MouseButtonFlags m_pressedButtons = 0;

        void Dispatch(MouseEvent event);
        MouseEvent MakeEvent(MouseEventType type,
                             MouseButton button,
                             double x,
                             double y,
                             double previousX,
                             double previousY,
                             double scrollX,
                             double scrollY,
                             MouseModifierFlags modifiers,
                             double timestampSeconds) const;
        ButtonState *StateFor(MouseButton button);
        const ButtonState *StateFor(MouseButton button) const;
        static size_t ButtonIndex(MouseButton button);
    };

    class MouseListenerGroup {
    public:
        MouseListenerGroup() = default;
        explicit MouseListenerGroup(MouseInput &input);
        ~MouseListenerGroup();

        MouseListenerGroup(const MouseListenerGroup &) = delete;
        MouseListenerGroup &operator=(const MouseListenerGroup &) = delete;

        MouseListenerGroup(MouseListenerGroup &&other) noexcept;
        MouseListenerGroup &operator=(MouseListenerGroup &&other) noexcept;

        MouseInput::ListenerId Add(MouseEventType type,
                                   MouseInput::Callback callback,
                                   int priority = 0);
        void Clear();
        bool Empty() const { return m_listenerIds.empty(); }

    private:
        MouseInput *m_input = nullptr;
        std::vector<MouseInput::ListenerId> m_listenerIds;
    };

} // namespace Engine::Render
```

- [ ] **Step 2: Create `src/Engine/Render/MouseInput.cpp`**

```cpp
#include "Engine/Render/MouseInput.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace Engine::Render {

    bool MouseEvent::IsDown(MouseButton queryButton) const {
        return (pressedButtons & MouseInput::ButtonMask(queryButton)) != 0u;
    }

    bool MouseEvent::HasModifier(MouseModifierBits modifier) const {
        return (modifiers & static_cast<MouseModifierFlags>(modifier)) != 0u;
    }

    MouseInput::ListenerId MouseInput::AddListener(MouseEventType type,
                                                   Callback callback,
                                                   int priority) {
        if (!callback)
            throw std::runtime_error("MouseInput::AddListener received empty callback");

        const ListenerId id = m_nextListenerId++;
        m_listeners.emplace(id, Listener{id, type, priority, std::move(callback)});
        return id;
    }

    bool MouseInput::RemoveListener(ListenerId id) {
        return m_listeners.erase(id) != 0;
    }

    void MouseInput::ClearListeners() {
        m_listeners.clear();
    }

    void MouseInput::OnMouseMove(double x,
                                 double y,
                                 MouseModifierFlags modifiers,
                                 double timestampSeconds) {
        const double previousX = m_hasPosition ? m_x : x;
        const double previousY = m_hasPosition ? m_y : y;
        m_x = x;
        m_y = y;
        m_hasPosition = true;

        Dispatch(MakeEvent(MouseEventType::Move,
                           MouseButton::None,
                           x, y,
                           previousX, previousY,
                           0.0, 0.0,
                           modifiers,
                           timestampSeconds));

        const double deltaX = x - previousX;
        const double deltaY = y - previousY;
        if (deltaX == 0.0 && deltaY == 0.0)
            return;

        for (uint32_t buttonValue = static_cast<uint32_t>(MouseButton::Left);
             buttonValue <= static_cast<uint32_t>(MouseButton::Button5);
             ++buttonValue) {
            MouseButton button = static_cast<MouseButton>(buttonValue);
            ButtonState *state = StateFor(button);
            if (!state || !state->down)
                continue;

            if (!state->dragging) {
                state->dragging = true;
                Dispatch(MakeEvent(MouseEventType::DragBegin,
                                   button,
                                   x, y,
                                   previousX, previousY,
                                   0.0, 0.0,
                                   modifiers,
                                   timestampSeconds));
            }

            Dispatch(MakeEvent(MouseEventType::Drag,
                               button,
                               x, y,
                               previousX, previousY,
                               0.0, 0.0,
                               modifiers,
                               timestampSeconds));
        }
    }

    void MouseInput::OnButton(MouseButton button,
                              bool pressed,
                              double x,
                              double y,
                              MouseModifierFlags modifiers,
                              double timestampSeconds) {
        const double previousX = m_hasPosition ? m_x : x;
        const double previousY = m_hasPosition ? m_y : y;
        m_x = x;
        m_y = y;
        m_hasPosition = true;

        ButtonState *state = StateFor(button);
        if (state) {
            if (pressed) {
                state->down = true;
                state->dragging = false;
                state->dragStartX = x;
                state->dragStartY = y;
                m_pressedButtons |= ButtonMask(button);
            } else {
                if (state->dragging) {
                    Dispatch(MakeEvent(MouseEventType::DragEnd,
                                       button,
                                       x, y,
                                       previousX, previousY,
                                       0.0, 0.0,
                                       modifiers,
                                       timestampSeconds));
                }
                state->down = false;
                state->dragging = false;
                m_pressedButtons &= ~ButtonMask(button);
            }
        }

        Dispatch(MakeEvent(pressed ? MouseEventType::ButtonDown : MouseEventType::ButtonUp,
                           button,
                           x, y,
                           previousX, previousY,
                           0.0, 0.0,
                           modifiers,
                           timestampSeconds));
    }

    void MouseInput::OnScroll(double x,
                              double y,
                              double scrollX,
                              double scrollY,
                              MouseModifierFlags modifiers,
                              double timestampSeconds) {
        const double previousX = m_hasPosition ? m_x : x;
        const double previousY = m_hasPosition ? m_y : y;
        m_x = x;
        m_y = y;
        m_hasPosition = true;

        Dispatch(MakeEvent(MouseEventType::Scroll,
                           MouseButton::None,
                           x, y,
                           previousX, previousY,
                           scrollX, scrollY,
                           modifiers,
                           timestampSeconds));
    }

    void MouseInput::OnCursorEnter(double x,
                                   double y,
                                   MouseModifierFlags modifiers,
                                   double timestampSeconds) {
        const double previousX = m_hasPosition ? m_x : x;
        const double previousY = m_hasPosition ? m_y : y;
        m_x = x;
        m_y = y;
        m_hasPosition = true;

        Dispatch(MakeEvent(MouseEventType::Enter,
                           MouseButton::None,
                           x, y,
                           previousX, previousY,
                           0.0, 0.0,
                           modifiers,
                           timestampSeconds));
    }

    void MouseInput::OnCursorLeave(double x,
                                   double y,
                                   MouseModifierFlags modifiers,
                                   double timestampSeconds) {
        const double previousX = m_hasPosition ? m_x : x;
        const double previousY = m_hasPosition ? m_y : y;
        m_x = x;
        m_y = y;

        Dispatch(MakeEvent(MouseEventType::Leave,
                           MouseButton::None,
                           x, y,
                           previousX, previousY,
                           0.0, 0.0,
                           modifiers,
                           timestampSeconds));
    }

    bool MouseInput::IsButtonDown(MouseButton button) const {
        return (m_pressedButtons & ButtonMask(button)) != 0u;
    }

    MouseButtonFlags MouseInput::ButtonMask(MouseButton button) {
        switch (button) {
            case MouseButton::Left:
                return 1u << 0u;
            case MouseButton::Right:
                return 1u << 1u;
            case MouseButton::Middle:
                return 1u << 2u;
            case MouseButton::Button4:
                return 1u << 3u;
            case MouseButton::Button5:
                return 1u << 4u;
            default:
                return 0u;
        }
    }

    void MouseInput::Dispatch(MouseEvent event) {
        std::vector<Listener> listeners;
        listeners.reserve(m_listeners.size());
        for (const auto &entry: m_listeners) {
            const Listener &listener = entry.second;
            if (listener.type == MouseEventType::Any || listener.type == event.type)
                listeners.push_back(listener);
        }

        std::sort(listeners.begin(), listeners.end(),
                  [](const Listener &a, const Listener &b) {
                      if (a.priority != b.priority)
                          return a.priority > b.priority;
                      return a.id < b.id;
                  });

        for (const Listener &listener: listeners) {
            listener.callback(event);
            if (event.handled)
                break;
        }
    }

    MouseEvent MouseInput::MakeEvent(MouseEventType type,
                                     MouseButton button,
                                     double x,
                                     double y,
                                     double previousX,
                                     double previousY,
                                     double scrollX,
                                     double scrollY,
                                     MouseModifierFlags modifiers,
                                     double timestampSeconds) const {
        MouseEvent event{};
        event.type = type;
        event.button = button;
        event.x = x;
        event.y = y;
        event.previousX = previousX;
        event.previousY = previousY;
        event.deltaX = x - previousX;
        event.deltaY = y - previousY;
        event.scrollX = scrollX;
        event.scrollY = scrollY;
        event.modifiers = modifiers;
        event.timestampSeconds = timestampSeconds;
        event.pressedButtons = m_pressedButtons;

        const ButtonState *state = StateFor(button);
        if (state) {
            event.dragStartX = state->dragStartX;
            event.dragStartY = state->dragStartY;
        } else {
            event.dragStartX = x;
            event.dragStartY = y;
        }
        return event;
    }

    MouseInput::ButtonState *MouseInput::StateFor(MouseButton button) {
        const size_t index = ButtonIndex(button);
        if (index >= m_buttons.size())
            return nullptr;
        return &m_buttons[index];
    }

    const MouseInput::ButtonState *MouseInput::StateFor(MouseButton button) const {
        const size_t index = ButtonIndex(button);
        if (index >= m_buttons.size())
            return nullptr;
        return &m_buttons[index];
    }

    size_t MouseInput::ButtonIndex(MouseButton button) {
        switch (button) {
            case MouseButton::Left:
                return 0;
            case MouseButton::Right:
                return 1;
            case MouseButton::Middle:
                return 2;
            case MouseButton::Button4:
                return 3;
            case MouseButton::Button5:
                return 4;
            default:
                return static_cast<size_t>(-1);
        }
    }

    MouseListenerGroup::MouseListenerGroup(MouseInput &input)
        : m_input(&input) {
    }

    MouseListenerGroup::~MouseListenerGroup() {
        Clear();
    }

    MouseListenerGroup::MouseListenerGroup(MouseListenerGroup &&other) noexcept
        : m_input(other.m_input),
          m_listenerIds(std::move(other.m_listenerIds)) {
        other.m_input = nullptr;
    }

    MouseListenerGroup &MouseListenerGroup::operator=(MouseListenerGroup &&other) noexcept {
        if (this == &other)
            return *this;

        Clear();
        m_input = other.m_input;
        m_listenerIds = std::move(other.m_listenerIds);
        other.m_input = nullptr;
        return *this;
    }

    MouseInput::ListenerId MouseListenerGroup::Add(MouseEventType type,
                                                   MouseInput::Callback callback,
                                                   int priority) {
        if (!m_input)
            throw std::runtime_error("MouseListenerGroup requires a MouseInput");

        MouseInput::ListenerId id = m_input->AddListener(type, std::move(callback), priority);
        m_listenerIds.push_back(id);
        return id;
    }

    void MouseListenerGroup::Clear() {
        if (m_input) {
            for (MouseInput::ListenerId id: m_listenerIds)
                m_input->RemoveListener(id);
        }
        m_listenerIds.clear();
    }

} // namespace Engine::Render
```

Byte-for-byte port of `src/vkRender/MouseInput.h`/`.cpp` aside from the namespace.

- [ ] **Step 3: Create `src/Engine/Render/KeyInput.h`**

```cpp
#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace Engine::Render {

    enum class KeyEventType : uint32_t {
        Any = 0,
        Press,
        Release,
        Repeat,
    };

    // Common keys only; anything a window backend can't map to one of these becomes
    // Unknown (see GlfwWindow in Task 5).
    enum class KeyCode : uint32_t {
        Unknown = 0,
        Escape, Space, Enter, Tab, Backspace,
        A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        Left, Right, Up, Down,
        LeftShift, LeftControl, LeftAlt, LeftSuper,
    };

    struct KeyEvent {
        KeyEventType type = KeyEventType::Press;
        KeyCode keyCode = KeyCode::Unknown;
        uint32_t modifiers = 0;
        double timestampSeconds = 0.0;
        bool handled = false;
    };

    class KeyInput {
    public:
        using ListenerId = uint64_t;
        using Callback = std::function<void(KeyEvent &)>;

        ListenerId AddListener(KeyEventType type,
                               Callback callback,
                               int priority = 0);
        bool RemoveListener(ListenerId id);
        void ClearListeners();

        void OnKey(KeyCode keyCode,
                   KeyEventType type,
                   uint32_t modifiers = 0,
                   double timestampSeconds = 0.0);

        bool IsKeyDown(KeyCode keyCode) const;

    private:
        struct Listener {
            ListenerId id = 0;
            KeyEventType type = KeyEventType::Any;
            int priority = 0;
            Callback callback;
        };

        std::unordered_map<ListenerId, Listener> m_listeners;
        std::unordered_map<KeyCode, bool> m_keyStates;
        ListenerId m_nextListenerId = 1;

        void Dispatch(KeyEvent event);
    };

    class KeyListenerGroup {
    public:
        KeyListenerGroup() = default;
        explicit KeyListenerGroup(KeyInput &input);
        ~KeyListenerGroup();

        KeyListenerGroup(const KeyListenerGroup &) = delete;
        KeyListenerGroup &operator=(const KeyListenerGroup &) = delete;

        KeyListenerGroup(KeyListenerGroup &&other) noexcept;
        KeyListenerGroup &operator=(KeyListenerGroup &&other) noexcept;

        KeyInput::ListenerId Add(KeyEventType type,
                                 KeyInput::Callback callback,
                                 int priority = 0);
        void Clear();
        bool Empty() const { return m_listenerIds.empty(); }

    private:
        KeyInput *m_input = nullptr;
        std::vector<KeyInput::ListenerId> m_listenerIds;
    };

} // namespace Engine::Render
```

- [ ] **Step 4: Create `src/Engine/Render/KeyInput.cpp`**

```cpp
#include "Engine/Render/KeyInput.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace Engine::Render {

    KeyInput::ListenerId KeyInput::AddListener(KeyEventType type,
                                               Callback callback,
                                               int priority) {
        if (!callback)
            throw std::runtime_error("KeyInput::AddListener received empty callback");

        const ListenerId id = m_nextListenerId++;
        m_listeners.emplace(id, Listener{id, type, priority, std::move(callback)});
        return id;
    }

    bool KeyInput::RemoveListener(ListenerId id) {
        return m_listeners.erase(id) != 0;
    }

    void KeyInput::ClearListeners() {
        m_listeners.clear();
    }

    void KeyInput::OnKey(KeyCode keyCode,
                         KeyEventType type,
                         uint32_t modifiers,
                         double timestampSeconds) {
        if (type == KeyEventType::Press)
            m_keyStates[keyCode] = true;
        else if (type == KeyEventType::Release)
            m_keyStates[keyCode] = false;

        KeyEvent event{};
        event.type = type;
        event.keyCode = keyCode;
        event.modifiers = modifiers;
        event.timestampSeconds = timestampSeconds;

        Dispatch(event);
    }

    bool KeyInput::IsKeyDown(KeyCode keyCode) const {
        const auto it = m_keyStates.find(keyCode);
        return it != m_keyStates.end() && it->second;
    }

    void KeyInput::Dispatch(KeyEvent event) {
        std::vector<Listener> listeners;
        listeners.reserve(m_listeners.size());
        for (const auto &entry: m_listeners) {
            const Listener &listener = entry.second;
            if (listener.type == KeyEventType::Any || listener.type == event.type)
                listeners.push_back(listener);
        }

        std::sort(listeners.begin(), listeners.end(),
                  [](const Listener &a, const Listener &b) {
                      if (a.priority != b.priority)
                          return a.priority > b.priority;
                      return a.id < b.id;
                  });

        for (const Listener &listener: listeners) {
            listener.callback(event);
            if (event.handled)
                break;
        }
    }

    KeyListenerGroup::KeyListenerGroup(KeyInput &input)
        : m_input(&input) {
    }

    KeyListenerGroup::~KeyListenerGroup() {
        Clear();
    }

    KeyListenerGroup::KeyListenerGroup(KeyListenerGroup &&other) noexcept
        : m_input(other.m_input),
          m_listenerIds(std::move(other.m_listenerIds)) {
        other.m_input = nullptr;
    }

    KeyListenerGroup &KeyListenerGroup::operator=(KeyListenerGroup &&other) noexcept {
        if (this == &other)
            return *this;

        Clear();
        m_input = other.m_input;
        m_listenerIds = std::move(other.m_listenerIds);
        other.m_input = nullptr;
        return *this;
    }

    KeyInput::ListenerId KeyListenerGroup::Add(KeyEventType type,
                                               KeyInput::Callback callback,
                                               int priority) {
        if (!m_input)
            throw std::runtime_error("KeyListenerGroup requires a KeyInput");

        KeyInput::ListenerId id = m_input->AddListener(type, std::move(callback), priority);
        m_listenerIds.push_back(id);
        return id;
    }

    void KeyListenerGroup::Clear() {
        if (m_input) {
            for (KeyInput::ListenerId id: m_listenerIds)
                m_input->RemoveListener(id);
        }
        m_listenerIds.clear();
    }

} // namespace Engine::Render
```

Same as `src/vkRender/KeyInput.h`/`.cpp` except `OnKey`/`IsKeyDown`/`KeyEvent.keyCode`/`m_keyStates` use `KeyCode` instead of raw `int`.

- [ ] **Step 5: Append MouseInput/KeyInput tests to `test/test_engineRender.cpp`**

Add these includes near the top:

```cpp
#include "Engine/Render/KeyInput.h"
#include "Engine/Render/MouseInput.h"
```

Add these tests at the end of the file:

```cpp
TEST(MouseInputTest, ListenerReceivesMoveEvent) {
    MouseInput input;
    bool received = false;
    input.AddListener(MouseEventType::Move, [&](MouseEvent &e) {
        received = true;
        EXPECT_DOUBLE_EQ(e.x, 10.0);
    });
    input.OnMouseMove(10.0, 20.0);
    EXPECT_TRUE(received);
}

TEST(MouseInputTest, HigherPriorityListenerRunsFirstAndCanStopPropagation) {
    MouseInput input;
    std::vector<int> order;
    input.AddListener(MouseEventType::Move, [&](MouseEvent &e) {
        order.push_back(1);
        e.handled = true;
    }, 10);
    input.AddListener(MouseEventType::Move, [&](MouseEvent &) {
        order.push_back(2);
    }, 0);
    input.OnMouseMove(0.0, 0.0);
    EXPECT_EQ(order, (std::vector<int>{1}));
}

TEST(MouseListenerGroupTest, ClearsListenersOnDestruction) {
    MouseInput input;
    int callCount = 0;
    {
        MouseListenerGroup group(input);
        group.Add(MouseEventType::Move, [&](MouseEvent &) { ++callCount; });
        input.OnMouseMove(0.0, 0.0);
    }
    input.OnMouseMove(1.0, 1.0);
    EXPECT_EQ(callCount, 1);
}

TEST(KeyInputTest, ListenerReceivesPressAndTracksState) {
    KeyInput input;
    bool received = false;
    input.AddListener(KeyEventType::Press, [&](KeyEvent &e) {
        received = true;
        EXPECT_EQ(e.keyCode, KeyCode::Escape);
    });
    input.OnKey(KeyCode::Escape, KeyEventType::Press);
    EXPECT_TRUE(received);
    EXPECT_TRUE(input.IsKeyDown(KeyCode::Escape));

    input.OnKey(KeyCode::Escape, KeyEventType::Release);
    EXPECT_FALSE(input.IsKeyDown(KeyCode::Escape));
}

TEST(KeyListenerGroupTest, ClearsListenersOnDestruction) {
    KeyInput input;
    int callCount = 0;
    {
        KeyListenerGroup group(input);
        group.Add(KeyEventType::Press, [&](KeyEvent &) { ++callCount; });
        input.OnKey(KeyCode::A, KeyEventType::Press);
    }
    input.OnKey(KeyCode::A, KeyEventType::Press);
    EXPECT_EQ(callCount, 1);
}
```

- [ ] **Step 6: Build and run**

```bash
cmake --build build --target vkspatial_tests --parallel
ctest --test-dir build --output-on-failure -R "MouseInputTest|MouseListenerGroupTest|KeyInputTest|KeyListenerGroupTest"
```

Expected: 6/6 new tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/Engine/Render/MouseInput.h src/Engine/Render/MouseInput.cpp src/Engine/Render/KeyInput.h src/Engine/Render/KeyInput.cpp test/test_engineRender.cpp
git commit -m "Add Engine::Render MouseInput/KeyInput"
```

---

### Task 5: Window + GlfwWindow

**Files:**
- Create: `src/Engine/Render/Window.h`
- Create: `src/Engine/Render/Window.cpp`
- Create: `src/Engine/Render/GlfwWindow.h`
- Create: `src/Engine/Render/GlfwWindow.cpp`
- Modify: `src/Engine/CMakeLists.txt`

**Interfaces:**
- Consumes: `Engine::Render::MouseInput`/`KeyInput`/`KeyCode`/`MouseButton` (Task 4). GLFW (`<GLFW/glfw3.h>`) — only inside `GlfwWindow.cpp`, never in a public header.
- Produces: `Engine::Render::WindowBackend`, `Engine::Render::WindowDescriptor`, `Engine::Render::Window` (abstract), `Engine::Render::GlfwWindow`. Consumed by Task 6 (`Application`).

This task is not unit-testable without a real display (creating a `GlfwWindow` opens an actual window). Verification is "does it build," with functional verification deferred to Task 7's visual run.

- [ ] **Step 1: Create `src/Engine/Render/Window.h`**

```cpp
#pragma once

#include "Engine/Render/KeyInput.h"
#include "Engine/Render/MouseInput.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Render {

    enum class WindowBackend {
        GLFW,
    };

    struct WindowDescriptor {
        uint32_t width = 1280;
        uint32_t height = 720;
        std::string title = "Engine::Render";
    };

    class Window {
    public:
        using UniquePtr = std::unique_ptr<Window>;

        static UniquePtr Create(WindowBackend backend, const WindowDescriptor &descriptor);
        virtual ~Window() = default;

        Window(const Window &) = delete;
        Window &operator=(const Window &) = delete;

        virtual bool ShouldClose() const = 0;
        virtual void RequestClose() = 0;
        virtual void PollEvents() = 0;
        virtual VkExtent2D FramebufferSize() const = 0;

        // Used internally by Application when constructing the Context — app code does
        // not call these directly.
        virtual std::vector<const char *> RequiredInstanceExtensions() const = 0;
        virtual VkSurfaceKHR CreateSurface(VkInstance instance) const = 0;

        MouseInput &Mouse() { return m_mouseInput; }
        KeyInput &Keys() { return m_keyInput; }

    protected:
        Window() = default;

        MouseInput m_mouseInput;
        KeyInput m_keyInput;
    };

} // namespace Engine::Render
```

- [ ] **Step 2: Create `src/Engine/Render/Window.cpp`**

```cpp
#include "Engine/Render/Window.h"
#include "Engine/Render/GlfwWindow.h"

#include <stdexcept>

namespace Engine::Render {

    Window::UniquePtr Window::Create(WindowBackend backend, const WindowDescriptor &descriptor) {
        switch (backend) {
            case WindowBackend::GLFW:
                return std::make_unique<GlfwWindow>(descriptor);
        }
        throw std::runtime_error("Window::Create received an unknown WindowBackend");
    }

} // namespace Engine::Render
```

- [ ] **Step 3: Create `src/Engine/Render/GlfwWindow.h`**

```cpp
#pragma once

#include "Engine/Render/Window.h"

struct GLFWwindow;

namespace Engine::Render {

    class GlfwWindow : public Window {
    public:
        explicit GlfwWindow(const WindowDescriptor &descriptor);
        ~GlfwWindow() override;

        bool ShouldClose() const override;
        void RequestClose() override;
        void PollEvents() override;
        VkExtent2D FramebufferSize() const override;

        std::vector<const char *> RequiredInstanceExtensions() const override;
        VkSurfaceKHR CreateSurface(VkInstance instance) const override;

    private:
        GLFWwindow *m_window = nullptr;

        static void CursorPosCallback(GLFWwindow *window, double x, double y);
        static void MouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
        static void ScrollCallback(GLFWwindow *window, double xoffset, double yoffset);
        static void CursorEnterCallback(GLFWwindow *window, int entered);
        static void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    };

} // namespace Engine::Render
```

- [ ] **Step 4: Create `src/Engine/Render/GlfwWindow.cpp`**

```cpp
#include "Engine/Render/GlfwWindow.h"

#include <GLFW/glfw3.h>

#include <stdexcept>

namespace Engine::Render {

    namespace {

        uint32_t QueryMouseModifiers(GLFWwindow *window) {
            uint32_t flags = 0;
            if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
                flags |= MouseModifierShift;
            if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS)
                flags |= MouseModifierControl;
            if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS)
                flags |= MouseModifierAlt;
            if (glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS)
                flags |= MouseModifierSuper;
            return flags;
        }

        MouseButton ToMouseButton(int glfwButton) {
            switch (glfwButton) {
                case GLFW_MOUSE_BUTTON_LEFT: return MouseButton::Left;
                case GLFW_MOUSE_BUTTON_RIGHT: return MouseButton::Right;
                case GLFW_MOUSE_BUTTON_MIDDLE: return MouseButton::Middle;
                case GLFW_MOUSE_BUTTON_4: return MouseButton::Button4;
                case GLFW_MOUSE_BUTTON_5: return MouseButton::Button5;
                default: return MouseButton::Other;
            }
        }

        KeyCode ToKeyCode(int glfwKey) {
            switch (glfwKey) {
                case GLFW_KEY_ESCAPE: return KeyCode::Escape;
                case GLFW_KEY_SPACE: return KeyCode::Space;
                case GLFW_KEY_ENTER: return KeyCode::Enter;
                case GLFW_KEY_TAB: return KeyCode::Tab;
                case GLFW_KEY_BACKSPACE: return KeyCode::Backspace;
                case GLFW_KEY_LEFT: return KeyCode::Left;
                case GLFW_KEY_RIGHT: return KeyCode::Right;
                case GLFW_KEY_UP: return KeyCode::Up;
                case GLFW_KEY_DOWN: return KeyCode::Down;
                case GLFW_KEY_LEFT_SHIFT: return KeyCode::LeftShift;
                case GLFW_KEY_LEFT_CONTROL: return KeyCode::LeftControl;
                case GLFW_KEY_LEFT_ALT: return KeyCode::LeftAlt;
                case GLFW_KEY_LEFT_SUPER: return KeyCode::LeftSuper;
                default:
                    break;
            }
            if (glfwKey >= GLFW_KEY_A && glfwKey <= GLFW_KEY_Z)
                return static_cast<KeyCode>(static_cast<uint32_t>(KeyCode::A) + (glfwKey - GLFW_KEY_A));
            if (glfwKey >= GLFW_KEY_0 && glfwKey <= GLFW_KEY_9)
                return static_cast<KeyCode>(static_cast<uint32_t>(KeyCode::Num0) + (glfwKey - GLFW_KEY_0));
            return KeyCode::Unknown;
        }

        KeyEventType ToKeyEventType(int action) {
            switch (action) {
                case GLFW_PRESS: return KeyEventType::Press;
                case GLFW_RELEASE: return KeyEventType::Release;
                case GLFW_REPEAT: return KeyEventType::Repeat;
                default: return KeyEventType::Press;
            }
        }

    } // namespace

    GlfwWindow::GlfwWindow(const WindowDescriptor &descriptor) {
        if (!glfwInit())
            throw std::runtime_error("GlfwWindow: glfwInit failed");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        m_window = glfwCreateWindow(static_cast<int>(descriptor.width),
                                    static_cast<int>(descriptor.height),
                                    descriptor.title.c_str(), nullptr, nullptr);
        if (!m_window) {
            glfwTerminate();
            throw std::runtime_error("GlfwWindow: glfwCreateWindow failed");
        }

        glfwSetWindowUserPointer(m_window, this);
        glfwSetCursorPosCallback(m_window, CursorPosCallback);
        glfwSetMouseButtonCallback(m_window, MouseButtonCallback);
        glfwSetScrollCallback(m_window, ScrollCallback);
        glfwSetCursorEnterCallback(m_window, CursorEnterCallback);
        glfwSetKeyCallback(m_window, KeyCallback);
    }

    GlfwWindow::~GlfwWindow() {
        if (m_window)
            glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    bool GlfwWindow::ShouldClose() const {
        return glfwWindowShouldClose(m_window);
    }

    void GlfwWindow::RequestClose() {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }

    void GlfwWindow::PollEvents() {
        glfwPollEvents();
    }

    VkExtent2D GlfwWindow::FramebufferSize() const {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(m_window, &width, &height);
        return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    }

    std::vector<const char *> GlfwWindow::RequiredInstanceExtensions() const {
        uint32_t count = 0;
        const char **extensions = glfwGetRequiredInstanceExtensions(&count);
        if (!extensions || count == 0)
            throw std::runtime_error("GlfwWindow: GLFW did not provide Vulkan extensions");
        return std::vector<const char *>(extensions, extensions + count);
    }

    VkSurfaceKHR GlfwWindow::CreateSurface(VkInstance instance) const {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (glfwCreateWindowSurface(instance, m_window, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error("GlfwWindow: failed to create window surface");
        return surface;
    }

    void GlfwWindow::CursorPosCallback(GLFWwindow *window, double x, double y) {
        auto *self = static_cast<GlfwWindow *>(glfwGetWindowUserPointer(window));
        if (!self)
            return;
        self->m_mouseInput.OnMouseMove(x, y, QueryMouseModifiers(window), glfwGetTime());
    }

    void GlfwWindow::MouseButtonCallback(GLFWwindow *window, int button, int action, int) {
        auto *self = static_cast<GlfwWindow *>(glfwGetWindowUserPointer(window));
        if (!self)
            return;
        double x = 0.0;
        double y = 0.0;
        glfwGetCursorPos(window, &x, &y);
        self->m_mouseInput.OnButton(ToMouseButton(button), action == GLFW_PRESS, x, y,
                                    QueryMouseModifiers(window), glfwGetTime());
    }

    void GlfwWindow::ScrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
        auto *self = static_cast<GlfwWindow *>(glfwGetWindowUserPointer(window));
        if (!self)
            return;
        double x = 0.0;
        double y = 0.0;
        glfwGetCursorPos(window, &x, &y);
        self->m_mouseInput.OnScroll(x, y, xoffset, yoffset, QueryMouseModifiers(window), glfwGetTime());
    }

    void GlfwWindow::CursorEnterCallback(GLFWwindow *window, int entered) {
        auto *self = static_cast<GlfwWindow *>(glfwGetWindowUserPointer(window));
        if (!self)
            return;
        double x = 0.0;
        double y = 0.0;
        glfwGetCursorPos(window, &x, &y);
        if (entered)
            self->m_mouseInput.OnCursorEnter(x, y, QueryMouseModifiers(window), glfwGetTime());
        else
            self->m_mouseInput.OnCursorLeave(x, y, QueryMouseModifiers(window), glfwGetTime());
    }

    void GlfwWindow::KeyCallback(GLFWwindow *window, int key, int, int action, int mods) {
        auto *self = static_cast<GlfwWindow *>(glfwGetWindowUserPointer(window));
        if (!self)
            return;
        self->m_keyInput.OnKey(ToKeyCode(key), ToKeyEventType(action),
                               static_cast<uint32_t>(mods), glfwGetTime());
    }

} // namespace Engine::Render
```

- [ ] **Step 5: Link `glfw` into `EngineRender`**

In `src/Engine/CMakeLists.txt`, find the `EngineRender` block's `target_link_libraries` and add `glfw` to `PRIVATE` (GLFW headers must not leak into `Engine::Render`'s public headers, so this is `PRIVATE`, not `PUBLIC`):

```cmake
target_link_libraries(EngineRender
        PUBLIC Engine::Core Vulkan::Vulkan
        PRIVATE glfw)
```

- [ ] **Step 6: Build**

```bash
cmake -S . -B build
cmake --build build --target EngineRender --parallel
```

Expected: builds with no errors.

- [ ] **Step 7: Commit**

```bash
git add src/Engine/Render/Window.h src/Engine/Render/Window.cpp src/Engine/Render/GlfwWindow.h src/Engine/Render/GlfwWindow.cpp src/Engine/CMakeLists.txt
git commit -m "Add Engine::Render Window abstraction with GLFW backend"
```

---

### Task 6: Application

**Files:**
- Create: `src/Engine/Render/Application.h`
- Create: `src/Engine/Render/Application.cpp`

**Interfaces:**
- Consumes: `Engine::Core::Context`, `Engine::Render::SwapChain`/`SwapChainDescriptor`, `Engine::Render::Renderer`, `Engine::Render::View`, `Engine::Render::Window`/`WindowBackend`/`WindowDescriptor` (Tasks 1, 3, 5, 6-in-progress).
- Produces: `Engine::Render::ApplicationDescriptor`, `Engine::Render::Application`. Consumed by Task 7 (`example2/cube_render.cpp`).

Like Tasks 3 and 5, this cannot be unit-tested without a real window — verification is "does it build," deferred functional verification to Task 7.

- [ ] **Step 1: Create `src/Engine/Render/Application.h`**

```cpp
#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Render/Renderer.h"
#include "Engine/Render/SwapChain.h"
#include "Engine/Render/View.h"
#include "Engine/Render/Window.h"

#include <memory>

namespace Engine::Render {

    struct ApplicationDescriptor {
        WindowBackend backend = WindowBackend::GLFW;
        WindowDescriptor window;
        VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    };

    class Application {
    public:
        explicit Application(const ApplicationDescriptor &descriptor = {});
        ~Application();

        Application(const Application &) = delete;
        Application &operator=(const Application &) = delete;

        Engine::Core::Context &GetContext() { return *m_context; }
        SwapChain &GetSwapChain() { return *m_swapChain; }
        Renderer &GetRenderer() { return *m_renderer; }
        Window &GetWindow() { return *m_window; }
        View &GetView() { return m_view; }

        // Loops until GetWindow().ShouldClose(). Polls events, skips frames while
        // minimized, and calls Renderer::BeginFrame/Render/EndFrame each iteration. If
        // anything throws mid-loop, waits for the device to idle before rethrowing (since
        // Context::~Context() does not itself wait for the device to idle).
        void Run();

    private:
        Window::UniquePtr m_window;
        std::unique_ptr<Engine::Core::Context> m_context;
        std::unique_ptr<SwapChain> m_swapChain;
        std::unique_ptr<Renderer> m_renderer;
        View m_view;
    };

} // namespace Engine::Render
```

Member declaration order matters here: `m_window` first, then `m_context`, `m_swapChain`, `m_renderer`, `m_view` — C++ destroys members in reverse declaration order, so on teardown `m_view` (no-op) destructs first, then `m_renderer` (waits idle, tears down sync objects + depth image), then `m_swapChain`, then `m_context` (which destroys the `VkSurfaceKHR` it was given), and finally `m_window` (which destroys the actual platform window) — the surface is torn down before the window that produced it, which is the safe order.

- [ ] **Step 2: Create `src/Engine/Render/Application.cpp`**

```cpp
#include "Engine/Render/Application.h"

namespace Engine::Render {

    Application::Application(const ApplicationDescriptor &descriptor) {
        m_window = Window::Create(descriptor.backend, descriptor.window);

        Window *windowPtr = m_window.get();
        m_context = std::make_unique<Engine::Core::Context>(
                true,
                windowPtr->RequiredInstanceExtensions(),
                [windowPtr](VkInstance instance) { return windowPtr->CreateSurface(instance); });

        const VkExtent2D framebufferSize = m_window->FramebufferSize();
        SwapChainDescriptor swapChainDescriptor{};
        swapChainDescriptor.width = framebufferSize.width;
        swapChainDescriptor.height = framebufferSize.height;
        m_swapChain = std::make_unique<SwapChain>(*m_context, swapChainDescriptor);

        m_renderer = std::make_unique<Renderer>(*m_context, *m_swapChain, descriptor.depthFormat);
    }

    Application::~Application() = default;

    void Application::Run() {
        try {
            while (!m_window->ShouldClose()) {
                m_window->PollEvents();

                const VkExtent2D size = m_window->FramebufferSize();
                if (size.width == 0 || size.height == 0)
                    continue;

                if (!m_renderer->BeginFrame(size.width, size.height))
                    continue;
                m_renderer->Render(m_view);
                m_renderer->EndFrame();
            }
        } catch (...) {
            vkDeviceWaitIdle(m_context->device);
            throw;
        }
        vkDeviceWaitIdle(m_context->device);
    }

} // namespace Engine::Render
```

- [ ] **Step 3: Build**

```bash
cmake --build build --target EngineRender --parallel
```

Expected: builds with no errors.

- [ ] **Step 4: Commit**

```bash
git add src/Engine/Render/Application.h src/Engine/Render/Application.cpp
git commit -m "Add Engine::Render Application (owns Window+Context+SwapChain+Renderer+View, Run())"
```

---

### Task 7: CubePass + example2/cube_render.cpp rewrite

**Files:**
- Create: `example2/CubePass.h`
- Create: `example2/CubePass.cpp`
- Modify: `example2/cube_render.cpp` (full rewrite)
- Modify: `example2/CMakeLists.txt`

**Interfaces:**
- Consumes: everything from Tasks 1-6 (`Application`, `View`, `Scene`, `Camera`, `RenderGraph`/`RenderPass`, `MouseListenerGroup`/`KeyListenerGroup`, `Engine::Core::Buffer`/`Context`/`QueueRole`, `Engine::Render::GraphicsPipeline`/`GraphicsPipelineDescriptor`/`RenderingDescriptor`/`RenderingScope`/`ClearOptions`). Also `vkMath::Quat`/`Vec3`/`MapToArcball` (already exists in `utilities/Math.h`), `SimpleResource::CreateCube`/`Primitives` (already exists, unchanged).

- [ ] **Step 1: Create `example2/CubePass.h`**

```cpp
#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Render/GraphicsPipeline.h"
#include "Engine/Render/RenderGraph.h"

#include <cstdint>
#include <string>

class CubePass : public Engine::Render::RenderPass {
public:
    CubePass(Engine::Core::Context &context, VkFormat colorFormat, const std::string &shaderDir);

    const char *Name() const override { return "CubePass"; }
    void Execute(Engine::Render::RenderContext &ctx) override;

private:
    Engine::Core::Buffer m_vertexBuffer;
    Engine::Core::Buffer m_indexBuffer;
    Engine::Render::GraphicsPipeline m_pipeline;
    uint32_t m_indexCount = 0;
};
```

- [ ] **Step 2: Create `example2/CubePass.cpp`**

```cpp
#include "CubePass.h"

#include "Engine/Render/Camera.h"
#include "Engine/Render/Rendering.h"
#include "Engine/Render/View.h"

#include "utilities/Math.h"
#include "utilities/SimpleResource.h"

#include <cstddef>

namespace {

    struct Vertex {
        float position[3];
        float color[3];
    };

    struct PushConstants {
        vkMath::Mat4 mvp;
    };

} // namespace

CubePass::CubePass(Engine::Core::Context &context, VkFormat colorFormat, const std::string &shaderDir)
    : m_vertexBuffer(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT),
      m_indexBuffer(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT),
      m_pipeline(context) {
    Primitives cube = SimpleResource::CreateCube(1.0f);
    std::vector<Vertex> vertices;
    vertices.reserve(cube.vertices.size());
    for (size_t i = 0; i < cube.vertices.size(); ++i) {
        const Eigen::Vector3f color =
                i < cube.colors.size() ? cube.colors[i] : Eigen::Vector3f(1.0f, 1.0f, 1.0f);
        vertices.push_back({
                {cube.vertices[i].x(), cube.vertices[i].y(), cube.vertices[i].z()},
                {color.x(), color.y(), color.z()},
        });
    }
    m_indexCount = static_cast<uint32_t>(cube.indices.size());

    const uint32_t vertexBytes = static_cast<uint32_t>(vertices.size() * sizeof(Vertex));
    m_vertexBuffer.Allocate(vertexBytes);
    m_vertexBuffer.Upload(vertices.data(), vertexBytes, Engine::Core::QueueRole::Graphics);

    const uint32_t indexBytes = static_cast<uint32_t>(cube.indices.size() * sizeof(uint32_t));
    m_indexBuffer.Allocate(indexBytes);
    m_indexBuffer.Upload(cube.indices.data(), indexBytes, Engine::Core::QueueRole::Graphics);

    Engine::Render::GraphicsPipelineDescriptor descriptor;
    descriptor.VertexShader(shaderDir + "/cube.vert.spv")
            .FragmentShader(shaderDir + "/cube.frag.spv")
            .VertexBinding<Vertex>()
            .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position))
            .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
            .ColorTarget(colorFormat)
            .DepthTarget(VK_FORMAT_D32_SFLOAT)
            .PushConstant<PushConstants>(VK_SHADER_STAGE_VERTEX_BIT);
    m_pipeline.Build(descriptor);
}

void CubePass::Execute(Engine::Render::RenderContext &ctx) {
    Engine::Render::ClearOptions clear{};
    clear.color[0] = 0.025f;
    clear.color[1] = 0.027f;
    clear.color[2] = 0.032f;
    clear.color[3] = 1.0f;

    const VkExtent2D extent = ctx.swapChain->Extent();
    auto renderingDescriptor = Engine::Render::RenderingDescriptor::ColorDepth(
            extent, ctx.swapChain->ImageView(ctx.imageIndex), ctx.depthImage->View(), clear);

    Engine::Render::RenderingScope scope(ctx.commandBuffer, renderingDescriptor);

    m_pipeline.Bind(ctx.commandBuffer);
    VkBuffer vertexHandle = m_vertexBuffer.Handle();
    VkDeviceSize vertexOffset = 0;
    vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &vertexHandle, &vertexOffset);
    vkCmdBindIndexBuffer(ctx.commandBuffer, m_indexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

    Engine::Render::Camera *camera = ctx.view->GetCamera();
    PushConstants push{camera->GetProjectionMatrix() * camera->GetViewMatrix()};
    m_pipeline.PushConstants(ctx.commandBuffer, VK_SHADER_STAGE_VERTEX_BIT, push);
    vkCmdDrawIndexed(ctx.commandBuffer, m_indexCount, 1, 0, 0, 0);
}
```

Note this is a `RenderPass` opening its own `RenderingScope` against the default color/depth targets that `Renderer::Render()` already transitioned to `ATTACHMENT_OPTIMAL` before calling `graph->Execute()` — no layout transition code appears in `CubePass` itself.

- [ ] **Step 3: Rewrite `example2/cube_render.cpp`**

```cpp
#include "CubePass.h"

#include "Engine/Render/Application.h"

#include "utilities/Math.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main() {
    try {
        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {900, 700, "Engine::Render Cube"};
        Engine::Render::Application app(descriptor);

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D extent = app.GetSwapChain().Extent();
        const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        camera.SetPerspective(60.0f * 3.14159265f / 180.0f, aspect, 0.1f, 100.0f);

        vkMath::Quat orientation = vkMath::Quat::Identity();
        vkMath::Vec3 target(0.0f, 0.0f, 0.0f);
        float distance = 4.5f;
        bool dragging = false;
        vkMath::Vec3 lastBall = vkMath::Vec3::Zero();

        auto updateCamera = [&]() {
            const vkMath::Vec3 eye = target + orientation * vkMath::Vec3(0.0f, 0.0f, distance);
            camera.LookAt(eye, target);
        };
        updateCamera();

        const std::string shaderDir = CUBE_RENDER2_SHADER_DIR;
        Engine::Render::RenderGraph graph;
        graph.AddPass(std::make_unique<CubePass>(app.GetContext(), app.GetSwapChain().Format(), shaderDir));

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

            vkMath::Vec3 current = vkMath::MapToArcball(
                    e.x, e.y, static_cast<int>(size.width), static_cast<int>(size.height));
            if (!dragging) {
                dragging = true;
                lastBall = current;
                return;
            }

            vkMath::Vec3 axis = lastBall.cross(current);
            if (axis.dot(axis) > 1e-8f) {
                const float w = std::clamp(lastBall.dot(current), -1.0f, 1.0f);
                vkMath::Quat delta = vkMath::Quat(w, axis.x(), axis.y(), axis.z()).normalized();
                orientation = (delta * orientation).normalized();
                updateCamera();
            }
            lastBall = current;
            e.handled = true;
        });
        trackball.Add(Engine::Render::MouseEventType::ButtonUp, [&](Engine::Render::MouseEvent &e) {
            if (e.button == Engine::Render::MouseButton::Left)
                dragging = false;
        });
        trackball.Add(Engine::Render::MouseEventType::Scroll, [&](Engine::Render::MouseEvent &e) {
            distance = std::clamp(distance * std::exp(static_cast<float>(-e.scrollY) * 0.08f), 2.0f, 10.0f);
            updateCamera();
            e.handled = true;
        });

        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape)
                app.GetWindow().RequestClose();
        });

        app.Run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
```

Note what changed from the old `example2/cube_render.cpp`: no `#include <GLFW/glfw3.h>`, no `VkInstance`/`VkDevice`/semaphores/command buffers anywhere, no manual resize detection, no manual `try/catch` around a `while` loop (that's inside `Application::Run()` now). The camera is trackball-orbit driven (left-drag to rotate via arcball, scroll to zoom) instead of a fixed auto-rotation, using the same `vkMath::Quat`/`MapToArcball`-based approach as `example/bvh_path_tracer.cpp`'s `TrackballCamera`. Escape closes the window via `KeyInput`.

- [ ] **Step 4: Update `example2/CMakeLists.txt`**

Replace the whole file with:

```cmake
cmake_minimum_required(VERSION 3.18)
project(vkspatial_example2 LANGUAGES CXX)

find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/bin")
if (GLSLC_EXECUTABLE)
    set(CUBE_RENDER2_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(CUBE_RENDER2_VERT_SPV "${CUBE_RENDER2_SHADER_OUTPUT_DIR}/cube.vert.spv")
    set(CUBE_RENDER2_FRAG_SPV "${CUBE_RENDER2_SHADER_OUTPUT_DIR}/cube.frag.spv")

    add_custom_command(
            OUTPUT "${CUBE_RENDER2_VERT_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER2_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_SOURCE_DIR}/example/cube.vert"
                    -o "${CUBE_RENDER2_VERT_SPV}"
            DEPENDS "${CMAKE_SOURCE_DIR}/example/cube.vert"
            VERBATIM)

    add_custom_command(
            OUTPUT "${CUBE_RENDER2_FRAG_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER2_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_SOURCE_DIR}/example/cube.frag"
                    -o "${CUBE_RENDER2_FRAG_SPV}"
            DEPENDS "${CMAKE_SOURCE_DIR}/example/cube.frag"
            VERBATIM)

    add_custom_target(cube_render2_shaders
            DEPENDS "${CUBE_RENDER2_VERT_SPV}" "${CUBE_RENDER2_FRAG_SPV}")

    add_executable(cube_render2
            cube_render.cpp
            CubePass.cpp
            "${CMAKE_SOURCE_DIR}/src/utilities/SimpleResource.cpp"
            "${CMAKE_SOURCE_DIR}/src/utilities/Math.cpp")
    add_dependencies(cube_render2 cube_render2_shaders)
    target_link_libraries(cube_render2 PRIVATE Engine::Render Engine::Core)
    target_compile_definitions(cube_render2 PRIVATE
            CUBE_RENDER2_SHADER_DIR=\"${CUBE_RENDER2_SHADER_OUTPUT_DIR}\")
else ()
    message(WARNING "cube_render2 example skipped because glslc was not found")
endif ()
```

Two things removed from the old file: the explicit Eigen `target_include_directories` (now transitive via `Engine::Render`'s `PUBLIC` Eigen include from Task 1) and the explicit `glfw` link (no longer needed — `cube_render.cpp`/`CubePass.cpp` don't call any GLFW function directly; GLFW's symbols still get linked into the final executable because `EngineRender`, a static library, privately depends on `glfw`, and CMake pulls a static library's own link dependencies into whatever finally links against it).

- [ ] **Step 5: Build**

```bash
cmake -S . -B build
cmake --build build --target cube_render2 --parallel
```

Expected: builds with no errors.

- [ ] **Step 6: Run and visually verify**

```bash
./build/example2/cube_render2
```

Expected: a window titled "Engine::Render Cube" opens showing a stationary cube (no auto-rotation now) on a dark background. Left-click and drag rotates the cube via arcball. Scroll wheel zooms in/out (clamped between distance 2.0 and 10.0). Pressing Escape closes the window. Resizing the window keeps rendering without crashing or validation errors. Closing via the window's close button or Escape exits the process cleanly (no hang, no crash).

- [ ] **Step 7: Run the existing test suite as a regression check**

```bash
ctest --test-dir build --output-on-failure
```

Expected: same 50/51 baseline plus whatever new tests Tasks 1, 2, 4 added, all passing (only `WideBVHTest.RadiusMatchesCpuReference` failing, pre-existing and unrelated).

- [ ] **Step 8: Commit**

```bash
git add example2/CubePass.h example2/CubePass.cpp example2/cube_render.cpp example2/CMakeLists.txt
git commit -m "Rewrite example2/cube_render.cpp on Application/RenderGraph/trackball input"
```

---

### Task 8: Full verification pass

**Files:**
- None expected to change (verification-only), unless a real regression is found and fixed.

**Interfaces:**
- Consumes: all of Tasks 1-7.

- [ ] **Step 1: Full clean rebuild**

```bash
rm -rf build
cmake -S . -B build --fresh
cmake --build build --parallel
```

Expected: every target builds with no errors, including all pre-existing targets alongside the new test cases in `vkspatial_tests` and the rewritten `cube_render2`.

- [ ] **Step 2: Run the full test suite**

```bash
ctest --test-dir build --output-on-failure
```

Expected: only `WideBVHTest.RadiusMatchesCpuReference` fails (pre-existing, unrelated); every other test — including all new `SceneTest`/`CameraTest`/`ViewTest`/`RenderGraphTest`/`MouseInputTest`/`MouseListenerGroupTest`/`KeyInputTest`/`KeyListenerGroupTest` cases from Tasks 1, 2, 4 — passes.

- [ ] **Step 3: Run `cube_render2` once more end-to-end**

```bash
./build/example2/cube_render2
```

Expected: same as Task 7 Step 6 — window opens, cube is visible, left-drag orbits it, scroll zooms, Escape and the window close button both exit cleanly, resize works. This is the one thing worth re-confirming after a full clean rebuild specifically (shader paths, `CUBE_RENDER2_SHADER_DIR`, and the `glslc` custom-command dependency chain all get exercised fresh here).

- [ ] **Step 4: Confirm no existing file was touched beyond the allowed edits**

```bash
git diff --stat main -- src/vkCommon src/vkSpatial src/vkRender src/Engine/Core example test/test_bvh* test/test_vkCompute.cpp test/test_wideBVH.cpp test/test_engineCore.cpp
```

Expected: empty output. (`src/Engine/CMakeLists.txt`, `example2/CMakeLists.txt`, `example2/cube_render.cpp`, and `test/CMakeLists.txt` are intentionally excluded from this check since they're explicitly in scope per the Global Constraints.)

- [ ] **Step 5: Commit any fixes found during verification, or confirm none needed**

If Steps 1-4 surfaced no issues, there's nothing to commit — this task is verification-only. If an issue was found and fixed, commit it with a message describing exactly what broke and how it was fixed, referencing which step caught it.

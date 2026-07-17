# Engine/Render Minimal + example2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the minimum `Engine::Render` pieces needed to draw a triangle-mesh to a window (`RenderTypes`, `RenderAttachments`, `Rendering`, `SwapChain`, `GraphicsPipeline`), then add `example2/cube_render.cpp` — a rotating cube rendered with a hand-written frame loop (no `Renderer`/`RenderGraph`/`Scene`/`View`/`Camera`), reusing `example/cube.vert`/`cube.frag`.

**Architecture:** `RenderTypes`/`RenderAttachments`/`Rendering` are pure Vulkan value/RAII types with zero `Context` dependency — namespace-only ports from `vkRender`. `SwapChain` and `GraphicsPipeline` are ported with their constructors changed to take `Engine::Core::Context&` instead of `vkCommon::VkContext*`/raw `VkDevice`, matching the convention `Engine::Core`'s five classes already established. `example2/cube_render.cpp` composes `Engine::Core::Context`/`Buffer`/`Image` with `Engine::Render::SwapChain`/`GraphicsPipeline`/`Rendering`, writing the acquire→record→submit→present frame loop directly in `main()`.

**Tech Stack:** C++17, Vulkan 1.3 (dynamic rendering), GLFW, existing `Engine::Core` (already built and merged).

## Global Constraints

- This plan creates only new files, plus these specific existing-file edits: `src/Engine/CMakeLists.txt` (add `EngineRender` target), root `CMakeLists.txt` (add `add_subdirectory(example2)`). `vkCommon`, `vkSpatial`, `vkRender`, `example/`, `test/`, and everything under `src/Engine/Core/` are untouched.
- `RenderTypes.h`, `RenderAttachments.h`, `Rendering.h`/`.cpp` are namespace-only ports (`vkRender` → `Engine::Render`) — no logic changes at all.
- `SwapChain` and `GraphicsPipeline` constructors take `Engine::Core::Context &context` (a reference, not a pointer) — matching `Engine::Core::Image`'s established precedent of dropping the "is context null" throw-check that a raw-pointer constructor needed (a reference can't be null through normal use).
- No automated tests are added in this plan. This project's established convention (see `docs/superpowers/specs/2026-07-17-engine-render-minimal-design.md`'s testing section) is that Vulkan window/swapchain code is verified by actually running the example, not GTest — `SwapChain`/`GraphicsPipeline` both require a live window+surface to construct meaningfully, which GTest's headless fixtures (used by `ContextTest`/`BufferTest`/etc. in `Engine::Core`) cannot provide. Task 4's manual run of `cube_render2` is what actually exercises Tasks 1-3.
- `example2`'s CMake target is named `cube_render2` (the name `cube_render` is already taken by `example/CMakeLists.txt`). It reuses `example/cube.vert`/`example/cube.frag` by referencing their existing paths — it does not copy them.

---

### Task 1: Port `RenderTypes`, `RenderAttachments`, `Rendering` + wire up the `EngineRender` CMake target

**Files:**
- Create: `src/Engine/Render/RenderTypes.h`
- Create: `src/Engine/Render/RenderAttachments.h`
- Create: `src/Engine/Render/Rendering.h`
- Create: `src/Engine/Render/Rendering.cpp`
- Modify: `src/Engine/CMakeLists.txt`

**Interfaces:**
- Produces: `Engine::Render::Extent`, `Viewport`, `ClearOptions`, `FrameInfo`, `SwapChainDescriptor` (RenderTypes.h); `Engine::Render::ColorAttachment`, `DepthAttachment` (RenderAttachments.h); `Engine::Render::RenderingDescriptor`, `RenderingScope` (Rendering.h) — for Tasks 2-4 to consume. Also produces the `EngineRender` CMake target (alias `Engine::Render`) that Tasks 2-3's files build into.

- [ ] **Step 1: Create `RenderTypes.h`**

Copy `src/vkRender/RenderTypes.h` verbatim into `src/Engine/Render/RenderTypes.h`, changing only `namespace vkRender {` → `namespace Engine::Render {` (and the matching closing comment `// namespace vkRender` → `// namespace Engine::Render`):

```cpp
#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

namespace Engine::Render {

    struct Extent {
        uint32_t width = 0;
        uint32_t height = 0;

        bool Empty() const { return width == 0 || height == 0; }
    };

    struct Viewport {
        int32_t x = 0;
        int32_t y = 0;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    struct ClearOptions {
        bool clearColor = true;
        bool clearDepth = true;
        bool clearStencil = false;
        float color[4] = {0.02f, 0.02f, 0.025f, 1.0f};
        float depth = 1.0f;
        uint32_t stencil = 0;
    };

    struct FrameInfo {
        uint64_t frameIndex = 0;
        double elapsedSeconds = 0.0;
        float deltaSeconds = 0.0f;
    };

    struct SwapChainDescriptor {
        uint32_t width = 0;
        uint32_t height = 0;
        VkFormat preferredFormat = VK_FORMAT_B8G8R8A8_UNORM;
        VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
        VkImageUsageFlags imageUsage =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    };

} // namespace Engine::Render
```

- [ ] **Step 2: Create `RenderAttachments.h`**

Copy `src/vkRender/RenderAttachments.h` verbatim into `src/Engine/Render/RenderAttachments.h`, changing only the namespace:

```cpp
#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

namespace Engine::Render {

    class ColorAttachment {
    public:
        explicit ColorAttachment(
                VkImageView imageView,
                VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            m_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            m_info.imageView = imageView;
            m_info.imageLayout = layout;
            m_info.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            m_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }

        ColorAttachment &Clear(const float color[4], bool enabled = true) {
            m_info.loadOp = enabled ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                    : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            m_clear.color = {{
                    color[0],
                    color[1],
                    color[2],
                    color[3],
            }};
            return *this;
        }

        ColorAttachment &Load() {
            m_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            return *this;
        }

        ColorAttachment &DontCareLoad() {
            m_info.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            return *this;
        }

        ColorAttachment &Store(bool enabled = true) {
            m_info.storeOp = enabled ? VK_ATTACHMENT_STORE_OP_STORE
                                     : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            return *this;
        }

        VkRenderingAttachmentInfo Build() const {
            VkRenderingAttachmentInfo info = m_info;
            info.clearValue = m_clear;
            return info;
        }

    private:
        VkRenderingAttachmentInfo m_info{};
        VkClearValue m_clear{};
    };

    class DepthAttachment {
    public:
        explicit DepthAttachment(VkImageView imageView,
                                 VkImageLayout layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
            m_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            m_info.imageView = imageView;
            m_info.imageLayout = layout;
            m_info.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            m_info.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        }

        DepthAttachment &Clear(float depth,
                               bool enabled = true,
                               uint32_t stencil = 0) {
            m_info.loadOp = enabled ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                    : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            m_clear.depthStencil = {depth, stencil};
            return *this;
        }

        DepthAttachment &Load() {
            m_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            return *this;
        }

        DepthAttachment &Store(bool enabled = true) {
            m_info.storeOp = enabled ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            return *this;
        }

        VkRenderingAttachmentInfo Build() const {
            VkRenderingAttachmentInfo info = m_info;
            info.clearValue = m_clear;
            return info;
        }

    private:
        VkRenderingAttachmentInfo m_info{};
        VkClearValue m_clear{};
    };

} // namespace Engine::Render
```

- [ ] **Step 3: Create `Rendering.h`**

Copy `src/vkRender/Rendering.h` verbatim into `src/Engine/Render/Rendering.h`, changing the namespace and the include path for `RenderAttachments.h`/`RenderTypes.h`:

```cpp
#pragma once

#include "Engine/Render/RenderAttachments.h"
#include "Engine/Render/RenderTypes.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Render {

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

} // namespace Engine::Render
```

- [ ] **Step 4: Create `Rendering.cpp`**

Copy `src/vkRender/Rendering.cpp` verbatim into `src/Engine/Render/Rendering.cpp`, changing only the include path and namespace:

```cpp
#include "Engine/Render/Rendering.h"

#include <stdexcept>

namespace Engine::Render {

    RenderingDescriptor::RenderingDescriptor(VkExtent2D extent) {
        SetExtent(extent);
    }

    RenderingDescriptor RenderingDescriptor::ColorDepth(VkExtent2D extent,
                                                        VkImageView colorView,
                                                        VkImageView depthView,
                                                        const ClearOptions &clear) {
        RenderingDescriptor descriptor(extent);
        descriptor
                .AddColorAttachment(
                        ColorAttachment(colorView)
                                .Clear(clear.color, clear.clearColor)
                                .Build())
                .SetDepthAttachment(
                        DepthAttachment(depthView)
                                .Clear(clear.depth, clear.clearDepth, clear.stencil)
                                .Build());
        return descriptor;
    }

    RenderingDescriptor &RenderingDescriptor::SetExtent(VkExtent2D extent) {
        renderArea = {{0, 0}, extent};
        return UseDefaultViewportAndScissor(extent);
    }

    RenderingDescriptor &RenderingDescriptor::SetRenderArea(VkRect2D area) {
        renderArea = area;
        return *this;
    }

    RenderingDescriptor &RenderingDescriptor::AddColorAttachment(
            const VkRenderingAttachmentInfo &attachment) {
        colorAttachments.push_back(attachment);
        return *this;
    }

    RenderingDescriptor &RenderingDescriptor::SetDepthAttachment(
            const VkRenderingAttachmentInfo &attachment) {
        depthAttachment = attachment;
        hasDepthAttachment = true;
        return *this;
    }

    RenderingDescriptor &RenderingDescriptor::SetViewport(VkViewport value) {
        viewport = value;
        setViewport = true;
        return *this;
    }

    RenderingDescriptor &RenderingDescriptor::SetScissor(VkRect2D value) {
        scissor = value;
        setScissor = true;
        return *this;
    }

    RenderingDescriptor &RenderingDescriptor::UseDefaultViewportAndScissor(
            VkExtent2D extent) {
        viewport = {
                0.0f,
                0.0f,
                static_cast<float>(extent.width),
                static_cast<float>(extent.height),
                0.0f,
                1.0f,
        };
        scissor = {{0, 0}, extent};
        setViewport = true;
        setScissor = true;
        return *this;
    }

    RenderingScope::RenderingScope(VkCommandBuffer commandBuffer,
                                   const RenderingDescriptor &descriptor)
        : m_commandBuffer(commandBuffer) {
        if (m_commandBuffer == VK_NULL_HANDLE)
            throw std::runtime_error("RenderingScope requires a valid VkCommandBuffer");
        if (descriptor.renderArea.extent.width == 0 ||
            descriptor.renderArea.extent.height == 0)
            throw std::runtime_error("RenderingScope requires a non-empty render area");

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = descriptor.renderArea;
        renderingInfo.layerCount = descriptor.layerCount;
        renderingInfo.colorAttachmentCount =
                static_cast<uint32_t>(descriptor.colorAttachments.size());
        renderingInfo.pColorAttachments =
                descriptor.colorAttachments.empty()
                        ? nullptr
                        : descriptor.colorAttachments.data();
        renderingInfo.pDepthAttachment =
                descriptor.hasDepthAttachment ? &descriptor.depthAttachment : nullptr;

        vkCmdBeginRendering(m_commandBuffer, &renderingInfo);
        m_active = true;

        if (descriptor.setViewport)
            vkCmdSetViewport(m_commandBuffer, 0, 1, &descriptor.viewport);
        if (descriptor.setScissor)
            vkCmdSetScissor(m_commandBuffer, 0, 1, &descriptor.scissor);
    }

    RenderingScope::~RenderingScope() {
        End();
    }

    void RenderingScope::End() {
        if (!m_active)
            return;
        vkCmdEndRendering(m_commandBuffer);
        m_active = false;
    }

} // namespace Engine::Render
```

- [ ] **Step 5: Add the `EngineRender` target to `src/Engine/CMakeLists.txt`**

Append after the existing `EngineCore` target block:

```cmake

file(GLOB_RECURSE ENGINE_RENDER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Render/*.cpp")

add_library(EngineRender STATIC ${ENGINE_RENDER_SOURCES})
add_library(Engine::Render ALIAS EngineRender)

target_link_libraries(EngineRender
        PUBLIC Engine::Core Vulkan::Vulkan)

target_include_directories(EngineRender
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        PRIVATE
            "${VULKAN_SDK}/include")
```

- [ ] **Step 6: Build**

Run:
```bash
cmake -S . -B build
cmake --build build --target EngineRender --parallel
```
Expected: builds with no errors.

- [ ] **Step 7: Run the existing test suite as a regression check**

Run:
```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
Expected: same 50/51 baseline as before this task (`WideBVHTest.RadiusMatchesCpuReference` the only, pre-existing, unrelated failure).

- [ ] **Step 8: Commit**

```bash
git add src/Engine/Render/RenderTypes.h src/Engine/Render/RenderAttachments.h src/Engine/Render/Rendering.h src/Engine/Render/Rendering.cpp src/Engine/CMakeLists.txt
git commit -m "Port RenderTypes/RenderAttachments/Rendering into Engine::Render"
```

---

### Task 2: Port `SwapChain`

**Files:**
- Create: `src/Engine/Render/SwapChain.h`
- Create: `src/Engine/Render/SwapChain.cpp`

**Interfaces:**
- Consumes: `Engine::Core::Context` (already built), `Engine::Render::SwapChainDescriptor` (Task 1).
- Produces, for Task 4 to consume:
  ```cpp
  namespace Engine::Render {
      class SwapChain {
      public:
          explicit SwapChain(Engine::Core::Context &context, const SwapChainDescriptor &descriptor = {});
          ~SwapChain();
          void Recreate(uint32_t width = 0, uint32_t height = 0);
          VkResult AcquireNextImage(VkSemaphore signalSemaphore, VkFence signalFence, uint32_t *imageIndex, uint64_t timeout = UINT64_MAX);
          VkResult Present(uint32_t imageIndex, VkSemaphore waitSemaphore);
          VkSwapchainKHR Handle() const;
          VkFormat Format() const;
          VkExtent2D Extent() const;
          uint32_t ImageCount() const;
          VkImage Image(uint32_t index) const;
          VkImageView ImageView(uint32_t index) const;
          bool RequiresRedBlueSwap() const;
      };
  }
  ```

- [ ] **Step 1: Create `SwapChain.h`**

```cpp
#pragma once

#include "Engine/Render/RenderTypes.h"

#include "Engine/Core/Context.h"

#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Render {

    class SwapChain {
    public:
        using UniquePtr = std::unique_ptr<SwapChain>;

        explicit SwapChain(Engine::Core::Context &context,
                           const SwapChainDescriptor &descriptor = {});
        ~SwapChain();

        SwapChain(const SwapChain &) = delete;
        SwapChain &operator=(const SwapChain &) = delete;

        void Recreate(uint32_t width = 0, uint32_t height = 0);

        VkResult AcquireNextImage(VkSemaphore signalSemaphore,
                                  VkFence signalFence,
                                  uint32_t *imageIndex,
                                  uint64_t timeout = UINT64_MAX);
        VkResult Present(uint32_t imageIndex, VkSemaphore waitSemaphore);

        VkSwapchainKHR Handle() const { return m_handle; }
        VkFormat Format() const { return m_format; }
        VkExtent2D Extent() const { return m_extent; }
        uint32_t ImageCount() const { return static_cast<uint32_t>(m_images.size()); }
        VkImage Image(uint32_t index) const { return m_images[index]; }
        VkImageView ImageView(uint32_t index) const { return m_imageViews[index]; }
        bool RequiresRedBlueSwap() const;

    private:
        Engine::Core::Context *m_context = nullptr;
        SwapChainDescriptor m_descriptor;
        VkSwapchainKHR m_handle = VK_NULL_HANDLE;
        VkFormat m_format = VK_FORMAT_UNDEFINED;
        VkExtent2D m_extent{};
        std::vector<VkImage> m_images;
        std::vector<VkImageView> m_imageViews;

        void Create(VkSwapchainKHR oldSwapchain);
        void DestroyViews();
        void Destroy();
    };

} // namespace Engine::Render
```

- [ ] **Step 2: Create `SwapChain.cpp`**

```cpp
#include "Engine/Render/SwapChain.h"

#include <algorithm>
#include <stdexcept>

namespace Engine::Render {
    namespace {

        VkSurfaceFormatKHR ChooseSurfaceFormat(
                const std::vector<VkSurfaceFormatKHR> &formats,
                VkFormat preferredFormat) {
            for (const auto &format: formats) {
                if (format.format == preferredFormat &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                    return format;
            }

            for (const auto &format: formats) {
                if (format.format == preferredFormat)
                    return format;
            }

            for (const auto &format: formats) {
                if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                    return format;
            }

            return formats.front();
        }

        VkPresentModeKHR ChoosePresentMode(
                const std::vector<VkPresentModeKHR> &presentModes,
                VkPresentModeKHR preferredMode) {
            for (VkPresentModeKHR mode: presentModes) {
                if (mode == preferredMode)
                    return mode;
            }
            return VK_PRESENT_MODE_FIFO_KHR;
        }

        VkExtent2D ChooseExtent(const VkSurfaceCapabilitiesKHR &caps,
                                uint32_t requestedWidth,
                                uint32_t requestedHeight) {
            if (caps.currentExtent.width != UINT32_MAX)
                return caps.currentExtent;

            VkExtent2D extent{};
            extent.width = std::max(1u, requestedWidth);
            extent.height = std::max(1u, requestedHeight);
            extent.width = std::clamp(extent.width,
                                      caps.minImageExtent.width,
                                      caps.maxImageExtent.width);
            extent.height = std::clamp(extent.height,
                                       caps.minImageExtent.height,
                                       caps.maxImageExtent.height);
            return extent;
        }

    } // namespace

    SwapChain::SwapChain(Engine::Core::Context &context,
                         const SwapChainDescriptor &descriptor)
        : m_context(&context), m_descriptor(descriptor) {
        if (m_context->surface == VK_NULL_HANDLE)
            throw std::runtime_error("SwapChain requires Context::surface");
        if (m_context->graphicsQueue == VK_NULL_HANDLE)
            throw std::runtime_error("SwapChain requires Context::graphicsQueue");

        Create(VK_NULL_HANDLE);
    }

    SwapChain::~SwapChain() {
        Destroy();
    }

    void SwapChain::Recreate(uint32_t width, uint32_t height) {
        if (width != 0) m_descriptor.width = width;
        if (height != 0) m_descriptor.height = height;

        VkSwapchainKHR oldSwapchain = m_handle;
        DestroyViews();
        m_images.clear();
        m_handle = VK_NULL_HANDLE;

        Create(oldSwapchain);

        if (oldSwapchain != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(m_context->device, oldSwapchain, nullptr);
    }

    VkResult SwapChain::AcquireNextImage(VkSemaphore signalSemaphore,
                                         VkFence signalFence,
                                         uint32_t *imageIndex,
                                         uint64_t timeout) {
        return vkAcquireNextImageKHR(m_context->device, m_handle, timeout,
                                     signalSemaphore, signalFence, imageIndex);
    }

    VkResult SwapChain::Present(uint32_t imageIndex, VkSemaphore waitSemaphore) {
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = waitSemaphore != VK_NULL_HANDLE ? 1u : 0u;
        presentInfo.pWaitSemaphores = waitSemaphore != VK_NULL_HANDLE ? &waitSemaphore : nullptr;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &m_handle;
        presentInfo.pImageIndices = &imageIndex;
        return vkQueuePresentKHR(m_context->graphicsQueue, &presentInfo);
    }

    bool SwapChain::RequiresRedBlueSwap() const {
        return m_format == VK_FORMAT_B8G8R8A8_UNORM ||
               m_format == VK_FORMAT_B8G8R8A8_SRGB;
    }

    void SwapChain::Create(VkSwapchainKHR oldSwapchain) {
        VkSurfaceCapabilitiesKHR caps{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_context->physicalDevice,
                                                  m_context->surface, &caps);

        if ((m_descriptor.imageUsage & caps.supportedUsageFlags) != m_descriptor.imageUsage)
            throw std::runtime_error("SwapChain image usage is not supported by surface");

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_context->physicalDevice,
                                             m_context->surface,
                                             &formatCount, nullptr);

        if (formatCount == 0)
            throw std::runtime_error("SwapChain surface has no supported formats");
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_context->physicalDevice,
                                             m_context->surface,
                                             &formatCount, formats.data());

        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(m_context->physicalDevice,
                                                  m_context->surface,
                                                  &presentModeCount, nullptr);
        if (presentModeCount == 0)
            throw std::runtime_error("SwapChain surface has no present modes");
        std::vector<VkPresentModeKHR> presentModes(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(m_context->physicalDevice,
                                                  m_context->surface,
                                                  &presentModeCount,
                                                  presentModes.data());

        const VkSurfaceFormatKHR surfaceFormat =
                ChooseSurfaceFormat(formats, m_descriptor.preferredFormat);
        const VkPresentModeKHR presentMode =
                ChoosePresentMode(presentModes, m_descriptor.presentMode);
        const VkExtent2D extent =
                ChooseExtent(caps, m_descriptor.width, m_descriptor.height);

        uint32_t imageCount = caps.minImageCount + 1;
        if (caps.maxImageCount > 0)
            imageCount = std::min(imageCount, caps.maxImageCount);

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = m_context->surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = m_descriptor.imageUsage;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform = caps.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = oldSwapchain;

        if (vkCreateSwapchainKHR(m_context->device, &createInfo,
                                 nullptr, &m_handle) != VK_SUCCESS)
            throw std::runtime_error("SwapChain failed to create VkSwapchainKHR");

        m_format = surfaceFormat.format;
        m_extent = extent;
        m_descriptor.width = extent.width;
        m_descriptor.height = extent.height;

        uint32_t actualImageCount = 0;
        vkGetSwapchainImagesKHR(m_context->device, m_handle,
                                &actualImageCount, nullptr);
        m_images.resize(actualImageCount);
        vkGetSwapchainImagesKHR(
                m_context->device,
                m_handle,
                &actualImageCount,
                m_images.data());

        m_imageViews.resize(actualImageCount);
        for (uint32_t i = 0; i < actualImageCount; ++i) {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_images[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = m_format;
            viewInfo.components = {
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY};
            viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            if (vkCreateImageView(m_context->device, &viewInfo,
                                  nullptr, &m_imageViews[i]) != VK_SUCCESS)
                throw std::runtime_error("SwapChain failed to create image view");
        }
    }

    void SwapChain::DestroyViews() {
        for (VkImageView view: m_imageViews)
            vkDestroyImageView(m_context->device, view, nullptr);
        m_imageViews.clear();
    }

    void SwapChain::Destroy() {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            return;

        DestroyViews();
        m_images.clear();
        if (m_handle != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(m_context->device, m_handle, nullptr);
        m_handle = VK_NULL_HANDLE;
    }

} // namespace Engine::Render
```

This is a faithful port of `src/vkRender/SwapChain.cpp` — the only changes are the namespace, the constructor parameter (`Engine::Core::Context &context` instead of `vkCommon::VkContext *context`, storing `&context` into the same `m_context` pointer member), dropping the `if (!m_context) throw ...` null check (impossible via a reference), and every `m_context->physDevice` becoming `m_context->physicalDevice` (five call sites: `vkGetPhysicalDeviceSurfaceCapabilitiesKHR`, two `vkGetPhysicalDeviceSurfaceFormatsKHR` calls, two `vkGetPhysicalDeviceSurfacePresentModesKHR` calls).

- [ ] **Step 3: Build**

Run:
```bash
cmake --build build --target EngineRender --parallel
```
Expected: builds with no errors.

- [ ] **Step 4: Run the existing test suite as a regression check**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: same 50/51 baseline (no test exercises `SwapChain` — it needs a live window, see Global Constraints — so this just confirms nothing else broke).

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Render/SwapChain.h src/Engine/Render/SwapChain.cpp
git commit -m "Port SwapChain into Engine::Render"
```

---

### Task 3: Port `GraphicsPipeline`

**Files:**
- Create: `src/Engine/Render/GraphicsPipeline.h`
- Create: `src/Engine/Render/GraphicsPipeline.cpp`

**Interfaces:**
- Consumes: `Engine::Core::Context`.
- Produces, for Task 4 to consume: `Engine::Render::GraphicsPipelineDescriptor` (fluent builder: `Shader`/`VertexShader`/`FragmentShader`/`VertexBinding`/`VertexAttribute`/`ColorTarget`/`DepthTarget`/`DepthBias`/`PushConstant`) and `Engine::Render::GraphicsPipeline` (`explicit GraphicsPipeline(Engine::Core::Context &context)`, `.Build(descriptor)`, `.Bind(cmd)`, `.PushConstants<T>(cmd, stage, value)`, `.ColorFormat()`, `.DepthFormat()`, `.MatchesColorTarget()`).

- [ ] **Step 1: Create `GraphicsPipeline.h`**

```cpp
#pragma once

#include "Engine/Core/Context.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Render {

    struct ShaderStageDescriptor {
        VkShaderStageFlagBits stage = VK_SHADER_STAGE_VERTEX_BIT;
        std::string path;
        std::string entryPoint = "main";
    };

    struct PipelineColorTarget {
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkPipelineColorBlendAttachmentState blend{};

        static PipelineColorTarget Opaque(VkFormat format);
    };

    struct GraphicsPipelineDescriptor {
        std::vector<ShaderStageDescriptor> shaderStages;
        std::vector<VkVertexInputBindingDescription> vertexBindings;
        std::vector<VkVertexInputAttributeDescription> vertexAttributes;
        std::vector<PipelineColorTarget> colorTargets;
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
        std::vector<VkPushConstantRange> pushConstantRanges;
        std::vector<VkDynamicState> dynamicStates = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR,
        };

        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;
        VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
        VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        bool depthBiasEnable = false;
        float depthBiasConstantFactor = 0.0f;
        float depthBiasClamp = 0.0f;
        float depthBiasSlopeFactor = 0.0f;

        VkFormat depthFormat = VK_FORMAT_UNDEFINED;
        bool depthTestEnable = false;
        bool depthWriteEnable = false;
        VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS;

        GraphicsPipelineDescriptor &Shader(VkShaderStageFlagBits stage,
                                           const std::string &path,
                                           const std::string &entryPoint = "main");
        GraphicsPipelineDescriptor &VertexShader(const std::string &path,
                                                 const std::string &entryPoint = "main");
        GraphicsPipelineDescriptor &FragmentShader(const std::string &path,
                                                   const std::string &entryPoint = "main");

        GraphicsPipelineDescriptor &VertexBinding(uint32_t binding,
                                                  uint32_t stride,
                                                  VkVertexInputRate inputRate =
                                                          VK_VERTEX_INPUT_RATE_VERTEX);
        template<typename VertexT>
        GraphicsPipelineDescriptor &VertexBinding(uint32_t binding = 0,
                                                  VkVertexInputRate inputRate =
                                                          VK_VERTEX_INPUT_RATE_VERTEX) {
            return VertexBinding(binding, static_cast<uint32_t>(sizeof(VertexT)), inputRate);
        }

        GraphicsPipelineDescriptor &VertexAttribute(uint32_t location,
                                                    uint32_t binding,
                                                    VkFormat format,
                                                    uint32_t offset);
        GraphicsPipelineDescriptor &ColorTarget(VkFormat format);
        GraphicsPipelineDescriptor &ColorTarget(const PipelineColorTarget &target);
        GraphicsPipelineDescriptor &DepthTarget(VkFormat format,
                                                bool writeDepth = true,
                                                VkCompareOp compareOp = VK_COMPARE_OP_LESS);
        GraphicsPipelineDescriptor &DepthBias(float constantFactor,
                                              float slopeFactor,
                                              float clamp = 0.0f);
        GraphicsPipelineDescriptor &PushConstant(VkShaderStageFlags stageFlags,
                                                 uint32_t size,
                                                 uint32_t offset = 0);
        template<typename PushT>
        GraphicsPipelineDescriptor &PushConstant(VkShaderStageFlags stageFlags,
                                                 uint32_t offset = 0) {
            return PushConstant(stageFlags, static_cast<uint32_t>(sizeof(PushT)), offset);
        }
    };

    class GraphicsPipeline {
    public:
        using UniquePtr = std::unique_ptr<GraphicsPipeline>;

        explicit GraphicsPipeline(Engine::Core::Context &context);
        ~GraphicsPipeline();

        GraphicsPipeline(const GraphicsPipeline &) = delete;
        GraphicsPipeline &operator=(const GraphicsPipeline &) = delete;

        GraphicsPipeline &Build(const GraphicsPipelineDescriptor &descriptor);
        void Destroy();

        void Bind(VkCommandBuffer commandBuffer) const;

        template<typename T>
        void PushConstants(VkCommandBuffer commandBuffer,
                           VkShaderStageFlags stageFlags,
                           const T &value,
                           uint32_t offset = 0) const {
            vkCmdPushConstants(commandBuffer, m_layout, stageFlags, offset,
                               static_cast<uint32_t>(sizeof(T)), &value);
        }

        VkPipeline Pipeline() const { return m_pipeline; }
        VkPipelineLayout Layout() const { return m_layout; }
        VkFormat ColorFormat(uint32_t index = 0) const;
        VkFormat DepthFormat() const { return m_depthFormat; }
        bool MatchesColorTarget(uint32_t index, VkFormat format) const;

    private:
        VkDevice m_device = VK_NULL_HANDLE;
        VkPipelineLayout m_layout = VK_NULL_HANDLE;
        VkPipeline m_pipeline = VK_NULL_HANDLE;
        std::vector<VkFormat> m_colorFormats;
        VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;

        static std::vector<uint32_t> LoadSPIRV(const std::string &path);
        VkShaderModule CreateShaderModule(const std::string &path) const;
    };

} // namespace Engine::Render
```

- [ ] **Step 2: Create `GraphicsPipeline.cpp`**

Faithful port of `src/vkRender/GraphicsPipeline.cpp` — only the namespace and the constructor change (`GraphicsPipeline::GraphicsPipeline(VkDevice device) : m_device(device) { if (m_device == VK_NULL_HANDLE) throw ...; }` becomes `GraphicsPipeline::GraphicsPipeline(Engine::Core::Context &context) : m_device(context.device) {}` — the null check is dropped because `context.device` is guaranteed valid by `Context`'s own constructor).

```cpp
#include "Engine/Render/GraphicsPipeline.h"

#include <fstream>
#include <stdexcept>

namespace Engine::Render {

    PipelineColorTarget PipelineColorTarget::Opaque(VkFormat format) {
        PipelineColorTarget target{};
        target.format = format;
        target.blend.blendEnable = VK_FALSE;
        target.blend.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT |
                VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;
        return target;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::Shader(
            VkShaderStageFlagBits stage,
            const std::string &path,
            const std::string &entryPoint) {
        shaderStages.push_back({stage, path, entryPoint});
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::VertexShader(
            const std::string &path,
            const std::string &entryPoint) {
        return Shader(VK_SHADER_STAGE_VERTEX_BIT, path, entryPoint);
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::FragmentShader(
            const std::string &path,
            const std::string &entryPoint) {
        return Shader(VK_SHADER_STAGE_FRAGMENT_BIT, path, entryPoint);
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::VertexBinding(
            uint32_t binding,
            uint32_t stride,
            VkVertexInputRate inputRate) {
        VkVertexInputBindingDescription description{};
        description.binding = binding;
        description.stride = stride;
        description.inputRate = inputRate;
        vertexBindings.push_back(description);
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::VertexAttribute(
            uint32_t location,
            uint32_t binding,
            VkFormat format,
            uint32_t offset) {
        VkVertexInputAttributeDescription description{};
        description.location = location;
        description.binding = binding;
        description.format = format;
        description.offset = offset;
        vertexAttributes.push_back(description);
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::ColorTarget(VkFormat format) {
        return ColorTarget(PipelineColorTarget::Opaque(format));
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::ColorTarget(
            const PipelineColorTarget &target) {
        colorTargets.push_back(target);
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::DepthTarget(
            VkFormat format,
            bool writeDepth,
            VkCompareOp compareOp) {
        depthFormat = format;
        depthTestEnable = format != VK_FORMAT_UNDEFINED;
        depthWriteEnable = writeDepth;
        depthCompareOp = compareOp;
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::DepthBias(
            float constantFactor,
            float slopeFactor,
            float clamp) {
        depthBiasEnable = true;
        depthBiasConstantFactor = constantFactor;
        depthBiasSlopeFactor = slopeFactor;
        depthBiasClamp = clamp;
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::PushConstant(
            VkShaderStageFlags stageFlags,
            uint32_t size,
            uint32_t offset) {
        VkPushConstantRange range{};
        range.stageFlags = stageFlags;
        range.offset = offset;
        range.size = size;
        pushConstantRanges.push_back(range);
        return *this;
    }

    GraphicsPipeline::GraphicsPipeline(Engine::Core::Context &context)
        : m_device(context.device) {}

    GraphicsPipeline::~GraphicsPipeline() {
        Destroy();
    }

    GraphicsPipeline &GraphicsPipeline::Build(
            const GraphicsPipelineDescriptor &descriptor) {
        if (descriptor.shaderStages.empty())
            throw std::runtime_error("GraphicsPipeline::Build requires at least one shader");

        Destroy();

        std::vector<VkShaderModule> shaderModules;
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
        shaderModules.reserve(descriptor.shaderStages.size());
        shaderStages.reserve(descriptor.shaderStages.size());

        auto destroyShaderModules = [&]() {
            for (VkShaderModule module: shaderModules)
                vkDestroyShaderModule(m_device, module, nullptr);
            shaderModules.clear();
        };

        try {
            for (const ShaderStageDescriptor &shader: descriptor.shaderStages) {
                VkShaderModule module = CreateShaderModule(shader.path);
                shaderModules.push_back(module);

                VkPipelineShaderStageCreateInfo stage{};
                stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                stage.stage = shader.stage;
                stage.module = module;
                stage.pName = shader.entryPoint.c_str();
                shaderStages.push_back(stage);
            }

            VkPipelineVertexInputStateCreateInfo vertexInput{};
            vertexInput.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInput.vertexBindingDescriptionCount =
                    static_cast<uint32_t>(descriptor.vertexBindings.size());
            vertexInput.pVertexBindingDescriptions =
                    descriptor.vertexBindings.empty() ? nullptr
                                                      : descriptor.vertexBindings.data();
            vertexInput.vertexAttributeDescriptionCount =
                    static_cast<uint32_t>(descriptor.vertexAttributes.size());
            vertexInput.pVertexAttributeDescriptions =
                    descriptor.vertexAttributes.empty() ? nullptr
                                                        : descriptor.vertexAttributes.data();

            VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
            inputAssembly.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.topology = descriptor.topology;

            VkPipelineViewportStateCreateInfo viewportState{};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rasterizer{};
            rasterizer.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.polygonMode = descriptor.polygonMode;
            rasterizer.cullMode = descriptor.cullMode;
            rasterizer.frontFace = descriptor.frontFace;
            rasterizer.lineWidth = 1.0f;
            rasterizer.depthBiasEnable = descriptor.depthBiasEnable ? VK_TRUE : VK_FALSE;
            rasterizer.depthBiasConstantFactor = descriptor.depthBiasConstantFactor;
            rasterizer.depthBiasClamp = descriptor.depthBiasClamp;
            rasterizer.depthBiasSlopeFactor = descriptor.depthBiasSlopeFactor;

            VkPipelineMultisampleStateCreateInfo multisampling{};
            multisampling.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.rasterizationSamples = descriptor.samples;

            VkPipelineDepthStencilStateCreateInfo depthStencil{};
            depthStencil.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = descriptor.depthTestEnable ? VK_TRUE : VK_FALSE;
            depthStencil.depthWriteEnable = descriptor.depthWriteEnable ? VK_TRUE : VK_FALSE;
            depthStencil.depthCompareOp = descriptor.depthCompareOp;

            std::vector<VkPipelineColorBlendAttachmentState> blendAttachments;
            blendAttachments.reserve(descriptor.colorTargets.size());
            m_colorFormats.clear();
            m_colorFormats.reserve(descriptor.colorTargets.size());
            for (const PipelineColorTarget &target: descriptor.colorTargets) {
                blendAttachments.push_back(target.blend);
                m_colorFormats.push_back(target.format);
            }

            VkPipelineColorBlendStateCreateInfo colorBlending{};
            colorBlending.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.attachmentCount =
                    static_cast<uint32_t>(blendAttachments.size());
            colorBlending.pAttachments =
                    blendAttachments.empty() ? nullptr : blendAttachments.data();

            VkPipelineDynamicStateCreateInfo dynamicState{};
            dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamicState.dynamicStateCount =
                    static_cast<uint32_t>(descriptor.dynamicStates.size());
            dynamicState.pDynamicStates =
                    descriptor.dynamicStates.empty() ? nullptr
                                                     : descriptor.dynamicStates.data();

            VkPipelineLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layoutInfo.setLayoutCount =
                    static_cast<uint32_t>(descriptor.descriptorSetLayouts.size());
            layoutInfo.pSetLayouts =
                    descriptor.descriptorSetLayouts.empty()
                            ? nullptr
                            : descriptor.descriptorSetLayouts.data();
            layoutInfo.pushConstantRangeCount =
                    static_cast<uint32_t>(descriptor.pushConstantRanges.size());
            layoutInfo.pPushConstantRanges =
                    descriptor.pushConstantRanges.empty()
                            ? nullptr
                            : descriptor.pushConstantRanges.data();
            if (vkCreatePipelineLayout(m_device, &layoutInfo,
                                       nullptr, &m_layout) != VK_SUCCESS)
                throw std::runtime_error("GraphicsPipeline: failed to create pipeline layout");

            VkPipelineRenderingCreateInfo renderingInfo{};
            renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            renderingInfo.colorAttachmentCount =
                    static_cast<uint32_t>(m_colorFormats.size());
            renderingInfo.pColorAttachmentFormats =
                    m_colorFormats.empty() ? nullptr : m_colorFormats.data();
            renderingInfo.depthAttachmentFormat = descriptor.depthFormat;

            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.pNext = &renderingInfo;
            pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
            pipelineInfo.pStages = shaderStages.data();
            pipelineInfo.pVertexInputState = &vertexInput;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = &depthStencil;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = m_layout;

            if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1,
                                          &pipelineInfo, nullptr,
                                          &m_pipeline) != VK_SUCCESS)
                throw std::runtime_error("GraphicsPipeline: failed to create graphics pipeline");

            m_depthFormat = descriptor.depthFormat;
            destroyShaderModules();
        } catch (...) {
            destroyShaderModules();
            Destroy();
            throw;
        }

        return *this;
    }

    void GraphicsPipeline::Destroy() {
        if (m_pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
        if (m_layout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(m_device, m_layout, nullptr);
        m_pipeline = VK_NULL_HANDLE;
        m_layout = VK_NULL_HANDLE;
        m_colorFormats.clear();
        m_depthFormat = VK_FORMAT_UNDEFINED;
    }

    void GraphicsPipeline::Bind(VkCommandBuffer commandBuffer) const {
        if (m_pipeline == VK_NULL_HANDLE)
            throw std::runtime_error("GraphicsPipeline::Bind called before Build");
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    }

    VkFormat GraphicsPipeline::ColorFormat(uint32_t index) const {
        if (index >= m_colorFormats.size())
            return VK_FORMAT_UNDEFINED;
        return m_colorFormats[index];
    }

    bool GraphicsPipeline::MatchesColorTarget(uint32_t index, VkFormat format) const {
        return ColorFormat(index) == format;
    }

    std::vector<uint32_t> GraphicsPipeline::LoadSPIRV(const std::string &path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file)
            throw std::runtime_error("GraphicsPipeline: failed to open shader " + path);

        const std::streamsize size = file.tellg();
        if (size <= 0 || size % 4 != 0)
            throw std::runtime_error("GraphicsPipeline: invalid shader bytecode " + path);

        std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(code.data()), size);
        return code;
    }

    VkShaderModule GraphicsPipeline::CreateShaderModule(
            const std::string &path) const {
        const std::vector<uint32_t> code = LoadSPIRV(path);

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size() * sizeof(uint32_t);
        createInfo.pCode = code.data();

        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(m_device, &createInfo, nullptr, &module) != VK_SUCCESS)
            throw std::runtime_error("GraphicsPipeline: failed to create shader module " + path);
        return module;
    }

} // namespace Engine::Render
```

- [ ] **Step 3: Build**

Run:
```bash
cmake --build build --target EngineRender --parallel
```
Expected: builds with no errors.

- [ ] **Step 4: Run the existing test suite as a regression check**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: same 50/51 baseline.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Render/GraphicsPipeline.h src/Engine/Render/GraphicsPipeline.cpp
git commit -m "Port GraphicsPipeline into Engine::Render"
```

---

### Task 4: `example2/cube_render.cpp`

**Files:**
- Create: `example2/CMakeLists.txt`
- Create: `example2/cube_render.cpp`
- Modify: `CMakeLists.txt` (root)

**Interfaces:**
- Consumes: `Engine::Core::Context`/`Buffer`/`Image`/`QueueRole` (already built), `Engine::Render::SwapChain`/`GraphicsPipeline`/`GraphicsPipelineDescriptor`/`RenderingDescriptor`/`RenderingScope`/`ClearOptions` (Tasks 1-3), `utilities/Math.h` (`vkMath`, unchanged), `utilities/SimpleResource.h` (`SimpleResource::CreateCube`, unchanged), the existing `example/cube.vert`/`example/cube.frag` shader sources (unchanged, reused not copied).

- [ ] **Step 1: Create `example2/CMakeLists.txt`**

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
            "${CMAKE_SOURCE_DIR}/src/utilities/SimpleResource.cpp"
            "${CMAKE_SOURCE_DIR}/src/utilities/Math.cpp")
    add_dependencies(cube_render2 cube_render2_shaders)
    target_link_libraries(cube_render2 PRIVATE Engine::Render Engine::Core glfw)
    target_include_directories(cube_render2 PRIVATE /opt/homebrew/opt/eigen/include/eigen3)
    target_compile_definitions(cube_render2 PRIVATE
            CUBE_RENDER2_SHADER_DIR=\"${CUBE_RENDER2_SHADER_OUTPUT_DIR}\")
else ()
    message(WARNING "cube_render2 example skipped because glslc was not found")
endif ()
```

(Mirrors `example/CMakeLists.txt`'s `cube_render` target exactly, except the shader `add_custom_command`s point at `${CMAKE_SOURCE_DIR}/example/cube.vert`/`cube.frag` — the existing files — instead of `${CMAKE_CURRENT_SOURCE_DIR}` in this new directory, since those files aren't duplicated here. `Engine::Render` already brings in `Engine::Core` transitively via its own `PUBLIC` link, but linking both explicitly here is harmless and clearer about what's used directly.

**Eigen include path is required and explicit here**: this target compiles `utilities/Math.cpp`/`SimpleResource.cpp` directly (as source files, same as `example/CMakeLists.txt`'s `cube_render` target does), and both `#include <Eigen/...>`. The old `cube_render` target gets Eigen's include path *transitively* through `vkRender::vkRender` → `vkSpatial::vkSpatial`, whose `target_include_directories` hardcodes `/opt/homebrew/opt/eigen/include/eigen3` (`src/CMakeLists.txt:47`). Neither `Engine::Core` nor `Engine::Render` link anything that provides this path — they have no Eigen dependency internally — so `cube_render2` must add it directly, using the same hardcoded path for consistency with the existing convention (not a portability improvement in scope for this task).)

- [ ] **Step 2: Add `example2` to the root `CMakeLists.txt`**

Find `add_subdirectory(example)` and add immediately after it:
```cmake
add_subdirectory(example2)
```

- [ ] **Step 3: Create `example2/cube_render.cpp`**

```cpp
#include "utilities/Math.h"
#include "utilities/SimpleResource.h"

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/Image.h"
#include "Engine/Render/GraphicsPipeline.h"
#include "Engine/Render/Rendering.h"
#include "Engine/Render/SwapChain.h"

#include <GLFW/glfw3.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    constexpr float kPi = 3.14159265358979323846f;

    struct Vertex {
        float position[3];
        float color[3];
    };

    struct PushConstants {
        vkMath::Mat4 mvp;
    };

    vkMath::Mat4 BuildViewProjection(VkExtent2D extent) {
        const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        const vkMath::Mat4 projection = vkMath::Perspective(60.0f * kPi / 180.0f, aspect, 0.1f, 100.0f);
        const vkMath::Mat4 view = vkMath::Translation(0.0f, 0.0f, -4.5f);
        return projection * view;
    }

    vkMath::Mat4 AnimatedModel(float timeSeconds) {
        return vkMath::RotationY(timeSeconds * 0.8f) * vkMath::RotationX(timeSeconds * 0.55f);
    }

} // namespace

int main() {
    if (!glfwInit()) {
        std::cerr << "cube_render2: failed to initialize GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(900, 700, "Engine::Render Cube", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        std::cerr << "cube_render2: failed to create window\n";
        return 1;
    }

    try {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions || glfwExtensionCount == 0)
            throw std::runtime_error("cube_render2: GLFW did not provide Vulkan extensions");
        std::vector<const char *> instanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        Engine::Core::Context context(
                true, instanceExtensions,
                [&](VkInstance instance) {
                    VkSurfaceKHR surface = VK_NULL_HANDLE;
                    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
                        throw std::runtime_error("cube_render2: failed to create window surface");
                    return surface;
                });

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

        Engine::Render::SwapChainDescriptor swapChainDescriptor{};
        swapChainDescriptor.width = static_cast<uint32_t>(framebufferWidth);
        swapChainDescriptor.height = static_cast<uint32_t>(framebufferHeight);
        Engine::Render::SwapChain swapChain(context, swapChainDescriptor);

        // Cube geometry
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
        const uint32_t indexCount = static_cast<uint32_t>(cube.indices.size());

        Engine::Core::Buffer vertexBuffer(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        const uint32_t vertexBytes = static_cast<uint32_t>(vertices.size() * sizeof(Vertex));
        vertexBuffer.Allocate(vertexBytes);
        vertexBuffer.Upload(vertices.data(), vertexBytes, Engine::Core::QueueRole::Graphics);

        Engine::Core::Buffer indexBuffer(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        const uint32_t indexBytes = static_cast<uint32_t>(cube.indices.size() * sizeof(uint32_t));
        indexBuffer.Allocate(indexBytes);
        indexBuffer.Upload(cube.indices.data(), indexBytes, Engine::Core::QueueRole::Graphics);

        Engine::Core::Image depthImage(context);
        depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(swapChain.Extent()));

        const std::string shaderDir = CUBE_RENDER2_SHADER_DIR;
        Engine::Render::GraphicsPipelineDescriptor pipelineDescriptor;
        pipelineDescriptor
                .VertexShader(shaderDir + "/cube.vert.spv")
                .FragmentShader(shaderDir + "/cube.frag.spv")
                .VertexBinding<Vertex>()
                .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position))
                .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
                .ColorTarget(swapChain.Format())
                .DepthTarget(VK_FORMAT_D32_SFLOAT)
                .PushConstant<PushConstants>(VK_SHADER_STAGE_VERTEX_BIT);

        Engine::Render::GraphicsPipeline pipeline(context);
        pipeline.Build(pipelineDescriptor);

        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlightFence = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &imageAvailable) != VK_SUCCESS ||
            vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &renderFinished) != VK_SUCCESS)
            throw std::runtime_error("cube_render2: failed to create semaphores");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(context.device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS)
            throw std::runtime_error("cube_render2: failed to create fence");

        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = context.graphicsCmdPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(context.device, &cmdAllocInfo, &cmd) != VK_SUCCESS)
            throw std::runtime_error("cube_render2: failed to allocate command buffer");

        // The whole frame loop is wrapped so that, if anything throws mid-loop (after a
        // vkQueueSubmit put GPU work in flight), we wait for the device to go idle before
        // unwinding into `context`'s destructor — Context::~Context() does not itself call
        // vkDeviceWaitIdle, so the caller must guarantee no in-flight work before it runs.
        try {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();

                glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
                if (framebufferWidth == 0 || framebufferHeight == 0) {
                    glfwWaitEvents();
                    continue;
                }

                const VkExtent2D currentExtent = swapChain.Extent();
                if (currentExtent.width != static_cast<uint32_t>(framebufferWidth) ||
                    currentExtent.height != static_cast<uint32_t>(framebufferHeight)) {
                    vkDeviceWaitIdle(context.device);
                    swapChain.Recreate(static_cast<uint32_t>(framebufferWidth),
                                       static_cast<uint32_t>(framebufferHeight));
                    depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(swapChain.Extent()));
                    continue;
                }

                vkWaitForFences(context.device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);

                uint32_t imageIndex = 0;
                VkResult acquireResult = swapChain.AcquireNextImage(imageAvailable, VK_NULL_HANDLE, &imageIndex);
                if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
                    vkDeviceWaitIdle(context.device);
                    swapChain.Recreate(static_cast<uint32_t>(framebufferWidth),
                                       static_cast<uint32_t>(framebufferHeight));
                    depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(swapChain.Extent()));
                    continue;
                }
                if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR)
                    throw std::runtime_error("cube_render2: failed to acquire swapchain image");

                vkResetFences(context.device, 1, &inFlightFence);
                vkResetCommandBuffer(cmd, 0);

                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                vkBeginCommandBuffer(cmd, &beginInfo);

                VkImage swapImage = swapChain.Image(imageIndex);
                Engine::Core::Image::TransitionLayout(
                        cmd, swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
                depthImage.TransitionLayout(
                        cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                        0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

                Engine::Render::ClearOptions clear{};
                clear.color[0] = 0.025f;
                clear.color[1] = 0.027f;
                clear.color[2] = 0.032f;
                clear.color[3] = 1.0f;
                auto renderingDescriptor = Engine::Render::RenderingDescriptor::ColorDepth(
                        swapChain.Extent(), swapChain.ImageView(imageIndex), depthImage.View(), clear);

                {
                    Engine::Render::RenderingScope scope(cmd, renderingDescriptor);

                    pipeline.Bind(cmd);
                    VkBuffer vertexHandle = vertexBuffer.Handle();
                    VkDeviceSize vertexOffset = 0;
                    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexHandle, &vertexOffset);
                    vkCmdBindIndexBuffer(cmd, indexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

                    PushConstants push{
                            BuildViewProjection(swapChain.Extent()) *
                            AnimatedModel(static_cast<float>(glfwGetTime()))};
                    pipeline.PushConstants(cmd, VK_SHADER_STAGE_VERTEX_BIT, push);
                    vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
                }

                Engine::Core::Image::TransitionLayout(
                        cmd, swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0);

                vkEndCommandBuffer(cmd);

                VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                VkSubmitInfo submitInfo{};
                submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submitInfo.waitSemaphoreCount = 1;
                submitInfo.pWaitSemaphores = &imageAvailable;
                submitInfo.pWaitDstStageMask = &waitStage;
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &cmd;
                submitInfo.signalSemaphoreCount = 1;
                submitInfo.pSignalSemaphores = &renderFinished;
                if (vkQueueSubmit(context.graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS)
                    throw std::runtime_error("cube_render2: failed to submit frame");

                VkResult presentResult = swapChain.Present(imageIndex, renderFinished);
                if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
                    vkDeviceWaitIdle(context.device);
                    swapChain.Recreate(static_cast<uint32_t>(framebufferWidth),
                                       static_cast<uint32_t>(framebufferHeight));
                    depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(swapChain.Extent()));
                } else if (presentResult != VK_SUCCESS) {
                    throw std::runtime_error("cube_render2: failed to present frame");
                }
            }
        } catch (...) {
            vkDeviceWaitIdle(context.device);
            throw;
        }

        vkDeviceWaitIdle(context.device);

        vkDestroyFence(context.device, inFlightFence, nullptr);
        vkDestroySemaphore(context.device, renderFinished, nullptr);
        vkDestroySemaphore(context.device, imageAvailable, nullptr);
        // pipeline, depthImage, indexBuffer, vertexBuffer, swapChain, context all clean up
        // via their own destructors (RAII) in reverse declaration order as this scope ends.
    } catch (const std::exception &e) {
        glfwDestroyWindow(window);
        glfwTerminate();
        std::cerr << e.what() << "\n";
        return 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
```

Note what's different from the old `example/cube_render.cpp`: no manual `context.shutdown()` anywhere — `Engine::Core::Context` is RAII, so it cleans itself up when it goes out of scope, both on the normal path and (via the inner `try`/`catch`'s `vkDeviceWaitIdle` before rethrow) on the exception path. There is no `CubePass` class, no `RenderPass`/`RenderGraph`, no `Renderer` — the whole frame lifecycle is the `while` loop above.

- [ ] **Step 4: Build**

Run:
```bash
cmake -S . -B build
cmake --build build --target cube_render2 --parallel
```
Expected: builds with no errors. If `glslc` isn't found, the target won't exist and CMake will print the warning from Step 1 — this project already requires `glslc` for the existing `cube_render`/`realtime_shadow` examples, so if those built successfully before, `glslc` is present and this should too.

- [ ] **Step 5: Run and visually confirm**

Run:
```bash
./build/example2/cube_render2
```
Expected: a window opens titled "Engine::Render Cube" showing a rotating cube (blue-ish faces, gray floor... actually no floor in this minimal version, just the cube) on a dark background, with correct depth testing (faces occlude each other correctly). Resize the window and confirm it keeps rendering without crashing or validation errors. Close the window and confirm the process exits cleanly (no hang, no crash) — this specifically confirms the RAII teardown order and the inner try/catch's `vkDeviceWaitIdle` are correct.

- [ ] **Step 6: Run the existing test suite as a regression check**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: same 50/51 baseline — this task doesn't touch anything the test suite covers, this just confirms the overall build didn't regress.

- [ ] **Step 7: Commit**

```bash
git add example2/ CMakeLists.txt
git commit -m "Add example2/cube_render.cpp using Engine::Core + Engine::Render"
```

---

### Task 5: Full verification pass

**Files:**
- None expected to change (verification-only), unless a real regression is found and fixed.

**Interfaces:**
- Consumes: all of Tasks 1-4.

- [ ] **Step 1: Full clean rebuild**

Run:
```bash
rm -rf build
cmake -S . -B build --fresh
cmake --build build --parallel
```
Expected: every target builds with no errors, including all pre-existing targets alongside the new `EngineRender` and `cube_render2`.

- [ ] **Step 2: Run the full test suite**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: 50/51 passing, same pre-existing `WideBVHTest.RadiusMatchesCpuReference` failure, no new ones.

- [ ] **Step 3: Run `cube_render2` once more end-to-end**

Run:
```bash
./build/example2/cube_render2
```
Expected: same as Task 4 Step 5 — window opens, cube renders and rotates, resize works, clean close. This is the one thing worth re-confirming after a full clean rebuild specifically (shader paths, `CUBE_RENDER2_SHADER_DIR` compile definition, and the `glslc` custom-command dependency chain all get exercised fresh here).

- [ ] **Step 4: Confirm no existing file was touched beyond the two allowed edits**

Run:
```bash
git diff --stat main -- src/vkCommon src/vkSpatial src/vkRender src/Engine/Core example test/test_bvh* test/test_vkCompute.cpp test/test_wideBVH.cpp test/test_engineCore.cpp
```
Expected: empty output. (The two allowed edits — `src/Engine/CMakeLists.txt` and root `CMakeLists.txt` — are intentionally excluded from this check since they're explicitly in scope per the Global Constraints.)

- [ ] **Step 5: Commit any fixes found during verification, or confirm none needed**

If Steps 1-4 surfaced no issues, there's nothing to commit — this task is verification-only. If an issue was found and fixed, commit it with a message describing exactly what broke and how it was fixed, referencing which step caught it.

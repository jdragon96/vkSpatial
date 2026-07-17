# Engine/Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/Engine/Core/` — a new, standalone foundation layer (`Context`, `OneShotCommands`, `Buffer`, `Image`, `ComputePipeline`) that will eventually replace `vkCommon`, backed by vk-bootstrap (already proven in `vkCommon::VkContext`) and VMA (new).

**Architecture:** Five classes in namespace `Engine::Core`, each in its own header/source pair under `src/Engine/Core/`, built as a new static library target `EngineCore`. `Context` owns the device/queues/allocator everything else depends on. `OneShotCommands` provides a single shared "record → submit → wait" helper that `Buffer::Upload/Download` and `ComputePipeline::Dispatch` both use, replacing several independent copies of the same pattern found across the existing codebase. `Buffer` and `Image` are VMA-backed replacements for `vkGPUMemory` and `vkRender::Image` respectively. `ComputePipeline` is `vkComputeBase` relocated with a simplified constructor.

**Tech Stack:** C++17, Vulkan 1.3, vk-bootstrap (already integrated), VMA (new), GoogleTest.

## Global Constraints

- This plan creates **only new files**. `vkCommon`, `vkSpatial`, `vkRender`, and every existing `example/`/`test/` file are untouched — `EngineCore` is a new, currently-unconsumed library. Migrating existing consumers onto it is explicitly out of scope (future spec).
- All error handling in the new code throws `std::runtime_error` with a `"<ClassName>: <message>"`-prefixed message — no `bool`-returning fallible methods (this is a deliberate departure from `vkGPUMemory`'s old convention, matching the rest of the codebase).
- `Engine::Core::Context` is RAII: construction performs what `VkContext::init()` used to do, destruction performs what `VkContext::shutdown()` used to do. No separate `init()`/`shutdown()` methods.
- `vkb::` (vk-bootstrap) types never appear in any `Engine::Core` header — only in `.cpp` files, matching the pattern already proven in `vkCommon/vkContext.cpp`. `Vma` types (`VmaAllocator`, `VmaAllocation`) **do** appear in headers where needed (`Context.h`'s `allocator` field, `Buffer.h`/`Image.h`'s private members) since VMA's types are meant to be used this way (unlike vk-bootstrap's builder types, which are single-use construction helpers).
- `vk-bootstrap` and VMA both link `PRIVATE` to the `EngineCore` CMake target.
- Every new class's public queue-role-taking methods (`Buffer::Upload/Download`, and `ComputePipeline`'s internal dispatch) route their one-shot GPU submission through `Engine::Core::SubmitOneShot` — no class hand-rolls its own command-buffer-allocate-submit-wait sequence.

---

### Task 1: Vendor VMA and create the `EngineCore` CMake target

**Files:**
- Create: `lib/vma/vk_mem_alloc.h`
- Create: `src/Engine/CMakeLists.txt`
- Create: `src/Engine/Core/VmaImplementation.cpp`
- Modify: `src/CMakeLists.txt`

**Interfaces:**
- Produces: a buildable `EngineCore` static library CMake target (alias `Engine::Core`) with VMA's implementation compiled in. Nothing depends on it yet — this task's only deliverable is "the target exists and builds."

- [ ] **Step 1: Download the VMA single header**

Run:
```bash
mkdir -p lib/vma
curl -fsSL -o lib/vma/vk_mem_alloc.h \
    https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/master/include/vk_mem_alloc.h
```
Expected: the command succeeds (no `curl: (22)` or similar error) and `lib/vma/vk_mem_alloc.h` is created. Verify:
```bash
grep -c "vmaCreateAllocator" lib/vma/vk_mem_alloc.h
```
Expected: a positive count (at least 1) — confirms the downloaded file is really the VMA header and not an HTML error page. If the count is 0, the download failed silently (e.g. GitHub redirected to an error page) — stop and report BLOCKED rather than proceeding with a bad file.

- [ ] **Step 2: Create `src/Engine/Core/VmaImplementation.cpp`**

```cpp
#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"
```

This is the one, and only, translation unit in the whole project that defines `VMA_IMPLEMENTATION` (mirrors `Capture.cpp`'s existing `#define STB_IMAGE_WRITE_IMPLEMENTATION` pattern for `stb_image_write.h`).

- [ ] **Step 3: Create `src/Engine/CMakeLists.txt`**

```cmake
file(GLOB_RECURSE ENGINE_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Core/*.cpp")

add_library(EngineCore STATIC ${ENGINE_CORE_SOURCES})
add_library(Engine::Core ALIAS EngineCore)

target_link_libraries(EngineCore
        PUBLIC  Vulkan::Vulkan
        PRIVATE ${SHADERC_LIB} spirv-reflect vk-bootstrap::vk-bootstrap)

target_include_directories(EngineCore
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
        PRIVATE
            "${VULKAN_SDK}/include")

target_compile_definitions(EngineCore PRIVATE
        VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")
```

Note: `${SHADERC_LIB}` and `${VKBVH_SHADER_DIR}` are variables already defined earlier in `src/CMakeLists.txt` (used by the existing `vkCommon`/`vkSpatial` targets) — since `add_subdirectory(Engine)` is called from within `src/CMakeLists.txt` (Step 4 below), these variables are already in scope and don't need to be redefined.

- [ ] **Step 4: Wire `Engine` into `src/CMakeLists.txt`**

Open `src/CMakeLists.txt`. Find the line `find_library(SHADERC_LIB shaderc_combined HINTS "${VULKAN_SDK}/lib")` (this defines `SHADERC_LIB`, needed by Step 3's target). Immediately after the `VKBVH_SHADER_DIR` cache variable is set (a few lines below `SHADERC_LIB`), add:
```cmake
add_subdirectory(Engine)
```
This must come after both `SHADERC_LIB` and `VKBVH_SHADER_DIR` are defined (Step 3's `CMakeLists.txt` uses both), and can come before or after the existing `vkCommon`/`vkSpatial`/`vkRender` target definitions since `EngineCore` doesn't depend on any of them yet.

- [ ] **Step 5: Build**

Run:
```bash
cmake -S . -B build && cmake --build build --target EngineCore --parallel
```
Expected: builds with no errors. This compiles `VmaImplementation.cpp` (which pulls in all of `vk_mem_alloc.h`'s implementation) — if there are any compiler errors here, they're almost always missing Vulkan headers or a too-old compiler; report the exact error if this fails rather than guessing a fix.

- [ ] **Step 6: Run the existing test suite as a regression check**

Run:
```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
Expected: same 44/45 baseline as before this task (`WideBVHTest.RadiusMatchesCpuReference` is the one known pre-existing, unrelated failure) — this task adds a new unused library, so nothing else should change.

- [ ] **Step 7: Commit**

```bash
git add lib/vma src/Engine/CMakeLists.txt src/Engine/Core/VmaImplementation.cpp src/CMakeLists.txt
git commit -m "Vendor VMA and create the EngineCore CMake target"
```

---

### Task 2: `Engine::Core::Context`

**Files:**
- Create: `src/Engine/Core/Context.h`
- Create: `src/Engine/Core/Context.cpp`
- Create: `test/test_engineCore.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `vk-bootstrap` (via `#include "VkBootstrap.h"` in `Context.cpp` only), VMA (`vmaCreateAllocator`/`vmaDestroyAllocator`).
- Produces, for later tasks to consume:
  ```cpp
  namespace Engine::Core {
      class Context {
      public:
          VkInstance instance = VK_NULL_HANDLE;
          VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
          VkDevice device = VK_NULL_HANDLE;
          VkQueue computeQueue = VK_NULL_HANDLE;
          uint32_t computeFamily = 0;
          VkCommandPool cmdPool = VK_NULL_HANDLE;
          VkQueue graphicsQueue = VK_NULL_HANDLE;
          uint32_t graphicsFamily = 0;
          VkCommandPool graphicsCmdPool = VK_NULL_HANDLE;
          VkSurfaceKHR surface = VK_NULL_HANDLE;
          VmaAllocator allocator = VK_NULL_HANDLE;

          explicit Context(bool enablePresent = false,
                           const std::vector<const char *> &extraInstanceExtensions = {},
                           const std::function<VkSurfaceKHR(VkInstance)> &surfaceFactory = nullptr);
          ~Context();
          Context(const Context &) = delete;
          Context &operator=(const Context &) = delete;
      };
  }
  ```

- [ ] **Step 1: Write `Context.h`**

```cpp
#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <functional>
#include <vector>

namespace Engine::Core {

    class Context {
    public:
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;

        VkQueue computeQueue = VK_NULL_HANDLE;
        uint32_t computeFamily = 0;
        VkCommandPool cmdPool = VK_NULL_HANDLE;

        // Populated only when enablePresent=true.
        VkQueue graphicsQueue = VK_NULL_HANDLE;
        uint32_t graphicsFamily = 0;
        VkCommandPool graphicsCmdPool = VK_NULL_HANDLE;
        VkSurfaceKHR surface = VK_NULL_HANDLE;

        VmaAllocator allocator = VK_NULL_HANDLE;

        explicit Context(bool enablePresent = false,
                         const std::vector<const char *> &extraInstanceExtensions = {},
                         const std::function<VkSurfaceKHR(VkInstance)> &surfaceFactory = nullptr);
        ~Context();

        Context(const Context &) = delete;
        Context &operator=(const Context &) = delete;

    private:
        bool m_presentEnabled = false;

        void createCommandPools();
        void createAllocator();
    };

} // namespace Engine::Core
```

- [ ] **Step 2: Write `Context.cpp`**

```cpp
#include "Engine/Core/Context.h"

#include "VkBootstrap.h"

#include <iostream>
#include <stdexcept>

namespace Engine::Core {

    Context::Context(bool enablePresent,
                     const std::vector<const char *> &extraInstanceExtensions,
                     const std::function<VkSurfaceKHR(VkInstance)> &surfaceFactory) {
        m_presentEnabled = enablePresent;

        vkb::InstanceBuilder instanceBuilder;
        instanceBuilder.set_app_name("vkbvh")
                       .require_api_version(1, 3, 0);

        if (enablePresent)
            for (const char *ext : extraInstanceExtensions)
                instanceBuilder.enable_extension(ext);

        // This vk-bootstrap version has no InstanceBuilder::enable_extension_if_present().
        // Its InstanceBuilder::build() already auto-detects and enables
        // VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME (and sets
        // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR) internally when the extension is
        // supported, so no explicit call is needed here (verified against the fetched
        // vk-bootstrap source and against vkCommon::VkContext's identical, already-working
        // implementation).

        auto instRet = instanceBuilder.build();
        if (!instRet)
            throw std::runtime_error(
                    "Context: failed to create instance: " + instRet.error().message());
        vkb::Instance vkbInstance = instRet.value();
        instance = vkbInstance.instance;

        if (enablePresent) {
            if (!surfaceFactory)
                throw std::runtime_error("Context: enablePresent requires a surfaceFactory");
            surface = surfaceFactory(instance);
            if (surface == VK_NULL_HANDLE)
                throw std::runtime_error("Context: surfaceFactory returned VK_NULL_HANDLE");
        }

        vkb::PhysicalDeviceSelector selector(vkbInstance, surface);
        selector.set_minimum_version(1, 3)
                .require_present(enablePresent);

        VkPhysicalDeviceVulkan13Features features13{};
        features13.dynamicRendering = VK_TRUE;
        if (enablePresent) {
            selector.add_required_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            selector.set_required_features_13(features13);
        }

        auto physRet = selector.select();
        if (!physRet)
            throw std::runtime_error(
                    "Context: failed to select physical device: " + physRet.error().message());
        vkb::PhysicalDevice vkbPhysDevice = physRet.value();
        physicalDevice = vkbPhysDevice.physical_device;

        std::cout << "[Engine::Core::Context] Device: " << vkbPhysDevice.properties.deviceName << "\n";

        vkb::DeviceBuilder deviceBuilder(vkbPhysDevice);
        auto devRet = deviceBuilder.build();
        if (!devRet)
            throw std::runtime_error(
                    "Context: failed to create device: " + devRet.error().message());
        vkb::Device vkbDevice = devRet.value();
        device = vkbDevice.device;

        auto computeQueueRet = vkbDevice.get_queue(vkb::QueueType::compute);
        auto computeFamilyRet = vkbDevice.get_queue_index(vkb::QueueType::compute);
        if (!computeQueueRet || !computeFamilyRet)
            throw std::runtime_error("Context: no compute queue available");
        computeQueue = computeQueueRet.value();
        computeFamily = computeFamilyRet.value();

        if (enablePresent) {
            auto presentQueueRet = vkbDevice.get_queue(vkb::QueueType::present);
            auto presentFamilyRet = vkbDevice.get_queue_index(vkb::QueueType::present);
            if (!presentQueueRet || !presentFamilyRet)
                throw std::runtime_error("Context: no present-capable graphics queue available");
            uint32_t presentFamilyIndex = presentFamilyRet.value();
            if (presentFamilyIndex >= vkbDevice.queue_families.size() ||
                !(vkbDevice.queue_families[presentFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT))
                throw std::runtime_error(
                        "Context: present-capable queue family does not support graphics");
            graphicsQueue = presentQueueRet.value();
            graphicsFamily = presentFamilyIndex;
        }

        createCommandPools();
        createAllocator();
    }

    Context::~Context() {
        if (allocator != VK_NULL_HANDLE) vmaDestroyAllocator(allocator);
        if (graphicsCmdPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, graphicsCmdPool, nullptr);
        if (cmdPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, cmdPool, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    }

    void Context::createCommandPools() {
        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = computeFamily;
        if (vkCreateCommandPool(device, &pci, nullptr, &cmdPool) != VK_SUCCESS)
            throw std::runtime_error("Context: failed to create compute VkCommandPool");

        if (m_presentEnabled) {
            pci.queueFamilyIndex = graphicsFamily;
            if (vkCreateCommandPool(device, &pci, nullptr, &graphicsCmdPool) != VK_SUCCESS)
                throw std::runtime_error("Context: failed to create graphics VkCommandPool");
        }
    }

    void Context::createAllocator() {
        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.instance = instance;
        allocatorInfo.physicalDevice = physicalDevice;
        allocatorInfo.device = device;
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

        if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS)
            throw std::runtime_error("Context: failed to create VmaAllocator");
    }

} // namespace Engine::Core
```

- [ ] **Step 3: Write the failing test**

Create `test/test_engineCore.cpp`:
```cpp
#include <gtest/gtest.h>

#include "Engine/Core/Context.h"

using namespace Engine::Core;

TEST(ContextTest, ComputeOnlyConstructionProducesValidHandles) {
    Context ctx;  // enablePresent=false by default

    EXPECT_NE(ctx.instance, VK_NULL_HANDLE);
    EXPECT_NE(ctx.physicalDevice, VK_NULL_HANDLE);
    EXPECT_NE(ctx.device, VK_NULL_HANDLE);
    EXPECT_NE(ctx.computeQueue, VK_NULL_HANDLE);
    EXPECT_NE(ctx.cmdPool, VK_NULL_HANDLE);
    EXPECT_NE(ctx.allocator, VK_NULL_HANDLE);

    EXPECT_EQ(ctx.graphicsQueue, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.graphicsCmdPool, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.surface, VK_NULL_HANDLE);
}
```

- [ ] **Step 4: Add the test file to `test/CMakeLists.txt` and link `EngineCore`**

Open `test/CMakeLists.txt`. Find:
```cmake
target_link_libraries(vkspatial_tests
        PRIVATE
        vkSpatial::vkSpatial
        GTest::gtest
        GTest::gtest_main)
```
Replace with:
```cmake
target_link_libraries(vkspatial_tests
        PRIVATE
        vkSpatial::vkSpatial
        Engine::Core
        GTest::gtest
        GTest::gtest_main)
```
`file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")` already picks up the new `test_engineCore.cpp` automatically — no other change needed. (The `VKBVH_SHADER_DIR` compile definition is already set on `vkspatial_tests`, harmless/unused by this specific test but needed later once `ComputePipelineTest` is added in Task 6.)

- [ ] **Step 5: Build and run — confirm the test passes**

Run:
```bash
cmake -S . -B build
cmake --build build --target vkspatial_tests --parallel
ctest --test-dir build --output-on-failure -R ContextTest
```
Expected: `PASSED`.

- [ ] **Step 6: Run the full suite as a regression check**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: 45/46 passing now (44/45 previous baseline + 1 new `ContextTest`), with `WideBVHTest.RadiusMatchesCpuReference` as the only failure.

- [ ] **Step 7: Commit**

```bash
git add src/Engine/Core/Context.h src/Engine/Core/Context.cpp test/test_engineCore.cpp test/CMakeLists.txt
git commit -m "Add Engine::Core::Context (vk-bootstrap + VMA allocator, RAII)"
```

---

### Task 3: `Engine::Core::OneShotCommands`

**Files:**
- Create: `src/Engine/Core/OneShotCommands.h`
- Create: `src/Engine/Core/OneShotCommands.cpp`
- Modify: `test/test_engineCore.cpp`

**Interfaces:**
- Consumes: `Engine::Core::Context` from Task 2.
- Produces, for Tasks 4 and 6 to consume:
  ```cpp
  namespace Engine::Core {
      enum class QueueRole { Compute, Graphics };
      void SubmitOneShot(Context &context, QueueRole role,
                         const std::function<void(VkCommandBuffer)> &record);
  }
  ```

- [ ] **Step 1: Write `OneShotCommands.h`**

```cpp
#pragma once

#include "Engine/Core/Context.h"

#include <functional>
#include <vulkan/vulkan.h>

namespace Engine::Core {

    enum class QueueRole {
        Compute,
        Graphics,
    };

    // Allocates a transient command buffer from the command pool matching `role`
    // (context.cmdPool for Compute, context.graphicsCmdPool for Graphics), begins it,
    // invokes `record(cmd)` to fill it in, then ends, submits, and blocks until the
    // corresponding queue is idle before freeing the command buffer. Throws
    // std::runtime_error on any Vulkan failure or if QueueRole::Graphics is requested
    // on a Context that wasn't constructed with enablePresent=true.
    void SubmitOneShot(Context &context,
                       QueueRole role,
                       const std::function<void(VkCommandBuffer)> &record);

} // namespace Engine::Core
```

- [ ] **Step 2: Write `OneShotCommands.cpp`**

```cpp
#include "Engine/Core/OneShotCommands.h"

#include <stdexcept>

namespace Engine::Core {

    void SubmitOneShot(Context &context,
                       QueueRole role,
                       const std::function<void(VkCommandBuffer)> &record) {
        VkQueue queue = (role == QueueRole::Compute) ? context.computeQueue : context.graphicsQueue;
        VkCommandPool pool = (role == QueueRole::Compute) ? context.cmdPool : context.graphicsCmdPool;

        if (queue == VK_NULL_HANDLE || pool == VK_NULL_HANDLE)
            throw std::runtime_error(
                    "SubmitOneShot: requested QueueRole is unavailable on this Context "
                    "(Graphics requires enablePresent=true)");

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(context.device, &allocInfo, &cmd) != VK_SUCCESS)
            throw std::runtime_error("SubmitOneShot: failed to allocate command buffer");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
            vkFreeCommandBuffers(context.device, pool, 1, &cmd);
            throw std::runtime_error("SubmitOneShot: failed to begin command buffer");
        }

        record(cmd);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            vkFreeCommandBuffers(context.device, pool, 1, &cmd);
            throw std::runtime_error("SubmitOneShot: failed to end command buffer");
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            vkFreeCommandBuffers(context.device, pool, 1, &cmd);
            throw std::runtime_error("SubmitOneShot: failed to submit command buffer");
        }

        vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(context.device, pool, 1, &cmd);
    }

} // namespace Engine::Core
```

- [ ] **Step 3: Write the failing test**

Append to `test/test_engineCore.cpp` (add `#include "Engine/Core/OneShotCommands.h"` to the top of the file alongside the existing include):

```cpp
TEST(OneShotCommandsTest, FillBufferRoundTripsThroughSubmit) {
    Context ctx;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = 256;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo{};
    ASSERT_EQ(vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, &allocationInfo),
              VK_SUCCESS);

    // Zero the buffer up front so the post-fill check below is meaningful.
    std::memset(allocationInfo.pMappedData, 0xFF, 256);

    SubmitOneShot(ctx, QueueRole::Compute, [&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, buffer, 0, 256, 0x2A2A2A2Au);
    });

    const uint32_t *data = static_cast<const uint32_t *>(allocationInfo.pMappedData);
    for (int i = 0; i < 64; ++i)
        EXPECT_EQ(data[i], 0x2A2A2A2Au) << "word " << i;

    vmaDestroyBuffer(ctx.allocator, buffer, allocation);
}
```

This test deliberately doesn't depend on `Buffer` (Task 4, not yet implemented) — it exercises `SubmitOneShot` directly against a raw VMA-allocated buffer with `vkCmdFillBuffer`, so it's a genuine, self-contained verification that command recording/submission/waiting actually works, not just that the code compiles.

- [ ] **Step 4: Build and run — confirm both tests pass**

Run:
```bash
cmake --build build --target vkspatial_tests --parallel
ctest --test-dir build --output-on-failure -R "ContextTest|OneShotCommandsTest"
```
Expected: both `PASSED`.

- [ ] **Step 5: Run the full suite as a regression check**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: 46/47 passing (previous 45/46 + 1 new `OneShotCommandsTest`), same lone pre-existing failure.

- [ ] **Step 6: Commit**

```bash
git add src/Engine/Core/OneShotCommands.h src/Engine/Core/OneShotCommands.cpp test/test_engineCore.cpp
git commit -m "Add Engine::Core::OneShotCommands"
```

---

### Task 4: `Engine::Core::Buffer`

**Files:**
- Create: `src/Engine/Core/Buffer.h`
- Create: `src/Engine/Core/Buffer.cpp`
- Modify: `test/test_engineCore.cpp`

**Interfaces:**
- Consumes: `Engine::Core::Context` (Task 2), `Engine::Core::SubmitOneShot`/`QueueRole` (Task 3).
- Produces, for Task 6 to consume:
  ```cpp
  namespace Engine::Core {
      class Buffer {
      public:
          explicit Buffer(Context &context, VkBufferUsageFlags extraUsage = 0);
          ~Buffer();
          void Allocate(uint32_t bytes);
          void Upload(const void *data, uint32_t bytes, QueueRole role = QueueRole::Compute);
          void Download(void *data, uint32_t bytes, QueueRole role = QueueRole::Compute);
          VkBuffer Handle() const;
          uint32_t Size() const;
      };
  }
  ```

- [ ] **Step 1: Write `Buffer.h`**

```cpp
#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Core/OneShotCommands.h"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace Engine::Core {

    class Buffer {
    public:
        explicit Buffer(Context &context, VkBufferUsageFlags extraUsage = 0);
        ~Buffer();

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;

        // (Re)allocates a device-local buffer of `bytes` size. Any prior allocation is freed
        // first. Throws std::runtime_error on failure.
        void Allocate(uint32_t bytes);

        // Uploads `bytes` from `data` via a temporary host-visible staging buffer and a
        // one-shot GPU copy on `role`'s queue. Throws if `bytes` exceeds the current
        // allocation.
        void Upload(const void *data, uint32_t bytes, QueueRole role = QueueRole::Compute);

        // Downloads `bytes` into `data` via a temporary host-visible staging buffer and a
        // one-shot GPU copy on `role`'s queue. Throws if `bytes` exceeds the current
        // allocation.
        void Download(void *data, uint32_t bytes, QueueRole role = QueueRole::Compute);

        VkBuffer Handle() const { return m_buffer; }
        uint32_t Size() const { return m_size; }

    private:
        Context &m_context;
        VkBufferUsageFlags m_extraUsage;
        VkBuffer m_buffer = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;
        uint32_t m_size = 0;

        void free();
    };

} // namespace Engine::Core
```

- [ ] **Step 2: Write `Buffer.cpp`**

```cpp
#include "Engine/Core/Buffer.h"

#include <stdexcept>

namespace Engine::Core {

    Buffer::Buffer(Context &context, VkBufferUsageFlags extraUsage)
        : m_context(context), m_extraUsage(extraUsage) {}

    Buffer::~Buffer() {
        free();
    }

    void Buffer::free() {
        if (m_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_context.allocator, m_buffer, m_allocation);
            m_buffer = VK_NULL_HANDLE;
            m_allocation = VK_NULL_HANDLE;
        }
        m_size = 0;
    }

    void Buffer::Allocate(uint32_t bytes) {
        free();

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bytes;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           m_extraUsage;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

        if (vmaCreateBuffer(m_context.allocator, &bufferInfo, &allocInfo,
                            &m_buffer, &m_allocation, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Buffer: failed to allocate");

        m_size = bytes;
    }

    void Buffer::Upload(const void *data, uint32_t bytes, QueueRole role) {
        if (m_buffer == VK_NULL_HANDLE || bytes > m_size)
            throw std::runtime_error("Buffer::Upload: not allocated or bytes exceeds capacity");

        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = bytes;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                 VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingAllocation = VK_NULL_HANDLE;
        VmaAllocationInfo stagingAllocationInfo{};
        if (vmaCreateBuffer(m_context.allocator, &stagingInfo, &stagingAllocInfo,
                            &stagingBuffer, &stagingAllocation, &stagingAllocationInfo) != VK_SUCCESS)
            throw std::runtime_error("Buffer::Upload: failed to create staging buffer");

        std::memcpy(stagingAllocationInfo.pMappedData, data, bytes);

        VkBuffer dstBuffer = m_buffer;
        SubmitOneShot(m_context, role, [&](VkCommandBuffer cmd) {
            VkBufferCopy region{};
            region.size = bytes;
            vkCmdCopyBuffer(cmd, stagingBuffer, dstBuffer, 1, &region);
        });

        vmaDestroyBuffer(m_context.allocator, stagingBuffer, stagingAllocation);
    }

    void Buffer::Download(void *data, uint32_t bytes, QueueRole role) {
        if (m_buffer == VK_NULL_HANDLE || bytes > m_size)
            throw std::runtime_error("Buffer::Download: not allocated or bytes exceeds capacity");

        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = bytes;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                 VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingAllocation = VK_NULL_HANDLE;
        VmaAllocationInfo stagingAllocationInfo{};
        if (vmaCreateBuffer(m_context.allocator, &stagingInfo, &stagingAllocInfo,
                            &stagingBuffer, &stagingAllocation, &stagingAllocationInfo) != VK_SUCCESS)
            throw std::runtime_error("Buffer::Download: failed to create staging buffer");

        VkBuffer srcBuffer = m_buffer;
        SubmitOneShot(m_context, role, [&](VkCommandBuffer cmd) {
            VkBufferCopy region{};
            region.size = bytes;
            vkCmdCopyBuffer(cmd, srcBuffer, stagingBuffer, 1, &region);
        });

        std::memcpy(data, stagingAllocationInfo.pMappedData, bytes);

        vmaDestroyBuffer(m_context.allocator, stagingBuffer, stagingAllocation);
    }

} // namespace Engine::Core
```

Add `#include <cstring>` to the top of `Buffer.cpp` alongside `<stdexcept>` (needed for `std::memcpy`).

- [ ] **Step 3: Write the failing test**

Append to `test/test_engineCore.cpp` (add `#include "Engine/Core/Buffer.h"` to the includes, and `#include <numeric>` and `#include <vector>` if not already present):

```cpp
TEST(BufferTest, AllocateUploadDownloadRoundTrip) {
    Context ctx;
    Buffer buffer(ctx);

    constexpr uint32_t kCount = 1024;
    std::vector<float> source(kCount);
    std::iota(source.begin(), source.end(), 0.0f);

    buffer.Allocate(kCount * sizeof(float));
    buffer.Upload(source.data(), kCount * sizeof(float));

    std::vector<float> result(kCount, -1.0f);
    buffer.Download(result.data(), kCount * sizeof(float));

    for (uint32_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(result[i], source[i]) << "index " << i;
}

TEST(BufferTest, DownloadBeyondCapacityThrows) {
    Context ctx;
    Buffer buffer(ctx);
    buffer.Allocate(16);

    std::vector<uint8_t> dst(64);
    EXPECT_THROW(buffer.Download(dst.data(), 64), std::runtime_error);
}
```

- [ ] **Step 4: Build and run — confirm all `Engine::Core` tests pass**

Run:
```bash
cmake --build build --target vkspatial_tests --parallel
ctest --test-dir build --output-on-failure -R "ContextTest|OneShotCommandsTest|BufferTest"
```
Expected: all `PASSED`.

- [ ] **Step 5: Run the full suite as a regression check**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: 48/49 passing (previous 46/47 + 2 new `BufferTest` cases), same lone pre-existing failure.

- [ ] **Step 6: Commit**

```bash
git add src/Engine/Core/Buffer.h src/Engine/Core/Buffer.cpp test/test_engineCore.cpp
git commit -m "Add Engine::Core::Buffer (VMA-backed, replaces vkGPUMemory)"
```

---

### Task 5: `Engine::Core::Image`

**Files:**
- Create: `src/Engine/Core/Image.h`
- Create: `src/Engine/Core/Image.cpp`
- Modify: `test/test_engineCore.cpp`

**Interfaces:**
- Consumes: `Engine::Core::Context` (Task 2).
- Produces:
  ```cpp
  namespace Engine::Core {
      struct ImageDescriptor { /* see below */
          static ImageDescriptor Depth2D(VkExtent2D extent, VkFormat format = VK_FORMAT_D32_SFLOAT);
          static ImageDescriptor Color2D(VkExtent2D extent, VkFormat format,
                                         VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
      };
      class Image { /* Create/Destroy/Handle/View/Format/AspectMask/CurrentLayout/Extent/Extent2D/Descriptor/Valid/HasView/Matches/TransitionLayout — see below */ };
  }
  ```

- [ ] **Step 1: Write `Image.h`**

```cpp
#pragma once

#include "Engine/Core/Context.h"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace Engine::Core {

    struct ImageDescriptor {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t depth = 1;
        uint32_t mipLevels = 1;
        uint32_t arrayLayers = 1;

        VkFormat format = VK_FORMAT_UNDEFINED;
        VkImageUsageFlags usage = 0;
        VkImageAspectFlags aspectMask = 0;
        VkImageType imageType = VK_IMAGE_TYPE_2D;
        VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
        VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bool createView = true;

        static ImageDescriptor Depth2D(VkExtent2D extent,
                                       VkFormat format = VK_FORMAT_D32_SFLOAT);
        static ImageDescriptor Color2D(VkExtent2D extent,
                                       VkFormat format,
                                       VkImageUsageFlags usage =
                                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                               VK_IMAGE_USAGE_SAMPLED_BIT);
    };

    class Image {
    public:
        explicit Image(Context &context);
        Image(Context &context, const ImageDescriptor &descriptor);
        ~Image();

        Image(const Image &) = delete;
        Image &operator=(const Image &) = delete;
        Image(Image &&rhs) noexcept;
        Image &operator=(Image &&rhs) noexcept;

        Image &Create(const ImageDescriptor &descriptor);
        void Destroy();

        bool Valid() const { return m_image != VK_NULL_HANDLE; }
        bool HasView() const { return m_view != VK_NULL_HANDLE; }
        bool Matches(VkExtent2D extent, VkFormat format = VK_FORMAT_UNDEFINED) const;

        VkImage Handle() const { return m_image; }
        VkImageView View() const { return m_view; }
        VkFormat Format() const { return m_descriptor.format; }
        VkImageAspectFlags AspectMask() const { return m_descriptor.aspectMask; }
        VkImageLayout CurrentLayout() const { return m_currentLayout; }
        VkExtent3D Extent() const { return {m_descriptor.width, m_descriptor.height, m_descriptor.depth}; }
        VkExtent2D Extent2D() const { return {m_descriptor.width, m_descriptor.height}; }
        const ImageDescriptor &Descriptor() const { return m_descriptor; }

        void TransitionLayout(VkCommandBuffer cmd,
                              VkImageLayout newLayout,
                              VkPipelineStageFlags srcStage,
                              VkPipelineStageFlags dstStage,
                              VkAccessFlags srcAccess,
                              VkAccessFlags dstAccess);

        static void TransitionLayout(VkCommandBuffer cmd,
                                     VkImage image,
                                     VkImageAspectFlags aspectMask,
                                     VkImageLayout oldLayout,
                                     VkImageLayout newLayout,
                                     VkPipelineStageFlags srcStage,
                                     VkPipelineStageFlags dstStage,
                                     VkAccessFlags srcAccess,
                                     VkAccessFlags dstAccess);

    private:
        Context *m_context = nullptr;
        VkImage m_image = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;
        VkImageView m_view = VK_NULL_HANDLE;
        ImageDescriptor m_descriptor{};
        VkImageLayout m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        void createImageAndAllocation();
        void createImageView();
        void moveFrom(Image &&rhs) noexcept;
    };

} // namespace Engine::Core
```

Note this is identical in shape to `vkRender::Image` except: `VkDeviceMemory m_memory` is replaced by `VmaAllocation m_allocation`, there is no `FindMemoryType` (VMA replaces it), and `ImageDescriptor` drops the `memoryProperties` field (VMA's `VMA_MEMORY_USAGE_AUTO` picks it automatically).

- [ ] **Step 2: Write `Image.cpp`**

```cpp
#include "Engine/Core/Image.h"

#include <stdexcept>
#include <utility>

namespace Engine::Core {

    ImageDescriptor ImageDescriptor::Depth2D(VkExtent2D extent, VkFormat format) {
        ImageDescriptor descriptor{};
        descriptor.width = extent.width;
        descriptor.height = extent.height;
        descriptor.format = format;
        descriptor.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        descriptor.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        return descriptor;
    }

    ImageDescriptor ImageDescriptor::Color2D(VkExtent2D extent, VkFormat format, VkImageUsageFlags usage) {
        ImageDescriptor descriptor{};
        descriptor.width = extent.width;
        descriptor.height = extent.height;
        descriptor.format = format;
        descriptor.usage = usage;
        descriptor.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        return descriptor;
    }

    Image::Image(Context &context) : m_context(&context) {}

    Image::Image(Context &context, const ImageDescriptor &descriptor) : Image(context) {
        Create(descriptor);
    }

    Image::~Image() {
        Destroy();
    }

    Image::Image(Image &&rhs) noexcept {
        moveFrom(std::move(rhs));
    }

    Image &Image::operator=(Image &&rhs) noexcept {
        if (this != &rhs) {
            Destroy();
            moveFrom(std::move(rhs));
        }
        return *this;
    }

    Image &Image::Create(const ImageDescriptor &descriptor) {
        if (descriptor.width == 0 || descriptor.height == 0 || descriptor.depth == 0)
            throw std::runtime_error("Image::Create requires a non-empty extent");
        if (descriptor.format == VK_FORMAT_UNDEFINED)
            throw std::runtime_error("Image::Create requires a valid VkFormat");
        if (descriptor.usage == 0)
            throw std::runtime_error("Image::Create requires VkImageUsageFlags");
        if (descriptor.createView && descriptor.aspectMask == 0)
            throw std::runtime_error("Image::Create view requires aspectMask");

        Destroy();
        m_descriptor = descriptor;

        try {
            createImageAndAllocation();
            if (m_descriptor.createView)
                createImageView();
        } catch (...) {
            Destroy();
            throw;
        }

        m_currentLayout = m_descriptor.initialLayout;
        return *this;
    }

    void Image::Destroy() {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            return;

        if (m_view != VK_NULL_HANDLE)
            vkDestroyImageView(m_context->device, m_view, nullptr);
        if (m_image != VK_NULL_HANDLE)
            vmaDestroyImage(m_context->allocator, m_image, m_allocation);

        m_view = VK_NULL_HANDLE;
        m_image = VK_NULL_HANDLE;
        m_allocation = VK_NULL_HANDLE;
        m_descriptor = {};
        m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void Image::TransitionLayout(VkCommandBuffer cmd,
                                 VkImageLayout newLayout,
                                 VkPipelineStageFlags srcStage,
                                 VkPipelineStageFlags dstStage,
                                 VkAccessFlags srcAccess,
                                 VkAccessFlags dstAccess) {
        if (m_currentLayout == newLayout)
            return;

        TransitionLayout(cmd, m_image, m_descriptor.aspectMask,
                         m_currentLayout, newLayout,
                         srcStage, dstStage, srcAccess, dstAccess);
        m_currentLayout = newLayout;
    }

    void Image::TransitionLayout(VkCommandBuffer cmd,
                                 VkImage image,
                                 VkImageAspectFlags aspectMask,
                                 VkImageLayout oldLayout,
                                 VkImageLayout newLayout,
                                 VkPipelineStageFlags srcStage,
                                 VkPipelineStageFlags dstStage,
                                 VkAccessFlags srcAccess,
                                 VkAccessFlags dstAccess) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {aspectMask, 0, 1, 0, 1};
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    bool Image::Matches(VkExtent2D extent, VkFormat format) const {
        if (!Valid())
            return false;
        if (m_descriptor.width != extent.width || m_descriptor.height != extent.height)
            return false;
        return format == VK_FORMAT_UNDEFINED || m_descriptor.format == format;
    }

    void Image::createImageAndAllocation() {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = m_descriptor.imageType;
        imageInfo.format = m_descriptor.format;
        imageInfo.extent = {m_descriptor.width, m_descriptor.height, m_descriptor.depth};
        imageInfo.mipLevels = m_descriptor.mipLevels;
        imageInfo.arrayLayers = m_descriptor.arrayLayers;
        imageInfo.samples = m_descriptor.samples;
        imageInfo.tiling = m_descriptor.tiling;
        imageInfo.usage = m_descriptor.usage;
        imageInfo.sharingMode = m_descriptor.sharingMode;
        imageInfo.initialLayout = m_descriptor.initialLayout;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

        if (vmaCreateImage(m_context->allocator, &imageInfo, &allocInfo,
                           &m_image, &m_allocation, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Image: failed to create image");
    }

    void Image::createImageView() {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_image;
        viewInfo.viewType = m_descriptor.viewType;
        viewInfo.format = m_descriptor.format;
        viewInfo.subresourceRange.aspectMask = m_descriptor.aspectMask;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = m_descriptor.mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = m_descriptor.arrayLayers;

        if (vkCreateImageView(m_context->device, &viewInfo, nullptr, &m_view) != VK_SUCCESS)
            throw std::runtime_error("Image: failed to create image view");
    }

    void Image::moveFrom(Image &&rhs) noexcept {
        m_context = rhs.m_context;
        m_image = rhs.m_image;
        m_allocation = rhs.m_allocation;
        m_view = rhs.m_view;
        m_descriptor = rhs.m_descriptor;
        m_currentLayout = rhs.m_currentLayout;

        rhs.m_context = nullptr;
        rhs.m_image = VK_NULL_HANDLE;
        rhs.m_allocation = VK_NULL_HANDLE;
        rhs.m_view = VK_NULL_HANDLE;
        rhs.m_descriptor = {};
        rhs.m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

} // namespace Engine::Core
```

- [ ] **Step 3: Write the failing test**

Append to `test/test_engineCore.cpp` (add `#include "Engine/Core/Image.h"`):

```cpp
TEST(ImageTest, CreateDepth2DProducesValidHandles) {
    Context ctx;
    Image image(ctx, ImageDescriptor::Depth2D({256, 256}));

    EXPECT_TRUE(image.Valid());
    EXPECT_TRUE(image.HasView());
    EXPECT_EQ(image.Format(), VK_FORMAT_D32_SFLOAT);
    EXPECT_EQ(image.AspectMask(), static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT));
    EXPECT_TRUE(image.Matches({256, 256}, VK_FORMAT_D32_SFLOAT));
    EXPECT_FALSE(image.Matches({512, 512}));
}
```

- [ ] **Step 4: Build and run — confirm all `Engine::Core` tests pass**

Run:
```bash
cmake --build build --target vkspatial_tests --parallel
ctest --test-dir build --output-on-failure -R "ContextTest|OneShotCommandsTest|BufferTest|ImageTest"
```
Expected: all `PASSED`.

- [ ] **Step 5: Run the full suite as a regression check**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: 49/50 passing, same lone pre-existing failure.

- [ ] **Step 6: Commit**

```bash
git add src/Engine/Core/Image.h src/Engine/Core/Image.cpp test/test_engineCore.cpp
git commit -m "Add Engine::Core::Image (VMA-backed, replaces vkRender::Image's allocation)"
```

---

### Task 6: `Engine::Core::ComputePipeline`

**Files:**
- Create: `src/Engine/Core/ComputePipeline.h`
- Create: `src/Engine/Core/ComputePipeline.cpp`
- Modify: `test/test_engineCore.cpp`

**Interfaces:**
- Consumes: `Engine::Core::Context` (Task 2), `Engine::Core::SubmitOneShot`/`QueueRole` (Task 3), `Engine::Core::Buffer` (Task 4, for the `Bind(binding, Buffer&)` convenience overload).
- Produces: `Engine::Core::ComputePipeline`, same fluent surface as `vkCommon::vkComputeBase` (`AddInclude`, `Build` ×2, `Bind` ×2, `Args<T>`, `Dispatch` ×2, `DispatchElements`, `Sync`, `GetLocalSize`), constructed from `Context&` instead of 4 raw handles, with its internal one-shot submission routed through `SubmitOneShot`.

- [ ] **Step 1: Write `ComputePipeline.h`**

```cpp
#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/OneShotCommands.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Core {

    enum class ShaderInput {
        SpvFile,
        GlslSrc,
    };

    class ComputePipeline {
    public:
        explicit ComputePipeline(Context &context);
        ~ComputePipeline();

        ComputePipeline(const ComputePipeline &) = delete;
        ComputePipeline &operator=(const ComputePipeline &) = delete;

        ComputePipeline &AddInclude(const std::string &name, const std::string &src);

        ComputePipeline &Build(const std::string &source, ShaderInput inputType);
        ComputePipeline &Build(const std::string &path);

        ComputePipeline &Bind(uint32_t binding, VkBuffer buffer, VkDeviceSize sizeBytes);

        ComputePipeline &Bind(uint32_t binding, Buffer &buffer) {
            return Bind(binding, buffer.Handle(), static_cast<VkDeviceSize>(buffer.Size()));
        }

        template<typename T>
        ComputePipeline &Args(const T &value) {
            static_assert(sizeof(T) <= 256, "push constant must be <= 256 bytes");
            m_pushData.resize(sizeof(T));
            std::memcpy(m_pushData.data(), &value, sizeof(T));
            return *this;
        }

        void Dispatch(VkExtent3D grid);
        void Dispatch(uint32_t gridX, uint32_t gridY = 1, uint32_t gridZ = 1);
        void DispatchElements(uint32_t numElements);

        // Retained for call-site compatibility with the vkComputeBase API this replaces.
        // Dispatch() already blocks via SubmitOneShot, so this is a no-op safety net, not
        // a required call.
        void Sync();

        VkExtent3D GetLocalSize() const { return m_localSize; }

    private:
        Context &m_context;

        VkShaderModule m_shaderModule = VK_NULL_HANDLE;
        VkDescriptorPool m_descPool = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_descLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_descSet = VK_NULL_HANDLE;
        VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_pipeline = VK_NULL_HANDLE;

        std::vector<uint8_t> m_pushData;

        struct BufferBinding {
            uint32_t binding;
            VkBuffer buffer;
            VkDeviceSize size;
        };
        std::vector<BufferBinding> m_bindings;
        bool m_dirty = true;

        VkExtent3D m_localSize{};

        std::unordered_map<std::string, std::string> m_includes;

        void destroyShaderResources();
        void ensurePipeline();
        void updateDescriptors();
        void submit(uint32_t gridX, uint32_t gridY, uint32_t gridZ);
        void reflectLocalSize(const std::vector<uint32_t> &spv);

        static std::vector<uint32_t> loadSPIRV(const std::string &path);
        std::vector<uint32_t> compileGlslToSpv(const std::string &src) const;
    };

} // namespace Engine::Core
```

- [ ] **Step 2: Write `ComputePipeline.cpp`**

This is `vkCommon::vkComputeBase`'s implementation (`src/vkCommon/vkComputeBase.cpp`) relocated with three changes: (1) namespace `vkCommon` → `Engine::Core`, class name `vkComputeBase` → `ComputePipeline`; (2) the constructor takes `Context&` instead of four raw handles, and every use of `m_device`/`m_physDevice`/`m_queue`/`m_cmdPool` becomes `m_context.device`/`.physicalDevice`/`.computeQueue`/`.cmdPool`; (3) `submit()`'s body is replaced to call `SubmitOneShot` instead of hand-rolling command-buffer allocate/submit/wait. Everything else (the two includer classes, `Build()` ×2, `reflectLocalSize`, `Bind`, `Dispatch`/`DispatchElements`, `ensurePipeline`, `updateDescriptors`, `loadSPIRV`, `compileGlslToSpv`) is copied verbatim.

```cpp
#include "Engine/Core/ComputePipeline.h"

#include "SPIRV-Reflect/spirv_reflect.h"
#include <fstream>
#include <shaderc/shaderc.hpp>
#include <sstream>
#include <stdexcept>

#ifndef VKBVH_SHADER_DIR
#define VKBVH_SHADER_DIR "."
#endif

namespace Engine::Core {

    class FilesystemIncluder : public shaderc::CompileOptions::IncluderInterface {
    public:
        explicit FilesystemIncluder(std::string dir) : m_dir(std::move(dir)) {}

        shaderc_include_result *GetInclude(const char *requested,
                                           shaderc_include_type,
                                           const char * /*requesting*/,
                                           size_t) override {
            std::string fullPath = m_dir + "/" + requested;
            std::ifstream f(fullPath, std::ios::binary);

            auto *r = new shaderc_include_result{};
            if (f.is_open()) {
                auto *content = new std::string(
                        std::istreambuf_iterator<char>(f),
                        std::istreambuf_iterator<char>());
                auto *name = new std::string(fullPath);
                r->source_name = name->c_str();
                r->source_name_length = name->size();
                r->content = content->c_str();
                r->content_length = content->size();
                r->user_data = new std::pair<std::string *, std::string *>(name, content);
            } else {
                static const char kErr[] = "file not found";
                r->source_name = "";
                r->source_name_length = 0;
                r->content = kErr;
                r->content_length = sizeof(kErr) - 1;
                r->user_data = nullptr;
            }
            return r;
        }

        void ReleaseInclude(shaderc_include_result *r) override {
            if (r->user_data) {
                auto *p = static_cast<std::pair<std::string *, std::string *> *>(r->user_data);
                delete p->first;
                delete p->second;
                delete p;
            }
            delete r;
        }

    private:
        std::string m_dir;
    };

    class InMemoryIncluder : public shaderc::CompileOptions::IncluderInterface {
    public:
        explicit InMemoryIncluder(const std::unordered_map<std::string, std::string> &srcs)
            : m_srcs(srcs) {}

        shaderc_include_result *GetInclude(const char *requested,
                                           shaderc_include_type,
                                           const char * /*requesting*/,
                                           size_t) override {
            auto it = m_srcs.find(requested);
            auto *r = new shaderc_include_result{};
            if (it != m_srcs.end()) {
                r->source_name = it->first.c_str();
                r->source_name_length = it->first.size();
                r->content = it->second.c_str();
                r->content_length = it->second.size();
            } else {
                static const char kErr[] = "include not found";
                r->source_name = "";
                r->source_name_length = 0;
                r->content = kErr;
                r->content_length = sizeof(kErr) - 1;
            }
            return r;
        }

        void ReleaseInclude(shaderc_include_result *r) override { delete r; }

    private:
        const std::unordered_map<std::string, std::string> &m_srcs;
    };

    ComputePipeline &ComputePipeline::AddInclude(const std::string &name, const std::string &src) {
        m_includes[name] = src;
        return *this;
    }

    std::vector<uint32_t> ComputePipeline::compileGlslToSpv(const std::string &src) const {
        shaderc::Compiler compiler;
        shaderc::CompileOptions opts;
        opts.SetOptimizationLevel(shaderc_optimization_level_performance);
        opts.SetIncluder(std::make_unique<InMemoryIncluder>(m_includes));

        auto result = compiler.CompileGlslToSpv(src, shaderc_compute_shader, "inline", opts);
        if (result.GetCompilationStatus() != shaderc_compilation_status_success)
            throw std::runtime_error("GLSL compile: " + result.GetErrorMessage());

        return std::vector<uint32_t>(result.cbegin(), result.cend());
    }

    ComputePipeline::ComputePipeline(Context &context) : m_context(context) {}

    void ComputePipeline::destroyShaderResources() {
        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_context.device, m_pipeline, nullptr);
            m_pipeline = VK_NULL_HANDLE;
        }
        if (m_pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_context.device, m_pipelineLayout, nullptr);
            m_pipelineLayout = VK_NULL_HANDLE;
        }
        if (m_descPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_context.device, m_descPool, nullptr);
            m_descPool = VK_NULL_HANDLE;
            m_descSet = VK_NULL_HANDLE;
        }
        if (m_descLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_context.device, m_descLayout, nullptr);
            m_descLayout = VK_NULL_HANDLE;
        }
        if (m_shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_context.device, m_shaderModule, nullptr);
            m_shaderModule = VK_NULL_HANDLE;
        }
        m_dirty = true;
    }

    ComputePipeline &ComputePipeline::Build(const std::string &filename) {
        destroyShaderResources();

        const std::string fullPath = std::string(VKBVH_SHADER_DIR) + "/" + filename;

        std::ifstream f(fullPath, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("ComputePipeline::Build: cannot open " + fullPath);
        std::ostringstream ss;
        ss << f.rdbuf();
        std::string src = ss.str();

        auto lastSlash = fullPath.rfind('/');
        std::string dir = (lastSlash != std::string::npos) ? fullPath.substr(0, lastSlash) : ".";

        shaderc::Compiler compiler;
        shaderc::CompileOptions opts;
        opts.SetOptimizationLevel(shaderc_optimization_level_performance);
        opts.SetIncluder(std::make_unique<FilesystemIncluder>(dir));

        auto result = compiler.CompileGlslToSpv(src, shaderc_compute_shader, fullPath.c_str(), opts);
        if (result.GetCompilationStatus() != shaderc_compilation_status_success)
            throw std::runtime_error("ComputePipeline::Build: " + result.GetErrorMessage());

        std::vector<uint32_t> spv(result.cbegin(), result.cend());

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size() * sizeof(uint32_t);
        smci.pCode = spv.data();
        if (vkCreateShaderModule(m_context.device, &smci, nullptr, &m_shaderModule) != VK_SUCCESS)
            throw std::runtime_error("ComputePipeline::Build: vkCreateShaderModule failed");

        reflectLocalSize(spv);
        return *this;
    }

    ComputePipeline &ComputePipeline::Build(const std::string &source, ShaderInput inputType) {
        destroyShaderResources();

        std::vector<uint32_t> spv = (inputType == ShaderInput::GlslSrc)
                                            ? compileGlslToSpv(source)
                                            : loadSPIRV(source);

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size() * sizeof(uint32_t);
        smci.pCode = spv.data();
        if (vkCreateShaderModule(m_context.device, &smci, nullptr, &m_shaderModule) != VK_SUCCESS)
            throw std::runtime_error("ComputePipeline::Build: vkCreateShaderModule failed");

        reflectLocalSize(spv);
        return *this;
    }

    void ComputePipeline::reflectLocalSize(const std::vector<uint32_t> &spv) {
        SpvReflectShaderModule module{};
        SpvReflectResult r = spvReflectCreateShaderModule(spv.size() * sizeof(uint32_t), spv.data(), &module);
        if (r != SPV_REFLECT_RESULT_SUCCESS) {
            fprintf(stderr, "[ComputePipeline] SPIRV-Reflect failed (code=%d), "
                            "local_size will be 0 — DispatchElements will throw\n",
                    static_cast<int>(r));
            return;
        }

        const SpvReflectEntryPoint *ep = spvReflectGetEntryPoint(&module, "main");
        if (ep) {
            m_localSize = {ep->local_size.x, ep->local_size.y, ep->local_size.z};
        } else {
            fprintf(stderr, "[ComputePipeline] SPIRV-Reflect: no 'main' entry point found\n");
        }

        spvReflectDestroyShaderModule(&module);
    }

    ComputePipeline::~ComputePipeline() {
        destroyShaderResources();
    }

    ComputePipeline &ComputePipeline::Bind(uint32_t binding, VkBuffer buffer, VkDeviceSize sizeBytes) {
        for (auto &b: m_bindings) {
            if (b.binding == binding) {
                b.buffer = buffer;
                b.size = sizeBytes;
                m_dirty = true;
                return *this;
            }
        }
        m_bindings.push_back({binding, buffer, sizeBytes});
        m_dirty = true;
        return *this;
    }

    void ComputePipeline::Dispatch(VkExtent3D grid) {
        Dispatch(grid.width, grid.height, grid.depth);
    }

    void ComputePipeline::Dispatch(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        ensurePipeline();
        if (m_dirty) updateDescriptors();
        submit(gridX, gridY, gridZ);
    }

    void ComputePipeline::DispatchElements(uint32_t numElements) {
        if (m_localSize.width == 0)
            throw std::runtime_error(
                    "ComputePipeline::DispatchElements: local_size.width=0 "
                    "(SPIR-V reflection failed — check Build() was called with valid shader)");
        uint32_t gridX = (numElements + m_localSize.width - 1) / m_localSize.width;
        Dispatch(gridX, 1, 1);
    }

    void ComputePipeline::Sync() {
        // Dispatch() already blocks via SubmitOneShot's vkQueueWaitIdle. Retained as a
        // no-op-equivalent call for API-shape compatibility with vkComputeBase.
        vkQueueWaitIdle(m_context.computeQueue);
    }

    void ComputePipeline::ensurePipeline() {
        if (m_pipeline != VK_NULL_HANDLE) return;

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (auto &b: m_bindings) {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = b.binding;
            lb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            lb.descriptorCount = 1;
            lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(lb);
        }

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(m_context.device, &dlci, nullptr, &m_descLayout) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreateDescriptorSetLayout failed");

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(bindings.size());

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(m_context.device, &dpci, nullptr, &m_descPool) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreateDescriptorPool failed");

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = m_descPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &m_descLayout;
        if (vkAllocateDescriptorSets(m_context.device, &dsai, &m_descSet) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkAllocateDescriptorSets failed");

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = m_pushData.empty() ? 0 : static_cast<uint32_t>(m_pushData.size());

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &m_descLayout;
        plci.pushConstantRangeCount = pcRange.size > 0 ? 1 : 0;
        plci.pPushConstantRanges = pcRange.size > 0 ? &pcRange : nullptr;
        if (vkCreatePipelineLayout(m_context.device, &plci, nullptr, &m_pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreatePipelineLayout failed");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = m_shaderModule;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = m_pipelineLayout;
        if (vkCreateComputePipelines(m_context.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreateComputePipelines failed");
    }

    void ComputePipeline::updateDescriptors() {
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorBufferInfo> bufInfos(m_bindings.size());

        for (size_t i = 0; i < m_bindings.size(); i++) {
            bufInfos[i].buffer = m_bindings[i].buffer;
            bufInfos[i].offset = 0;
            bufInfos[i].range = m_bindings[i].size;

            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = m_descSet;
            w.dstBinding = m_bindings[i].binding;
            w.dstArrayElement = 0;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &bufInfos[i];
            writes.push_back(w);
        }

        vkUpdateDescriptorSets(m_context.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        m_dirty = false;
    }

    void ComputePipeline::submit(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        VkPipeline pipeline = m_pipeline;
        VkPipelineLayout layout = m_pipelineLayout;
        VkDescriptorSet descSet = m_descSet;
        const std::vector<uint8_t> &pushData = m_pushData;

        SubmitOneShot(m_context, QueueRole::Compute, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);
            if (!pushData.empty())
                vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   static_cast<uint32_t>(pushData.size()), pushData.data());
            vkCmdDispatch(cmd, gridX, gridY, gridZ);
        });
    }

    std::vector<uint32_t> ComputePipeline::loadSPIRV(const std::string &path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("ComputePipeline: cannot open SPIR-V: " + path);

        size_t byteSize = static_cast<size_t>(file.tellg());
        if (byteSize % 4 != 0)
            throw std::runtime_error("ComputePipeline: SPIR-V size not 4-byte aligned");

        std::vector<uint32_t> code(byteSize / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char *>(code.data()), static_cast<std::streamsize>(byteSize));
        return code;
    }

} // namespace Engine::Core
```

- [ ] **Step 3: Write the failing test**

Append to `test/test_engineCore.cpp` (add `#include "Engine/Core/ComputePipeline.h"` and `#include <numeric>`). This reproduces `test/test_vkCompute.cpp`'s `SumZeroToTenThousand` test against the new type, reusing the same GLSL source pattern (an atomic-add reduction shader compiled inline via `ShaderInput::GlslSrc`, so it needs no new `.comp` file on disk):

```cpp
TEST(ComputePipelineTest, DispatchSimpleShaderProducesExpectedOutput) {
    Context ctx;

    constexpr uint32_t N = 10001u;  // sum(0..N-1) = 50,005,000
    std::vector<uint32_t> input(N);
    std::iota(input.begin(), input.end(), 0u);

    Buffer inputBuffer(ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    inputBuffer.Allocate(N * sizeof(uint32_t));
    inputBuffer.Upload(input.data(), N * sizeof(uint32_t));

    Buffer outputBuffer(ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    outputBuffer.Allocate(sizeof(uint32_t));
    uint32_t zero = 0;
    outputBuffer.Upload(&zero, sizeof(uint32_t));

    static const char *kShader = R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) readonly buffer InputBuf { uint values[]; } inputBuf;
        layout(binding = 1) buffer OutputBuf { uint total; } outputBuf;
        layout(push_constant) uniform PC { uint count; } pc;
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= pc.count) return;
            atomicAdd(outputBuf.total, inputBuf.values[idx]);
        }
    )";

    struct PushConstants { uint32_t count; };

    ComputePipeline pipeline(ctx);
    pipeline.Build(kShader, ShaderInput::GlslSrc)
            .Bind(0, inputBuffer)
            .Bind(1, outputBuffer)
            .Args(PushConstants{N})
            .DispatchElements(N);

    uint32_t result = 0;
    outputBuffer.Download(&result, sizeof(uint32_t));

    EXPECT_EQ(result, 50005000u);
}
```

- [ ] **Step 4: Build and run — confirm all `Engine::Core` tests pass**

Run:
```bash
cmake --build build --target vkspatial_tests --parallel
ctest --test-dir build --output-on-failure -R "ContextTest|OneShotCommandsTest|BufferTest|ImageTest|ComputePipelineTest"
```
Expected: all `PASSED`.

- [ ] **Step 5: Run the full suite as a regression check**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: 50/51 passing, same lone pre-existing failure.

- [ ] **Step 6: Commit**

```bash
git add src/Engine/Core/ComputePipeline.h src/Engine/Core/ComputePipeline.cpp test/test_engineCore.cpp
git commit -m "Add Engine::Core::ComputePipeline (replaces vkComputeBase)"
```

---

### Task 7: Full verification pass

**Files:**
- None expected to change (verification-only), unless a real regression is found and fixed.

**Interfaces:**
- Consumes: all of Tasks 1-6.

- [ ] **Step 1: Full clean rebuild**

Run:
```bash
rm -rf build
cmake -S . -B build --fresh
cmake --build build --parallel
```
Expected: every target builds with no errors, including all pre-existing `vkCommon`/`vkSpatial`/`vkRender`/example/test targets (untouched by this plan) alongside the new `EngineCore`.

- [ ] **Step 2: Run the full test suite**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: 51 tests total (44 pre-existing + `ContextTest`(1) + `OneShotCommandsTest`(1) + `BufferTest`(2) + `ImageTest`(1) + `ComputePipelineTest`(1) = 50 passing, 1 failing — the same pre-existing `WideBVHTest.RadiusMatchesCpuReference`, unrelated to this work).

- [ ] **Step 3: Confirm no existing file was touched**

Run:
```bash
git diff --stat main -- src/vkCommon src/vkSpatial src/vkRender example test/test_bvh* test/test_vkCompute.cpp test/test_wideBVH.cpp
```
Expected: empty output — this plan's Global Constraint ("only new files") holds. If anything shows up here, that's a real problem to investigate before proceeding, not something to wave through.

- [ ] **Step 4: Commit any fixes found during verification, or confirm none needed**

If Steps 1-3 surfaced no issues, there's nothing to commit — this task is verification-only. If an issue was found and fixed, commit it with a message describing exactly what broke and how it was fixed, referencing which step caught it.

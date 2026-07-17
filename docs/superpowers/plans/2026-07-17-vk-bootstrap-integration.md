# vk-bootstrap Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-rolled Vulkan instance/physical-device/device creation inside `vkCommon::VkContext` with [vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap), without changing anything any other module (`vkSpatial`, `vkRender`, examples, tests) observes.

**Architecture:** `VkContext::init()` currently calls four private helpers (`createInstance`, `pickPhysicalDevice`, `createDevice`, `createCommandPool`) that hand-roll `VkInstanceCreateInfo`/`VkDeviceCreateInfo`/manual queue-family scanning. vk-bootstrap's `InstanceBuilder` → `PhysicalDeviceSelector` → `DeviceBuilder` chain produces value types (`vkb::Instance`, `vkb::PhysicalDevice`, `vkb::Device`) that must flow through one another, so the three helpers collapse into a single new `init()` body. `createCommandPool()` is untouched (vk-bootstrap doesn't manage command pools).

**Tech Stack:** C++17, Vulkan 1.3, CMake `FetchContent`, vk-bootstrap, GoogleTest (existing `vkspatial_tests` target).

## Global Constraints

- `vkContext.h`'s PUBLIC fields (`instance`, `physDevice`, `device`, `computeQueue`, `computeFamily`, `cmdPool`, `graphicsQueue`, `graphicsFamily`, `surface`) and the `init()`/`shutdown()` signatures never change — every other module reads these directly.
- vk-bootstrap links `PRIVATE` to the `vkCommon` CMake target only — it must never propagate to `vkSpatial`, `vkRender`, `example`, or `test`.
- No validation layers / debug messenger are enabled in this work (matches current behavior; out of scope per the design spec).
- `createCommandPool()` is not touched.
- Physical device selection must not call `disable_portability_subset()` — the automatic `VK_KHR_portability_subset` enablement replaces the current manual `#ifdef __APPLE__` extension push.
- The present-capable queue is retrieved via `vkb::QueueType::present`, not `vkb::QueueType::graphics` — the current code's real requirement is "a queue that can present to this surface," which `present` expresses directly.
- **Deviation from the design spec** (`docs/superpowers/specs/2026-07-17-vk-bootstrap-integration-design.md`): that spec assumed `vkContext.h` would need zero changes. In practice, `vkb::PhysicalDeviceSelector` requires a `vkb::Instance`, and `vkb::DeviceBuilder` requires the `vkb::PhysicalDevice` the selector produced — these value types don't fit cleanly into three separate `void`-returning private methods without adding new private member state to carry them between calls. This plan instead removes the `createInstance`/`pickPhysicalDevice`/`createDevice` private method declarations and does the whole vk-bootstrap chain inline in `init()`. This only touches the *private* section of `vkContext.h` — no public field, method signature, or downstream-visible behavior changes.

---

### Task 1: Add vk-bootstrap as a build dependency (no behavior change)

**Files:**
- Modify: `CMakeLists.txt` (root)
- Modify: `src/CMakeLists.txt`
- Modify: `src/vkCommon/vkContext.cpp` (add include only)

**Interfaces:**
- Produces: a linkable `vk-bootstrap::vk-bootstrap` CMake target available to `vkCommon`, and a confirmed-working `#include "VkBootstrap.h"` in `vkContext.cpp`. No runtime behavior changes yet — this task only proves the dependency builds and links.

- [ ] **Step 1: Find the latest vk-bootstrap release tag**

Run:
```bash
git ls-remote --tags --refs https://github.com/charles-lunarg/vk-bootstrap | sort -t/ -k3 -V | tail -5
```
Expected: a list of `refs/tags/vX.Y.Z` lines. Take the highest version tag (last line) — call it `<VKB_TAG>` for the next step.

- [ ] **Step 2: Add FetchContent declaration to the root `CMakeLists.txt`**

Add after `find_package(glfw3 REQUIRED)` and before `add_subdirectory(lib)`:

```cmake
include(FetchContent)
FetchContent_Declare(
        fetch_vk_bootstrap
        GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
        GIT_TAG        <VKB_TAG>
)
FetchContent_MakeAvailable(fetch_vk_bootstrap)
```

Replace `<VKB_TAG>` with the tag found in Step 1 (e.g. `v1.3.301` — use the actual value you found, not this example).

- [ ] **Step 3: Link vk-bootstrap privately to the `vkCommon` target**

In `src/CMakeLists.txt`, find:
```cmake
target_link_libraries(vkCommon
        PUBLIC  Vulkan::Vulkan
        PRIVATE ${SHADERC_LIB} spirv-reflect)
```
Replace with:
```cmake
target_link_libraries(vkCommon
        PUBLIC  Vulkan::Vulkan
        PRIVATE ${SHADERC_LIB} spirv-reflect vk-bootstrap::vk-bootstrap)
```

- [ ] **Step 4: Add a no-op include to `vkContext.cpp` to prove the dependency compiles**

In `src/vkCommon/vkContext.cpp`, add after `#include "vkContext.h"`:
```cpp
#include "VkBootstrap.h"
```

- [ ] **Step 5: Rebuild the whole project from scratch**

Run:
```bash
rm -rf build
cmake -S . -B build --fresh
cmake --build build --parallel
```
Expected: build succeeds (this downloads and builds vk-bootstrap via `FetchContent`). No compile errors from the unused include (an unused-include is not a compile error in C++; if your compiler is set to `-Werror` on unused warnings for something else, that's unrelated to this change).

- [ ] **Step 6: Run the existing test suite to confirm nothing broke**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: all existing tests pass (same pass count as before this task — this task changes zero runtime behavior).

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/CMakeLists.txt src/vkCommon/vkContext.cpp
git commit -m "Add vk-bootstrap dependency via FetchContent (no behavior change yet)"
```

---

### Task 2: Add a characterization test for `VkContext::init()`

**Files:**
- Create: `test/test_vkContext.cpp`

**Interfaces:**
- Consumes: `vkCommon::VkContext` — public fields and `init()`/`shutdown()` exactly as declared in `src/vkCommon/vkContext.h` (unchanged by this task).
- Produces: a GTest test (`VkContextTest.ComputeOnlyInitProducesValidHandles`) that pins down `VkContext`'s current observable behavior on the compute-only path (`enablePresent=false`, the default). This is the safety net Task 3's refactor must keep green. `test/CMakeLists.txt` already does `file(GLOB TEST_SOURCES "*.cpp")`, so no CMake edit is needed — the new file is picked up automatically.

- [ ] **Step 1: Write the characterization test**

Create `test/test_vkContext.cpp`:
```cpp
#include <gtest/gtest.h>

#include "vkCommon/vkContext.h"

using namespace vkCommon;

TEST(VkContextTest, ComputeOnlyInitProducesValidHandles) {
    VkContext ctx;
    ctx.init();  // enablePresent=false by default

    EXPECT_NE(ctx.instance, VK_NULL_HANDLE);
    EXPECT_NE(ctx.physDevice, VK_NULL_HANDLE);
    EXPECT_NE(ctx.device, VK_NULL_HANDLE);
    EXPECT_NE(ctx.computeQueue, VK_NULL_HANDLE);
    EXPECT_NE(ctx.cmdPool, VK_NULL_HANDLE);

    // enablePresent=false일 때는 그래픽스/프레젠트 관련 필드가 채워지지 않는다.
    EXPECT_EQ(ctx.graphicsQueue, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.surface, VK_NULL_HANDLE);

    ctx.shutdown();

    // shutdown()이 실제로 해제하는 핸들만 검증한다 (computeQueue/physDevice는
    // shutdown()이 재설정하지 않는 값이라 여기서 단언하지 않는다 — vkContext.cpp:31-40 참고).
    EXPECT_EQ(ctx.instance, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.device, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.cmdPool, VK_NULL_HANDLE);
}
```

- [ ] **Step 2: Rebuild and run this test against the CURRENT (pre-refactor) implementation**

Run:
```bash
cmake --build build --parallel --target vkspatial_tests
ctest --test-dir build --output-on-failure -R VkContextTest
```
Expected: `PASSED` — this test must pass against today's hand-rolled `VkContext` implementation before you touch anything in Task 3. If it fails here, the test itself is wrong (fix the test, not `VkContext`) — Task 3 hasn't started yet.

- [ ] **Step 3: Commit**

```bash
git add test/test_vkContext.cpp
git commit -m "Add characterization test for VkContext::init() before vk-bootstrap refactor"
```

---

### Task 3: Replace instance/physical-device/device creation with vk-bootstrap

**Files:**
- Modify: `src/vkCommon/vkContext.h`
- Modify: `src/vkCommon/vkContext.cpp`

**Interfaces:**
- Consumes: `vk-bootstrap::vk-bootstrap` target from Task 1; `VkContextTest.ComputeOnlyInitProducesValidHandles` from Task 2 as the regression gate.
- Produces: `VkContext::init()`/`VkContext::shutdown()` with identical public signatures and identical observable postconditions, now implemented via `vkb::InstanceBuilder`/`vkb::PhysicalDeviceSelector`/`vkb::DeviceBuilder`. `createCommandPool()` keeps its exact current signature and body.

- [ ] **Step 1: Update the private section of `vkContext.h`**

In `src/vkCommon/vkContext.h`, replace:
```cpp
    private:
        void createInstance(bool enablePresent,
                            const std::vector<const char *> &extraInstanceExtensions);
        void pickPhysicalDevice();
        void createDevice(bool enablePresent);
        void createCommandPool();

        bool m_presentEnabled = false;
```
with:
```cpp
    private:
        void createCommandPool();

        bool m_presentEnabled = false;
```

- [ ] **Step 2: Rewrite `init()` and delete the three old private methods in `vkContext.cpp`**

Replace the entire `VkContext::init(...)`, `VkContext::createInstance(...)`, `VkContext::pickPhysicalDevice()`, and `VkContext::createDevice(...)` definitions (currently `vkContext.cpp:9-183`, i.e. everything between `shutdown()` and `createCommandPool()`) with:

```cpp
void VkContext::init(bool enablePresent,
                      const std::vector<const char *> &extraInstanceExtensions,
                      const std::function<VkSurfaceKHR(VkInstance)> &surfaceFactory) {
    m_presentEnabled = enablePresent;

    vkb::InstanceBuilder instanceBuilder;
    instanceBuilder.set_app_name("vkbvh")
                   .require_api_version(1, 3, 0);

    if (enablePresent)
        for (const char *ext : extraInstanceExtensions)
            instanceBuilder.enable_extension(ext);

#ifdef __APPLE__
    instanceBuilder.enable_extension_if_present(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

    auto instRet = instanceBuilder.build();
    if (!instRet)
        throw std::runtime_error(
                "VkContext: failed to create instance: " + instRet.error().message());
    vkb::Instance vkbInstance = instRet.value();
    instance = vkbInstance.instance;

    if (enablePresent) {
        if (!surfaceFactory)
            throw std::runtime_error(
                    "VkContext: enablePresent requires a surfaceFactory");
        surface = surfaceFactory(instance);
        if (surface == VK_NULL_HANDLE)
            throw std::runtime_error(
                    "VkContext: surfaceFactory returned VK_NULL_HANDLE");
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
                "VkContext: failed to select physical device: " + physRet.error().message());
    vkb::PhysicalDevice vkbPhysDevice = physRet.value();
    physDevice = vkbPhysDevice.physical_device;

    std::cout << "[VkContext] Device: " << vkbPhysDevice.properties.deviceName << "\n";

    vkb::DeviceBuilder deviceBuilder(vkbPhysDevice);
    auto devRet = deviceBuilder.build();
    if (!devRet)
        throw std::runtime_error(
                "VkContext: failed to create device: " + devRet.error().message());
    vkb::Device vkbDevice = devRet.value();
    device = vkbDevice.device;

    auto computeQueueRet = vkbDevice.get_queue(vkb::QueueType::compute);
    auto computeFamilyRet = vkbDevice.get_queue_index(vkb::QueueType::compute);
    if (!computeQueueRet || !computeFamilyRet)
        throw std::runtime_error("VkContext: no compute queue available");
    computeQueue = computeQueueRet.value();
    computeFamily = computeFamilyRet.value();

    if (enablePresent) {
        auto presentQueueRet = vkbDevice.get_queue(vkb::QueueType::present);
        auto presentFamilyRet = vkbDevice.get_queue_index(vkb::QueueType::present);
        if (!presentQueueRet || !presentFamilyRet)
            throw std::runtime_error("VkContext: no present-capable graphics queue available");
        graphicsQueue = presentQueueRet.value();
        graphicsFamily = presentFamilyRet.value();
    }

    createCommandPool();
}

void VkContext::shutdown() {
    if (cmdPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, cmdPool, nullptr);
    if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
    if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    cmdPool = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    surface = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
}
```

(`shutdown()` is reproduced verbatim, unchanged — it's included here only because it sits between the deleted methods and `createCommandPool()` in the file; don't modify its body.) Leave `VkContext::createCommandPool()` exactly as it is below this.

Make sure `#include "VkBootstrap.h"` (added in Task 1) is still present at the top of the file.

- [ ] **Step 3: Rebuild**

Run:
```bash
cmake --build build --parallel
```
Expected: build succeeds. If `vkb::PhysicalDeviceSelector`/`vkb::DeviceBuilder`/`vkb::QueueType` fail to resolve, confirm the vk-bootstrap version fetched in Task 1 is recent enough to expose `set_required_features_13` and `QueueType::present` (both used above) — check `build/_deps/fetch_vk_bootstrap-src/src/VkBootstrap.h` for these symbols if the build fails.

- [ ] **Step 4: Run the characterization test — it must still pass**

Run:
```bash
ctest --test-dir build --output-on-failure -R VkContextTest
```
Expected: `PASSED`. This proves the compute-only path (`enablePresent=false`) behaves identically after the refactor.

- [ ] **Step 5: Run the full existing test suite**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: all tests pass (same pass count as Task 1's baseline) — this exercises real GPU BVH/compute work through the new `VkContext`, not just the characterization test's field checks.

- [ ] **Step 6: Manually verify the present/windowed path**

Run:
```bash
cmake --build build --target cube_render --parallel
./build/example/cube_render
```
Expected: a window opens showing the rotating cube example rendering normally, with no validation errors printed to the console. This exercises `enablePresent=true` — GLFW surface extensions, swapchain creation, `VK_KHR_portability_subset` (on macOS), and the `dynamicRendering` feature — none of which the characterization test covers. Close the window when confirmed.

If the window fails to open or a MoltenVK/portability-related error is printed to the console (macOS only), check `build/_deps/fetch_vk_bootstrap-src/src/VkBootstrap.h` for how it treats `VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME` — some versions require the instance creation flag `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR` to be set explicitly even when the extension is enabled. If so, this is the one spot in this plan that may need an extra line calling whatever flag-setting method that vk-bootstrap version exposes (grep the header for `PORTABILITY` or `flags` in `InstanceBuilder`) — add it next to the existing `enable_extension_if_present` call in Step 2 above.

- [ ] **Step 7: Commit**

```bash
git add src/vkCommon/vkContext.h src/vkCommon/vkContext.cpp
git commit -m "Replace VkContext instance/device creation with vk-bootstrap"
```

---

### Task 4: Full verification pass and cleanup

**Files:**
- None expected to change (verification-only), unless Step 3 below surfaces a real regression to fix.

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: confidence that the whole project (all examples, all tests) behaves correctly with vk-bootstrap-backed `VkContext` before considering this migration done.

- [ ] **Step 1: Full clean rebuild of every target**

Run:
```bash
rm -rf build
cmake -S . -B build --fresh
cmake --build build --parallel
```
Expected: every target (all libraries, all examples, all tests) builds with no errors.

- [ ] **Step 2: Run the full test suite one more time**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: 100% pass, same test count as the pre-migration baseline from Task 1 Step 6 plus the one new `VkContextTest` from Task 2.

- [ ] **Step 3: Run at least two windowed examples end-to-end**

Run:
```bash
./build/example/cube_render
./build/example/realtime_shadow
```
Expected: both open a window and render correctly (cube spinning; shadow scene with moving light), matching their pre-migration appearance. `realtime_shadow` additionally exercises the Enter-key screenshot capture (`docs/superpowers/specs/2026-07-13-key-input-screenshot-design.md`) — press Enter in each and confirm a `screenshots/*.png`/`.ppm` file is still produced without error, since that feature depends on `VkContext`'s graphics queue/surface being set up correctly.

- [ ] **Step 4: Confirm the physical device selection log**

While running Step 3, check the console output for the `[VkContext] Device: <name>` line. Confirm it names a real, sensible GPU for this machine (on a single-GPU machine this is trivially the only option; on a multi-GPU machine, confirm it's the discrete GPU per the design spec's risk #2 — `docs/superpowers/specs/2026-07-17-vk-bootstrap-integration-design.md`).

- [ ] **Step 5: Commit any fixes found during verification, or confirm no changes needed**

If Steps 1-4 surfaced no issues, there's nothing to commit — this task is verification-only. If any issue was found and fixed, commit it with a message describing exactly what broke and how it was fixed, referencing which step caught it.

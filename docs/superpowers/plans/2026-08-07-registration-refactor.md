# Registration & Features Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate all registration code into one flat home (`src/Engine/Pipeline/Registration/`) with one file per algorithm and per tracker strategy, and extract the feature algorithms into a new `Engine::Features` module — without changing any behavior.

**Architecture:** A behavior-preserving reorg done in four sequenced tasks, each of which leaves the build and the full test suite green. Files move with `git mv`; `#include` paths, namespaces (feature files only), and CMake are updated to match; the existing ≈240-test suite is the acceptance gate at every task boundary.

**Tech Stack:** C++17, CMake (GLOB_RECURSE static libs), Eigen, Ceres, Vulkan (unchanged), GoogleTest.

## Global Constraints

- **Behavior-preserving.** No algorithm, threading, or public-name change. Tracker registry names stay exactly `identity` / `icp` / `icp-cpu` / `global`. The full suite must stay green (**≈237 pass / 1 skip**, run `./build/test/vkspatial_tests`) at the end of every task.
- **Namespaces preserved except feature files.** Shared types (`PointCloud`, `RegistrationResult`, `RegistrationParam`) and the CPU/global registration algorithms keep `Engine::Registration`. Trackers + `GpuPointToPlaneIcp` keep `Engine::Pipeline`. ONLY the feature algorithms (`Fpfh`, `FeatureMatching`, `Downsample`) move to the new `Engine::Features` namespace.
- **`RegistrationTypes.h` lives in `Engine/Features/`** (lowest lib; both features and registration include it). `RegistrationParam` moves out of `Icp.h` into it.
- **Flat layout** under `src/Engine/Pipeline/Registration/` (no subfolders).
- **CMake globs, so reconfigure** after every move: `cmake -S . -B build` before building (a bare `--build` will not re-glob).
- **macOS/BSD `sed`**: in-place edits use `sed -i '' 's|OLD|NEW|g' <files>` (note the empty `''`); use `|` as the delimiter since paths contain `/`.
- **Everything is uncommitted-WIP-free in the tracker/GpuIcp area is committed**, but the working tree also contains the USER's own concurrent edits to `src/Engine/Spatial/AdvancedTSDF.cpp` and `src/shader/advanced_tsdf_{compact,integrate}.comp.glsl` — DO NOT touch, stage, or revert those. Stage only each task's own files. Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do NOT push.
- **Verification build target set** (every task): `cmake --build build --target vkspatial_tests voxel_fill_debugger registration_chair_demo -j8`. This compiles all tests (incl. the legacy `test_alignment.cpp` → `example2/Alignment.h`) and both example2 registration consumers, so any missed include/namespace surfaces immediately.

---

## File Structure

**New `src/Engine/Features/`** (lib `Engine::Features`, links Eigen only):
- `RegistrationTypes.h` — `PointCloud`, `RegistrationResult`, `RegistrationParam` (namespace `Engine::Registration`)
- `Fpfh.h/.cpp`, `FeatureMatching.h/.cpp`, `Downsample.h/.cpp` (namespace `Engine::Features`)

**`src/Engine/Pipeline/Registration/`** (flat, compiled into `EnginePipeline`):
- `PointToPlaneIcp.h` (CPU, from `Icp.h`), `GpuPointToPlaneIcp.h/.cpp` (from `GpuIcp.*`), `GlobalRegistration.h/.cpp`
- `Tracker.h` (interface + `TrackerRegistry`), `TrackerRegistry.cpp` (central `Default()`)
- `IdentityTracker.h/.cpp`, `PointToPlaneIcpTracker.h/.cpp`, `GpuIcpTracker.h/.cpp`, `GlobalRegistrationTracker.h/.cpp`
- `RegistrationThread.h/.cpp` (unchanged)

**Deleted:** `src/Engine/Registration/` (emptied); `EngineRegistration` CMake lib.

---

## Task 1: Create `Engine::Features` + relocate shared types

Extract the feature algorithms and shared types into a new low-level lib. After this task `src/Engine/Registration/` still holds `Icp.h` + `GlobalRegistration.*` (moved in Task 2).

**Files:**
- Create dir: `src/Engine/Features/`
- Move: `Engine/Registration/{RegistrationTypes.h,Fpfh.h,Fpfh.cpp,FeatureMatching.h,FeatureMatching.cpp,Downsample.h,Downsample.cpp}` → `Engine/Features/`
- Modify: `src/Engine/CMakeLists.txt` (add `EngineFeatures`; `EngineRegistration` links it)
- Modify (includes/namespaces): the moved files + every includer of the moved headers (`GlobalRegistration.*`, `Icp.h`, `Pipeline/Registration/Tracker.cpp`, `Pipeline/Registration/GpuIcp.h`, `test/test_icp.cpp`, `test/test_gpuIcp.cpp`, `test/test_registration.cpp`, `example2/registration_chair_demo.cpp`, `example2/Alignment.h`)

**Interfaces:**
- Produces: `Engine::Features::ComputeFpfh`, `Engine::Features::MatchFeatures`, `Engine::Features::DownsampleVoxel`, `Engine::Features::Fpfh33`, `Engine::Features::Correspondence`; and `Engine::Registration::{PointCloud,RegistrationResult,RegistrationParam}` now at `Engine/Features/RegistrationTypes.h`.

- [ ] **Step 1: Move the files and add RegistrationParam to the types header**

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
mkdir -p src/Engine/Features
git mv src/Engine/Registration/RegistrationTypes.h src/Engine/Features/RegistrationTypes.h
git mv src/Engine/Registration/Fpfh.h            src/Engine/Features/Fpfh.h
git mv src/Engine/Registration/Fpfh.cpp          src/Engine/Features/Fpfh.cpp
git mv src/Engine/Registration/FeatureMatching.h   src/Engine/Features/FeatureMatching.h
git mv src/Engine/Registration/FeatureMatching.cpp src/Engine/Features/FeatureMatching.cpp
git mv src/Engine/Registration/Downsample.h      src/Engine/Features/Downsample.h
git mv src/Engine/Registration/Downsample.cpp    src/Engine/Features/Downsample.cpp
```

Then move `RegistrationParam` into `RegistrationTypes.h`: cut this exact block out of `src/Engine/Registration/Icp.h` (it sits just inside `namespace Engine::Registration {`)…

```cpp
    struct RegistrationParam {
        int maxIters = 20;
        // correspondence gate (world units); set to the data scale
        float maxCorrDist = 0.1f;
        int minInliers = 10;
        // stop when the incremental update norm drops below this
        float convEps = 1e-6f;
    };
```

…and paste it into `src/Engine/Features/RegistrationTypes.h` inside its `namespace Engine::Registration { … }` (next to `PointCloud`/`RegistrationResult`). `Icp.h` keeps `#include "Engine/Registration/RegistrationTypes.h"` (rewritten in Step 3), so it still sees `RegistrationParam`.

- [ ] **Step 2: Rename the feature namespace + import the shared PointCloud**

In the six feature files, change the namespace and (in the headers that use `PointCloud`) import it, since `PointCloud` stays in `Engine::Registration`:

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
FEAT="src/Engine/Features/Fpfh.h src/Engine/Features/Fpfh.cpp src/Engine/Features/FeatureMatching.h src/Engine/Features/FeatureMatching.cpp src/Engine/Features/Downsample.h src/Engine/Features/Downsample.cpp"
sed -i '' 's|namespace Engine::Registration|namespace Engine::Features|g' $FEAT
sed -i '' 's|// namespace Engine::Registration|// namespace Engine::Features|g' $FEAT
```

Then, in each of `Fpfh.h`, `FeatureMatching.h`, `Downsample.h`, add — as the first line inside `namespace Engine::Features {` — an import so the moved signatures still resolve `PointCloud`:

```cpp
        using Engine::Registration::PointCloud;
```

(`Fpfh33`/`Correspondence` are defined inside `Engine::Features` and need no import.)

- [ ] **Step 3: Rewrite all include paths for the moved headers**

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
# every file that includes any moved header (feature files include each other + the types header too)
FILES=$(grep -rEl 'Engine/Registration/(RegistrationTypes|Fpfh|FeatureMatching|Downsample)\.h' src test example2)
for pat in RegistrationTypes Fpfh FeatureMatching Downsample; do
  sed -i '' "s|Engine/Registration/${pat}.h|Engine/Features/${pat}.h|g" $FILES
done
```

- [ ] **Step 4: Requalify feature-symbol references to `Engine::Features`**

The feature FUNCTIONS/TYPES moved namespace. Their callers must be requalified. Known callers: `src/Engine/Registration/GlobalRegistration.cpp` and `test/test_registration.cpp`.

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
CALLERS="src/Engine/Registration/GlobalRegistration.cpp test/test_registration.cpp"
for sym in ComputeFpfh MatchFeatures DownsampleVoxel Fpfh33 Correspondence; do
  sed -i '' "s|Engine::Registration::${sym}|Engine::Features::${sym}|g" $CALLERS
done
```

`GlobalRegistration.cpp` also calls features unqualified from inside `namespace Engine::Registration`; add `using namespace Engine::Features;` immediately after its `#include` block so those resolve. (`Estimate`/`RegistrationConfig`/the shared types stay `Engine::Registration` — do not touch those.)

- [ ] **Step 5: CMake — add `EngineFeatures`, make `EngineRegistration` link it**

In `src/Engine/CMakeLists.txt`, insert BEFORE the `EngineRegistration` block (line ~83):

```cmake
file(GLOB_RECURSE ENGINE_FEATURES_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Features/*.cpp")
add_library(EngineFeatures STATIC ${ENGINE_FEATURES_SOURCES})
add_library(Engine::Features ALIAS EngineFeatures)
target_link_libraries(EngineFeatures PUBLIC Eigen3::Eigen)
target_include_directories(EngineFeatures
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3)
```

And add `Engine::Features` to `EngineRegistration`'s link line (line ~86):

```cmake
target_link_libraries(EngineRegistration PUBLIC Eigen3::Eigen Ceres::ceres Engine::Features)
```

- [ ] **Step 6: Reconfigure, build, run the full suite (green gate)**

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
cmake -S . -B build >/dev/null && cmake --build build --target vkspatial_tests voxel_fill_debugger registration_chair_demo -j8 2>&1 | grep -E "error:|Built target (vkspatial_tests|voxel_fill_debugger|registration_chair_demo)$"
./build/test/vkspatial_tests 2>&1 | grep -E '\[  PASSED  \]|\[  FAILED  \]'
```
Expected: all three targets built; `[  PASSED  ]` ≈237, no `[  FAILED  ]`. Fix any compile error the requalification/includes missed (the build pinpoints them), then re-run.

- [ ] **Step 7: Commit**

```bash
git add src/Engine/Features src/Engine/Registration/Icp.h src/Engine/Registration/GlobalRegistration.cpp src/Engine/CMakeLists.txt src/Engine/Pipeline/Registration/Tracker.cpp src/Engine/Pipeline/Registration/GpuIcp.h test/test_icp.cpp test/test_gpuIcp.cpp test/test_registration.cpp example2/registration_chair_demo.cpp example2/Alignment.h
git commit -m "refactor(features): extract Fpfh/FeatureMatching/Downsample + shared types to Engine::Features

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Move CPU ICP + GlobalRegistration into Pipeline/Registration; dissolve `Engine::Registration` lib

**Files:**
- Move: `Engine/Registration/Icp.h` → `Engine/Pipeline/Registration/PointToPlaneIcp.h`
- Move: `Engine/Registration/GlobalRegistration.{h,cpp}` → `Engine/Pipeline/Registration/`
- Delete: empty `src/Engine/Registration/`
- Modify: `src/Engine/CMakeLists.txt` (remove `EngineRegistration`; `EnginePipeline` links `Engine::Features` + `Ceres`), `example2/CMakeLists.txt` (`registration_chair_demo` link), and every includer of the two moved headers.

**Interfaces:**
- Consumes: `Engine::Features` (Task 1).
- Produces: `Engine::Registration::AlignPointToPlaneIcp` at `Engine/Pipeline/Registration/PointToPlaneIcp.h`; `Engine::Registration::Estimate` at `Engine/Pipeline/Registration/GlobalRegistration.h`.

- [ ] **Step 1: Move the files**

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
git mv src/Engine/Registration/Icp.h                 src/Engine/Pipeline/Registration/PointToPlaneIcp.h
git mv src/Engine/Registration/GlobalRegistration.h   src/Engine/Pipeline/Registration/GlobalRegistration.h
git mv src/Engine/Registration/GlobalRegistration.cpp src/Engine/Pipeline/Registration/GlobalRegistration.cpp
rmdir src/Engine/Registration 2>/dev/null || ls -la src/Engine/Registration   # must be empty now
```

- [ ] **Step 2: Rewrite include paths for the two moved headers**

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
FILES=$(grep -rEl 'Engine/Registration/(Icp|GlobalRegistration)\.h' src test example2)
sed -i '' 's|Engine/Registration/Icp.h|Engine/Pipeline/Registration/PointToPlaneIcp.h|g' $FILES
sed -i '' 's|Engine/Registration/GlobalRegistration.h|Engine/Pipeline/Registration/GlobalRegistration.h|g' $FILES
```

(`PointToPlaneIcp.h`'s own `#include "Engine/Features/RegistrationTypes.h"` was already fixed in Task 1 and is unaffected. `GpuIcp.h` still includes `PointToPlaneIcp.h` only for `RegistrationParam`; leave that here — Task 3 drops it.)

- [ ] **Step 3: CMake — dissolve `EngineRegistration`, relink `EnginePipeline`**

In `src/Engine/CMakeLists.txt`, DELETE the entire `EngineRegistration` block (the `file(GLOB_RECURSE ...Registration...)`, `add_library(EngineRegistration ...)`, `add_library(Engine::Registration ALIAS ...)`, `target_link_libraries(EngineRegistration ...)`, and its `target_include_directories(...)` — lines ~83–92).

Change `EnginePipeline`'s link line (was `PUBLIC Engine::Spatial Engine::Registration Engine::Core`) to:

```cmake
target_link_libraries(EnginePipeline
        PUBLIC Engine::Spatial Engine::Features Engine::Core Ceres::ceres
        PRIVATE Engine::Render)
```

In `example2/CMakeLists.txt`, change `registration_chair_demo`'s link (line ~58) from `Engine::Registration` to `Engine::Pipeline`:

```cmake
target_link_libraries(registration_chair_demo PRIVATE Engine::Pipeline)
```

- [ ] **Step 4: Reconfigure, build, run the full suite (green gate)**

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
cmake -S . -B build >/dev/null && cmake --build build --target vkspatial_tests voxel_fill_debugger registration_chair_demo -j8 2>&1 | grep -E "error:|Built target (vkspatial_tests|voxel_fill_debugger|registration_chair_demo)$"
./build/test/vkspatial_tests 2>&1 | grep -E '\[  PASSED  \]|\[  FAILED  \]'
```
Expected: all built; ≈237 passed, 0 failed. If the linker reports undefined `Engine::Registration::Estimate`/`AlignPointToPlaneIcp` symbols, a consumer still links the deleted `Engine::Registration` — fix its `target_link_libraries` to `Engine::Pipeline`.

- [ ] **Step 5: Commit**

```bash
git add -A -- src/Engine/Pipeline/Registration src/Engine/CMakeLists.txt example2/CMakeLists.txt src/Engine/Pipeline/Registration/GpuIcp.h src/Engine/Pipeline/Registration/Tracker.cpp test example2/registration_chair_demo.cpp example2/Alignment.h
git commit -m "refactor(registration): move CPU ICP + GlobalRegistration into Pipeline/Registration; dissolve Engine::Registration lib

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Rename `GpuIcp` → `GpuPointToPlaneIcp`

**Files:**
- Move: `Engine/Pipeline/Registration/GpuIcp.{h,cpp}` → `GpuPointToPlaneIcp.{h,cpp}`
- Modify: includers of `GpuIcp.h` (`Tracker.cpp`, `test/test_gpuIcp.cpp`), and `GpuPointToPlaneIcp.h` itself (drop the now-unnecessary `PointToPlaneIcp.h` include).

**Interfaces:**
- Consumes: `Engine::Pipeline::GpuPointToPlaneIcp` (class name unchanged), `Engine::Registration::RegistrationParam` (from `RegistrationTypes.h`).

- [ ] **Step 1: Rename + fix the include self-reference**

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
git mv src/Engine/Pipeline/Registration/GpuIcp.h   src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.h
git mv src/Engine/Pipeline/Registration/GpuIcp.cpp src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.cpp
FILES=$(grep -rl 'Engine/Pipeline/Registration/GpuIcp.h' src test example2)
sed -i '' 's|Engine/Pipeline/Registration/GpuIcp.h|Engine/Pipeline/Registration/GpuPointToPlaneIcp.h|g' $FILES $([ -f src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.cpp ] && echo src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.cpp)
```

- [ ] **Step 2: Drop the redundant `PointToPlaneIcp.h` include from the header**

In `src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.h`, delete the line
`#include "Engine/Registration/Icp.h" // RegistrationParam` (now `…/PointToPlaneIcp.h` after Task 2's sed). It was only for `RegistrationParam`, which `RegistrationTypes.h` (already included) now provides — the GPU solver uses no CPU-ICP symbol.

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
sed -i '' '/#include "Engine\/Pipeline\/Registration\/PointToPlaneIcp.h" \/\/ RegistrationParam/d' src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.h
grep -n 'RegistrationTypes.h' src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.h   # must still be present
```

- [ ] **Step 3: Reconfigure, build, run the full suite (green gate)**

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
cmake -S . -B build >/dev/null && cmake --build build --target vkspatial_tests voxel_fill_debugger registration_chair_demo -j8 2>&1 | grep -E "error:|Built target (vkspatial_tests|voxel_fill_debugger|registration_chair_demo)$"
./build/test/vkspatial_tests 2>&1 | grep -E '\[  PASSED  \]|\[  FAILED  \]'
```
Expected: built; ≈237 passed, 0 failed. If `RegistrationParam` is now undefined in the GPU header, re-add `#include "Engine/Features/RegistrationTypes.h"` (it should already be there).

- [ ] **Step 4: Commit**

```bash
git add -A -- src/Engine/Pipeline/Registration test/test_gpuIcp.cpp
git commit -m "refactor(registration): rename GpuIcp -> GpuPointToPlaneIcp

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Split `Tracker.cpp` into a hub + one file per strategy

Currently `Tracker.cpp` holds all four strategy classes (in an anonymous namespace) plus `TrackerRegistry::Default()`. Split into named per-strategy files + a central registry file.

**Files:**
- Modify: `src/Engine/Pipeline/Registration/Tracker.h` (unchanged interface; verify it still only declares `Tracker`/`TrackingResult`/`TrackerRegistry`)
- Create: `IdentityTracker.{h,cpp}`, `PointToPlaneIcpTracker.{h,cpp}`, `GpuIcpTracker.{h,cpp}`, `GlobalRegistrationTracker.{h,cpp}`, `TrackerRegistry.cpp`
- Delete: `src/Engine/Pipeline/Registration/Tracker.cpp`

**Interfaces:**
- Consumes: `Tracker`, `TrackingResult`, `TrackerRegistry` (`Tracker.h`); `AlignPointToPlaneIcp`, `GpuPointToPlaneIcp`, `Estimate`, `AdvancedEntry`.
- Produces: unchanged registry names `identity`/`icp`/`icp-cpu`/`global` via `TrackerRegistry::Default()` in `TrackerRegistry.cpp`.

- [ ] **Step 1: Create one header+source per strategy (named classes, not anonymous)**

For each strategy, lift its class body verbatim from the current `Tracker.cpp` into a header declaring the class and a `.cpp` defining `Track()`. Each header includes `Tracker.h` and whatever the strategy needs. Example — `PointToPlaneIcpTracker.h`:

```cpp
#pragma once
#include "Engine/Pipeline/Registration/Tracker.h"
#include "Engine/Pipeline/Registration/PointToPlaneIcp.h"

namespace Engine::Pipeline {
    // Local point-to-plane ICP against the latest model's occupied voxels (centres + normals).
    class PointToPlaneIcpTracker : public Tracker {
    public:
        const char *Name() const override { return "icp-cpu"; }
        TrackingResult Track(const Frame &frame, const ModelSnapshot *model,
                             const Eigen::Isometry3f &priorPose) override;
    private:
        Engine::Registration::RegistrationParam m_params;
    };
} // namespace Engine::Pipeline
```

`PointToPlaneIcpTracker.cpp` then holds the `Track()` body currently in `Tracker.cpp` (the crop-free full-model target build + voxel-scaled `maxCorrDist` + `AlignPointToPlaneIcp` call). Do the same for:
- `IdentityTracker.{h,cpp}` (`"identity"`, returns identity pose)
- `GpuIcpTracker.{h,cpp}` (`"icp"`, includes `GpuPointToPlaneIcp.h` + `Engine/Core/Context.h`; holds `m_ctx`/`m_gpu`; the world-frame crop + voxel-scaled `maxCorrDist` + `Solve` body verbatim)
- `GlobalRegistrationTracker.{h,cpp}` (`"global"`, includes `GlobalRegistration.h`)

Each strategy that iterates `model->entries` includes `Engine/Spatial/AdvancedTSDF.h` (for `AdvancedEntry`), as `Tracker.cpp` does today. Copy the bodies EXACTLY — no logic changes.

- [ ] **Step 2: Create `TrackerRegistry.cpp` with the central `Default()`**

```cpp
#include "Engine/Pipeline/Registration/Tracker.h"
#include "Engine/Pipeline/Registration/IdentityTracker.h"
#include "Engine/Pipeline/Registration/PointToPlaneIcpTracker.h"
#include "Engine/Pipeline/Registration/GpuIcpTracker.h"
#include "Engine/Pipeline/Registration/GlobalRegistrationTracker.h"
#include <memory>

namespace Engine::Pipeline {
    TrackerRegistry TrackerRegistry::Default() {
        TrackerRegistry reg;
        reg.Register("identity", [] { return std::make_unique<IdentityTracker>(); });
        reg.Register("icp",      [] { return std::make_unique<GpuIcpTracker>(); });
        reg.Register("icp-cpu",  [] { return std::make_unique<PointToPlaneIcpTracker>(); });
        reg.Register("global",   [] { return std::make_unique<GlobalRegistrationTracker>(); });
        return reg;
    }
} // namespace Engine::Pipeline
```

(Copy the exact `Register(...)` lines from the current `Tracker.cpp::Default()` so the names/factories match byte-for-byte.)

- [ ] **Step 3: Delete the monolith**

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
git rm src/Engine/Pipeline/Registration/Tracker.cpp
```

Confirm `Tracker.h` still declares only the interface + registry (no strategy classes). If any `TrackingResult` field or `Tracker`/`TrackerRegistry` method was defined inline in `Tracker.cpp`, it must already be in `Tracker.h` or move there — check `git grep` for definitions.

- [ ] **Step 4: Reconfigure, build, run the full suite (green gate)**

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
cmake -S . -B build >/dev/null && cmake --build build --target vkspatial_tests voxel_fill_debugger registration_chair_demo -j8 2>&1 | grep -E "error:|Built target (vkspatial_tests|voxel_fill_debugger|registration_chair_demo)$"
./build/test/vkspatial_tests 2>&1 | grep -E '\[  PASSED  \]|\[  FAILED  \]'
```
Expected: built; ≈237 passed, 0 failed. Registry-driven tests (`Pipeline.GpuIcpTrackerRuns` asserting `Create("icp")`/`Create("icp-cpu")` non-null, `test_pipeline` identity/icp) prove the hub still wires all four names.

- [ ] **Step 5: Commit**

```bash
git add -A -- src/Engine/Pipeline/Registration
git commit -m "refactor(registration): split Tracker.cpp into hub (TrackerRegistry) + per-strategy files

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification

After Task 4, `src/Engine/Registration/` no longer exists, `src/Engine/Features/` holds the feature module + shared types, and `src/Engine/Pipeline/Registration/` holds one file per algorithm and per tracker strategy plus the hub. Confirm the end state:

```bash
cd /Users/sjy/Desktop/VulkanProject/VkLBVH
test ! -d src/Engine/Registration && echo "Engine/Registration removed OK"
ls src/Engine/Features src/Engine/Pipeline/Registration
grep -rl 'Engine/Registration/' src test example2 && echo "STRAGGLER old include remains (fix it)" || echo "no stale Engine/Registration includes"
./build/test/vkspatial_tests 2>&1 | tail -3
```
Expected: `Engine/Registration removed OK`, no stale includes, suite ≈237 pass / 1 skip.

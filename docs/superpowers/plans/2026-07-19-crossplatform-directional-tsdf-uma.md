# Cross-Platform Directional TSDF (UMA + Discrete) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `Engine::Spatial::DirectionalTSDF` so its residency machinery lives behind an `IResidencyBackend` interface, add a `UnifiedResidencyBackend` for UMA (zero-copy whole-model residency), auto-select the backend at runtime, and prove both backends produce identical reconstructions.

**Architecture:** The `DirectionalTSDF` core keeps the algorithm (integrate/extract/merge kernels, local-base math, point cloud). All "where does a group live and how does it become device-addressable" logic moves into a swappable backend. `StreamingResidencyBackend` = today's host-store + active-pool + staging + classify/register streaming (behavior-preserving move). `UnifiedResidencyBackend` = one coherent `DEVICE_LOCAL|HOST_VISIBLE` pool holding the whole model, so `EnsureResident` is append + indexGrid relabel with no copies. A factory probes `VkPhysicalDeviceMemoryProperties` to choose.

**Tech Stack:** C++17, Vulkan 1.x + VMA (`vk_mem_alloc.h`), Eigen, GoogleTest. Compute in GLSL comp shaders. macOS via MoltenVK.

## Global Constraints

- Namespace `Engine::Spatial`, no `vk` prefix (matches `SimpleTSDF`/`DirectionalTSDF` convention).
- Single fixed-point accumulator wire/GPU format is **out of scope for this plan** — keep the existing `HostTsdfVoxel{value,weight}` (host) and `GpuTsdfVoxel{sumDW,sumW}` (GPU) formats and their conversion unchanged. (The spec's §2 8B unification is a *later* plan; doing it here would entangle a format change with the interface extraction.)
- Behavior-preserving refactor: every existing test in `test/test_directionalTSDF.cpp` and `test/test_directionalTSDFEval.cpp` MUST still pass unchanged after Phase 1. They are the characterization safety net — do not edit them in Phase 1.
- Architecture constants come from `DirectionalTSDFTypes.h` (`kGroupDim=8`, `kVoxelsPerGroup=512`, `kLocalGroupGrid=50`, `kNumDirections=6`, `kIndexGridCells=750000`, `kInvalidPoolIndex=0xFFFFFFFF`, `kTsdfFixedScale=10000`). Do not redefine them.
- Every GPU submission stays synchronous (no async/Phase-5 optimizations in this plan).
- Build: `cmake -S . -B build && cmake --build build --parallel`. Test binary: `./build/test/vkspatial_tests` (GoogleTest, supports `--gtest_filter=`).
- `static_assert(std::is_standard_layout_v<...>)` + `sizeof`/`offsetof` checks accompany every GPU-shared struct (existing `DirectionalTSDFTypes.h:105-114` convention).
- Commit after every task with a passing build + passing tests.

---

## File Structure

**New files:**
- `src/Engine/Spatial/IResidencyBackend.h` — pure-virtual residency interface + `ResidencyStats` struct + `MakeResidencyBackend` factory declaration.
- `src/Engine/Spatial/StreamingResidencyBackend.h` / `.cpp` — the existing streaming machinery, moved verbatim behind the interface.
- `src/Engine/Spatial/UnifiedResidencyBackend.h` / `.cpp` — UMA coherent-pool backend.
- `src/Engine/Spatial/ResidencyBackendFactory.cpp` — `MakeResidencyBackend` (memory-property probe + config override).
- `test/test_residencyBackend.cpp` — unit tests for the unified backend + the cross-backend correctness test.

**Modified files:**
- `src/Engine/Spatial/DirectionalTSDF.h` / `.cpp` — drop the moved members, hold an `std::unique_ptr<IResidencyBackend>`, delegate.
- `src/Engine/Spatial/CMakeLists.txt` — add the new sources (verify glob vs explicit list first).

---

## Task 1: Define `IResidencyBackend` interface + `ResidencyStats`

**Files:**
- Create: `src/Engine/Spatial/IResidencyBackend.h`
- Test: `test/test_residencyBackend.cpp`

**Interfaces:**
- Produces:
  - `struct Engine::Spatial::ResidencyStats { uint32_t residentCount, missingCount, writeBackCount, h2dBytes, d2hBytes; float overlapRatio; };`
  - `class Engine::Spatial::IResidencyBackend` with the pure-virtual methods listed below.
  - `enum class Engine::Spatial::ResidencyMode { Auto, Streaming, Unified };`
  - `std::unique_ptr<IResidencyBackend> Engine::Spatial::MakeResidencyBackend(Engine::Core::Context&, uint32_t poolCapacity, ResidencyMode mode = ResidencyMode::Auto);` (declaration only; defined in Task 5).

- [ ] **Step 1: Write the failing test** (compile-level contract — a stub backend must satisfy the interface)

`test/test_residencyBackend.cpp`:
```cpp
#include "Engine/Spatial/IResidencyBackend.h"
#include <gtest/gtest.h>

using namespace Engine::Spatial;

namespace {
// Minimal in-memory fake proving the interface is implementable without Vulkan.
class FakeBackend : public IResidencyBackend {
public:
    void BeginFrame(const Eigen::Vector3i &base) override { m_base = base; }
    void EnsureResident(const std::vector<DirectionalGroupKey> &r) override { m_last = r.size(); }
    void EndFrame() override {}
    VkBuffer IndexGridBuffer() const override { return VK_NULL_HANDLE; }
    VkBuffer PoolVoxelBuffer() const override { return VK_NULL_HANDLE; }
    VkBuffer MetaBuffer() const override { return VK_NULL_HANDLE; }
    uint32_t PoolCapacity() const override { return 0; }
    Eigen::Vector3i LocalBase() const override { return m_base; }
    bool IsUnified() const override { return true; }
    bool SupportsZeroCopyCpuAccess() const override { return true; }
    DirectionalHostStore &HostStore() override { return m_store; }
    ResidencyStats FrameStats() const override { return {}; }
    std::vector<uint32_t> DebugDownloadIndexGrid() override { return {}; }
    uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &) override { return kInvalidPoolIndex; }
    DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &) override { return {}; }
private:
    Eigen::Vector3i m_base = Eigen::Vector3i::Zero();
    size_t m_last = 0;
    DirectionalHostStore m_store;
};
} // namespace

TEST(ResidencyBackend, InterfaceIsImplementable) {
    FakeBackend b;
    IResidencyBackend &iface = b;
    iface.BeginFrame(Eigen::Vector3i(1, 2, 3));
    iface.EnsureResident({});
    EXPECT_EQ(iface.LocalBase(), Eigen::Vector3i(1, 2, 3));
    EXPECT_TRUE(iface.IsUnified());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --parallel --target vkspatial_tests`
Expected: FAIL to compile — `IResidencyBackend.h` not found.

- [ ] **Step 3: Write the interface header**

`src/Engine/Spatial/IResidencyBackend.h`:
```cpp
#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalHostStore.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"

#include <Eigen/Core>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Spatial {

    // Residency-side counters the core aggregates into DirectionalTSDF::Stats.
    struct ResidencyStats {
        uint32_t residentCount = 0;
        uint32_t missingCount = 0;
        uint32_t writeBackCount = 0;
        uint32_t h2dBytes = 0;
        uint32_t d2hBytes = 0;
        float overlapRatio = 0.0f;
    };

    // Abstracts "where a directional group lives and how it becomes device-addressable".
    // Streaming (discrete/PCIe) and Unified (UMA/zero-copy) implement this identically to
    // the core, which never issues vkCmdCopyBuffer/vkCmdFillBuffer/allocations itself.
    class IResidencyBackend {
    public:
        virtual ~IResidencyBackend() = default;

        // Reset the indexGrid for `localBase`, classify the pool, re-register reusable slots.
        virtual void BeginFrame(const Eigen::Vector3i &localBase) = 0;
        // Make `required` (write-set + halo) resident. Streaming: upload missing. Unified: no-op.
        virtual void EnsureResident(const std::vector<DirectionalGroupKey> &required) = 0;
        // End-of-frame cleanup. Streaming: dirty write-back + eviction. Unified: no-op (or cold compact).
        virtual void EndFrame() = 0;

        // GPU buffers the integrate/extract kernels bind (owned by the backend).
        virtual VkBuffer IndexGridBuffer() const = 0;
        virtual VkBuffer PoolVoxelBuffer() const = 0;
        virtual VkBuffer MetaBuffer() const = 0;
        virtual uint32_t PoolCapacity() const = 0;
        virtual Eigen::Vector3i LocalBase() const = 0;

        // Capability flags (core uses these to gate quality bonuses in later plans).
        virtual bool IsUnified() const = 0;
        virtual bool SupportsZeroCopyCpuAccess() const = 0;

        // Authoritative group store (streaming: the map; unified: a view over the pool).
        // Used by demos to seed geometry and by debug downloads.
        virtual DirectionalHostStore &HostStore() = 0;

        // Residency counters for the frame just processed.
        virtual ResidencyStats FrameStats() const = 0;

        // Synchronous debug/test downloads (not for per-frame use).
        virtual std::vector<uint32_t> DebugDownloadIndexGrid() = 0;
        virtual uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &key) = 0;
        virtual DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &key) = 0;
    };

    // Explicit backend choice. Auto probes memory topology (UMA→Unified, else Streaming).
    enum class ResidencyMode { Auto, Streaming, Unified };

    // Builds the requested backend. `Auto` chooses UnifiedResidencyBackend when the device
    // exposes a DEVICE_LOCAL|HOST_VISIBLE heap large enough for the model, else Streaming.
    // The env var VKLBVH_RESIDENCY=streaming|unified overrides `mode` at runtime (escape hatch;
    // defined in ResidencyBackendFactory.cpp, Task 5).
    std::unique_ptr<IResidencyBackend>
    MakeResidencyBackend(Engine::Core::Context &ctx, uint32_t poolCapacity,
                         ResidencyMode mode = ResidencyMode::Auto);

} // namespace Engine::Spatial
```

- [ ] **Step 4: Add the test file to the build and compile**

Confirm `test/CMakeLists.txt` globs `*.cpp` (it does: `file(GLOB TEST_SOURCES ...)`) so the new test is picked up automatically. Then:
Run: `cmake -S . -B build && cmake --build build --parallel --target vkspatial_tests`
Expected: compiles.

- [ ] **Step 5: Run test to verify it passes**

Run: `./build/test/vkspatial_tests --gtest_filter='ResidencyBackend.InterfaceIsImplementable'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/Engine/Spatial/IResidencyBackend.h test/test_residencyBackend.cpp
git commit -m "feat(spatial): add IResidencyBackend interface + ResidencyStats"
```

---

## Task 2: Move the streaming machinery into `StreamingResidencyBackend`

This is a behavior-preserving relocation. The existing `DirectionalTSDF` members and private methods listed below move into the new class *unchanged in body*; only their owning class changes. Do NOT reimplement them — cut and paste, then fix the `this->`/member references.

**Files:**
- Create: `src/Engine/Spatial/StreamingResidencyBackend.h`, `src/Engine/Spatial/StreamingResidencyBackend.cpp`
- Read (source of the moved code): `src/Engine/Spatial/DirectionalTSDF.cpp`, `src/Engine/Spatial/DirectionalTSDF.h:97-157`
- Modify: `src/Engine/Spatial/CMakeLists.txt` (add sources if it uses an explicit source list; skip if it globs)

**Interfaces:**
- Consumes: `IResidencyBackend`, `ResidencyStats` (Task 1); existing `DirectionalHostStore`, `DirectionalTSDFTypes.h`, `Engine::Core::{Context,Buffer,ComputePipeline}`, `Engine::Compute::{StagingBuffer,CommandBatch}`.
- Produces: `class StreamingResidencyBackend : public IResidencyBackend` with an added `void Build(Engine::Core::Context&, uint32_t poolCapacity)` (one-time setup, mirroring the residency half of today's `DirectionalTSDF::Build`).

**Members to move** from `DirectionalTSDF` (header `DirectionalTSDF.h`) into `StreamingResidencyBackend`:
`m_ctx, m_poolCapacity, m_hostStore, m_localBase, m_indexGrid, m_poolVoxels, m_metaBuffer, m_slotListBuffer, m_reusableList, m_cleanFreeList, m_writeBackList, m_countsBuffer, m_registerKernel, m_classifyKernel, m_stageCounts, m_stageLists, m_stageCleanFree, m_stageGroups, m_stageMeta, m_stageSlotList, m_reusableSlots, m_residentIndex, m_slotKeys, m_freeSlots, m_requiredThisFrame, m_lastCounts`, plus the residency-related fields of `m_stats` (surface them via `ResidencyStats`).

**Methods to move** (bodies unchanged): `fillIndexGridInvalid`, `updateOverlapRatio`, `recordResidency`, and the classify/register/local-base portions of `BeginFrame`/`EnsureResident`. `quantizeLocalBase` stays in the core (it is algorithm, not residency) — the backend's `BeginFrame` now receives an already-quantized `Eigen::Vector3i localBase`.

- [ ] **Step 1: Write the failing test** — the streaming backend builds and reports its capabilities

Append to `test/test_residencyBackend.cpp`:
```cpp
#include "Engine/Core/Context.h"
#include "Engine/Spatial/StreamingResidencyBackend.h"

TEST(ResidencyBackend, StreamingBuildsAndClassifiesEmpty) {
    Engine::Core::Context ctx;
    StreamingResidencyBackend be;
    be.Build(ctx, /*poolCapacity=*/1024);
    EXPECT_FALSE(be.IsUnified());
    EXPECT_EQ(be.PoolCapacity(), 1024u);
    be.BeginFrame(Eigen::Vector3i(0, 0, 0));
    be.EnsureResident({}); // nothing required → no missing
    EXPECT_EQ(be.FrameStats().missingCount, 0u);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --parallel --target vkspatial_tests`
Expected: FAIL to compile — `StreamingResidencyBackend.h` not found.

- [ ] **Step 3: Create `StreamingResidencyBackend.h`**

Declare the class implementing every `IResidencyBackend` method plus `Build`, and holding the moved members. Skeleton (fill member declarations by moving them from `DirectionalTSDF.h:97-157`):
```cpp
#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Compute/StagingBuffer.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/IResidencyBackend.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Engine::Spatial {

    class StreamingResidencyBackend : public IResidencyBackend {
    public:
        void Build(Engine::Core::Context &ctx, uint32_t poolCapacity);

        void BeginFrame(const Eigen::Vector3i &localBase) override;
        void EnsureResident(const std::vector<DirectionalGroupKey> &required) override;
        void EndFrame() override;

        VkBuffer IndexGridBuffer() const override { return m_indexGrid->Handle(); }
        VkBuffer PoolVoxelBuffer() const override { return m_poolVoxels->Handle(); }
        VkBuffer MetaBuffer() const override { return m_metaBuffer->Handle(); }
        uint32_t PoolCapacity() const override { return m_poolCapacity; }
        Eigen::Vector3i LocalBase() const override { return m_localBase; }
        bool IsUnified() const override { return false; }
        bool SupportsZeroCopyCpuAccess() const override { return false; }
        DirectionalHostStore &HostStore() override { return m_hostStore; }
        ResidencyStats FrameStats() const override { return m_stats; }
        std::vector<uint32_t> DebugDownloadIndexGrid() override;
        uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &key) override;
        DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &key) override;

    private:
        // (moved verbatim from DirectionalTSDF.h:97-157 — pool/indexGrid/meta/lists/staging,
        //  m_residentIndex/m_slotKeys/m_freeSlots/m_requiredThisFrame, classify/register kernels)
        // ... member declarations ...
        ResidencyStats m_stats;

        void fillIndexGridInvalid();
        void updateOverlapRatio();
        uint32_t recordResidency(const std::vector<DirectionalGroupKey> &required,
                                 Engine::Compute::CommandBatch &batch);
    };

} // namespace Engine::Spatial
```

- [ ] **Step 4: Create `StreamingResidencyBackend.cpp`** by moving the residency method bodies from `DirectionalTSDF.cpp`

Cut these from `DirectionalTSDF.cpp` into the new `.cpp`, changing only the class qualifier (`DirectionalTSDF::` → `StreamingResidencyBackend::`) and nothing else in the bodies:
- The buffer/kernel/staging allocation block of `Build` (everything except `m_voxelSize`/`m_truncation`/`m_maxPoints`/`m_maxCandidates`/integrate+extract kernels+point/candidate buffers, which stay in the core) → into `StreamingResidencyBackend::Build`.
- `BeginFrame` body (local-base assignment now takes the passed `localBase` instead of calling `quantizeLocalBase`; keep `fillIndexGridInvalid` + classify dispatch + reusable download + register).
- `EnsureResident` body (record + submit).
- `fillIndexGridInvalid`, `updateOverlapRatio`, `recordResidency`, `DebugDownloadIndexGrid`, `DebugQueryPoolIndex`, `DebugDownloadGroupVoxels`.
- Add an empty `void StreamingResidencyBackend::EndFrame() {}` for now (write-back is still Phase 4 in the original spec; unchanged behavior = no-op).

Assign residency counters into `m_stats` (a `ResidencyStats`) where the old code wrote into `DirectionalTSDF::Stats`.

- [ ] **Step 5: Register sources in CMake**

Inspect `src/Engine/Spatial/CMakeLists.txt`. If it lists sources explicitly, add `StreamingResidencyBackend.cpp` (and `IResidencyBackend.h` is header-only). If it globs, no change.

- [ ] **Step 6: Build and run the new test**

Run: `cmake --build build --parallel --target vkspatial_tests && ./build/test/vkspatial_tests --gtest_filter='ResidencyBackend.StreamingBuildsAndClassifiesEmpty'`
Expected: PASS. (`DirectionalTSDF.cpp` will not yet compile because its bodies were removed — that is fixed in Task 3. To keep this task independently green, temporarily exclude `DirectionalTSDF.cpp` from the `Engine::Spatial` target, OR fold Tasks 2+3 into one commit. Prefer folding: proceed to Task 3 before building the full library.)

> **Note:** Tasks 2 and 3 share one build gate (the library only compiles once the core is rewired). Treat them as a single commit boundary; the two-task split is for reviewer clarity, not for independent builds.

---

## Task 3: Rewire `DirectionalTSDF` to delegate to the backend

**Files:**
- Modify: `src/Engine/Spatial/DirectionalTSDF.h`, `src/Engine/Spatial/DirectionalTSDF.cpp`

**Interfaces:**
- Consumes: `IResidencyBackend` (Task 1), `StreamingResidencyBackend` (Task 2).
- Produces: `DirectionalTSDF` with the same public API (`Build/BeginFrame/EnsureResident/Integrate/PointCloud/...`) unchanged, now delegating residency to `m_backend` (hardcoded Streaming in this task).

- [ ] **Step 1: Replace moved members with a backend pointer** in `DirectionalTSDF.h`

Remove the members listed in Task 2. Keep only the algorithm-side members: `m_ctx, m_voxelSize, m_truncation, m_maxPoints, m_maxCandidates, m_pointBuffer, m_candidateBuffer, m_candidateCounter, m_integrateKernel, m_extractKernel, m_stagePoints, m_stageCandidates, m_pointCloud, m_stats`. Add:
```cpp
        std::unique_ptr<IResidencyBackend> m_backend;
```
Keep the public `Stats`/`ClassifyCounts` structs and accessors. `HostStore()`/`LocalBase()`/`PoolCapacity()`/`DebugDownload*` now forward to `m_backend`.

- [ ] **Step 2: Rewire the method bodies** in `DirectionalTSDF.cpp`

- `Build`: after setting the algorithm params + creating integrate/extract kernels + point/candidate buffers, create the backend **directly as Streaming** (the factory is not implemented until Task 5, and Phase 1 must be behavior-preserving on every platform — auto-selecting Unified on a UMA machine like the M4 Max dev box would break the streaming-specific existing tests). Include `StreamingResidencyBackend.h` and write:
```cpp
        auto be = std::make_unique<StreamingResidencyBackend>();
        be->Build(*m_ctx, poolCapacity);
        m_backend = std::move(be);
```
Task 5 replaces this with `MakeResidencyBackend`.
- `BeginFrame(hint)`: `m_backend->BeginFrame(quantizeLocalBase(hint));` (`quantizeLocalBase` stays here).
- `EnsureResident(required)`: `m_backend->EnsureResident(required);`
- `Integrate(...)`: same pipeline, but bind `m_backend->IndexGridBuffer()/PoolVoxelBuffer()/MetaBuffer()/PoolCapacity()/LocalBase()` where it used the old members for the integrate/extract dispatches; call `m_backend->EndFrame()` at the end. Aggregate `m_backend->FrameStats()` (a `ResidencyStats`) into `m_stats`.
- Accessors: `LocalBase()`→`m_backend->LocalBase()`, `PoolCapacity()`→`m_backend->PoolCapacity()`, `HostStore()`→`m_backend->HostStore()`, `DebugDownloadIndexGrid/DebugQueryPoolIndex/DebugDownloadGroupVoxels`→forward.
- `DebugLastClassifyCounts()`: map from `ResidencyStats` (residentCount/missingCount/writeBackCount) or add a `ClassifyCounts LastClassify()` to the interface if the existing tests assert exact reusable/cleanFree/writeBack; check `test_directionalTSDF.cpp` for which fields it reads and expose exactly those.

- [ ] **Step 3: Build the test target**

Run: `cmake --build build --parallel --target vkspatial_tests`
Expected: compiles. (Build only `vkspatial_tests`, not the whole project — `example2` is broken at HEAD, referencing an uncommitted `ShadowMap.cpp`; that is pre-existing and unrelated to this work.)

- [ ] **Step 4: Run the full existing DirectionalTSDF suite — the characterization gate**

Run: `./build/test/vkspatial_tests --gtest_filter='*Directional*'`
Expected: PASS — identical results to before the refactor. If any test fails, the move changed behavior; diff against `git show HEAD:src/Engine/Spatial/DirectionalTSDF.cpp` and reconcile. Do not edit the tests.

- [ ] **Step 5: Commit (Tasks 2+3 together)**

```bash
git add src/Engine/Spatial/StreamingResidencyBackend.h src/Engine/Spatial/StreamingResidencyBackend.cpp \
        src/Engine/Spatial/DirectionalTSDF.h src/Engine/Spatial/DirectionalTSDF.cpp \
        src/Engine/Spatial/CMakeLists.txt
git commit -m "refactor(spatial): extract streaming residency behind IResidencyBackend"
```

---

## Task 4: Make `IResidencyBackend` sufficient for `Integrate`, then add `UnifiedResidencyBackend`

> **Why this task grew (Task 2+3 review finding):** the Task-3 refactor kept `DirectionalTSDF::Integrate`'s hot path coupled to a concrete `StreamingResidencyBackend*` (`m_streaming`), because the frozen `IResidencyBackend` cannot express what `Integrate` needs: the batched residency-record (`recordResidency` folded into the shared `CommandBatch`), resident-slot iteration (`ResidentIndex`), and the `gpuSubmits`/classify counters. That is a null-deref landmine the moment the backend is Unified. **Part A expands the interface and decouples the core (verified by the unchanged streaming suite); Part B then implements `UnifiedResidencyBackend` against the complete interface**, so Task 6's cross-backend `Integrate` works without any concrete-type branching.

### Part A: Expand `IResidencyBackend` + decouple `DirectionalTSDF` from the concrete backend

**Files:**
- Modify: `src/Engine/Spatial/IResidencyBackend.h`, `src/Engine/Spatial/StreamingResidencyBackend.h`, `src/Engine/Spatial/StreamingResidencyBackend.cpp`, `src/Engine/Spatial/DirectionalTSDF.h`, `src/Engine/Spatial/DirectionalTSDF.cpp`

**A1. Expand the interface** (`IResidencyBackend.h`):
- Add includes: `#include "Engine/Compute/CommandBatch.h"` and `#include <unordered_map>`.
- Extend `ResidencyStats` with the counters the core reads back (keep the existing five + `overlapRatio`):
```cpp
        uint32_t reusableCount = 0;   // classify: resident & inside new window
        uint32_t cleanFreeCount = 0;  // classify: free or clean-evicted slots
        uint32_t gpuSubmits = 0;      // queue submissions the backend made this frame
```
- Add two pure-virtual methods to `IResidencyBackend`:
```cpp
        // Fold this frame's residency work into an existing command batch (Phase-5 batched path).
        // Streaming: stage missing groups + record copies/register into `batch`; returns slots recorded.
        // Unified: CPU slot bookkeeping only, records nothing to `batch`; returns slots touched.
        virtual uint32_t RecordResidency(const std::vector<DirectionalGroupKey> &required,
                                         Engine::Compute::CommandBatch &batch) = 0;
        // Enumerate resident (key -> pool slot) so the core can pick recompute slots for extraction.
        virtual const std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> &
        ResidentIndex() const = 0;
```

**A2. Make `StreamingResidencyBackend` satisfy the expanded interface** (`StreamingResidencyBackend.{h,cpp}`):
- Its existing public `recordResidency(required, batch)` already has the exact signature — rename to `RecordResidency` and mark `override` (or add a one-line `RecordResidency` override that forwards to it).
- Add `const std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> &ResidentIndex() const override { return m_residentIndex; }`.
- Fold the standalone `GpuSubmits()`/`LastReusableCount()`/`LastCleanFreeCount()`/`LastWriteBackCount()` accessors added in Task 3 INTO `FrameStats()`: set `m_stats.gpuSubmits`, `m_stats.reusableCount`, `m_stats.cleanFreeCount` (and existing `writeBackCount`) at the same points those internal counters are updated, and delete the four extra public methods.

**A3. Decouple `DirectionalTSDF`** (`DirectionalTSDF.{h,cpp}`):
- Delete the `StreamingResidencyBackend *m_streaming;` field and its assignment in `Build`.
- Route every former `m_streaming->X` call through the interface `m_backend`:
  - `m_streaming->recordResidency(req, batch)` → `m_backend->RecordResidency(req, batch)`
  - `m_streaming->ResidentIndex()` → `m_backend->ResidentIndex()`
  - `m_streaming->GpuSubmits()` / `Last*Count()` → read `m_backend->FrameStats()` (`.gpuSubmits`, `.reusableCount`, `.cleanFreeCount`, `.writeBackCount`).
  - `updateOverlapRatio()` was moved to the backend in Task 3 — it must run inside the backend's own `BeginFrame`/`RecordResidency` so `FrameStats().overlapRatio` is populated; the core reads it via `FrameStats()`, it does not call the backend's `updateOverlapRatio` directly. (If Task 3 left `updateOverlapRatio` public and core-called, move that call into the backend now.)
- `DebugLastClassifyCounts()` builds its `ClassifyCounts` from `m_backend->FrameStats()` (`reusableCount`, `cleanFreeCount`, `writeBackCount`).

**A4. Gate (behavior-preserving — identical to Task 3's gate):**
```
cmake -S . -B build && cmake --build build --parallel --target vkspatial_tests
./build/test/vkspatial_tests --gtest_filter='*Directional*:*HostStore*:ResidencyBackend.*'
```
Expected: all PASS, unchanged (the core still runs on Streaming; only the call path changed from concrete to interface). Do NOT edit existing tests.

**A5. Commit Part A** before starting Part B:
```bash
git add src/Engine/Spatial/IResidencyBackend.h src/Engine/Spatial/StreamingResidencyBackend.h \
        src/Engine/Spatial/StreamingResidencyBackend.cpp src/Engine/Spatial/DirectionalTSDF.h \
        src/Engine/Spatial/DirectionalTSDF.cpp
git commit -m "refactor(spatial): expand IResidencyBackend so Integrate is backend-agnostic"
```

### Part B: `UnifiedResidencyBackend` (UMA coherent pool)

**Files:**
- Create: `src/Engine/Spatial/UnifiedResidencyBackend.h`, `src/Engine/Spatial/UnifiedResidencyBackend.cpp`
- Test: `test/test_residencyBackend.cpp` (append)

**Part B note:** `UnifiedResidencyBackend` implements the **expanded** interface from Part A, i.e. also `RecordResidency` and `ResidentIndex` and fills `ResidencyStats.{reusableCount,cleanFreeCount,gpuSubmits}`:
- `RecordResidency(required, batch)` = the same CPU slot bookkeeping as `EnsureResident` (append first-seen slots, relabel indexGrid via the mapped pointers); it records **nothing** into `batch` (no GPU copy on UMA) and returns the number of slots touched. `gpuSubmits = 0`.
- `ResidentIndex()` returns the backend's own `m_slotOf` map (its type is already `unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash>`).
- `FrameStats()`: `missingCount=0`, `h2dBytes=0`, `d2hBytes=0`, `writeBackCount=0`, `gpuSubmits=0`; `reusableCount` = slots re-registered inside the window this frame; `cleanFreeCount` = 0.

**Interfaces:**
- Consumes: `IResidencyBackend`, `Engine::Core::Context` (`ctx.allocator`, `ctx.physicalDevice`, `ctx.device`).
- Produces: `class UnifiedResidencyBackend : public IResidencyBackend` + `void Build(Engine::Core::Context&, uint32_t initialCapacity)`.

**Design:** Allocate ONE coherent, persistently-mapped VMA buffer for the pool voxels (`GpuTsdfVoxel[capacity*512]`) and one for the indexGrid (`uint32[kIndexGridCells]`), plus a device meta buffer. Because CPU and GPU share it, `EnsureResident` never copies: it appends a slot for each first-seen key (writing zero-filled voxels straight through the mapped pointer) and relabels the indexGrid for the current window. `MetaBuffer` still needs `ActiveGroupMeta` for the integrate/extract kernels; on UMA it too is coherent-mapped and written by the CPU. `EndFrame` is a no-op (cold-compaction is a later plan).

- [ ] **Step 1: Write the failing test** — unified backend keeps groups resident with no host store round-trip

Append to `test/test_residencyBackend.cpp`:
```cpp
#include "Engine/Spatial/UnifiedResidencyBackend.h"

TEST(ResidencyBackend, UnifiedKeepsGroupsResidentZeroCopy) {
    Engine::Core::Context ctx;
    UnifiedResidencyBackend be;
    be.Build(ctx, /*initialCapacity=*/4096);
    ASSERT_TRUE(be.IsUnified());
    ASSERT_TRUE(be.SupportsZeroCopyCpuAccess());

    be.BeginFrame(Eigen::Vector3i(0, 0, 0));
    DirectionalGroupKey k{1, 2, 3, /*direction=*/0};
    be.EnsureResident({k});
    EXPECT_NE(be.DebugQueryPoolIndex(k), kInvalidPoolIndex);
    EXPECT_EQ(be.FrameStats().missingCount, 0u); // nothing "missing" — everything is resident
    EXPECT_EQ(be.FrameStats().h2dBytes, 0u);     // zero-copy: no upload bytes

    // Same key next frame → still resident, same slot (reuse), no growth.
    uint32_t slot = be.DebugQueryPoolIndex(k);
    be.BeginFrame(Eigen::Vector3i(0, 0, 0));
    be.EnsureResident({k});
    EXPECT_EQ(be.DebugQueryPoolIndex(k), slot);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --parallel --target vkspatial_tests`
Expected: FAIL to compile — `UnifiedResidencyBackend.h` not found.

- [ ] **Step 3: Create `UnifiedResidencyBackend.h`**

```cpp
#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/IResidencyBackend.h"

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <vk_mem_alloc.h>

namespace Engine::Spatial {

    // UMA backend: one coherent DEVICE_LOCAL|HOST_VISIBLE pool holds the whole model, so
    // residency is CPU-side slot bookkeeping + indexGrid relabel with no GPU copies.
    class UnifiedResidencyBackend : public IResidencyBackend {
    public:
        ~UnifiedResidencyBackend() override;
        void Build(Engine::Core::Context &ctx, uint32_t initialCapacity);

        void BeginFrame(const Eigen::Vector3i &localBase) override;
        void EnsureResident(const std::vector<DirectionalGroupKey> &required) override;
        void EndFrame() override {}

        // Expanded-interface methods (Part A). On UMA the batched path is the same CPU
        // bookkeeping as EnsureResident and records nothing into the command batch.
        uint32_t RecordResidency(const std::vector<DirectionalGroupKey> &required,
                                 Engine::Compute::CommandBatch &batch) override {
            EnsureResident(required);
            (void)batch;
            return uint32_t(required.size());
        }
        const std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> &
        ResidentIndex() const override { return m_slotOf; }

        VkBuffer IndexGridBuffer() const override { return m_indexGrid.buffer; }
        VkBuffer PoolVoxelBuffer() const override { return m_pool.buffer; }
        VkBuffer MetaBuffer() const override { return m_meta.buffer; }
        uint32_t PoolCapacity() const override { return m_capacity; }
        Eigen::Vector3i LocalBase() const override { return m_localBase; }
        bool IsUnified() const override { return true; }
        bool SupportsZeroCopyCpuAccess() const override { return true; }
        DirectionalHostStore &HostStore() override { return m_storeView; }
        ResidencyStats FrameStats() const override { return m_stats; }
        std::vector<uint32_t> DebugDownloadIndexGrid() override;
        uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &key) override;
        DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &key) override;

    private:
        struct MappedBuffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            void *mapped = nullptr;
        };
        MappedBuffer allocCoherent(uint32_t bytes, VkBufferUsageFlags usage);
        void freeBuffer(MappedBuffer &b);
        uint32_t offsetOf(const Eigen::Vector3i &g, uint8_t dir) const; // indexGrid cell, or kInvalidPoolIndex if outside window

        Engine::Core::Context *m_ctx = nullptr;
        uint32_t m_capacity = 0;
        Eigen::Vector3i m_localBase = Eigen::Vector3i::Zero();
        MappedBuffer m_pool;       // GpuTsdfVoxel[capacity*512]
        MappedBuffer m_indexGrid;  // uint32[kIndexGridCells]
        MappedBuffer m_meta;       // ActiveGroupMeta[capacity]
        std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> m_slotOf;
        DirectionalHostStore m_storeView; // unused authoritative-store placeholder for interface parity
        ResidencyStats m_stats;
    };

} // namespace Engine::Spatial
```

- [ ] **Step 4: Create `UnifiedResidencyBackend.cpp`**

```cpp
#include "Engine/Spatial/UnifiedResidencyBackend.h"

#include <cstring>
#include <stdexcept>

namespace Engine::Spatial {

    UnifiedResidencyBackend::MappedBuffer
    UnifiedResidencyBackend::allocCoherent(uint32_t bytes, VkBufferUsageFlags usage) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes;
        bi.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        // Ask for host access + persistent map; on UMA VMA returns a DEVICE_LOCAL|HOST_VISIBLE heap.
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        MappedBuffer b{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(m_ctx->allocator, &bi, &ai, &b.buffer, &b.alloc, &info) != VK_SUCCESS)
            throw std::runtime_error("UnifiedResidencyBackend: coherent alloc failed");
        b.mapped = info.pMappedData; // non-null because MAPPED_BIT was set
        std::memset(b.mapped, 0, bytes);
        return b;
    }

    void UnifiedResidencyBackend::freeBuffer(MappedBuffer &b) {
        if (b.buffer) vmaDestroyBuffer(m_ctx->allocator, b.buffer, b.alloc);
        b = {};
    }

    UnifiedResidencyBackend::~UnifiedResidencyBackend() {
        if (m_ctx) { freeBuffer(m_pool); freeBuffer(m_indexGrid); freeBuffer(m_meta); }
    }

    void UnifiedResidencyBackend::Build(Engine::Core::Context &ctx, uint32_t initialCapacity) {
        m_ctx = &ctx;
        m_capacity = initialCapacity;
        m_pool = allocCoherent(uint32_t(sizeof(GpuTsdfVoxel)) * kVoxelsPerGroup * m_capacity, 0);
        m_indexGrid = allocCoherent(uint32_t(sizeof(uint32_t)) * kIndexGridCells, 0);
        m_meta = allocCoherent(uint32_t(sizeof(ActiveGroupMeta)) * m_capacity, 0);
    }

    uint32_t UnifiedResidencyBackend::offsetOf(const Eigen::Vector3i &g, uint8_t dir) const {
        Eigen::Vector3i l = g - m_localBase;
        if (l.x() < 0 || l.y() < 0 || l.z() < 0 ||
            l.x() >= int(kLocalGroupGrid) || l.y() >= int(kLocalGroupGrid) || l.z() >= int(kLocalGroupGrid))
            return kInvalidPoolIndex;
        return IndexGridOffset(uint32_t(l.x()), uint32_t(l.y()), uint32_t(l.z()), dir);
    }

    void UnifiedResidencyBackend::BeginFrame(const Eigen::Vector3i &localBase) {
        m_localBase = localBase;
        m_stats = {};
        // Relabel: reset indexGrid, then re-register every resident slot that falls in the window.
        auto *grid = static_cast<uint32_t *>(m_indexGrid.mapped);
        for (uint32_t i = 0; i < kIndexGridCells; ++i) grid[i] = kInvalidPoolIndex;
        auto *meta = static_cast<ActiveGroupMeta *>(m_meta.mapped);
        for (const auto &kv : m_slotOf) {
            const DirectionalGroupKey &k = kv.first;
            uint32_t slot = kv.second;
            uint32_t cell = offsetOf(Eigen::Vector3i(k.gx, k.gy, k.gz), k.direction);
            if (cell != kInvalidPoolIndex) {
                grid[cell] = slot;
                meta[slot] = ActiveGroupMeta{k.gx, k.gy, k.gz,
                    PackMeta(k.direction, SlotState::ResidentClean, /*dirty=*/false, /*valid=*/true)};
                ++m_stats.residentCount;
            }
        }
    }

    void UnifiedResidencyBackend::EnsureResident(const std::vector<DirectionalGroupKey> &required) {
        auto *grid = static_cast<uint32_t *>(m_indexGrid.mapped);
        auto *meta = static_cast<ActiveGroupMeta *>(m_meta.mapped);
        auto *pool = static_cast<GpuTsdfVoxel *>(m_pool.mapped);
        for (const DirectionalGroupKey &k : required) {
            uint32_t cell = offsetOf(Eigen::Vector3i(k.gx, k.gy, k.gz), k.direction);
            if (cell == kInvalidPoolIndex)
                throw std::runtime_error("UnifiedResidencyBackend: required key outside window");
            auto it = m_slotOf.find(k);
            uint32_t slot;
            if (it == m_slotOf.end()) {
                if (m_slotOf.size() >= m_capacity)
                    throw std::runtime_error("UnifiedResidencyBackend: pool exhausted (grow not yet impl)");
                slot = uint32_t(m_slotOf.size());
                m_slotOf.emplace(k, slot);
                std::memset(pool + size_t(slot) * kVoxelsPerGroup, 0,
                            sizeof(GpuTsdfVoxel) * kVoxelsPerGroup); // zero-fill on first touch
            } else {
                slot = it->second;
            }
            grid[cell] = slot;
            meta[slot] = ActiveGroupMeta{k.gx, k.gy, k.gz,
                PackMeta(k.direction, SlotState::ResidentClean, false, true)};
        }
        // Zero-copy: no H2D bytes, nothing "missing".
        m_stats.h2dBytes = 0;
        m_stats.missingCount = 0;
    }

    std::vector<uint32_t> UnifiedResidencyBackend::DebugDownloadIndexGrid() {
        auto *grid = static_cast<uint32_t *>(m_indexGrid.mapped);
        return std::vector<uint32_t>(grid, grid + kIndexGridCells);
    }

    uint32_t UnifiedResidencyBackend::DebugQueryPoolIndex(const DirectionalGroupKey &key) {
        auto it = m_slotOf.find(key);
        return it == m_slotOf.end() ? kInvalidPoolIndex : it->second;
    }

    DirectionalHostStore::Group UnifiedResidencyBackend::DebugDownloadGroupVoxels(const DirectionalGroupKey &key) {
        DirectionalHostStore::Group out{}; // value/weight form
        auto it = m_slotOf.find(key);
        if (it == m_slotOf.end()) return out;
        auto *pool = static_cast<GpuTsdfVoxel *>(m_pool.mapped);
        const GpuTsdfVoxel *g = pool + size_t(it->second) * kVoxelsPerGroup;
        for (uint32_t i = 0; i < kVoxelsPerGroup; ++i) {
            out[i].weight = float(g[i].sumW) / float(kTsdfFixedScale);
            out[i].value = g[i].sumW ? float(g[i].sumDW) / float(g[i].sumW) : 0.0f;
        }
        return out;
    }

} // namespace Engine::Spatial
```

- [ ] **Step 5: Register in CMake** (same rule as Task 2, Step 5), then build.

Run: `cmake --build build --parallel --target vkspatial_tests`
Expected: compiles.

- [ ] **Step 6: Run test to verify it passes**

Run: `./build/test/vkspatial_tests --gtest_filter='ResidencyBackend.UnifiedKeepsGroupsResidentZeroCopy'`
Expected: PASS. (If it throws "coherent alloc failed", the device is not UMA — this test must run on Apple Silicon or an integrated GPU. Gate it with a runtime skip if the CI device is discrete: query memory props and `GTEST_SKIP()` when no `DEVICE_LOCAL|HOST_VISIBLE` heap exists.)

- [ ] **Step 7: Commit**

```bash
git add src/Engine/Spatial/UnifiedResidencyBackend.h src/Engine/Spatial/UnifiedResidencyBackend.cpp \
        src/Engine/Spatial/CMakeLists.txt test/test_residencyBackend.cpp
git commit -m "feat(spatial): add UnifiedResidencyBackend (UMA zero-copy residency)"
```

---

## Task 5: Backend auto-selection factory

**Files:**
- Create: `src/Engine/Spatial/ResidencyBackendFactory.cpp`
- Test: `test/test_residencyBackend.cpp` (append)

**Interfaces:**
- Consumes: `StreamingResidencyBackend`, `UnifiedResidencyBackend`.
- Produces: definition of `MakeResidencyBackend` (declared in Task 1).

**Selection rule:** env `VKLBVH_RESIDENCY` (`unified`/`streaming`) wins if set; else honor the explicit `mode` arg; else (`Auto`) pick `Unified` iff a memory heap is both `DEVICE_LOCAL` and `HOST_VISIBLE` and its size ≥ the model budget (`sizeof(GpuTsdfVoxel)*kVoxelsPerGroup*poolCapacity + indexGrid + meta`), else `Streaming`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_residencyBackend.cpp`:
```cpp
#include "Engine/Spatial/IResidencyBackend.h"
#include <cstdlib>

TEST(ResidencyBackend, FactoryHonorsOverride) {
    Engine::Core::Context ctx;
#ifdef _WIN32
    _putenv_s("VKLBVH_RESIDENCY", "streaming");
#else
    setenv("VKLBVH_RESIDENCY", "streaming", 1);
#endif
    auto be = MakeResidencyBackend(ctx, 1024);
    ASSERT_NE(be, nullptr);
    EXPECT_FALSE(be->IsUnified());
#ifndef _WIN32
    unsetenv("VKLBVH_RESIDENCY");
#endif
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --parallel --target vkspatial_tests`
Expected: FAIL to link — `MakeResidencyBackend` undefined.

- [ ] **Step 3: Implement the factory**

`src/Engine/Spatial/ResidencyBackendFactory.cpp`:
```cpp
#include "Engine/Spatial/IResidencyBackend.h"
#include "Engine/Spatial/StreamingResidencyBackend.h"
#include "Engine/Spatial/UnifiedResidencyBackend.h"

#include <cstdlib>
#include <cstring>

namespace Engine::Spatial {

    static bool hasUnifiedHeap(VkPhysicalDevice dev, VkDeviceSize needed) {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(dev, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            bool devLocal = f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            bool hostVis = f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            if (devLocal && hostVis &&
                mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size >= needed)
                return true;
        }
        return false;
    }

    std::unique_ptr<IResidencyBackend>
    MakeResidencyBackend(Engine::Core::Context &ctx, uint32_t poolCapacity, ResidencyMode mode) {
        const char *ov = std::getenv("VKLBVH_RESIDENCY");
        VkDeviceSize needed = VkDeviceSize(sizeof(GpuTsdfVoxel)) * kVoxelsPerGroup * poolCapacity +
                              VkDeviceSize(sizeof(uint32_t)) * kIndexGridCells +
                              VkDeviceSize(sizeof(ActiveGroupMeta)) * poolCapacity;

        bool useUnified;
        if (ov && std::strcmp(ov, "unified") == 0)         useUnified = true;   // env escape hatch wins
        else if (ov && std::strcmp(ov, "streaming") == 0)  useUnified = false;
        else if (mode == ResidencyMode::Unified)           useUnified = true;   // explicit request
        else if (mode == ResidencyMode::Streaming)         useUnified = false;
        else useUnified = hasUnifiedHeap(ctx.physicalDevice, needed);           // Auto: probe topology

        if (useUnified) {
            auto b = std::make_unique<UnifiedResidencyBackend>();
            b->Build(ctx, poolCapacity);
            return b;
        }
        auto b = std::make_unique<StreamingResidencyBackend>();
        b->Build(ctx, poolCapacity);
        return b;
    }

} // namespace Engine::Spatial
```

- [ ] **Step 4: Register in CMake** (same rule), build, run.

Run: `cmake --build build --parallel --target vkspatial_tests && ./build/test/vkspatial_tests --gtest_filter='ResidencyBackend.FactoryHonorsOverride'`
Expected: PASS.

- [ ] **Step 5: Wire the factory into `DirectionalTSDF::Build` behind a `ResidencyMode` arg**

In `DirectionalTSDF.h`, add a trailing defaulted arg to `Build` (backward-compatible — existing callers pass nothing new):
```cpp
        void Build(Engine::Core::Context &ctx,
                   float voxelSize = 0.1f, float truncation = 0.3f,
                   uint32_t poolCapacity = 32768, uint32_t maxPoints = 1u << 15,
                   uint32_t maxCandidates = 1u << 16,
                   ResidencyMode residency = ResidencyMode::Streaming);
```
Default is **Streaming** so every existing test (which calls `Build(ctx)` or `Build(ctx, ...)`) keeps the streaming backend on all platforms — including the UMA M4 Max — with **zero test edits**. In `DirectionalTSDF.cpp::Build`, replace the hardcoded `std::make_unique<StreamingResidencyBackend>()` from Task 3 with `m_backend = MakeResidencyBackend(*m_ctx, poolCapacity, residency);` (include `IResidencyBackend.h`). The `Auto`/product-facing default can flip to `ResidencyMode::Auto` in a later plan once quality parity is proven; keeping it Streaming here preserves the refactor's characterization gate.

- [ ] **Step 6: Confirm existing tests still pass unchanged (default Streaming)**

Run: `cmake --build build --parallel --target vkspatial_tests && ./build/test/vkspatial_tests --gtest_filter='*Directional*:*HostStore*'`
Expected: all PASS (default `Build` still selects Streaming; no existing test was edited).

- [ ] **Step 7: Commit**

```bash
git add src/Engine/Spatial/ResidencyBackendFactory.cpp src/Engine/Spatial/CMakeLists.txt \
        src/Engine/Spatial/DirectionalTSDF.h src/Engine/Spatial/DirectionalTSDF.cpp \
        test/test_residencyBackend.cpp
git commit -m "feat(spatial): auto-select residency backend + ResidencyMode on DirectionalTSDF"
```

---

## Task 6: Cross-backend correctness test (the abstraction contract)

**Files:**
- Test: `test/test_residencyBackend.cpp` (append)

**Interfaces:**
- Consumes: `DirectionalTSDF::Build(..., ResidencyMode)` (Task 5) — selects the backend explicitly, no env needed.

**Contract:** For identical input and parameters, the reconstruction is identical (within ε) whichever backend is used. This is what proves the residency abstraction is sound.

- [ ] **Step 1: Write the test** — run the same synthetic integration through both backends and compare point clouds

Append to `test/test_residencyBackend.cpp`:
```cpp
#include "Engine/Spatial/DirectionalTSDF.h"
#include <algorithm>

namespace {
// Small synthetic frame: a planar patch of samples with +Z normals near the origin.
void makePlane(std::vector<Eigen::Vector3f> &pts, std::vector<Eigen::Vector3f> &nrm) {
    pts.clear(); nrm.clear();
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) {
            pts.emplace_back(i * 0.05f, j * 0.05f, 0.0f);
            nrm.emplace_back(0.0f, 0.0f, 1.0f);
        }
}

std::vector<Engine::Spatial::ExtractedPoint> runWith(Engine::Spatial::ResidencyMode mode) {
    Engine::Core::Context ctx;
    Engine::Spatial::DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 32768, 1u << 15, 1u << 16, mode);
    std::vector<Eigen::Vector3f> pts, nrm;
    makePlane(pts, nrm);
    tsdf.Integrate(pts, nrm, Eigen::Vector3f(0, 0, 1), Eigen::Vector3f::Zero());
    auto pc = tsdf.PointCloud();
    std::sort(pc.begin(), pc.end(), [](auto &a, auto &b) {
        if (a.position.x() != b.position.x()) return a.position.x() < b.position.x();
        if (a.position.y() != b.position.y()) return a.position.y() < b.position.y();
        return a.position.z() < b.position.z();
    });
    return pc;
}

// Build a DirectionalTSDF forcing Unified; false if the device has no UMA heap.
bool unifiedAvailable() {
    Engine::Core::Context ctx;
    try {
        Engine::Spatial::DirectionalTSDF t;
        t.Build(ctx, 0.1f, 0.3f, 32768, 1u << 15, 1u << 16,
                Engine::Spatial::ResidencyMode::Unified);
    } catch (...) {
        return false;
    }
    return true;
}
} // namespace

TEST(ResidencyBackend, CrossBackendReconstructionMatches) {
    if (!unifiedAvailable())
        GTEST_SKIP() << "no UMA heap on this device; cross-backend test needs unified support";

    // Explicit modes — no env. (If VKLBVH_RESIDENCY is set it would override; the
    // FactoryHonorsOverride test clears it, so nothing leaks into this run.)
    auto a = runWith(Engine::Spatial::ResidencyMode::Streaming);
    auto b = runWith(Engine::Spatial::ResidencyMode::Unified);
    ASSERT_EQ(a.size(), b.size());
    const float eps = 1e-3f; // ε: positions match to 1 micron at 0.1mm voxel scale
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_LT((a[i].position - b[i].position).norm(), eps) << "point " << i;
        EXPECT_LT((a[i].normal - b[i].normal).norm(), eps) << "point " << i;
    }
}
```

- [ ] **Step 2: Build and run**

Run: `cmake --build build --parallel --target vkspatial_tests && ./build/test/vkspatial_tests --gtest_filter='ResidencyBackend.CrossBackendReconstructionMatches'`
Expected: PASS on a UMA device; SKIP on a discrete-only device.

- [ ] **Step 3: Run the whole suite to confirm nothing regressed**

Run: `./build/test/vkspatial_tests`
Expected: all PASS (or SKIP where UMA-gated).

- [ ] **Step 4: Commit**

```bash
git add test/test_residencyBackend.cpp
git commit -m "test(spatial): cross-backend reconstruction equivalence"
```

---

## Out of scope (follow-on plans)

These correspond to spec phases 4–6 and get their own plans after this one lands:
- **Fixed-point 8B unification** (`DirVoxel{sumDW,sumW}`, collapse host/GPU formats) — spec §2.
- **Quality core**: multi-direction soft integration, confidence-weighted fusion, directional candidate merge/split — spec §7.
- **UMA quality bonus**: global re-extraction + CPU/GPU zero-copy co-refinement; discrete recomputeMask partial extraction — spec §8.
- **Performance**: subgroup classification, chunked staging + H2D/D2H overlap (discrete), UMA cold compaction — spec §11 phase 6.

---

## Self-Review Notes

- **Spec coverage (this plan's slice):** §0 layered architecture → Tasks 1–3; §4 `IResidencyBackend` → Task 1; §4.1 `UnifiedResidencyBackend` → Task 4; §5 `StreamingResidencyBackend` → Tasks 2–3; §6 auto-selection → Task 5; §10 cross-backend correctness → Task 6. Spec §2/§7/§8/§11-phase6 are explicitly deferred above.
- **Open values pinned here:** cross-backend ε = `1e-3` (Task 6); soft-weight `p`/`maxDirections` belong to the deferred quality plan, not this one.
- **Type consistency:** interface method names (`BeginFrame(Eigen::Vector3i)`, `EnsureResident`, `EndFrame`, `PoolVoxelBuffer`, `IndexGridBuffer`, `MetaBuffer`, `PoolCapacity`, `LocalBase`, `FrameStats`→`ResidencyStats`, `DebugQueryPoolIndex`) are used identically across Tasks 1/2/4/5/6.
- **Known risk:** `Engine::Core` has a documented large-N (≳1000) GPU compute non-determinism bug (`docs/KNOWN_ISSUES_engine_core_large_n.md`). Task 6's synthetic frame uses 289 samples (17×17), well under that threshold, so the cross-backend comparison is not confounded by it.

# DirectionalTSDF Phase 1 (기본 streaming cache) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 slice of `Engine::Spatial::DirectionalTSDF` — shared types, the CPU-side `DirectionalHostStore`, GPU buffers (indexGrid / active pool / meta), and the missing-group upload path (host store → CPU fixed-point encode → staged multi-region copy → GPU register kernel → indexGrid), verified by GTest.

**Architecture:** Three new units under `src/Engine/Spatial/` (types header, host store, orchestrator) plus one new compute shader. Phase 1 uses "full reload" frame semantics: `BeginFrame` frees every pool slot and resets the indexGrid; `EnsureResident` treats every requested group as missing, uploads it from the host store into sequentially-allocated pool slots, and registers the slots in the indexGrid via `directional_tsdf_register_reusable.comp`. Phase 2 will replace the free-everything step with device-side classification + reuse; the register kernel is deliberately written to take *any* slot list so Phase 2 can reuse it unchanged.

**Tech Stack:** C++17, Vulkan 1.3 via `Engine::Core` (`Context`/`Buffer`/`ComputePipeline`/`SubmitOneShot`), GLSL compute (shaderc, runtime-compiled from `src/shader/`), Eigen, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-07-17-directional-tsdf-design.md` — read it if any step here is ambiguous. This plan covers only Phase 1 of the spec's 4-phase roadmap.

## Global Constraints

- All new code lives in `namespace Engine::Spatial`, no `vk` prefix on class names (matches `SimpleTSDF`).
- Error handling: throw `std::runtime_error("<ClassName>: <message>")`. No bool-returning fallible methods.
- GPU-shared structs must be `std::is_standard_layout_v` with `static_assert`ed sizes/offsets (GLSL std430 has no 8-bit members — pack small fields into a `uint32_t`).
- All GPU submissions are synchronous (`Engine::Core::Buffer::Upload/Download`, `ComputePipeline::Dispatch`, and raw `SubmitOneShot` all block via `vkQueueWaitIdle`). No barriers are needed between our single-command submissions.
- Scattered (non-contiguous) writes into the pool must NOT use `Buffer::Upload` directly (it always writes from offset 0). Use the pattern: temp `Engine::Core::Buffer` staging → `SubmitOneShot` + `vkCmdCopyBuffer` with one `VkBufferCopy` region per slot.
- CMake: `src/Engine/CMakeLists.txt` already GLOB-collects `Spatial/*.cpp` into the `EngineSpatial` target — no changes needed there. New shaders go in `src/shader/` (found via the `VKBVH_SHADER_DIR` define compiled into `EngineCore`).
- Test baseline before this plan: 51 tests, 50 pass, `WideBVHTest.RadiusMatchesCpuReference` is a known pre-existing unrelated failure. Every task's full-suite run must keep that baseline (plus its own new tests).
- Build commands used throughout (VULKAN_SDK must be exported):
  ```bash
  export VULKAN_SDK=/Users/sjy/VulkanSDK/1.4.328.1/macOS
  cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
  ```

---

### Task 1: `DirectionalTSDFTypes.h` + CPU-only tests

**Files:**
- Create: `src/Engine/Spatial/DirectionalTSDFTypes.h`
- Create: `test/test_directionalTSDF.cpp`
- Modify: `test/CMakeLists.txt` (add `Engine::Spatial` link)

**Interfaces:**
- Consumes: nothing (header-only, Eigen for `ExtractedPoint`).
- Produces (used by every later task): constants `kGroupDim`/`kVoxelsPerGroup`/`kLocalGroupGrid`/`kNumDirections`/`kIndexGridCells`/`kInvalidPoolIndex`/`kTsdfFixedScale`, `enum class SlotState`, `DirectionalGroupKey` + `DirectionalGroupKeyHash`, `HostTsdfVoxel{float value; float weight;}`, `GpuTsdfVoxel{int32_t sumDW; uint32_t sumW;}`, `ActiveGroupMeta{int32_t gx,gy,gz; uint32_t packed;}`, `PackMeta(direction, state, dirty, valid)` / `MetaDirection` / `MetaState` / `MetaDirty` / `MetaValid`, `IndexGridOffset(lx,ly,lz,direction)`, `DirectionalCandidate`, `ExtractedPoint`.

- [ ] **Step 1: Write the types header**

Create `src/Engine/Spatial/DirectionalTSDFTypes.h`:

```cpp
#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace Engine::Spatial {

    // Architecture constants from the DirectionalTSDF design doc (§3).
    constexpr uint32_t kGroupDim = 8;                                          // 8x8x8 voxels per group
    constexpr uint32_t kVoxelsPerGroup = kGroupDim * kGroupDim * kGroupDim;    // 512
    constexpr uint32_t kLocalGroupGrid = 50;                                   // 50x50x50 groups per local window
    constexpr uint32_t kNumDirections = 6;                                     // +X,-X,+Y,-Y,+Z,-Z
    constexpr uint32_t kIndexGridCells =
            kLocalGroupGrid * kLocalGroupGrid * kLocalGroupGrid * kNumDirections; // 750,000
    constexpr uint32_t kInvalidPoolIndex = 0xFFFFFFFFu;
    constexpr int32_t kTsdfFixedScale = 10000; // same fixed-point scale as SimpleTSDF

    enum class SlotState : uint8_t {
        Free = 0,
        ResidentClean = 1,
        ResidentDirty = 2,
        PendingUpload = 3,
        PendingWriteBack = 4,
    };

    struct DirectionalGroupKey {
        int32_t gx = 0;
        int32_t gy = 0;
        int32_t gz = 0;
        uint8_t direction = 0; // 0..5 = +X,-X,+Y,-Y,+Z,-Z

        bool operator==(const DirectionalGroupKey &o) const {
            return gx == o.gx && gy == o.gy && gz == o.gz && direction == o.direction;
        }
        bool operator!=(const DirectionalGroupKey &o) const { return !(*this == o); }
    };

    struct DirectionalGroupKeyHash {
        size_t operator()(const DirectionalGroupKey &k) const {
            uint64_t h = uint64_t(uint32_t(k.gx)) * 73856093ull;
            h ^= uint64_t(uint32_t(k.gy)) * 19349663ull;
            h ^= uint64_t(uint32_t(k.gz)) * 83492791ull;
            h ^= uint64_t(k.direction) * 2654435761ull;
            h ^= h >> 33;
            return size_t(h);
        }
    };

    // Host store / wire format: running-average form (value = avg SDF, weight = total weight).
    struct HostTsdfVoxel {
        float value = 0.0f;
        float weight = 0.0f;
    }; // 8B

    // GPU active-pool format: fixed-point accumulators so the integrate kernel can atomicAdd
    // (GLSL has no float atomics). Conversion: sumW = weight*scale, sumDW = value*weight*scale.
    struct GpuTsdfVoxel {
        int32_t sumDW = 0;
        uint32_t sumW = 0;
    }; // 8B

    // GLSL std430 has no 8-bit members, so direction/state/dirty/valid live in one packed uint.
    struct ActiveGroupMeta {
        int32_t gx = 0;
        int32_t gy = 0;
        int32_t gz = 0;
        uint32_t packed = 0; // bits 0-7 direction, 8-15 state, 16-23 dirty, 24-31 valid
    }; // 16B

    constexpr uint32_t PackMeta(uint8_t direction, SlotState state, bool dirty, bool valid) {
        return uint32_t(direction) | (uint32_t(state) << 8) |
               (uint32_t(dirty ? 1 : 0) << 16) | (uint32_t(valid ? 1 : 0) << 24);
    }
    constexpr uint8_t MetaDirection(uint32_t packed) { return uint8_t(packed & 0xFFu); }
    constexpr SlotState MetaState(uint32_t packed) { return SlotState((packed >> 8) & 0xFFu); }
    constexpr bool MetaDirty(uint32_t packed) { return ((packed >> 16) & 0xFFu) != 0; }
    constexpr bool MetaValid(uint32_t packed) { return ((packed >> 24) & 0xFFu) != 0; }

    // CPU mirror of the shader-side indexGrid addressing. lx/ly/lz are local group coords
    // (global - localBase), all in [0, kLocalGroupGrid).
    constexpr uint32_t IndexGridOffset(uint32_t lx, uint32_t ly, uint32_t lz, uint32_t direction) {
        return ((lz * kLocalGroupGrid + ly) * kLocalGroupGrid + lx) * kNumDirections + direction;
    }

    // Extraction candidate produced on the GPU (Phase 3); declared here so the GPU layout
    // is asserted alongside the other shared structs.
    struct DirectionalCandidate {
        float px = 0, py = 0, pz = 0;
        float nx = 0, ny = 0, nz = 0;
        int32_t gx = 0, gy = 0, gz = 0;
        uint32_t direction = 0;
    }; // 40B

    // CPU-side merged output point (Phase 3).
    struct ExtractedPoint {
        Eigen::Vector3f position = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal = Eigen::Vector3f::Zero();
        int32_t ownerGx = 0, ownerGy = 0, ownerGz = 0;
        uint8_t dirMask = 0;
    };

    static_assert(std::is_standard_layout_v<HostTsdfVoxel>);
    static_assert(sizeof(HostTsdfVoxel) == 8);
    static_assert(std::is_standard_layout_v<GpuTsdfVoxel>);
    static_assert(sizeof(GpuTsdfVoxel) == 8);
    static_assert(offsetof(GpuTsdfVoxel, sumW) == 4);
    static_assert(std::is_standard_layout_v<ActiveGroupMeta>);
    static_assert(sizeof(ActiveGroupMeta) == 16);
    static_assert(offsetof(ActiveGroupMeta, packed) == 12);
    static_assert(std::is_standard_layout_v<DirectionalCandidate>);
    static_assert(sizeof(DirectionalCandidate) == 40);

} // namespace Engine::Spatial
```

- [ ] **Step 2: Write the failing tests**

Create `test/test_directionalTSDF.cpp`:

```cpp
#include <gtest/gtest.h>

#include "Engine/Spatial/DirectionalTSDFTypes.h"

#include <unordered_set>

using namespace Engine::Spatial;

TEST(DirectionalGroupKeyTest, EqualityComparesAllFields) {
    DirectionalGroupKey a{1, 2, 3, 4};
    DirectionalGroupKey b{1, 2, 3, 4};
    DirectionalGroupKey c{1, 2, 3, 5};
    DirectionalGroupKey d{-1, 2, 3, 4};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);
}

TEST(DirectionalGroupKeyTest, HashSpreadsAcrossDirectionsAndCoords) {
    DirectionalGroupKeyHash hash;
    std::unordered_set<size_t> seen;
    for (int g = -2; g <= 2; ++g)
        for (uint8_t d = 0; d < kNumDirections; ++d)
            seen.insert(hash(DirectionalGroupKey{g, -g, g * 3, d}));
    // 30 keys — expect essentially no collisions from a decent hash.
    EXPECT_GT(seen.size(), 25u);
}

TEST(DirectionalTSDFTypesTest, MetaPackRoundTrips) {
    uint32_t packed = PackMeta(5, SlotState::ResidentDirty, true, true);
    EXPECT_EQ(MetaDirection(packed), 5);
    EXPECT_EQ(MetaState(packed), SlotState::ResidentDirty);
    EXPECT_TRUE(MetaDirty(packed));
    EXPECT_TRUE(MetaValid(packed));

    packed = PackMeta(0, SlotState::Free, false, false);
    EXPECT_EQ(MetaDirection(packed), 0);
    EXPECT_EQ(MetaState(packed), SlotState::Free);
    EXPECT_FALSE(MetaDirty(packed));
    EXPECT_FALSE(MetaValid(packed));
}

TEST(DirectionalTSDFTypesTest, IndexGridOffsetMatchesLayout) {
    // Layout: ((lz*50 + ly)*50 + lx)*6 + dir
    EXPECT_EQ(IndexGridOffset(0, 0, 0, 0), 0u);
    EXPECT_EQ(IndexGridOffset(0, 0, 0, 5), 5u);
    EXPECT_EQ(IndexGridOffset(1, 0, 0, 0), 6u);
    EXPECT_EQ(IndexGridOffset(0, 1, 0, 0), 50u * 6u);
    EXPECT_EQ(IndexGridOffset(0, 0, 1, 0), 50u * 50u * 6u);
    EXPECT_EQ(IndexGridOffset(49, 49, 49, 5), kIndexGridCells - 1u);
}
```

- [ ] **Step 3: Link `Engine::Spatial` into the test target**

In `test/CMakeLists.txt`, replace:
```cmake
target_link_libraries(vkspatial_tests
        PRIVATE
        vkSpatial::vkSpatial
        Engine::Core
        GTest::gtest
        GTest::gtest_main)
```
with:
```cmake
target_link_libraries(vkspatial_tests
        PRIVATE
        vkSpatial::vkSpatial
        Engine::Core
        Engine::Spatial
        GTest::gtest
        GTest::gtest_main)
```
(`file(GLOB TEST_SOURCES ...)` picks up the new test file automatically after a CMake reconfigure.)

- [ ] **Step 4: Build and run the new tests**

```bash
export VULKAN_SDK=/Users/sjy/VulkanSDK/1.4.328.1/macOS
cmake -S . -B build -DVULKAN_SDK="$VULKAN_SDK"
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="DirectionalGroupKeyTest.*:DirectionalTSDFTypesTest.*"
```
Expected: 4 tests, all PASS. (These are compile-time+CPU tests; failure means the header has a bug, not the GPU.)

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Spatial/DirectionalTSDFTypes.h test/test_directionalTSDF.cpp test/CMakeLists.txt
git commit -m "Add DirectionalTSDF shared types and constants"
```

---

### Task 2: `DirectionalHostStore`

**Files:**
- Create: `src/Engine/Spatial/DirectionalHostStore.h`
- Create: `src/Engine/Spatial/DirectionalHostStore.cpp`
- Modify: `test/test_directionalTSDF.cpp` (append tests)

**Interfaces:**
- Consumes: `DirectionalGroupKey`/`DirectionalGroupKeyHash`/`HostTsdfVoxel`/`kVoxelsPerGroup` from Task 1.
- Produces (used by Task 4):
  ```cpp
  class DirectionalHostStore {
      using Group = std::array<HostTsdfVoxel, kVoxelsPerGroup>;
      bool Contains(const DirectionalGroupKey &key) const;
      const Group &Get(const DirectionalGroupKey &key) const; // throws std::runtime_error if absent
      Group &GetOrCreate(const DirectionalGroupKey &key);      // zero-filled on first touch
      void Put(const DirectionalGroupKey &key, const Group &data);
      size_t Size() const;
  };
  ```

- [ ] **Step 1: Write the failing tests**

Append to `test/test_directionalTSDF.cpp` (add `#include "Engine/Spatial/DirectionalHostStore.h"` and `#include <stdexcept>` at the top):

```cpp
TEST(DirectionalHostStoreTest, GetOrCreateInitializesZeroedGroup) {
    DirectionalHostStore store;
    DirectionalGroupKey key{10, -3, 7, 2};
    EXPECT_FALSE(store.Contains(key));

    auto &group = store.GetOrCreate(key);
    EXPECT_TRUE(store.Contains(key));
    EXPECT_EQ(store.Size(), 1u);
    for (const auto &v : group) {
        EXPECT_EQ(v.value, 0.0f);
        EXPECT_EQ(v.weight, 0.0f);
    }
}

TEST(DirectionalHostStoreTest, PutGetRoundTrip) {
    DirectionalHostStore store;
    DirectionalGroupKey key{0, 0, 0, 0};

    DirectionalHostStore::Group group{};
    group[0] = {0.5f, 2.0f};
    group[511] = {-0.25f, 1.0f};
    store.Put(key, group);

    const auto &loaded = store.Get(key);
    EXPECT_FLOAT_EQ(loaded[0].value, 0.5f);
    EXPECT_FLOAT_EQ(loaded[0].weight, 2.0f);
    EXPECT_FLOAT_EQ(loaded[511].value, -0.25f);
    EXPECT_FLOAT_EQ(loaded[511].weight, 1.0f);
}

TEST(DirectionalHostStoreTest, GetMissingKeyThrows) {
    DirectionalHostStore store;
    EXPECT_THROW(store.Get(DirectionalGroupKey{1, 1, 1, 1}), std::runtime_error);
}

TEST(DirectionalHostStoreTest, KeysWithDifferentDirectionsAreDistinct) {
    DirectionalHostStore store;
    store.GetOrCreate(DirectionalGroupKey{3, 3, 3, 0})[0] = {0.1f, 1.0f};
    store.GetOrCreate(DirectionalGroupKey{3, 3, 3, 1})[0] = {0.9f, 1.0f};
    EXPECT_EQ(store.Size(), 2u);
    EXPECT_FLOAT_EQ(store.Get(DirectionalGroupKey{3, 3, 3, 0})[0].value, 0.1f);
    EXPECT_FLOAT_EQ(store.Get(DirectionalGroupKey{3, 3, 3, 1})[0].value, 0.9f);
}
```

- [ ] **Step 2: Run to verify they fail to compile**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)" 2>&1 | tail -5
```
Expected: compile error — `DirectionalHostStore.h` not found.

- [ ] **Step 3: Write the implementation**

Create `src/Engine/Spatial/DirectionalHostStore.h`:

```cpp
#pragma once

#include "Engine/Spatial/DirectionalTSDFTypes.h"

#include <array>
#include <cstddef>
#include <unordered_map>

namespace Engine::Spatial {

    // Host-side authoritative sparse TSDF store (DirectionalTSDF design doc §11/§12).
    // Pure CPU data structure: the GPU active pool is a cache over this map.
    class DirectionalHostStore {
    public:
        using Group = std::array<HostTsdfVoxel, kVoxelsPerGroup>;

        bool Contains(const DirectionalGroupKey &key) const;
        // Throws std::runtime_error if the key is absent.
        const Group &Get(const DirectionalGroupKey &key) const;
        // Creates a zero-filled group on first touch.
        Group &GetOrCreate(const DirectionalGroupKey &key);
        void Put(const DirectionalGroupKey &key, const Group &data);
        size_t Size() const { return m_groups.size(); }

    private:
        std::unordered_map<DirectionalGroupKey, Group, DirectionalGroupKeyHash> m_groups;
    };

} // namespace Engine::Spatial
```

Create `src/Engine/Spatial/DirectionalHostStore.cpp`:

```cpp
#include "Engine/Spatial/DirectionalHostStore.h"

#include <stdexcept>

namespace Engine::Spatial {

    bool DirectionalHostStore::Contains(const DirectionalGroupKey &key) const {
        return m_groups.find(key) != m_groups.end();
    }

    const DirectionalHostStore::Group &
    DirectionalHostStore::Get(const DirectionalGroupKey &key) const {
        auto it = m_groups.find(key);
        if (it == m_groups.end())
            throw std::runtime_error("DirectionalHostStore: group not found");
        return it->second;
    }

    DirectionalHostStore::Group &
    DirectionalHostStore::GetOrCreate(const DirectionalGroupKey &key) {
        // operator[] value-initializes: HostTsdfVoxel's member initializers zero every voxel.
        return m_groups[key];
    }

    void DirectionalHostStore::Put(const DirectionalGroupKey &key, const Group &data) {
        m_groups[key] = data;
    }

} // namespace Engine::Spatial
```

- [ ] **Step 4: Build and run**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="DirectionalHostStoreTest.*"
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Spatial/DirectionalHostStore.h src/Engine/Spatial/DirectionalHostStore.cpp test/test_directionalTSDF.cpp
git commit -m "Add DirectionalHostStore (CPU sparse TSDF store)"
```

---

### Task 3: Register kernel + `DirectionalTSDF::Build`/`BeginFrame`

**Files:**
- Create: `src/shader/directional_tsdf_register_reusable.comp`
- Create: `src/Engine/Spatial/DirectionalTSDF.h`
- Create: `src/Engine/Spatial/DirectionalTSDF.cpp`
- Modify: `test/test_directionalTSDF.cpp` (append test)

**Interfaces:**
- Consumes: Task 1 types; `Engine::Core::Context`/`Buffer`/`ComputePipeline`/`SubmitOneShot`; Task 2 `DirectionalHostStore` (member, unused until Task 4).
- Produces (Task 4 fills in `EnsureResident`; Phase 2+ reuses everything):
  ```cpp
  class DirectionalTSDF {
      struct Stats { uint32_t residentCount, missingCount, writeBackCount, h2dBytes, d2hBytes; float overlapRatio; };
      void Build(Engine::Core::Context &ctx, float voxelSize = 0.1f, float truncation = 0.3f,
                 uint32_t poolCapacity = 32768);
      void BeginFrame(const Eigen::Vector3f &aabbCenterHint);
      void EnsureResident(const std::vector<DirectionalGroupKey> &required); // Task 4
      DirectionalHostStore &HostStore();
      Eigen::Vector3i LocalBase() const;
      float VoxelSize() const;
      float GroupWorldSize() const;            // voxelSize * kGroupDim
      Stats LastFrameStats() const;
      std::vector<uint32_t> DebugDownloadIndexGrid();
      uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &key);          // Task 4
      DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &key); // Task 4
  };
  ```
- Shader contract (`directional_tsdf_register_reusable.comp`): binding 0 = `uint slots[]` (pool indices to register), binding 1 = `ActiveGroupMeta meta[]` (readonly; `gx/gy/gz/packed` must already be correct for each listed slot), binding 2 = `uint indexGrid[]`; push constants `{uint count; int baseX, baseY, baseZ;}`. Each thread registers one slot: `indexGrid[((lz*50+ly)*50+lx)*6+dir] = poolIndex` where `l* = m.g* - base*`, skipping out-of-window slots. Used both for reusable-slot re-registration (Phase 2) and freshly-uploaded-slot registration (Task 4) — same kernel, different slot lists.

- [ ] **Step 1: Write the shader**

Create `src/shader/directional_tsdf_register_reusable.comp`:

```glsl
#version 460
layout(local_size_x = 256) in;

// Registers a list of active-pool slots into the indexGrid (DirectionalTSDF design doc §10).
// Works for both reusable slots (classification output, Phase 2) and freshly-uploaded
// missing-group slots (Phase 1): each listed slot's meta must already hold its group key.

#define LOCAL_GRID 50
#define NUM_DIRS   6u

struct ActiveGroupMeta {
    int  gx;
    int  gy;
    int  gz;
    uint packed; // bits 0-7 direction, 8-15 state, 16-23 dirty, 24-31 valid
};

layout(push_constant) uniform PC {
    uint g_count;
    int  g_baseX;
    int  g_baseY;
    int  g_baseZ;
};

layout(std430, set = 0, binding = 0) readonly buffer PoolIndexList { uint g_slots[]; };
layout(std430, set = 0, binding = 1) readonly buffer Meta { ActiveGroupMeta g_meta[]; };
layout(std430, set = 0, binding = 2) buffer IndexGrid { uint g_indexGrid[]; };

void main() {
    uint k = gl_GlobalInvocationID.x;
    if (k >= g_count) return;

    uint poolIndex = g_slots[k];
    ActiveGroupMeta m = g_meta[poolIndex];

    int lx = m.gx - g_baseX;
    int ly = m.gy - g_baseY;
    int lz = m.gz - g_baseZ;
    if (lx < 0 || ly < 0 || lz < 0 ||
        lx >= LOCAL_GRID || ly >= LOCAL_GRID || lz >= LOCAL_GRID)
        return;

    uint dir = m.packed & 0xFFu;
    uint cell = ((uint(lz) * uint(LOCAL_GRID) + uint(ly)) * uint(LOCAL_GRID) + uint(lx)) * NUM_DIRS + dir;
    g_indexGrid[cell] = poolIndex;
}
```

- [ ] **Step 2: Write `DirectionalTSDF.h`**

Create `src/Engine/Spatial/DirectionalTSDF.h`:

```cpp
#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalHostStore.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"

#include <Eigen/Core>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Engine::Spatial {

    // Directional TSDF with a host/GPU streaming cache
    // (docs/superpowers/specs/2026-07-17-directional-tsdf-design.md).
    //
    // Phase 1 scope: BeginFrame uses full-reload semantics (every slot freed each frame);
    // EnsureResident uploads every requested group from the host store and registers it in
    // the indexGrid. Device-side classification/reuse (Phase 2), integration/extraction
    // (Phase 3) and dirty write-back (Phase 4) come later.
    class DirectionalTSDF {
    public:
        struct Stats {
            uint32_t residentCount = 0;
            uint32_t missingCount = 0;
            uint32_t writeBackCount = 0;
            uint32_t h2dBytes = 0;
            uint32_t d2hBytes = 0;
            float overlapRatio = 0.0f;
        };

        void Build(Engine::Core::Context &ctx,
                   float voxelSize = 0.1f,
                   float truncation = 0.3f,
                   uint32_t poolCapacity = 32768);

        // Starts a frame: recomputes the local base (window centred on the hint, snapped to
        // the group grid) and resets the indexGrid to kInvalidPoolIndex.
        void BeginFrame(const Eigen::Vector3f &aabbCenterHint);

        // Makes the given groups resident: fetches each from the host store (zero-filled on
        // first touch), uploads into free pool slots, and registers them in the indexGrid.
        // Keys already resident this frame are skipped. Throws if a key lies outside the
        // current local window or the pool is exhausted.
        void EnsureResident(const std::vector<DirectionalGroupKey> &required);

        DirectionalHostStore &HostStore() { return m_hostStore; }
        Eigen::Vector3i LocalBase() const { return m_localBase; }
        float VoxelSize() const { return m_voxelSize; }
        float Truncation() const { return m_truncation; }
        float GroupWorldSize() const { return m_voxelSize * float(kGroupDim); }
        uint32_t PoolCapacity() const { return m_poolCapacity; }
        Stats LastFrameStats() const { return m_stats; }

        // Test/debug helpers — synchronous GPU downloads, not for per-frame use.
        std::vector<uint32_t> DebugDownloadIndexGrid();
        uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &key);
        DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &key);

    private:
        Engine::Core::Context *m_ctx = nullptr;
        float m_voxelSize = 0.1f;
        float m_truncation = 0.3f;
        uint32_t m_poolCapacity = 0;

        DirectionalHostStore m_hostStore;
        Eigen::Vector3i m_localBase = Eigen::Vector3i::Zero();

        std::unique_ptr<Engine::Core::Buffer> m_indexGrid;      // uint32[kIndexGridCells]
        std::unique_ptr<Engine::Core::Buffer> m_poolVoxels;     // GpuTsdfVoxel[poolCapacity*512]
        std::unique_ptr<Engine::Core::Buffer> m_metaBuffer;     // ActiveGroupMeta[poolCapacity]
        std::unique_ptr<Engine::Core::Buffer> m_slotListBuffer; // uint32[poolCapacity]
        std::unique_ptr<Engine::Core::ComputePipeline> m_registerKernel;

        // CPU mirror of slot occupancy: which key each slot currently holds.
        std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> m_residentIndex;
        std::vector<DirectionalGroupKey> m_slotKeys;
        uint32_t m_nextFreeSlot = 0;

        Stats m_stats;

        void fillIndexGridInvalid();
        Eigen::Vector3i quantizeLocalBase(const Eigen::Vector3f &center) const;
    };

} // namespace Engine::Spatial
```

- [ ] **Step 3: Write `DirectionalTSDF.cpp` (Build/BeginFrame/fill/quantize only; the rest throws)**

Create `src/Engine/Spatial/DirectionalTSDF.cpp`:

```cpp
#include "Engine/Spatial/DirectionalTSDF.h"

#include "Engine/Core/OneShotCommands.h"

#include <cmath>
#include <stdexcept>

namespace Engine::Spatial {

    namespace {
        constexpr uint32_t kGroupBytes = kVoxelsPerGroup * uint32_t(sizeof(GpuTsdfVoxel)); // 4096

        struct RegisterPC {
            uint32_t count;
            int32_t baseX;
            int32_t baseY;
            int32_t baseZ;
        };
    } // namespace

    void DirectionalTSDF::Build(Engine::Core::Context &ctx,
                                float voxelSize,
                                float truncation,
                                uint32_t poolCapacity) {
        m_ctx = &ctx;
        m_voxelSize = voxelSize;
        m_truncation = truncation;
        m_poolCapacity = poolCapacity;

        m_indexGrid = std::make_unique<Engine::Core::Buffer>(ctx);
        m_poolVoxels = std::make_unique<Engine::Core::Buffer>(ctx);
        m_metaBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_slotListBuffer = std::make_unique<Engine::Core::Buffer>(ctx);

        m_indexGrid->Allocate(kIndexGridCells * sizeof(uint32_t));
        m_poolVoxels->Allocate(poolCapacity * kGroupBytes);
        m_metaBuffer->Allocate(poolCapacity * sizeof(ActiveGroupMeta));
        m_slotListBuffer->Allocate(poolCapacity * sizeof(uint32_t));

        m_registerKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_registerKernel->Build("directional_tsdf_register_reusable.comp")
                .Bind(0, *m_slotListBuffer)
                .Bind(1, *m_metaBuffer)
                .Bind(2, *m_indexGrid);

        m_slotKeys.assign(poolCapacity, {});
        m_residentIndex.clear();
        m_nextFreeSlot = 0;
        m_stats = {};

        fillIndexGridInvalid();
    }

    void DirectionalTSDF::BeginFrame(const Eigen::Vector3f &aabbCenterHint) {
        if (!m_ctx)
            throw std::runtime_error("DirectionalTSDF: Build() must be called first");

        m_localBase = quantizeLocalBase(aabbCenterHint);

        // Phase 1: full-reload semantics — every frame starts from an empty pool.
        // Phase 2 replaces this with device-side classification + reusable re-registration.
        m_residentIndex.clear();
        m_slotKeys.assign(m_poolCapacity, {});
        m_nextFreeSlot = 0;
        m_stats = {};

        fillIndexGridInvalid();
    }

    void DirectionalTSDF::EnsureResident(const std::vector<DirectionalGroupKey> &required) {
        (void) required;
        throw std::runtime_error("DirectionalTSDF: EnsureResident not implemented yet");
    }

    std::vector<uint32_t> DirectionalTSDF::DebugDownloadIndexGrid() {
        std::vector<uint32_t> grid(kIndexGridCells);
        m_indexGrid->Download(grid.data(), kIndexGridCells * sizeof(uint32_t));
        return grid;
    }

    uint32_t DirectionalTSDF::DebugQueryPoolIndex(const DirectionalGroupKey &key) {
        (void) key;
        throw std::runtime_error("DirectionalTSDF: DebugQueryPoolIndex not implemented yet");
    }

    DirectionalHostStore::Group
    DirectionalTSDF::DebugDownloadGroupVoxels(const DirectionalGroupKey &key) {
        (void) key;
        throw std::runtime_error("DirectionalTSDF: DebugDownloadGroupVoxels not implemented yet");
    }

    void DirectionalTSDF::fillIndexGridInvalid() {
        VkBuffer grid = m_indexGrid->Handle();
        Engine::Core::SubmitOneShot(*m_ctx, Engine::Core::QueueRole::Compute,
                                    [&](VkCommandBuffer cmd) {
                                        vkCmdFillBuffer(cmd, grid, 0, VK_WHOLE_SIZE,
                                                        kInvalidPoolIndex);
                                    });
    }

    Eigen::Vector3i DirectionalTSDF::quantizeLocalBase(const Eigen::Vector3f &center) const {
        const float g = GroupWorldSize();
        const int half = int(kLocalGroupGrid) / 2;
        return Eigen::Vector3i(int(std::floor(center.x() / g)) - half,
                               int(std::floor(center.y() / g)) - half,
                               int(std::floor(center.z() / g)) - half);
    }

} // namespace Engine::Spatial
```

- [ ] **Step 4: Write the failing test**

Append to `test/test_directionalTSDF.cpp` (add `#include "Engine/Core/Context.h"` and `#include "Engine/Spatial/DirectionalTSDF.h"` at the top):

```cpp
TEST(DirectionalTSDFTest, BeginFrameResetsIndexGridAndComputesLocalBase) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/256);
    tsdf.BeginFrame(Eigen::Vector3f::Zero());

    // groupWorldSize = 0.1 * 8 = 0.8; floor(0/0.8) - 25 = -25 per axis.
    EXPECT_EQ(tsdf.LocalBase().x(), -25);
    EXPECT_EQ(tsdf.LocalBase().y(), -25);
    EXPECT_EQ(tsdf.LocalBase().z(), -25);
    EXPECT_FLOAT_EQ(tsdf.GroupWorldSize(), 0.8f);

    auto grid = tsdf.DebugDownloadIndexGrid();
    ASSERT_EQ(grid.size(), kIndexGridCells);
    size_t invalid = 0;
    for (uint32_t v : grid)
        if (v == kInvalidPoolIndex) ++invalid;
    EXPECT_EQ(invalid, grid.size());
}
```

- [ ] **Step 5: Build and run**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="DirectionalTSDFTest.BeginFrameResetsIndexGridAndComputesLocalBase"
```
Expected: PASS. (This also proves the shader compiles at `ComputePipeline::Build` time — a GLSL syntax error would throw inside `Build()`.)

- [ ] **Step 6: Commit**

```bash
git add src/shader/directional_tsdf_register_reusable.comp src/Engine/Spatial/DirectionalTSDF.h src/Engine/Spatial/DirectionalTSDF.cpp test/test_directionalTSDF.cpp
git commit -m "Add DirectionalTSDF skeleton: buffers, register kernel, BeginFrame"
```

---

### Task 4: `EnsureResident` — missing-group upload + indexGrid registration

**Files:**
- Modify: `src/Engine/Spatial/DirectionalTSDF.cpp` (replace the three not-implemented stubs)
- Modify: `test/test_directionalTSDF.cpp` (append tests)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: working `EnsureResident`, `DebugQueryPoolIndex`, `DebugDownloadGroupVoxels`. Phase 2's classification will slot in between `BeginFrame` and `EnsureResident` without changing these signatures.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_directionalTSDF.cpp`:

```cpp
TEST(DirectionalTSDFTest, SingleFrameUploadRegistersIndexGrid) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/256);
    tsdf.BeginFrame(Eigen::Vector3f::Zero());

    std::vector<DirectionalGroupKey> required = {
            {0, 0, 0, 0},
            {0, 0, 0, 3},   // same spatial group, different direction → separate slot
            {-1, 4, 2, 5},
    };
    tsdf.EnsureResident(required);

    auto grid = tsdf.DebugDownloadIndexGrid();
    std::unordered_set<uint32_t> slots;
    for (const auto &key : required) {
        uint32_t lx = uint32_t(key.gx - tsdf.LocalBase().x());
        uint32_t ly = uint32_t(key.gy - tsdf.LocalBase().y());
        uint32_t lz = uint32_t(key.gz - tsdf.LocalBase().z());
        uint32_t poolIndex = grid[IndexGridOffset(lx, ly, lz, key.direction)];
        EXPECT_NE(poolIndex, kInvalidPoolIndex);
        EXPECT_LT(poolIndex, 256u);
        slots.insert(poolIndex);
    }
    EXPECT_EQ(slots.size(), required.size()); // invariant #1: no duplicate slots

    // A key never requested stays invalid.
    EXPECT_EQ(tsdf.DebugQueryPoolIndex({5, 5, 5, 1}), kInvalidPoolIndex);

    auto stats = tsdf.LastFrameStats();
    EXPECT_EQ(stats.missingCount, 3u);
    EXPECT_EQ(stats.residentCount, 3u);
    EXPECT_EQ(stats.h2dBytes, uint32_t(3u * kVoxelsPerGroup * sizeof(GpuTsdfVoxel)));
}

TEST(DirectionalTSDFTest, EnsureResidentSkipsAlreadyResidentKeys) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 64);
    tsdf.BeginFrame(Eigen::Vector3f::Zero());

    DirectionalGroupKey key{1, 1, 1, 0};
    tsdf.EnsureResident({key});
    uint32_t slotBefore = tsdf.DebugQueryPoolIndex(key);

    tsdf.EnsureResident({key}); // second call: nothing new to upload
    EXPECT_EQ(tsdf.DebugQueryPoolIndex(key), slotBefore);
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 1u);
    EXPECT_EQ(tsdf.LastFrameStats().residentCount, 1u);
}

TEST(DirectionalTSDFTest, HostStoreValuesRoundTripThroughGpuPool) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 64);

    DirectionalGroupKey key{1, 2, 3, 4};
    DirectionalHostStore::Group group{};
    group[7] = {0.5f, 2.0f};
    group[200] = {-0.75f, 4.0f};
    tsdf.HostStore().Put(key, group);

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident({key});

    auto out = tsdf.DebugDownloadGroupVoxels(key);
    EXPECT_NEAR(out[7].value, 0.5f, 1e-3f);
    EXPECT_NEAR(out[7].weight, 2.0f, 1e-3f);
    EXPECT_NEAR(out[200].value, -0.75f, 1e-3f);
    EXPECT_NEAR(out[200].weight, 4.0f, 1e-3f);
    EXPECT_EQ(out[0].weight, 0.0f);
}

TEST(DirectionalTSDFTest, PoolExhaustionThrows) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/2);
    tsdf.BeginFrame(Eigen::Vector3f::Zero());

    std::vector<DirectionalGroupKey> required = {
            {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}};
    EXPECT_THROW(tsdf.EnsureResident(required), std::runtime_error);
}

TEST(DirectionalTSDFTest, KeyOutsideLocalWindowThrows) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 64);
    tsdf.BeginFrame(Eigen::Vector3f::Zero()); // window covers groups [-25, 25)

    EXPECT_THROW(tsdf.EnsureResident({DirectionalGroupKey{100, 0, 0, 0}}),
                 std::runtime_error);
}
```

- [ ] **Step 2: Run to verify they fail**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="DirectionalTSDFTest.*"
```
Expected: the new tests FAIL with "EnsureResident not implemented yet" (the Task 3 test still passes).

- [ ] **Step 3: Implement `EnsureResident` and the two debug helpers**

In `src/Engine/Spatial/DirectionalTSDF.cpp`, replace the `EnsureResident` stub with:

```cpp
    void DirectionalTSDF::EnsureResident(const std::vector<DirectionalGroupKey> &required) {
        if (!m_ctx)
            throw std::runtime_error("DirectionalTSDF: Build() must be called first");

        struct Pending {
            DirectionalGroupKey key;
            uint32_t slot;
        };
        std::vector<Pending> pending;
        pending.reserve(required.size());

        for (const auto &key : required) {
            const int lx = key.gx - m_localBase.x();
            const int ly = key.gy - m_localBase.y();
            const int lz = key.gz - m_localBase.z();
            if (lx < 0 || ly < 0 || lz < 0 ||
                lx >= int(kLocalGroupGrid) || ly >= int(kLocalGroupGrid) ||
                lz >= int(kLocalGroupGrid))
                throw std::runtime_error(
                        "DirectionalTSDF: required group is outside the local window");

            if (m_residentIndex.find(key) != m_residentIndex.end())
                continue;
            if (m_nextFreeSlot >= m_poolCapacity)
                throw std::runtime_error("DirectionalTSDF: active pool exhausted");

            const uint32_t slot = m_nextFreeSlot++;
            m_residentIndex.emplace(key, slot);
            m_slotKeys[slot] = key;
            pending.push_back({key, slot});
        }
        m_stats.residentCount = uint32_t(m_residentIndex.size());
        if (pending.empty())
            return;

        // Encode host groups into the GPU fixed-point format and build their meta entries.
        std::vector<GpuTsdfVoxel> voxelData(pending.size() * kVoxelsPerGroup);
        std::vector<ActiveGroupMeta> metaData(pending.size());
        for (size_t i = 0; i < pending.size(); ++i) {
            const auto &key = pending[i].key;
            const auto &group = m_hostStore.GetOrCreate(key);
            for (uint32_t v = 0; v < kVoxelsPerGroup; ++v) {
                const HostTsdfVoxel &h = group[v];
                GpuTsdfVoxel &g = voxelData[i * kVoxelsPerGroup + v];
                g.sumW = uint32_t(std::lround(double(h.weight) * kTsdfFixedScale));
                g.sumDW = int32_t(
                        std::lround(double(h.value) * double(h.weight) * kTsdfFixedScale));
            }
            metaData[i] = ActiveGroupMeta{key.gx, key.gy, key.gz,
                                          PackMeta(key.direction, SlotState::ResidentClean,
                                                   /*dirty=*/false, /*valid=*/true)};
        }

        // Voxel payloads → scattered pool slots: staging buffer + one multi-region copy.
        {
            const uint32_t bytes = uint32_t(voxelData.size() * sizeof(GpuTsdfVoxel));
            Engine::Core::Buffer staging(*m_ctx);
            staging.Allocate(bytes);
            staging.Upload(voxelData.data(), bytes);

            std::vector<VkBufferCopy> regions(pending.size());
            for (size_t i = 0; i < pending.size(); ++i) {
                regions[i].srcOffset = VkDeviceSize(i) * kGroupBytes;
                regions[i].dstOffset = VkDeviceSize(pending[i].slot) * kGroupBytes;
                regions[i].size = kGroupBytes;
            }
            VkBuffer src = staging.Handle();
            VkBuffer dst = m_poolVoxels->Handle();
            Engine::Core::SubmitOneShot(*m_ctx, Engine::Core::QueueRole::Compute,
                                        [&](VkCommandBuffer cmd) {
                                            vkCmdCopyBuffer(cmd, src, dst,
                                                            uint32_t(regions.size()),
                                                            regions.data());
                                        });
            m_stats.h2dBytes += bytes;
        }

        // Meta entries → scattered meta slots: same staging + multi-region pattern.
        {
            const uint32_t bytes = uint32_t(metaData.size() * sizeof(ActiveGroupMeta));
            Engine::Core::Buffer staging(*m_ctx);
            staging.Allocate(bytes);
            staging.Upload(metaData.data(), bytes);

            std::vector<VkBufferCopy> regions(pending.size());
            for (size_t i = 0; i < pending.size(); ++i) {
                regions[i].srcOffset = VkDeviceSize(i) * sizeof(ActiveGroupMeta);
                regions[i].dstOffset = VkDeviceSize(pending[i].slot) * sizeof(ActiveGroupMeta);
                regions[i].size = sizeof(ActiveGroupMeta);
            }
            VkBuffer src = staging.Handle();
            VkBuffer dst = m_metaBuffer->Handle();
            Engine::Core::SubmitOneShot(*m_ctx, Engine::Core::QueueRole::Compute,
                                        [&](VkCommandBuffer cmd) {
                                            vkCmdCopyBuffer(cmd, src, dst,
                                                            uint32_t(regions.size()),
                                                            regions.data());
                                        });
        }

        // Register the freshly-filled slots in the indexGrid.
        std::vector<uint32_t> slots(pending.size());
        for (size_t i = 0; i < pending.size(); ++i)
            slots[i] = pending[i].slot;
        m_slotListBuffer->Upload(slots.data(), uint32_t(slots.size() * sizeof(uint32_t)));

        RegisterPC pc{uint32_t(slots.size()), m_localBase.x(), m_localBase.y(),
                      m_localBase.z()};
        m_registerKernel->Args(pc).DispatchElements(uint32_t(slots.size()));

        m_stats.missingCount += uint32_t(pending.size());
    }
```

Replace the `DebugQueryPoolIndex` stub with:

```cpp
    uint32_t DirectionalTSDF::DebugQueryPoolIndex(const DirectionalGroupKey &key) {
        const int lx = key.gx - m_localBase.x();
        const int ly = key.gy - m_localBase.y();
        const int lz = key.gz - m_localBase.z();
        if (lx < 0 || ly < 0 || lz < 0 ||
            lx >= int(kLocalGroupGrid) || ly >= int(kLocalGroupGrid) ||
            lz >= int(kLocalGroupGrid))
            return kInvalidPoolIndex;

        auto grid = DebugDownloadIndexGrid();
        return grid[IndexGridOffset(uint32_t(lx), uint32_t(ly), uint32_t(lz), key.direction)];
    }
```

Replace the `DebugDownloadGroupVoxels` stub with:

```cpp
    DirectionalHostStore::Group
    DirectionalTSDF::DebugDownloadGroupVoxels(const DirectionalGroupKey &key) {
        auto it = m_residentIndex.find(key);
        if (it == m_residentIndex.end())
            throw std::runtime_error("DirectionalTSDF: group is not resident");
        const uint32_t slot = it->second;

        Engine::Core::Buffer staging(*m_ctx);
        staging.Allocate(kGroupBytes);
        VkBuffer src = m_poolVoxels->Handle();
        VkBuffer dst = staging.Handle();
        Engine::Core::SubmitOneShot(*m_ctx, Engine::Core::QueueRole::Compute,
                                    [&](VkCommandBuffer cmd) {
                                        VkBufferCopy region{};
                                        region.srcOffset = VkDeviceSize(slot) * kGroupBytes;
                                        region.dstOffset = 0;
                                        region.size = kGroupBytes;
                                        vkCmdCopyBuffer(cmd, src, dst, 1, &region);
                                    });

        std::vector<GpuTsdfVoxel> raw(kVoxelsPerGroup);
        staging.Download(raw.data(), kGroupBytes);

        DirectionalHostStore::Group group{};
        for (uint32_t v = 0; v < kVoxelsPerGroup; ++v) {
            group[v].weight = float(raw[v].sumW) / float(kTsdfFixedScale);
            group[v].value = raw[v].sumW > 0
                                     ? float(double(raw[v].sumDW) / double(raw[v].sumW))
                                     : 0.0f;
        }
        return group;
    }
```

- [ ] **Step 4: Build and run all DirectionalTSDF tests**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="Directional*"
```
Expected: all Directional* tests PASS (4 key/types + 4 host store + 6 TSDF = 14).

- [ ] **Step 5: Run the full suite as a regression check**

```bash
./build/test/vkspatial_tests 2>&1 | tail -8
```
Expected: previous baseline plus the new tests — only `WideBVHTest.RadiusMatchesCpuReference` failing (known pre-existing).

- [ ] **Step 6: Commit**

```bash
git add src/Engine/Spatial/DirectionalTSDF.cpp test/test_directionalTSDF.cpp
git commit -m "Implement DirectionalTSDF Phase 1 missing-group upload path"
```

---

## Phase 1 completion checklist

- [ ] All `Directional*` tests pass; full suite keeps the known baseline.
- [ ] `DebugQueryPoolIndex` returns distinct valid slots for uploaded groups, `kInvalidPoolIndex` otherwise (spec Phase 1 검증 항목).
- [ ] Report back for review before starting the Phase 2 plan (classification/reuse).

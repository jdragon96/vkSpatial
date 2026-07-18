# Pluggable BVH + Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a swappable acceleration-structure architecture to `Engine::Spatial` — an abstract `SpatialIndex` base with `BinaryLBVH` and `WideBVH` backends — plus a benchmark executable that compares them on build time, query time, memory, and accuracy.

**Architecture:** Runtime polymorphism. `SpatialIndex` defines the common surface (`Build<T>` / `KNN` / `RadiusSearch` / `Length` / `NodeCount` / `MemoryBytes` / `Name`). A factory `MakeSpatialIndex(ctx, kind, params)` returns `std::unique_ptr<SpatialIndex>`. Both backends are clean Engine::Core rewrites of the old `vkSpatial::vkBVH` / `vkWideBVH`, reusing the existing `src/shader/bvh_*.comp` + `cmd_*` shaders. Query kernels are built once at `Build()` and re-dispatched per query (cache pattern) so query timings measure traversal, not pipeline creation. Ray/path tracing is a declared-only `RayTraceable` capability (not implemented this pass).

**Tech Stack:** C++17, Vulkan (via `Engine::Core::Context`/`Buffer`/`ComputePipeline`), shaderc (runtime GLSL→SPV inside ComputePipeline), Eigen (in `BVHTypes.h`), GoogleTest.

## Global Constraints

- Build only on `Engine::Core` (`Context`/`Buffer`/`ComputePipeline`). Do NOT touch `vkCommon`/`vkSpatial`/`vkRender`.
- All new library `.cpp` go under `src/Engine/Spatial/` — collected by `GLOB_RECURSE Spatial/*.cpp`; **no library CMake edits**. New test files under `test/` — collected by `GLOB *.cpp`; **no test CMake edits**. Only `example2/CMakeLists.txt` gets one new target.
- Shaders are found via `VKBVH_SHADER_DIR` (compiled into EngineCore, `= <repo>/src/shader`). Pass a shader filename (e.g. `"bvh_mortonCode.comp"`) to `ComputePipeline::Build(filename)`; `#include "bvh_common.glsl"` resolves from disk automatically. Reuse existing shaders — do NOT write new ones.
- Errors: `throw std::runtime_error("<ClassName>: <message>")`.
- GPU-shared structs (push constants, node/leaf/state structs): keep `static_assert(std::is_standard_layout_v<T>)` + exact `sizeof`/`offsetof` checks matching the GLSL std430 layout. Never reorder fields without matching the shader.
- `Engine::Core::ComputePipeline::Dispatch()` is fully synchronous (blocks on `vkQueueWaitIdle`). `.Sync()` is unnecessary; do not add it.
- Cached-pipeline rule: a `ComputePipeline` fixes its descriptor-set layout at first `Dispatch()`. You may re-`Bind()` different buffers to the **same binding indices** across dispatches (marks dirty → descriptors rewritten), but never change the *set* of binding indices on a cached pipeline.
- `Engine::Core::Buffer::Allocate/Upload/Download` take `uint32_t` byte counts. `sizeof(Primitive)==28`, `sizeof(MortonCode)==8`. Binary node = 36 bytes, binary constructionInfo = 8 bytes (raw byte sizes, matching the shaders — there is no C++ `Node` struct).

---

## File Structure

```
src/Engine/Spatial/
  BVHTypes.h                 (exists, unchanged) Primitive/MortonCode/MortonConstant/converters
  SpatialIndex.h             NEW  abstract base + RayTraceable + BVHKind + BVHParams + MakeSpatialIndex decl
  BinaryLBVH.h / .cpp        NEW  binary LBVH backend
  WideBVH.h / .cpp           NEW  wide BVH backend (build + query; ray tracing NOT implemented)
  SpatialIndexFactory.cpp    NEW  MakeSpatialIndex impl
  BVH.h                      DELETE (unreferenced stub; BinaryLBVH replaces it)

test/
  test_spatialIndex.cpp      NEW  parametrized correctness across backends

example2/
  bvh_benchmark.cpp          NEW  benchmark executable
  CMakeLists.txt             MODIFY  add bvh_benchmark target
```

---

## Task 1: SpatialIndex interface header

**Files:**
- Create: `src/Engine/Spatial/SpatialIndex.h`
- Delete: `src/Engine/Spatial/BVH.h`
- Test: `test/test_spatialIndex.cpp` (created here; grows in later tasks)

**Interfaces:**
- Produces: `class Engine::Spatial::SpatialIndex` (abstract) with public non-virtual `template<typename T> void Build(const std::vector<T>&)`; pure virtuals `std::vector<uint32_t> RadiusSearch(float,float,float,float)`, `std::vector<uint32_t> KNN(float,float,float,int)`, `uint32_t Length() const`, `uint32_t NodeCount() const`, `uint32_t MemoryBytes() const`, `const char* Name() const`; protected pure virtual `void BuildFromPrimitives(const std::vector<Primitive>&)`.
- Produces: `class Engine::Spatial::RayTraceable` (empty, declared only).
- Produces: `enum class Engine::Spatial::BVHKind { BinaryLBVH, Wide }`; `struct Engine::Spatial::BVHParams { uint32_t maxLeafPrimitives = 4; }`; `std::unique_ptr<SpatialIndex> Engine::Spatial::MakeSpatialIndex(Engine::Core::Context&, BVHKind, const BVHParams& = {})`.

- [ ] **Step 1: Write the failing test**

Create `test/test_spatialIndex.cpp`:

```cpp
#include <gtest/gtest.h>

#include "Engine/Spatial/SpatialIndex.h"

using namespace Engine::Spatial;

// Compile/shape guard for the interface header. Real backend behaviour is added
// in later tasks. This test only verifies the header is well-formed and the
// enum/params/defaults exist.
TEST(SpatialIndexInterface, EnumAndParamsExist) {
    EXPECT_EQ(BVHParams{}.maxLeafPrimitives, 4u);
    EXPECT_NE(static_cast<int>(BVHKind::BinaryLBVH),
              static_cast<int>(BVHKind::Wide));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests` (from repo root; configure `build/` first if absent with `cmake -S . -B build`)
Expected: FAIL — `fatal error: Engine/Spatial/SpatialIndex.h: No such file` (header does not exist yet).

- [ ] **Step 3: Create the header**

Create `src/Engine/Spatial/SpatialIndex.h`:

```cpp
#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Spatial/BVHTypes.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Spatial {

    // Common swappable surface every acceleration structure implements.
    class SpatialIndex {
    public:
        virtual ~SpatialIndex() = default;

        SpatialIndex(const SpatialIndex &) = delete;
        SpatialIndex &operator=(const SpatialIndex &) = delete;

        // Ergonomic build. T is Primitive, PointPrim, TrianglePrim, or any type with a
        // PrimitiveConverter<T> specialisation. Converts to Primitive then delegates to
        // the virtual BuildFromPrimitives so every backend shares this entry point.
        template<typename T>
        void Build(const std::vector<T> &primitives) {
            std::vector<Primitive> prims;
            prims.reserve(primitives.size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(primitives.size()); ++i)
                prims.push_back(PrimitiveConverter<T>::convert(primitives[i], i));
            BuildFromPrimitives(prims);
        }

        virtual std::vector<uint32_t> RadiusSearch(float cx, float cy, float cz, float r) = 0;
        virtual std::vector<uint32_t> KNN(float cx, float cy, float cz, int k) = 0;

        virtual uint32_t Length() const = 0;       // primitive count
        virtual uint32_t NodeCount() const = 0;    // structural node count
        virtual uint32_t MemoryBytes() const = 0;  // structural GPU buffer bytes
        virtual const char *Name() const = 0;      // benchmark label

    protected:
        SpatialIndex() = default;
        virtual void BuildFromPrimitives(const std::vector<Primitive> &prims) = 0;
    };

    // Optional capability — only algorithms that can trace rays implement it.
    // Declared-only this pass; WideBVH will inherit + implement it in future work.
    class RayTraceable {
    public:
        virtual ~RayTraceable() = default;
        // TODO(future): TraceRays / TracePath. See old vkSpatial::vkWideBVH for signatures.
    };

    enum class BVHKind {
        BinaryLBVH,
        Wide,
    };

    struct BVHParams {
        uint32_t maxLeafPrimitives = 4; // Wide only; ignored by BinaryLBVH.
    };

    std::unique_ptr<SpatialIndex>
    MakeSpatialIndex(Engine::Core::Context &ctx, BVHKind kind, const BVHParams &params = {});

} // namespace Engine::Spatial
```

- [ ] **Step 4: Delete the old stub**

Run: `git rm src/Engine/Spatial/BVH.h`
(The file is an unreferenced header-only stub — confirm no `#include "Engine/Spatial/BVH.h"` exists: `grep -rn "Spatial/BVH.h" src test example example2` returns nothing.)

- [ ] **Step 5: Run test to verify it fails to LINK (expected)**

Run: `cmake --build build --target vkspatial_tests`
Expected: FAIL at link — `undefined reference to Engine::Spatial::MakeSpatialIndex(...)` is NOT expected yet because nothing calls it. The header-only test compiles and links. Expected: PASS.

If instead it PASSES, that is the intended end state for this task.

- [ ] **Step 6: Run the test**

Run: `ctest --test-dir build -R SpatialIndexInterface --output-on-failure`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/Engine/Spatial/SpatialIndex.h test/test_spatialIndex.cpp
git commit -m "feat(spatial): add SpatialIndex interface + remove BVH stub"
```

---

## Task 2: BinaryLBVH build + metrics

**Files:**
- Create: `src/Engine/Spatial/BinaryLBVH.h`, `src/Engine/Spatial/BinaryLBVH.cpp`
- Test: `test/test_spatialIndex.cpp` (add cases)

**Interfaces:**
- Consumes: `SpatialIndex` (Task 1), `Engine::Core::Buffer`, `Engine::Core::ComputePipeline`, `BVHTypes.h` structs.
- Produces: `class Engine::Spatial::BinaryLBVH : public SpatialIndex` with `explicit BinaryLBVH(Engine::Core::Context&)`. After `Build`, `Length()==N`, `NodeCount()==2N-1`, `MemoryBytes()==node-buffer bytes`. Query kernels/buffers set up but `RadiusSearch`/`KNN` added in Tasks 3–4 (stub `return {};` for now).

- [ ] **Step 1: Write the failing test**

Add to `test/test_spatialIndex.cpp` (add `#include "Engine/Core/Context.h"`, `#include "Engine/Spatial/BinaryLBVH.h"`, `#include "Engine/Spatial/BVHTypes.h"`, `<random>` at top):

```cpp
namespace {
    // Builds a Context once; skips the test if Vulkan is unavailable.
    struct CtxHolder {
        std::unique_ptr<Engine::Core::Context> ctx;
        bool ok = false;
        CtxHolder() {
            try { ctx = std::make_unique<Engine::Core::Context>(); ok = true; }
            catch (const std::exception &) { ok = false; }
        }
    };

    std::vector<PointPrim> randomPoints(uint32_t n, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(-20.0f, 20.0f);
        std::vector<PointPrim> pts(n);
        for (auto &p : pts) { p.x = d(rng); p.y = d(rng); p.z = d(rng); }
        return pts;
    }
}

TEST(BinaryLBVHTest, BuildProducesExpectedMetrics) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(512, 1);
    BinaryLBVH bvh(*h.ctx);
    bvh.Build(pts);

    EXPECT_EQ(bvh.Length(), 512u);
    EXPECT_EQ(bvh.NodeCount(), 2u * 512u - 1u);
    EXPECT_GT(bvh.MemoryBytes(), 0u);
}

TEST(BinaryLBVHTest, RejectsFewerThanTwoPrimitives) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";
    BinaryLBVH bvh(*h.ctx);
    std::vector<PointPrim> one{{0.0f, 0.0f, 0.0f}};
    EXPECT_THROW(bvh.Build(one), std::runtime_error);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests`
Expected: FAIL — `Engine/Spatial/BinaryLBVH.h: No such file`.

- [ ] **Step 3: Create the header**

Create `src/Engine/Spatial/BinaryLBVH.h`:

```cpp
#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/SpatialIndex.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Core {
    class ComputePipeline;
}

namespace Engine::Spatial {

    // Binary LBVH (Karras-style: Morton codes -> radix sort -> hierarchy -> bounds),
    // built on Engine::Core. Query kernels are created once at Build() and re-dispatched
    // per query (cache pattern). Reuses src/shader/bvh_*.comp + cmd_knn/cmd_radiusSearch.
    class BinaryLBVH : public SpatialIndex {
    public:
        explicit BinaryLBVH(Engine::Core::Context &ctx);
        ~BinaryLBVH() override;

        std::vector<uint32_t> RadiusSearch(float cx, float cy, float cz, float r) override;
        std::vector<uint32_t> KNN(float cx, float cy, float cz, int k) override;

        uint32_t Length() const override { return m_count; }
        uint32_t NodeCount() const override { return m_count ? (2u * m_count - 1u) : 0u; }
        uint32_t MemoryBytes() const override;
        const char *Name() const override { return "BinaryLBVH"; }

    protected:
        void BuildFromPrimitives(const std::vector<Primitive> &prims) override;

    private:
        void stepSortMortonCodes();
        void setupQueryKernels();

        Engine::Core::Context *m_ctx = nullptr;
        uint32_t m_count = 0;
        bool m_built = false;

        std::unique_ptr<Engine::Core::Buffer> m_primBuf;
        std::unique_ptr<Engine::Core::Buffer> m_mortonBuf;     // sorted result ends here
        std::unique_ptr<Engine::Core::Buffer> m_mortonPingBuf; // radix pong
        std::unique_ptr<Engine::Core::Buffer> m_histBuf;
        std::unique_ptr<Engine::Core::Buffer> m_nodeBuf;
        std::unique_ptr<Engine::Core::Buffer> m_constructionBuf;

        std::unique_ptr<Engine::Core::ComputePipeline> m_knnKernel;
        std::unique_ptr<Engine::Core::ComputePipeline> m_radiusKernel;
        std::unique_ptr<Engine::Core::Buffer> m_knnResultBuf;    // uint[MAX_K]
        std::unique_ptr<Engine::Core::Buffer> m_knnDistBuf;      // float[MAX_K]
        std::unique_ptr<Engine::Core::Buffer> m_radiusResultBuf; // uint[N]
        std::unique_ptr<Engine::Core::Buffer> m_radiusCountBuf;  // uint[1]
    };

} // namespace Engine::Spatial
```

- [ ] **Step 4: Create the implementation (build only; query methods stubbed)**

Create `src/Engine/Spatial/BinaryLBVH.cpp`:

```cpp
#include "Engine/Spatial/BinaryLBVH.h"

#include "Engine/Core/ComputePipeline.h"

#include <algorithm>
#include <stdexcept>

namespace Engine::Spatial {

    using Engine::Core::Buffer;
    using Engine::Core::ComputePipeline;

    namespace {
        constexpr uint32_t MAX_K = 64;
        constexpr uint32_t INVALID_IDX = 0xFFFFFFFFu;

        struct RadixSortPC {
            uint32_t g_count;
            uint32_t g_shift;
        };
        struct RadixScanPC {
            uint32_t g_numWGs;
        };
        struct HierarchyPC {
            uint32_t g_count;
            uint32_t g_absolutePointers;
        };
        struct KNNPC {
            float cx, cy, cz;
            uint32_t k;
        };
        struct RadiusPC {
            float cx, cy, cz, r;
            uint32_t maxResults;
        };
    } // namespace

    BinaryLBVH::BinaryLBVH(Engine::Core::Context &ctx) : m_ctx(&ctx) {}
    BinaryLBVH::~BinaryLBVH() = default;

    void BinaryLBVH::BuildFromPrimitives(const std::vector<Primitive> &prims) {
        m_count = static_cast<uint32_t>(prims.size());
        if (m_count < 2)
            throw std::runtime_error("BinaryLBVH: need at least 2 primitives");

        const uint32_t N = m_count;
        const uint32_t NODES = N + N - 1;

        m_primBuf = std::make_unique<Buffer>(*m_ctx);
        m_mortonBuf = std::make_unique<Buffer>(*m_ctx);
        m_mortonPingBuf = std::make_unique<Buffer>(*m_ctx);
        m_nodeBuf = std::make_unique<Buffer>(*m_ctx);
        m_constructionBuf = std::make_unique<Buffer>(*m_ctx);

        m_primBuf->Allocate(N * static_cast<uint32_t>(sizeof(Primitive)));
        m_mortonBuf->Allocate(N * static_cast<uint32_t>(sizeof(MortonCode)));
        m_mortonPingBuf->Allocate(N * static_cast<uint32_t>(sizeof(MortonCode)));
        m_nodeBuf->Allocate(NODES * 36u);
        m_constructionBuf->Allocate(NODES * 8u);

        m_primBuf->Upload(prims.data(), N * static_cast<uint32_t>(sizeof(Primitive)));

        MortonConstant mpc;
        mpc.Extend(prims);
        ComputePipeline(*m_ctx)
                .Build("bvh_mortonCode.comp")
                .Bind(0, *m_mortonBuf)
                .Bind(1, *m_primBuf)
                .Args(mpc)
                .DispatchElements(N);

        stepSortMortonCodes(); // sorted MortonCode[] ends up in m_mortonBuf

        const HierarchyPC hpc{N, 1u};
        ComputePipeline(*m_ctx)
                .Build("bvh_hierarchy.comp")
                .Bind(0, *m_mortonBuf)
                .Bind(1, *m_primBuf)
                .Bind(2, *m_nodeBuf)
                .Bind(3, *m_constructionBuf)
                .Args(hpc)
                .DispatchElements(N);

        ComputePipeline(*m_ctx)
                .Build("bvh_boundingBox.comp")
                .Bind(0, *m_nodeBuf)
                .Bind(1, *m_constructionBuf)
                .Args(hpc)
                .DispatchElements(N);

        setupQueryKernels();
        m_built = true;
    }

    void BinaryLBVH::stepSortMortonCodes() {
        constexpr uint32_t WG_SIZE = 256;
        constexpr uint32_t RADIX = 16; // 4-bit radix
        constexpr uint32_t PASSES = 8; // 8 x 4 = 32 bits

        const uint32_t numWGs = (m_count + WG_SIZE - 1) / WG_SIZE;

        m_histBuf = std::make_unique<Buffer>(*m_ctx);
        m_histBuf->Allocate(RADIX * numWGs * static_cast<uint32_t>(sizeof(uint32_t)));

        Buffer *ping = m_mortonBuf.get();
        Buffer *pong = m_mortonPingBuf.get();

        ComputePipeline histogram(*m_ctx), prefixScan(*m_ctx), reorder(*m_ctx);
        histogram.Build("bvh_radixSort_histogram.comp");
        prefixScan.Build("bvh_radixSort_prefixScan.comp");
        reorder.Build("bvh_radixSort_reorder.comp");

        for (uint32_t pass = 0; pass < PASSES; ++pass) {
            const RadixSortPC pcSort{m_count, pass * 4u};
            const RadixScanPC pcScan{numWGs};

            histogram.Bind(0, *ping).Bind(1, *m_histBuf).Args(pcSort).Dispatch(numWGs);
            prefixScan.Bind(0, *m_histBuf).Args(pcScan).Dispatch(1);
            reorder.Bind(0, *ping).Bind(1, *m_histBuf).Bind(2, *pong).Args(pcSort).Dispatch(numWGs);

            std::swap(ping, pong);
        }
        // PASSES is even, so after the final swap `ping == m_mortonBuf` and it holds the
        // fully sorted MortonCode[]. bvh_hierarchy.comp reads from m_mortonBuf.
    }

    void BinaryLBVH::setupQueryKernels() {
        m_knnResultBuf = std::make_unique<Buffer>(*m_ctx);
        m_knnDistBuf = std::make_unique<Buffer>(*m_ctx);
        m_radiusResultBuf = std::make_unique<Buffer>(*m_ctx);
        m_radiusCountBuf = std::make_unique<Buffer>(*m_ctx);

        m_knnResultBuf->Allocate(MAX_K * static_cast<uint32_t>(sizeof(uint32_t)));
        m_knnDistBuf->Allocate(MAX_K * static_cast<uint32_t>(sizeof(float)));
        m_radiusResultBuf->Allocate(m_count * static_cast<uint32_t>(sizeof(uint32_t)));
        m_radiusCountBuf->Allocate(static_cast<uint32_t>(sizeof(uint32_t)));

        m_knnKernel = std::make_unique<ComputePipeline>(*m_ctx);
        m_knnKernel->Build("cmd_knn.comp")
                .Bind(0, *m_nodeBuf)
                .Bind(1, *m_knnResultBuf)
                .Bind(2, *m_knnDistBuf);

        m_radiusKernel = std::make_unique<ComputePipeline>(*m_ctx);
        m_radiusKernel->Build("cmd_radiusSearch.comp")
                .Bind(0, *m_nodeBuf)
                .Bind(1, *m_radiusResultBuf)
                .Bind(2, *m_radiusCountBuf);
    }

    uint32_t BinaryLBVH::MemoryBytes() const {
        return m_nodeBuf ? m_nodeBuf->Size() : 0u;
    }

    std::vector<uint32_t> BinaryLBVH::RadiusSearch(float, float, float, float) {
        return {}; // implemented in Task 3
    }

    std::vector<uint32_t> BinaryLBVH::KNN(float, float, float, int) {
        return {}; // implemented in Task 4
    }

} // namespace Engine::Spatial
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R BinaryLBVHTest --output-on-failure`
Expected: `BuildProducesExpectedMetrics` and `RejectsFewerThanTwoPrimitives` PASS (or SKIP if no Vulkan).

- [ ] **Step 6: Commit**

```bash
git add src/Engine/Spatial/BinaryLBVH.h src/Engine/Spatial/BinaryLBVH.cpp test/test_spatialIndex.cpp
git commit -m "feat(spatial): BinaryLBVH build + metrics on Engine::Core"
```

---

## Task 3: BinaryLBVH::RadiusSearch (cached kernel)

**Files:**
- Modify: `src/Engine/Spatial/BinaryLBVH.cpp` (replace the `RadiusSearch` stub)
- Test: `test/test_spatialIndex.cpp` (add case)

**Interfaces:**
- Consumes: cached `m_radiusKernel`, `m_radiusResultBuf` (uint[N]), `m_radiusCountBuf` (uint[1]) from Task 2.
- Produces: `RadiusSearch` returns primitive indices whose centre lies within `r` of `(cx,cy,cz)`, order unspecified.

- [ ] **Step 1: Write the failing test**

Add to `test/test_spatialIndex.cpp` (add a CPU reference helper in the anonymous namespace):

```cpp
namespace {
    std::vector<uint32_t> cpuRadius(const std::vector<PointPrim> &pts,
                                    float cx, float cy, float cz, float r) {
        const float r2 = r * r;
        std::vector<uint32_t> out;
        for (uint32_t i = 0; i < pts.size(); ++i) {
            const float dx = pts[i].x - cx, dy = pts[i].y - cy, dz = pts[i].z - cz;
            if (dx * dx + dy * dy + dz * dz <= r2) out.push_back(i);
        }
        return out;
    }
}

TEST(BinaryLBVHTest, RadiusMatchesCpu) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(512, 42);
    BinaryLBVH bvh(*h.ctx);
    bvh.Build(pts);

    auto gpu = bvh.RadiusSearch(1.0f, -2.0f, 0.5f, 7.5f);
    auto cpu = cpuRadius(pts, 1.0f, -2.0f, 0.5f, 7.5f);
    std::sort(gpu.begin(), gpu.end());
    EXPECT_EQ(gpu, cpu);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R BinaryLBVHTest.RadiusMatchesCpu --output-on-failure`
Expected: FAIL — stub returns `{}`, so `gpu` is empty but `cpu` is not.

- [ ] **Step 3: Implement RadiusSearch**

In `src/Engine/Spatial/BinaryLBVH.cpp`, replace the `RadiusSearch` stub with:

```cpp
    std::vector<uint32_t> BinaryLBVH::RadiusSearch(float cx, float cy, float cz, float r) {
        if (!m_built)
            throw std::runtime_error("BinaryLBVH: Build() must be called first");

        const uint32_t zero = 0;
        m_radiusCountBuf->Upload(&zero, sizeof(uint32_t));

        const RadiusPC pc{cx, cy, cz, r, m_count};
        m_radiusKernel->Args(pc).Dispatch(1);

        uint32_t count = 0;
        m_radiusCountBuf->Download(&count, sizeof(uint32_t));
        if (count == 0) return {};

        const uint32_t n = std::min(count, m_count);
        std::vector<uint32_t> out(n);
        m_radiusResultBuf->Download(out.data(), n * static_cast<uint32_t>(sizeof(uint32_t)));
        return out;
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R BinaryLBVHTest.RadiusMatchesCpu --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Spatial/BinaryLBVH.cpp test/test_spatialIndex.cpp
git commit -m "feat(spatial): BinaryLBVH RadiusSearch (cached kernel)"
```

---

## Task 4: BinaryLBVH::KNN (cached kernel)

**Files:**
- Modify: `src/Engine/Spatial/BinaryLBVH.cpp` (replace the `KNN` stub)
- Test: `test/test_spatialIndex.cpp` (add case)

**Interfaces:**
- Consumes: cached `m_knnKernel`, `m_knnResultBuf` (uint[MAX_K]), `m_knnDistBuf` (float[MAX_K]).
- Produces: `KNN` returns up to `k` nearest primitive indices in ascending distance; `k` in `[1,64]` else throws.

- [ ] **Step 1: Write the failing test**

Add to `test/test_spatialIndex.cpp` (add a CPU KNN helper):

```cpp
namespace {
    std::vector<uint32_t> cpuKNN(const std::vector<PointPrim> &pts,
                                 float cx, float cy, float cz, uint32_t k) {
        std::vector<std::pair<float, uint32_t>> d;
        d.reserve(pts.size());
        for (uint32_t i = 0; i < pts.size(); ++i) {
            const float dx = pts[i].x - cx, dy = pts[i].y - cy, dz = pts[i].z - cz;
            d.emplace_back(dx * dx + dy * dy + dz * dz, i);
        }
        std::sort(d.begin(), d.end());
        const uint32_t n = std::min<uint32_t>(k, static_cast<uint32_t>(d.size()));
        std::vector<uint32_t> out(n);
        for (uint32_t i = 0; i < n; ++i) out[i] = d[i].second;
        return out;
    }
}

TEST(BinaryLBVHTest, KNNMatchesCpuInDistanceOrder) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(400, 99);
    BinaryLBVH bvh(*h.ctx);
    bvh.Build(pts);

    EXPECT_EQ(bvh.KNN(0.5f, -1.0f, 2.0f, 32), cpuKNN(pts, 0.5f, -1.0f, 2.0f, 32));
    EXPECT_THROW(bvh.KNN(0.0f, 0.0f, 0.0f, 0), std::runtime_error);
    EXPECT_THROW(bvh.KNN(0.0f, 0.0f, 0.0f, 65), std::runtime_error);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R BinaryLBVHTest.KNNMatchesCpuInDistanceOrder --output-on-failure`
Expected: FAIL — stub returns `{}`.

- [ ] **Step 3: Implement KNN**

In `src/Engine/Spatial/BinaryLBVH.cpp`, replace the `KNN` stub with:

```cpp
    std::vector<uint32_t> BinaryLBVH::KNN(float cx, float cy, float cz, int k) {
        if (!m_built)
            throw std::runtime_error("BinaryLBVH: Build() must be called first");
        if (k <= 0 || static_cast<uint32_t>(k) > MAX_K)
            throw std::runtime_error("BinaryLBVH: KNN k must be in [1, 64]");

        const uint32_t uk = static_cast<uint32_t>(k);
        const KNNPC pc{cx, cy, cz, uk};
        m_knnKernel->Args(pc).Dispatch(1);

        std::vector<uint32_t> indices(uk);
        m_knnResultBuf->Download(indices.data(), uk * static_cast<uint32_t>(sizeof(uint32_t)));
        indices.erase(std::remove(indices.begin(), indices.end(), INVALID_IDX), indices.end());
        return indices;
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R BinaryLBVHTest.KNNMatchesCpuInDistanceOrder --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Spatial/BinaryLBVH.cpp test/test_spatialIndex.cpp
git commit -m "feat(spatial): BinaryLBVH KNN (cached kernel)"
```

---

## Task 5: Factory (binary) + parametrized harness

**Files:**
- Create: `src/Engine/Spatial/SpatialIndexFactory.cpp`
- Test: `test/test_spatialIndex.cpp` (add parametrized suite; Wide is added in Task 9)

**Interfaces:**
- Consumes: `MakeSpatialIndex` decl (Task 1), `BinaryLBVH` (Tasks 2–4).
- Produces: `MakeSpatialIndex(ctx, BVHKind::BinaryLBVH, ...)` returns a working `BinaryLBVH`. `BVHKind::Wide` throws `std::runtime_error("MakeSpatialIndex: Wide backend not implemented yet")` until Task 9.

- [ ] **Step 1: Write the failing test**

Add to `test/test_spatialIndex.cpp`:

```cpp
// Parametrized correctness across every registered backend. Wide is added in Task 9.
struct BackendCase {
    BVHKind kind;
    uint32_t leaf;
    const char *label;
};

class SpatialIndexBackend : public ::testing::TestWithParam<BackendCase> {};

TEST_P(SpatialIndexBackend, RadiusAndKnnMatchCpu) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(512, 7);
    const BackendCase c = GetParam();
    auto idx = MakeSpatialIndex(*h.ctx, c.kind, BVHParams{c.leaf});
    idx->Build(pts);

    EXPECT_EQ(idx->Length(), 512u);
    EXPECT_GT(idx->NodeCount(), 0u);
    EXPECT_GT(idx->MemoryBytes(), 0u);

    auto gpuR = idx->RadiusSearch(1.0f, -2.0f, 0.5f, 7.5f);
    auto cpuR = cpuRadius(pts, 1.0f, -2.0f, 0.5f, 7.5f);
    std::sort(gpuR.begin(), gpuR.end());
    EXPECT_EQ(gpuR, cpuR);

    EXPECT_EQ(idx->KNN(0.5f, -1.0f, 2.0f, 16), cpuKNN(pts, 0.5f, -1.0f, 2.0f, 16));
}

INSTANTIATE_TEST_SUITE_P(
        Backends, SpatialIndexBackend,
        ::testing::Values(BackendCase{BVHKind::BinaryLBVH, 0u, "BinaryLBVH"}),
        [](const ::testing::TestParamInfo<BackendCase> &i) { return i.param.label; });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests`
Expected: FAIL at link — `undefined reference to Engine::Spatial::MakeSpatialIndex`.

- [ ] **Step 3: Create the factory**

Create `src/Engine/Spatial/SpatialIndexFactory.cpp`:

```cpp
#include "Engine/Spatial/SpatialIndex.h"

#include "Engine/Spatial/BinaryLBVH.h"

#include <stdexcept>

namespace Engine::Spatial {

    std::unique_ptr<SpatialIndex>
    MakeSpatialIndex(Engine::Core::Context &ctx, BVHKind kind, const BVHParams &params) {
        switch (kind) {
            case BVHKind::BinaryLBVH:
                return std::make_unique<BinaryLBVH>(ctx);
            case BVHKind::Wide:
                throw std::runtime_error(
                        "MakeSpatialIndex: Wide backend not implemented yet");
        }
        throw std::runtime_error("MakeSpatialIndex: unknown BVHKind");
        (void) params;
    }

} // namespace Engine::Spatial
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R "SpatialIndexBackend" --output-on-failure`
Expected: PASS (`Backends/SpatialIndexBackend.RadiusAndKnnMatchCpu/BinaryLBVH`).

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Spatial/SpatialIndexFactory.cpp test/test_spatialIndex.cpp
git commit -m "feat(spatial): factory + parametrized backend test (binary)"
```

---

## Task 6: WideBVH build + metrics

**Files:**
- Create: `src/Engine/Spatial/WideBVH.h`, `src/Engine/Spatial/WideBVH.cpp`
- Test: `test/test_spatialIndex.cpp` (add cases)

**Interfaces:**
- Consumes: `SpatialIndex`, `Engine::Core::Buffer`/`ComputePipeline`, `BVHTypes.h`. Reuses shaders `bvh_wide_*.comp`, `bvh_radixSort_*.comp`, `bvh_hierarchy.comp`, `bvh_boundingBox.comp`.
- Produces: `class Engine::Spatial::WideBVH : public SpatialIndex` with `explicit WideBVH(Engine::Core::Context&, uint32_t maxLeafPrimitives = 4)`; throws if `maxLeafPrimitives` not in `[1,64]`. After Build: `Length()==N`, `NodeCount()` = actual wide node count (`< 2N-1` for N large enough), `MaxLeafPrimitives()==leaf`, `MemoryBytes()` = wide-node + leaf buffer bytes. `RadiusSearch`/`KNN` stubbed here (Tasks 7–8).

- [ ] **Step 1: Write the failing test**

Add to `test/test_spatialIndex.cpp` (add `#include "Engine/Spatial/WideBVH.h"`):

```cpp
TEST(WideBVHTest, BuildProducesFewerNodesThanBinary) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(1024, 7);
    WideBVH bvh(*h.ctx, 4);
    bvh.Build(pts);

    EXPECT_EQ(bvh.Length(), 1024u);
    EXPECT_EQ(bvh.MaxLeafPrimitives(), 4u);
    EXPECT_GT(bvh.NodeCount(), 0u);
    EXPECT_LT(bvh.NodeCount(), 2u * 1024u - 1u);
    EXPECT_GT(bvh.MemoryBytes(), 0u);
}

TEST(WideBVHTest, RejectsInvalidLeafSize) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";
    EXPECT_THROW(WideBVH(*h.ctx, 0), std::runtime_error);
    EXPECT_THROW(WideBVH(*h.ctx, 65), std::runtime_error);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests`
Expected: FAIL — `Engine/Spatial/WideBVH.h: No such file`.

- [ ] **Step 3: Create the header**

Create `src/Engine/Spatial/WideBVH.h`:

```cpp
#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/SpatialIndex.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Engine::Core {
    class ComputePipeline;
}

namespace Engine::Spatial {

    // N-wide BVH: builds a binary LBVH, then GPU-collapses it into a quantized wide
    // hierarchy (up to maxLeafPrimitives per leaf). Query kernels cached at Build().
    // Reuses src/shader/bvh_wide_*.comp + cmd_knn_wide/cmd_radiusSearch_wide.
    // Ray/path tracing is NOT implemented (future RayTraceable capability).
    class WideBVH : public SpatialIndex {
    public:
        explicit WideBVH(Engine::Core::Context &ctx, uint32_t maxLeafPrimitives = 4);
        ~WideBVH() override;

        std::vector<uint32_t> RadiusSearch(float cx, float cy, float cz, float r) override;
        std::vector<uint32_t> KNN(float cx, float cy, float cz, int k) override;

        uint32_t Length() const override { return m_count; }
        uint32_t NodeCount() const override { return m_nodeCount; }
        uint32_t MemoryBytes() const override;
        const char *Name() const override { return m_name.c_str(); }
        uint32_t MaxLeafPrimitives() const { return m_maxLeafPrimitives; }

    protected:
        void BuildFromPrimitives(const std::vector<Primitive> &prims) override;

    private:
        void setupQueryKernels();

        Engine::Core::Context *m_ctx = nullptr;
        uint32_t m_maxLeafPrimitives = 4;
        std::string m_name;
        uint32_t m_count = 0;
        uint32_t m_nodeCount = 0;
        bool m_built = false;

        std::unique_ptr<Engine::Core::Buffer> m_primitiveBuf;  // Primitive[]
        std::unique_ptr<Engine::Core::Buffer> m_sortedMortonBuf;
        std::unique_ptr<Engine::Core::Buffer> m_nodeBuf; // QuantizedWideNode[]
        std::unique_ptr<Engine::Core::Buffer> m_leafBuf; // LeafRange[]

        std::unique_ptr<Engine::Core::ComputePipeline> m_radiusKernel;
        std::unique_ptr<Engine::Core::ComputePipeline> m_knnKernel;
        std::unique_ptr<Engine::Core::Buffer> m_radiusResultBuf; // uint[N]
        std::unique_ptr<Engine::Core::Buffer> m_radiusStateBuf;  // QueryState
        std::unique_ptr<Engine::Core::Buffer> m_knnResultBuf;    // uint[MAX_K]
        std::unique_ptr<Engine::Core::Buffer> m_knnDistBuf;      // float[MAX_K]
        std::unique_ptr<Engine::Core::Buffer> m_knnStateBuf;     // QueryState
    };

} // namespace Engine::Spatial
```

- [ ] **Step 4: Create the implementation (build only; query methods stubbed)**

Create `src/Engine/Spatial/WideBVH.cpp`. This is a faithful port of `src/vkSpatial/vkWideBVH.cpp` (lines 20–174 for the anon-namespace structs/helpers, 270–483 for the build) onto Engine::Core. The GPU struct definitions and `static_assert`s must be copied exactly.

```cpp
#include "Engine/Spatial/WideBVH.h"

#include "Engine/Core/ComputePipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace Engine::Spatial {

    using Engine::Core::Buffer;
    using Engine::Core::ComputePipeline;

    namespace {
        constexpr uint32_t RADIX = 16;
        constexpr uint32_t RADIX_PASSES = 8;
        constexpr uint32_t RADIX_WORKGROUP_SIZE = 256;
        constexpr uint32_t MAX_K = 64;
        constexpr uint32_t INVALID_IDX = 0xFFFFFFFFu;

        struct CountPC { uint32_t count; };
        struct WideBuildPC { uint32_t binaryNodeCount; uint32_t maxLeafPrimitives; };
        struct RadixSortPC { uint32_t g_count; uint32_t g_shift; };
        struct RadixScanPC { uint32_t g_numWGs; };
        struct HierarchyPC { uint32_t g_count; uint32_t g_absolutePointers; };
        struct RadiusPC { float cx, cy, cz, radius; uint32_t maxResults; };
        struct KNNPC { float cx, cy, cz; uint32_t k; };

        struct QueryState { uint32_t count; uint32_t status; };
        struct WideBuildState { uint32_t nodeCount; uint32_t leafCount; uint32_t status; uint32_t reserved; };

        struct QuantizedWideNode {
            float originX, originY, originZ;
            float scaleX, scaleY, scaleZ;
            uint32_t child[8];
            uint32_t qBounds[12];
            uint32_t childCount;
            uint32_t leafMask;
        };
        struct LeafRange { uint32_t firstPrimitive; uint32_t primitiveCount; };
        struct BinaryRange {
            uint32_t firstPrimitive;
            uint32_t primitiveCount;
            uint32_t visitationCount;
            uint32_t reserved;
        };

        static_assert(std::is_standard_layout_v<QuantizedWideNode>);
        static_assert(sizeof(QuantizedWideNode) == 112);
        static_assert(offsetof(QuantizedWideNode, child) == 24);
        static_assert(offsetof(QuantizedWideNode, qBounds) == 56);
        static_assert(offsetof(QuantizedWideNode, childCount) == 104);
        static_assert(offsetof(QuantizedWideNode, leafMask) == 108);
        static_assert(sizeof(LeafRange) == 8);
        static_assert(sizeof(BinaryRange) == 16);
        static_assert(sizeof(WideBuildState) == 16);
        static_assert(sizeof(QueryState) == 8);

        uint32_t checkedBytes(uint64_t count, uint64_t elementSize, const char *name) {
            const uint64_t bytes = count * elementSize;
            if (count == 0 || bytes > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error(std::string("WideBVH: invalid buffer size for ") + name);
            return static_cast<uint32_t>(bytes);
        }

        std::unique_ptr<Buffer> makeBuffer(Engine::Core::Context &ctx, uint32_t bytes) {
            auto b = std::make_unique<Buffer>(ctx);
            b->Allocate(bytes);
            return b;
        }

        void validatePrimitive(const Primitive &p) {
            const std::array<float, 6> v{p.aabbMinX, p.aabbMinY, p.aabbMinZ,
                                         p.aabbMaxX, p.aabbMaxY, p.aabbMaxZ};
            for (float f : v)
                if (!std::isfinite(f))
                    throw std::runtime_error("WideBVH: primitive bounds must be finite");
            if (p.aabbMinX > p.aabbMaxX || p.aabbMinY > p.aabbMaxY || p.aabbMinZ > p.aabbMaxZ)
                throw std::runtime_error("WideBVH: primitive AABB min exceeds max");
        }
    } // namespace

    WideBVH::WideBVH(Engine::Core::Context &ctx, uint32_t maxLeafPrimitives)
        : m_ctx(&ctx), m_maxLeafPrimitives(maxLeafPrimitives) {
        if (maxLeafPrimitives == 0 || maxLeafPrimitives > 64)
            throw std::runtime_error("WideBVH: maxLeafPrimitives must be in [1, 64]");
        m_name = "WideBVH(leaf=" + std::to_string(maxLeafPrimitives) + ")";
    }

    WideBVH::~WideBVH() = default;

    void WideBVH::BuildFromPrimitives(const std::vector<Primitive> &prims) {
        if (prims.empty())
            throw std::runtime_error("WideBVH: need at least one primitive");
        for (const Primitive &p : prims) validatePrimitive(p);

        const uint32_t count = static_cast<uint32_t>(prims.size());
        const uint64_t binaryNodeCount64 = static_cast<uint64_t>(count) * 2u - 1u;
        if (binaryNodeCount64 > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("WideBVH: binary node count exceeds uint32");
        const uint32_t binaryNodeCount = static_cast<uint32_t>(binaryNodeCount64);
        const uint32_t radixWGs = (count + RADIX_WORKGROUP_SIZE - 1u) / RADIX_WORKGROUP_SIZE;

        auto primitiveBuffer = makeBuffer(*m_ctx, checkedBytes(count, sizeof(Primitive), "primitiveBuffer"));
        auto mortonBuffer = makeBuffer(*m_ctx, checkedBytes(count, sizeof(MortonCode), "mortonBuffer"));
        auto mortonPingBuffer = makeBuffer(*m_ctx, checkedBytes(count, sizeof(MortonCode), "mortonPingBuffer"));
        auto histogramBuffer = makeBuffer(*m_ctx, checkedBytes(static_cast<uint64_t>(RADIX) * radixWGs, sizeof(uint32_t), "histogramBuffer"));
        auto sceneBoundsBuffer = makeBuffer(*m_ctx, checkedBytes(6, sizeof(uint32_t), "sceneBoundsBuffer"));
        auto binaryNodeBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, 36, "binaryNodeBuffer"));
        auto constructionBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, 8, "constructionBuffer"));
        auto rangeBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, sizeof(BinaryRange), "rangeBuffer"));
        auto queueBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, sizeof(uint32_t), "queueBuffer"));
        auto buildStateBuffer = makeBuffer(*m_ctx, sizeof(WideBuildState));
        auto wideNodeBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, sizeof(QuantizedWideNode), "wideNodeBuffer"));
        auto leafBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, sizeof(LeafRange), "leafBuffer"));

        primitiveBuffer->Upload(prims.data(), checkedBytes(count, sizeof(Primitive), "primitiveBuffer"));

        // One pipeline object per shader, created once and reused across dispatches.
        auto mk = [&](const char *shader) {
            auto p = std::make_unique<ComputePipeline>(*m_ctx);
            p->Build(shader);
            return p;
        };
        auto boundsInit = mk("bvh_wide_bounds_init.comp");
        auto boundsReduce = mk("bvh_wide_bounds_reduce.comp");
        auto morton = mk("bvh_wide_morton.comp");
        auto histogram = mk("bvh_radixSort_histogram.comp");
        auto prefixScan = mk("bvh_radixSort_prefixScan.comp");
        auto reorder = mk("bvh_radixSort_reorder.comp");
        auto hierarchy = mk("bvh_hierarchy.comp");
        auto boundingBox = mk("bvh_boundingBox.comp");
        auto rangeInit = mk("bvh_wide_range_init.comp");
        auto range = mk("bvh_wide_range.comp");
        auto wideInit = mk("bvh_wide_build_init.comp");
        auto wideBuild = mk("bvh_wide_build.comp");

        boundsInit->Bind(0, *sceneBoundsBuffer).Dispatch(1);

        const CountPC countPC{count};
        boundsReduce->Bind(0, *primitiveBuffer).Bind(1, *sceneBoundsBuffer).Args(countPC).DispatchElements(count);
        morton->Bind(0, *mortonBuffer).Bind(1, *primitiveBuffer).Bind(2, *sceneBoundsBuffer).Args(countPC).DispatchElements(count);

        Buffer *radixInput = mortonBuffer.get();
        Buffer *radixOutput = mortonPingBuffer.get();
        for (uint32_t pass = 0; pass < RADIX_PASSES; ++pass) {
            const RadixSortPC sortPC{count, pass * 4u};
            const RadixScanPC scanPC{radixWGs};
            histogram->Bind(0, *radixInput).Bind(1, *histogramBuffer).Args(sortPC).Dispatch(radixWGs);
            prefixScan->Bind(0, *histogramBuffer).Args(scanPC).Dispatch(1);
            reorder->Bind(0, *radixInput).Bind(1, *histogramBuffer).Bind(2, *radixOutput).Args(sortPC).Dispatch(radixWGs);
            std::swap(radixInput, radixOutput);
        }

        const HierarchyPC hierarchyPC{count, 1u};
        hierarchy->Bind(0, *radixInput).Bind(1, *primitiveBuffer).Bind(2, *binaryNodeBuffer).Bind(3, *constructionBuffer).Args(hierarchyPC).DispatchElements(count);
        boundingBox->Bind(0, *binaryNodeBuffer).Bind(1, *constructionBuffer).Args(hierarchyPC).DispatchElements(count);

        const CountPC binaryNodeCountPC{binaryNodeCount};
        rangeInit->Bind(0, *rangeBuffer).Args(binaryNodeCountPC).DispatchElements(binaryNodeCount);
        range->Bind(0, *binaryNodeBuffer).Bind(1, *constructionBuffer).Bind(2, *rangeBuffer).Args(countPC).DispatchElements(count);

        wideInit->Bind(0, *queueBuffer).Bind(1, *buildStateBuffer).Dispatch(1);

        const WideBuildPC wideBuildPC{binaryNodeCount, m_maxLeafPrimitives};
        wideBuild->Bind(0, *binaryNodeBuffer).Bind(1, *rangeBuffer).Bind(2, *queueBuffer).Bind(3, *buildStateBuffer).Bind(4, *wideNodeBuffer).Bind(5, *leafBuffer).Args(wideBuildPC).Dispatch(1);

        WideBuildState state{};
        buildStateBuffer->Download(&state, sizeof(state));
        if (state.status != 0u)
            throw std::runtime_error("WideBVH: GPU wide collapse exceeded buffer capacity");
        if (state.nodeCount == 0u || state.nodeCount > binaryNodeCount ||
            state.leafCount == 0u || state.leafCount > binaryNodeCount)
            throw std::runtime_error("WideBVH: GPU wide collapse produced invalid counts");

        if (radixInput == mortonBuffer.get())
            m_sortedMortonBuf = std::move(mortonBuffer);
        else
            m_sortedMortonBuf = std::move(mortonPingBuffer);

        m_primitiveBuf = std::move(primitiveBuffer);
        m_nodeBuf = std::move(wideNodeBuffer);
        m_leafBuf = std::move(leafBuffer);
        m_count = count;
        m_nodeCount = state.nodeCount;

        setupQueryKernels();
        m_built = true;
    }

    void WideBVH::setupQueryKernels() {
        m_radiusResultBuf = makeBuffer(*m_ctx, checkedBytes(m_count, sizeof(uint32_t), "radiusResult"));
        m_radiusStateBuf = makeBuffer(*m_ctx, sizeof(QueryState));
        m_knnResultBuf = makeBuffer(*m_ctx, MAX_K * static_cast<uint32_t>(sizeof(uint32_t)));
        m_knnDistBuf = makeBuffer(*m_ctx, MAX_K * static_cast<uint32_t>(sizeof(float)));
        m_knnStateBuf = makeBuffer(*m_ctx, sizeof(QueryState));

        m_radiusKernel = std::make_unique<ComputePipeline>(*m_ctx);
        m_radiusKernel->Build("cmd_radiusSearch_wide.comp")
                .Bind(0, *m_nodeBuf).Bind(1, *m_leafBuf).Bind(2, *m_sortedMortonBuf)
                .Bind(3, *m_primitiveBuf).Bind(4, *m_radiusResultBuf).Bind(5, *m_radiusStateBuf);

        m_knnKernel = std::make_unique<ComputePipeline>(*m_ctx);
        m_knnKernel->Build("cmd_knn_wide.comp")
                .Bind(0, *m_nodeBuf).Bind(1, *m_leafBuf).Bind(2, *m_sortedMortonBuf)
                .Bind(3, *m_primitiveBuf).Bind(4, *m_knnResultBuf).Bind(5, *m_knnDistBuf)
                .Bind(6, *m_knnStateBuf);
    }

    uint32_t WideBVH::MemoryBytes() const {
        uint32_t bytes = 0;
        if (m_nodeBuf) bytes += m_nodeBuf->Size();
        if (m_leafBuf) bytes += m_leafBuf->Size();
        return bytes;
    }

    std::vector<uint32_t> WideBVH::RadiusSearch(float, float, float, float) {
        return {}; // implemented in Task 7
    }

    std::vector<uint32_t> WideBVH::KNN(float, float, float, int) {
        return {}; // implemented in Task 8
    }

} // namespace Engine::Spatial
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R WideBVHTest --output-on-failure`
Expected: `BuildProducesFewerNodesThanBinary` and `RejectsInvalidLeafSize` PASS (or SKIP).

- [ ] **Step 6: Commit**

```bash
git add src/Engine/Spatial/WideBVH.h src/Engine/Spatial/WideBVH.cpp test/test_spatialIndex.cpp
git commit -m "feat(spatial): WideBVH build + metrics on Engine::Core"
```

---

## Task 7: WideBVH::RadiusSearch (cached kernel)

**Files:**
- Modify: `src/Engine/Spatial/WideBVH.cpp` (replace the `RadiusSearch` stub)
- Test: `test/test_spatialIndex.cpp` (add case)

**Interfaces:**
- Consumes: cached `m_radiusKernel`, `m_radiusResultBuf` (uint[N]), `m_radiusStateBuf` (`QueryState`).
- Produces: `RadiusSearch` returns primitive indices within `r`, order unspecified.

- [ ] **Step 1: Write the failing test**

Add to `test/test_spatialIndex.cpp`:

```cpp
TEST(WideBVHTest, RadiusMatchesCpu) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(512, 42);
    WideBVH bvh(*h.ctx, 4);
    bvh.Build(pts);

    auto gpu = bvh.RadiusSearch(1.0f, -2.0f, 0.5f, 7.5f);
    auto cpu = cpuRadius(pts, 1.0f, -2.0f, 0.5f, 7.5f);
    std::sort(gpu.begin(), gpu.end());
    EXPECT_EQ(gpu, cpu);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R WideBVHTest.RadiusMatchesCpu --output-on-failure`
Expected: FAIL — stub returns `{}`.

- [ ] **Step 3: Implement RadiusSearch**

In `src/Engine/Spatial/WideBVH.cpp`, replace the `RadiusSearch` stub with:

```cpp
    std::vector<uint32_t> WideBVH::RadiusSearch(float cx, float cy, float cz, float radius) {
        if (!m_built)
            throw std::runtime_error("WideBVH: Build() must be called first");
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(cz) ||
            !std::isfinite(radius) || radius < 0.0f)
            throw std::runtime_error("WideBVH: RadiusSearch arguments are invalid");

        const QueryState zero{0u, 0u};
        m_radiusStateBuf->Upload(&zero, sizeof(zero));

        const RadiusPC pc{cx, cy, cz, radius, m_count};
        m_radiusKernel->Args(pc).Dispatch(1);

        QueryState state{};
        m_radiusStateBuf->Download(&state, sizeof(state));
        if (state.status != 0u)
            throw std::runtime_error("WideBVH: RadiusSearch traversal stack overflow");
        if (state.count > m_count)
            throw std::runtime_error("WideBVH: RadiusSearch result count is corrupt");
        if (state.count == 0u) return {};

        std::vector<uint32_t> out(state.count);
        m_radiusResultBuf->Download(out.data(), state.count * static_cast<uint32_t>(sizeof(uint32_t)));
        return out;
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R WideBVHTest.RadiusMatchesCpu --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Spatial/WideBVH.cpp test/test_spatialIndex.cpp
git commit -m "feat(spatial): WideBVH RadiusSearch (cached kernel)"
```

---

## Task 8: WideBVH::KNN (cached kernel)

**Files:**
- Modify: `src/Engine/Spatial/WideBVH.cpp` (replace the `KNN` stub)
- Test: `test/test_spatialIndex.cpp` (add case)

**Interfaces:**
- Consumes: cached `m_knnKernel`, `m_knnResultBuf` (uint[MAX_K]), `m_knnDistBuf` (float[MAX_K]), `m_knnStateBuf` (`QueryState`).
- Produces: `KNN` returns up to `k` nearest indices ascending; `k` in `[1,64]` else throws.

- [ ] **Step 1: Write the failing test**

Add to `test/test_spatialIndex.cpp`:

```cpp
TEST(WideBVHTest, KNNMatchesCpuInDistanceOrder) {
    CtxHolder h;
    if (!h.ok) GTEST_SKIP() << "Vulkan context unavailable";

    const auto pts = randomPoints(400, 99);
    WideBVH bvh(*h.ctx, 4);
    bvh.Build(pts);

    EXPECT_EQ(bvh.KNN(0.5f, -1.0f, 2.0f, 32), cpuKNN(pts, 0.5f, -1.0f, 2.0f, 32));
    EXPECT_THROW(bvh.KNN(0.0f, 0.0f, 0.0f, 0), std::runtime_error);
    EXPECT_THROW(bvh.KNN(0.0f, 0.0f, 0.0f, 65), std::runtime_error);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R WideBVHTest.KNNMatchesCpuInDistanceOrder --output-on-failure`
Expected: FAIL — stub returns `{}`.

- [ ] **Step 3: Implement KNN**

In `src/Engine/Spatial/WideBVH.cpp`, replace the `KNN` stub with:

```cpp
    std::vector<uint32_t> WideBVH::KNN(float cx, float cy, float cz, int k) {
        if (!m_built)
            throw std::runtime_error("WideBVH: Build() must be called first");
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(cz))
            throw std::runtime_error("WideBVH: KNN arguments are invalid");
        if (k <= 0 || static_cast<uint32_t>(k) > MAX_K)
            throw std::runtime_error("WideBVH: KNN k must be in [1, 64]");

        const uint32_t uk = static_cast<uint32_t>(k);
        const QueryState zero{0u, 0u};
        m_knnStateBuf->Upload(&zero, sizeof(zero));

        const KNNPC pc{cx, cy, cz, uk};
        m_knnKernel->Args(pc).Dispatch(1);

        QueryState state{};
        m_knnStateBuf->Download(&state, sizeof(state));
        if (state.status != 0u)
            throw std::runtime_error("WideBVH: KNN traversal stack overflow");

        std::vector<uint32_t> indices(uk);
        m_knnResultBuf->Download(indices.data(), uk * static_cast<uint32_t>(sizeof(uint32_t)));
        indices.erase(std::remove(indices.begin(), indices.end(), INVALID_IDX), indices.end());
        return indices;
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R WideBVHTest.KNNMatchesCpuInDistanceOrder --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Spatial/WideBVH.cpp test/test_spatialIndex.cpp
git commit -m "feat(spatial): WideBVH KNN (cached kernel)"
```

---

## Task 9: Factory wires WideBVH + parametrized suite covers all backends

**Files:**
- Modify: `src/Engine/Spatial/SpatialIndexFactory.cpp`
- Modify: `test/test_spatialIndex.cpp` (extend `INSTANTIATE_TEST_SUITE_P`)

**Interfaces:**
- Consumes: `WideBVH` (Tasks 6–8).
- Produces: `MakeSpatialIndex(ctx, BVHKind::Wide, {leaf})` returns a working `WideBVH`.

- [ ] **Step 1: Extend the parametrized instantiation (failing test)**

In `test/test_spatialIndex.cpp`, replace the `INSTANTIATE_TEST_SUITE_P(...)` block from Task 5 with:

```cpp
INSTANTIATE_TEST_SUITE_P(
        Backends, SpatialIndexBackend,
        ::testing::Values(
                BackendCase{BVHKind::BinaryLBVH, 0u, "BinaryLBVH"},
                BackendCase{BVHKind::Wide, 4u, "Wide4"},
                BackendCase{BVHKind::Wide, 8u, "Wide8"}),
        [](const ::testing::TestParamInfo<BackendCase> &i) { return i.param.label; });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R "SpatialIndexBackend" --output-on-failure`
Expected: FAIL — `Wide4`/`Wide8` throw `MakeSpatialIndex: Wide backend not implemented yet`.

- [ ] **Step 3: Wire the factory**

In `src/Engine/Spatial/SpatialIndexFactory.cpp`, add `#include "Engine/Spatial/WideBVH.h"` and replace the `BVHKind::Wide` case:

```cpp
            case BVHKind::Wide:
                return std::make_unique<WideBVH>(ctx, params.maxLeafPrimitives);
```

(Remove the now-unused `(void) params;` line at the end.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target vkspatial_tests && ctest --test-dir build -R "SpatialIndexBackend" --output-on-failure`
Expected: PASS for `BinaryLBVH`, `Wide4`, `Wide8`.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Spatial/SpatialIndexFactory.cpp test/test_spatialIndex.cpp
git commit -m "feat(spatial): factory wires WideBVH; parametrized suite covers all backends"
```

---

## Task 10: Benchmark executable

**Files:**
- Create: `example2/bvh_benchmark.cpp`
- Modify: `example2/CMakeLists.txt`

**Interfaces:**
- Consumes: `MakeSpatialIndex`, `BVHKind`, `BVHParams`, `SpatialIndex`, `PointPrim`.
- Produces: an executable `bvh_benchmark` printing a comparison table (build ms, KNN µs/query, radius µs/query, NodeCount, MemoryBytes, accuracy match) for each backend across an N sweep; `--csv <path>` writes the same as CSV.

- [ ] **Step 1: Add the CMake target**

In `example2/CMakeLists.txt`, after the `directional_tsdf_eval` block (line 14), add:

```cmake
add_executable(bvh_benchmark bvh_benchmark.cpp)
target_link_libraries(bvh_benchmark PRIVATE Engine::Spatial)
target_compile_definitions(bvh_benchmark PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")
```

- [ ] **Step 2: Write the benchmark**

Create `example2/bvh_benchmark.cpp`:

```cpp
// Benchmark: compares Engine::Spatial acceleration backends on build time, query
// time, structural size, and accuracy vs a CPU brute-force reference.
//
//   ./bvh_benchmark [--csv out.csv]
//
// Timing is wall-clock (std::chrono). Engine::Core::ComputePipeline::Dispatch blocks
// on vkQueueWaitIdle, so measured wall time includes full GPU execution.

#include "Engine/Core/Context.h"
#include "Engine/Spatial/SpatialIndex.h"
#include "Engine/Spatial/BVHTypes.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace Engine::Spatial;
using Clock = std::chrono::steady_clock;

namespace {

    struct BackendCase {
        BVHKind kind;
        uint32_t leaf;
        const char *label;
    };

    std::vector<PointPrim> randomPoints(uint32_t n, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(-100.0f, 100.0f);
        std::vector<PointPrim> pts(n);
        for (auto &p : pts) { p.x = d(rng); p.y = d(rng); p.z = d(rng); }
        return pts;
    }

    struct Query { float x, y, z; };
    std::vector<Query> randomQueries(uint32_t m, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(-100.0f, 100.0f);
        std::vector<Query> qs(m);
        for (auto &q : qs) { q.x = d(rng); q.y = d(rng); q.z = d(rng); }
        return qs;
    }

    std::vector<uint32_t> cpuKNN(const std::vector<PointPrim> &pts,
                                 float cx, float cy, float cz, uint32_t k) {
        std::vector<std::pair<float, uint32_t>> d;
        d.reserve(pts.size());
        for (uint32_t i = 0; i < pts.size(); ++i) {
            const float dx = pts[i].x - cx, dy = pts[i].y - cy, dz = pts[i].z - cz;
            d.emplace_back(dx * dx + dy * dy + dz * dz, i);
        }
        std::sort(d.begin(), d.end());
        const uint32_t n = std::min<uint32_t>(k, static_cast<uint32_t>(d.size()));
        std::vector<uint32_t> out(n);
        for (uint32_t i = 0; i < n; ++i) out[i] = d[i].second;
        return out;
    }

    struct Row {
        std::string backend;
        uint32_t n;
        double buildMs;
        double knnUs;
        double radiusUs;
        uint32_t nodeCount;
        uint32_t memBytes;
        bool knnMatches;
    };

    // Median build time over `repeats` builds (+ 1 warmup).
    double medianBuildMs(Engine::Core::Context &ctx, const BackendCase &c,
                         const std::vector<PointPrim> &pts, int repeats) {
        { auto warm = MakeSpatialIndex(ctx, c.kind, BVHParams{c.leaf}); warm->Build(pts); }
        std::vector<double> ms;
        ms.reserve(repeats);
        for (int i = 0; i < repeats; ++i) {
            auto idx = MakeSpatialIndex(ctx, c.kind, BVHParams{c.leaf});
            const auto t0 = Clock::now();
            idx->Build(pts);
            const auto t1 = Clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    }

    Row runBackend(Engine::Core::Context &ctx, const BackendCase &c,
                   const std::vector<PointPrim> &pts,
                   const std::vector<Query> &queries, uint32_t k, float radius) {
        Row row;
        row.backend = c.label;
        row.n = static_cast<uint32_t>(pts.size());
        row.buildMs = medianBuildMs(ctx, c, pts, 5);

        auto idx = MakeSpatialIndex(ctx, c.kind, BVHParams{c.leaf});
        idx->Build(pts);
        row.nodeCount = idx->NodeCount();
        row.memBytes = idx->MemoryBytes();

        // KNN timing + accuracy on the first query.
        const auto tk0 = Clock::now();
        for (const auto &q : queries) idx->KNN(q.x, q.y, q.z, static_cast<int>(k));
        const auto tk1 = Clock::now();
        row.knnUs = std::chrono::duration<double, std::micro>(tk1 - tk0).count() / queries.size();

        auto gpuKnn = idx->KNN(queries[0].x, queries[0].y, queries[0].z, static_cast<int>(k));
        auto refKnn = cpuKNN(pts, queries[0].x, queries[0].y, queries[0].z, k);
        row.knnMatches = (gpuKnn == refKnn);

        // Radius timing.
        const auto tr0 = Clock::now();
        for (const auto &q : queries) idx->RadiusSearch(q.x, q.y, q.z, radius);
        const auto tr1 = Clock::now();
        row.radiusUs = std::chrono::duration<double, std::micro>(tr1 - tr0).count() / queries.size();

        return row;
    }

} // namespace

int main(int argc, char **argv) {
    std::string csvPath;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) csvPath = argv[++i];
    }

    Engine::Core::Context ctx;

    const std::vector<BackendCase> backends = {
            {BVHKind::BinaryLBVH, 0u, "BinaryLBVH"},
            {BVHKind::Wide, 4u, "Wide(leaf=4)"},
            {BVHKind::Wide, 8u, "Wide(leaf=8)"},
    };
    const std::vector<uint32_t> nSweep = {1000u, 10000u, 100000u};
    const uint32_t k = 16;
    const float radius = 10.0f;
    const uint32_t queryCount = 200;

    std::vector<Row> rows;
    for (uint32_t n : nSweep) {
        const auto pts = randomPoints(n, 12345u);
        const auto queries = randomQueries(queryCount, 999u);
        for (const auto &b : backends)
            rows.push_back(runBackend(ctx, b, pts, queries, k, radius));
    }

    std::printf("%-14s %8s %10s %10s %10s %10s %10s %6s\n",
                "backend", "N", "build_ms", "knn_us", "radius_us", "nodes", "mem_KB", "knn_ok");
    for (const auto &r : rows)
        std::printf("%-14s %8u %10.3f %10.3f %10.3f %10u %10.1f %6s\n",
                    r.backend.c_str(), r.n, r.buildMs, r.knnUs, r.radiusUs,
                    r.nodeCount, r.memBytes / 1024.0, r.knnMatches ? "yes" : "NO");

    if (!csvPath.empty()) {
        std::ofstream f(csvPath);
        f << "backend,N,build_ms,knn_us,radius_us,nodes,mem_bytes,knn_ok\n";
        for (const auto &r : rows)
            f << r.backend << ',' << r.n << ',' << r.buildMs << ',' << r.knnUs << ','
              << r.radiusUs << ',' << r.nodeCount << ',' << r.memBytes << ','
              << (r.knnMatches ? 1 : 0) << '\n';
        std::printf("\nWrote CSV to %s\n", csvPath.c_str());
    }

    return 0;
}
```

- [ ] **Step 3: Build the benchmark**

Run: `cmake --build build --target bvh_benchmark`
Expected: builds with no errors.

- [ ] **Step 4: Run the benchmark and verify output**

Run: `./build/example2/bvh_benchmark` (path may vary by generator; find with `find build -name bvh_benchmark -type f`)
Expected: a table with 9 rows (3 backends × 3 N values). Every `knn_ok` column reads `yes`. `Wide(leaf=*)` rows show `nodes` lower than the `BinaryLBVH` row at the same N. If Vulkan is unavailable the process throws from the `Context` constructor — that is acceptable on headless CI; run it on a machine with a GPU.

- [ ] **Step 5: Commit**

```bash
git add example2/bvh_benchmark.cpp example2/CMakeLists.txt
git commit -m "feat(spatial): bvh_benchmark executable (build/query/memory/accuracy)"
```

---

## Self-Review

**Spec coverage:**
- Swappable interface (§2/§4 of spec) → Task 1.
- BinaryLBVH backend, cache pattern (§5.1/§5.2) → Tasks 2–4.
- WideBVH backend, cache pattern, ray-tracing excluded (§5.3) → Tasks 6–8; `RayTraceable` declared-only in Task 1.
- Factory + selection (§4) → Tasks 5, 9.
- Benchmark: build/query/memory/accuracy, wall-clock, table + CSV (§5 of spec) → Task 10.
- Correctness regression, parametrized over backends (§7) → Tasks 5, 9 (`SpatialIndexBackend` suite).
- Metrics: Build time, query time, node/memory, accuracy (spec decision 3) → all present in Task 10 columns + tests.
- No new shaders; reuse existing (global constraints) → all tasks pass shader filenames only.
- GLOB CMake, only example2 target added (global constraints) → Task 10 is the only CMake edit.

**Placeholder scan:** No "TBD"/"implement later" in delivered code. The `RadiusSearch`/`KNN` stubs in Tasks 2/6 are deliberate, replaced within the same task group (Tasks 3–4, 7–8) — each stub-then-implement step shows full code. `RayTraceable`'s `TODO(future)` is an intentional out-of-scope marker, not a plan gap.

**Type consistency:** `MakeSpatialIndex(Engine::Core::Context&, BVHKind, const BVHParams&)` identical across Tasks 1/5/9. `BuildFromPrimitives(const std::vector<Primitive>&)` protected virtual consistent in base + both backends. Query PC structs match the shader push-constant layouts read from `cmd_knn.comp`/`cmd_radiusSearch.comp` (binary) and the old wide `RadiusPC`/`KNNPC`/`QueryState`. `MAX_K==64`, `INVALID_IDX==0xFFFFFFFFu` consistent. `BackendCase` struct identical in test (Tasks 5/9) and benchmark (Task 10). Binary node byte size `36`, constructionInfo `8`, `QuantizedWideNode==112`, `LeafRange==8`, `BinaryRange==16` match the shaders and old `static_assert`s.

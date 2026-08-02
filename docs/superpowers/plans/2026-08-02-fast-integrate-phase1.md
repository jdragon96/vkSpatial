# Fast-Integrate (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the per-op staging-alloc + `vkQueueWaitIdle` overhead from the TSDF integrate path so `voxel_fill_debugger --dump` integrate time drops far below its current ~148 ms/frame, with GPU results bit-identical to today.

**Architecture:** Add an additive host-visible persistent-mapped mode to `Engine::Core::Buffer` (UMA zero-copy upload). Make `AdvancedTSDF` memcpy its point/normal data into mapped buffers and record its dispatch into a caller-owned `Engine::Compute::CommandBatch` instead of self-submitting. `TiledDirectionalTSDF` and `SubmapAdvancedTSDF` gain a batched `Integrate` that records every touched tile (base + detail) into ONE `CommandBatch` and submits once per frame. All existing public APIs and behavior are preserved (existing GPU tests must still pass).

**Tech Stack:** C++17, Vulkan + VMA, GoogleTest. Existing engine types: `Engine::Core::Buffer`, `Engine::Core::ComputePipeline` (`Args`/`RecordDispatch`/`GetLocalSize`), `Engine::Compute::CommandBatch` (`DispatchElements`/`Submit`).

## Global Constraints

- Build: `VULKAN_SDK=/usr/local`. Tests need conda workaround: `cmake -S . -B build -DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3 -Ugflags_DIR -Uglog_DIR -UGTest_DIR -UCeres_DIR`. Test binary: `build/test/vkspatial_tests` (GLOBs `test/*.cpp`; a new test file needs a reconfigure).
- Platform: Apple M4 Max, UMA — a `HOST_VISIBLE | DEVICE_LOCAL` memory type exists; a mapped buffer is also a valid storage buffer for the shader.
- **Additive only** for `Buffer`/`ComputePipeline`: existing `Allocate`/`Upload`/`Download`/`Dispatch` signatures and behavior unchanged.
- **Bit-identical GPU results:** `test_advancedTsdf`, `test_tiledAdvancedTsdf`, `test_submapAdvancedTsdf` must pass unchanged.
- `CommandBatch` constraint: never dispatch the SAME `ComputePipeline` object twice in one batch. (Each tile is a distinct `AdvancedTSDF` → distinct pipeline; routing yields one `RecordIntegrate` per tile → safe.)
- GLSL/host style per repo (`///` banners, Allman braces, tabs, descriptive names).
- Git: branch, ff-merge to local `main`, do NOT push. Trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Leave pre-existing uncommitted work untouched.

---

### Task 1: `Buffer` host-visible persistent-mapped mode (additive)

**Files:**
- Modify: `src/Engine/Core/Buffer.h` (add 3 methods + `void *m_mapped` member)
- Modify: `src/Engine/Core/Buffer.cpp` (implement; clear `m_mapped` in `free()`)
- Test: `test/test_buffer.cpp` (create)

**Interfaces:**
- Produces: `void Buffer::AllocateHostVisible(uint32_t bytes)`, `void *Buffer::MappedPtr() const`, `void Buffer::FlushMapped(uint32_t bytes) const`.

- [ ] **Step 1: Write the failing test** — `test/test_buffer.cpp`:

```cpp
#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

// A host-visible mapped buffer: writes through MappedPtr are visible to a GPU-side Download.
TEST(Buffer, HostVisibleMappedRoundTrip) {
    Engine::Core::Context ctx;
    Engine::Core::Buffer buf(ctx);
    buf.AllocateHostVisible(16 * sizeof(float));
    ASSERT_NE(buf.MappedPtr(), nullptr);

    std::vector<float> src(16);
    for (int i = 0; i < 16; ++i) src[i] = float(i) * 1.5f;
    std::memcpy(buf.MappedPtr(), src.data(), src.size() * sizeof(float));
    buf.FlushMapped(uint32_t(src.size() * sizeof(float)));

    std::vector<float> dst(16, 0.0f);
    buf.Download(dst.data(), uint32_t(dst.size() * sizeof(float))); // GPU copy back
    for (int i = 0; i < 16; ++i) EXPECT_FLOAT_EQ(dst[i], src[i]);
}

// Plain Allocate is device-local: no persistent mapping.
TEST(Buffer, PlainAllocateHasNoMappedPtr) {
    Engine::Core::Context ctx;
    Engine::Core::Buffer buf(ctx);
    buf.Allocate(16);
    EXPECT_EQ(buf.MappedPtr(), nullptr);
}
```

- [ ] **Step 2: Reconfigure + build, verify it FAILS to compile** (methods don't exist yet):

```bash
VULKAN_SDK=/usr/local cmake -S . -B build -DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3 -Ugflags_DIR -Uglog_DIR -UGTest_DIR -UCeres_DIR
VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8
```
Expected: compile error — `no member named 'AllocateHostVisible'`.

- [ ] **Step 3: Add declarations to `src/Engine/Core/Buffer.h`** — after the `Download(...)` declaration (line ~31):

```cpp
        // Allocates a HOST_VISIBLE (and, on UMA, DEVICE_LOCAL) buffer that stays persistently
        // mapped: MappedPtr() returns a CPU pointer you can memcpy into, and the same buffer is a
        // valid storage buffer for shaders (zero-copy upload — no staging, no submit). Any prior
        // allocation is freed first.
        void AllocateHostVisible(uint32_t bytes);

        // Persistent mapped pointer for AllocateHostVisible buffers; nullptr after plain Allocate.
        void *MappedPtr() const { return m_mapped; }

        // Flush `bytes` of host writes to the device (no-op on HOST_COHERENT memory; always safe).
        void FlushMapped(uint32_t bytes) const;
```

And add the member next to `m_size` (line ~41):

```cpp
        void *m_mapped = nullptr;
```

- [ ] **Step 4: Implement in `src/Engine/Core/Buffer.cpp`** — set `m_mapped = nullptr;` inside `free()` (after `m_size = 0;`), and add after `Allocate(...)`:

```cpp
    void Buffer::AllocateHostVisible(uint32_t bytes) {
        free();

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bytes;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | m_extraUsage;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo info{};
        if (vmaCreateBuffer(m_context.allocator, &bufferInfo, &allocInfo, &m_buffer, &m_allocation,
                            &info) != VK_SUCCESS)
            throw std::runtime_error("Buffer: failed to allocate host-visible");

        m_mapped = info.pMappedData;
        m_size = bytes;
    }

    void Buffer::FlushMapped(uint32_t bytes) const {
        if (m_allocation != VK_NULL_HANDLE)
            vmaFlushAllocation(m_context.allocator, m_allocation, 0, bytes);
    }
```

- [ ] **Step 5: Build + run the two tests, verify PASS**:

```bash
VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8
(cd build && VULKAN_SDK=/usr/local ./test/vkspatial_tests --gtest_filter='Buffer.*')
```
Expected: 2 passed.

- [ ] **Step 6: Commit**:

```bash
git add src/Engine/Core/Buffer.h src/Engine/Core/Buffer.cpp test/test_buffer.cpp
git commit -m "feat(core): Buffer host-visible persistent-mapped mode (additive)"
```

---

### Task 2: `AdvancedTSDF` mapped uploads + `RecordIntegrate(CommandBatch&)`

**Files:**
- Modify: `src/Engine/Spatial/AdvancedTSDF.h` (include CommandBatch; declare `RecordIntegrate`)
- Modify: `src/Engine/Spatial/AdvancedTSDF.cpp` (Build: point/normal → `AllocateHostVisible`; add `RecordIntegrate`; `Integrate` → wrapper)
- Test: `test/test_advancedTsdf.cpp` (add one equivalence test)

**Interfaces:**
- Consumes: `Buffer::AllocateHostVisible/MappedPtr/FlushMapped` (Task 1); `CommandBatch::DispatchElements(ComputePipeline&, uint32_t)` + `Submit()`.
- Produces: `void AdvancedTSDF::RecordIntegrate(const std::vector<Eigen::Vector3f>&, const std::vector<Eigen::Vector3f>&, const Eigen::Vector3f&, Engine::Compute::CommandBatch&)`.

- [ ] **Step 1: Write the failing test** — append to `test/test_advancedTsdf.cpp` (uses the file's existing includes/helpers; if it lacks a plane helper, build a small inline one):

```cpp
// RecordIntegrate into a CommandBatch produces the same hash as the self-submitting Integrate.
TEST(AdvancedTsdf, RecordIntegrateMatchesIntegrate) {
    Engine::Core::Context ctx;
    std::vector<Eigen::Vector3f> pts, nrm;
    for (int i = -10; i <= 10; ++i)
        for (int j = -10; j <= 10; ++j) {
            pts.emplace_back(i * 0.02f, j * 0.02f, 0.0f);
            nrm.emplace_back(0.0f, 0.0f, 1.0f);
        }
    const Eigen::Vector3f cam(0, 0, 1);

    Engine::Spatial::AdvancedTSDF a, b;
    a.Build(ctx, 0.02f, 0.06f);
    b.Build(ctx, 0.02f, 0.06f);

    a.Integrate(pts, nrm, cam);
    {
        Engine::Compute::CommandBatch batch(ctx);
        b.RecordIntegrate(pts, nrm, cam, batch);
        batch.Submit();
    }

    const auto ea = a.DownloadEntries();
    const auto eb = b.DownloadEntries();
    ASSERT_GT(ea.size(), 100u);
    EXPECT_EQ(ea.size(), eb.size());
}
```
(Add `#include "Engine/Compute/CommandBatch.h"` to the test if not present.)

- [ ] **Step 2: Build, verify FAIL** — `no member named 'RecordIntegrate'`:

```bash
VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8
```

- [ ] **Step 3: Edit `src/Engine/Spatial/AdvancedTSDF.h`** — add include near the top:

```cpp
#include "Engine/Compute/CommandBatch.h"
```
and declare after the existing `Integrate(...)` declaration (line ~85):

```cpp
        // Records upload (memcpy into mapped buffers) + the integrate dispatch into `batch` WITHOUT
        // submitting. The caller submits the batch (batches many tiles into one submit). Same result
        // as Integrate.
        void RecordIntegrate(const std::vector<Eigen::Vector3f> &points,
                             const std::vector<Eigen::Vector3f> &normals,
                             const Eigen::Vector3f &cameraPos,
                             Engine::Compute::CommandBatch &batch);
```

- [ ] **Step 4: Edit `src/Engine/Spatial/AdvancedTSDF.cpp`**:

(a) In `Build`, change the point/normal allocations (lines ~88-89) from `Allocate` to `AllocateHostVisible`:

```cpp
        m_pointBuffer->AllocateHostVisible(maxPoints * 3u * sizeof(float));
        m_normalBuffer->AllocateHostVisible(maxPoints * 3u * sizeof(float));
```

(b) Replace the body of `Integrate` (lines ~109-129) with a wrapper + the new `RecordIntegrate`:

```cpp
    void AdvancedTSDF::Integrate(const std::vector<Eigen::Vector3f> &points,
                                 const std::vector<Eigen::Vector3f> &normals,
                                 const Eigen::Vector3f &cameraPos) {
        if (points.empty()) return;
        Engine::Compute::CommandBatch batch(*m_ctx);
        RecordIntegrate(points, normals, cameraPos, batch);
        batch.Submit();
    }

    void AdvancedTSDF::RecordIntegrate(const std::vector<Eigen::Vector3f> &points,
                                       const std::vector<Eigen::Vector3f> &normals,
                                       const Eigen::Vector3f &cameraPos,
                                       Engine::Compute::CommandBatch &batch) {
        if (points.empty()) return;

        const uint32_t N = std::min({static_cast<uint32_t>(points.size()),
                                     static_cast<uint32_t>(normals.size()), m_maxPoints});
        if (N == 0) return;

        // Zero-copy upload: memcpy straight into the persistently mapped storage buffers.
        std::memcpy(m_pointBuffer->MappedPtr(), points.data(), N * 3u * sizeof(float));
        std::memcpy(m_normalBuffer->MappedPtr(), normals.data(), N * 3u * sizeof(float));
        m_pointBuffer->FlushMapped(N * 3u * sizeof(float));
        m_normalBuffer->FlushMapped(N * 3u * sizeof(float));

        IntegratePC pc{
                N, m_hashCapacity, m_voxelSize, m_truncation,
                cameraPos.x(), cameraPos.y(), cameraPos.z(),
                m_quality.maxDirections, m_quality.dirExponent,
                m_quality.viewAngleWeight ? 1u : 0u,
                m_originVoxel.x(), m_originVoxel.y(), m_originVoxel.z(),
                uint32_t(m_pointToPlane ? 1u : 0u), m_confWeight};
        m_kernel->Args(pc);
        batch.DispatchElements(*m_kernel, N);
    }
```
Add `#include "Engine/Compute/CommandBatch.h"` and `#include <cstring>` to the .cpp if not present.

- [ ] **Step 5: Build + run AdvancedTsdf tests (equivalence + all existing), verify PASS**:

```bash
VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8
(cd build && VULKAN_SDK=/usr/local ./test/vkspatial_tests --gtest_filter='AdvancedTsdf.*')
```
Expected: all pass (existing tests prove bit-identical; the new test confirms Record==Integrate).

- [ ] **Step 6: Commit**:

```bash
git add src/Engine/Spatial/AdvancedTSDF.h src/Engine/Spatial/AdvancedTSDF.cpp test/test_advancedTsdf.cpp
git commit -m "feat(spatial): AdvancedTSDF mapped uploads + RecordIntegrate(CommandBatch&)"
```

---

### Task 3: `TiledDirectionalTSDF` batched Integrate (one submit per Integrate)

**Files:**
- Modify: `src/Engine/Spatial/TiledDirectionalTSDF.h` (include CommandBatch; add batched overload; self-submit reuses it)
- Test: `test/test_tiledAdvancedTsdf.cpp` (existing tests must still pass; add a batched-equivalence test)

**Interfaces:**
- Consumes: `Backend::RecordIntegrate(pts, nrm, cam, CommandBatch&)` (Task 2 for `AdvancedTSDF`); `m_ctx` (already stored).
- Produces: `void Integrate(points, normals, cameraPos, Engine::Compute::CommandBatch &batch)` on `TiledDirectionalTSDF<Backend>`.

- [ ] **Step 1: Write the failing test** — append to `test/test_tiledAdvancedTsdf.cpp`:

```cpp
// The batched Integrate overload yields the same occupied count as the self-submitting one.
TEST(TiledAdvanced, BatchedIntegrateMatches) {
    Engine::Core::Context ctx;
    std::vector<Eigen::Vector3f> pts, nrm;
    for (int i = -30; i <= 30; ++i)
        for (int j = -30; j <= 30; ++j) {
            pts.emplace_back(i * 0.05f, j * 0.05f, 0.0f);
            nrm.emplace_back(0.0f, 0.0f, 1.0f);
        }
    const Eigen::Vector3f cam(0, 0, 1);

    Engine::Spatial::TiledAdvancedTSDF a, b;
    a.Build(ctx, 0.05f, 0.15f);
    b.Build(ctx, 0.05f, 0.15f);

    a.Integrate(pts, nrm, cam); // self-submitting
    {
        Engine::Compute::CommandBatch batch(ctx);
        b.Integrate(pts, nrm, cam, batch); // batched
        batch.Submit();
    }
    EXPECT_EQ(a.DownloadEntries().size(), b.DownloadEntries().size());
    EXPECT_GT(a.DownloadEntries().size(), 100u);
}
```
(Add `#include "Engine/Compute/CommandBatch.h"` to the test if not present.)

- [ ] **Step 2: Build, verify FAIL** (no 4-arg Integrate):

```bash
VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8
```

- [ ] **Step 3: Edit `src/Engine/Spatial/TiledDirectionalTSDF.h`** — add include near the top:

```cpp
#include "Engine/Compute/CommandBatch.h"
```
Replace the tile-dispatch loop at the end of the existing `Integrate` (lines ~121-124) so the self-submitting form delegates to a new batched overload. Change the existing function's tail from:

```cpp
            for (auto &kv : routed) {
                Backend *tile = tileFor(kv.first);
                tile->Integrate(kv.second.pts, kv.second.nrm, cameraPos);
            }
        }
```
to:

```cpp
            for (auto &kv : routed) {
                Backend *tile = tileFor(kv.first);
                tile->RecordIntegrate(kv.second.pts, kv.second.nrm, cameraPos, batch);
            }
        }
```
and change that function's signature to take a batch (it becomes the batched overload):

```cpp
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos, Engine::Compute::CommandBatch &batch) {
```
Then add the self-submitting overload just above it:

```cpp
        // Self-submitting: route + record every touched tile into one CommandBatch, submit once.
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero()) {
            if (std::min(points.size(), normals.size()) == 0) return;
            Engine::Compute::CommandBatch batch(*m_ctx);
            Integrate(points, normals, cameraPos, batch);
            batch.Submit();
        }
```
(The batched overload keeps the routing body; remove the old default arg from it since the self-submitting overload now owns the default.)

- [ ] **Step 4: Build + run TiledAdvanced tests, verify PASS**:

```bash
VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8
(cd build && VULKAN_SDK=/usr/local ./test/vkspatial_tests --gtest_filter='TiledAdvanced.*')
```
Expected: all pass (existing + batched-equivalence).

- [ ] **Step 5: Commit**:

```bash
git add src/Engine/Spatial/TiledDirectionalTSDF.h test/test_tiledAdvancedTsdf.cpp
git commit -m "feat(spatial): TiledDirectionalTSDF batched Integrate (one submit/frame)"
```

---

### Task 4: `SubmapAdvancedTSDF` batched Integrate (base + detail in one batch) + measure

**Files:**
- Modify: `src/Engine/Spatial/SubmapAdvancedTSDF.h` (store `m_ctx`; batch base + detail into one submit)
- Test: `test/test_submapAdvancedTsdf.cpp` (existing tests must still pass)
- Measure: `example2/voxel_fill_debugger --dump`

**Interfaces:**
- Consumes: `TiledAdvancedTSDF::Integrate(pts, nrm, cam, CommandBatch&)` (Task 3); `CommandBatch(ctx)` + `Submit()`.

- [ ] **Step 1: Edit `src/Engine/Spatial/SubmapAdvancedTSDF.h`** — store the context in `Build` (add `m_ctx = &ctx;` at the top of `Build`, and a member `Engine::Core::Context *m_ctx = nullptr;` near the other members). Include is already present (`Engine/Core/Context.h`); add `#include "Engine/Compute/CommandBatch.h"`.

Replace the body of `Integrate` (the base + detail calls) with a single batched submit:

```cpp
        void Integrate(const std::vector<Eigen::Vector3f> &pts,
                       const std::vector<Eigen::Vector3f> &nrm,
                       const Eigen::Vector3f &cam = Eigen::Vector3f::Zero()) {
            const std::size_t n = std::min(pts.size(), nrm.size());
            if (n == 0) return;

            Engine::Compute::CommandBatch batch(*m_ctx);
            m_base.Integrate(pts, nrm, cam, batch); // all points -> base tiles

            if (m_finalized && !m_dense.empty()) {
                std::vector<Eigen::Vector3f> dp, dn;
                dp.reserve(n);
                dn.reserve(n);
                for (std::size_t i = 0; i < n; ++i)
                    if (m_dense.count(blockOf(pts[i]))) {
                        dp.push_back(pts[i]);
                        dn.push_back(nrm[i]);
                    }
                if (!dp.empty()) m_detail.Integrate(dp, dn, cam, batch); // detail tiles (distinct pipelines)
            }
            batch.Submit(); // one submit for base + detail
        }
```
(Base tiles and detail tiles are all distinct `AdvancedTSDF` pipelines writing independent tile hashes, so one batch is safe and needs no intra-batch barrier.)

- [ ] **Step 2: Build + run Submap tests, verify PASS** (bit-identical results):

```bash
VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8
(cd build && VULKAN_SDK=/usr/local ./test/vkspatial_tests --gtest_filter='SubmapAdvanced.*')
```
Expected: 6/6 pass.

- [ ] **Step 3: Build the debugger and measure integrate before/after**:

```bash
VULKAN_SDK=/usr/local cmake --build build --target voxel_fill_debugger -j8
mkdir -p scan_out/perf && for i in $(seq -w 0 14); do cp scans/dragon/frame_00$i.ply scan_out/perf/ 2>/dev/null; done
VULKAN_SDK=/usr/local ./build/example2/voxel_fill_debugger --dir scan_out/perf --voxel 0.5 --dump 2>&1 | tail -6
rm -rf scan_out/perf
```
Expected: the `--dump` timing table's `integrate` avg is far below the pre-Phase-1 ~148 ms/frame; occupied counts and dense-block/tile counts unchanged from before (results identical).

- [ ] **Step 4: Commit**:

```bash
git add src/Engine/Spatial/SubmapAdvancedTSDF.h
git commit -m "feat(spatial): SubmapAdvancedTSDF batched Integrate (base+detail one submit)"
```

---

## Notes for the implementer

- If `AdvancedTSDF.cpp`'s `IntegratePC` struct is defined locally in the .cpp (it is), `RecordIntegrate` lives in the same .cpp so it sees the struct — keep it there.
- `CommandBatch::Submit()` does one `vkQueueWaitIdle`; that single per-frame sync is expected (Phase 2 removes even that from the render thread by moving Integrate to a worker). Do not add extra `vkDeviceWaitIdle`.
- Do NOT change `Buffer::Upload`/`Download` or `ComputePipeline::Dispatch` — other engine code depends on them.
- After all four tasks, run the full affected suite once: `--gtest_filter='Buffer.*:AdvancedTsdf.*:TiledAdvanced.*:SubmapAdvanced.*'`.

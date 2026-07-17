# DirectionalTSDF Phase 5 (submission batching / high-perf compute API) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the ~16–20 synchronous `vkQueueWaitIdle` GPU flushes per DirectionalTSDF frame down to 3 (4 with write-back) by recording each run of GPU ops between CPU readback points into a single command buffer and submitting once — using a new, render-independent high-performance compute batching library (`Engine::Compute`) plus persistent mapped staging buffers.

**Architecture:** Three layers. (1) One purely additive method on `Engine::Core::ComputePipeline` — `RecordDispatch(cmd,...)` — that records a dispatch into a caller-owned command buffer instead of self-submitting (existing `Dispatch()` behavior unchanged). (2) A new standalone `Engine::Compute` static library with `StagingBuffer` (persistently-mapped host-visible VMA buffer) and `CommandBatch` (records dispatches/copies/fills/barriers, submits once). It depends only on `Engine::Core`, never on `Engine::Render`, so it cannot conflict with the just-merged rendering work. (3) `DirectionalTSDF` reworked to route its whole GPU hot path through `CommandBatch`/`StagingBuffer`, grouped into 3 batches (classify+readback → upload+register+integrate → extract+readback), with a `Stats.gpuSubmits` counter that makes the reduction a deterministic test assertion rather than a flaky timing measurement.

**Tech Stack:** C++17, Vulkan 1.3, VMA (already vendored), `Engine::Core`, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-07-17-directional-tsdf-design.md` §20 Phase 5(1 H2D/D2H overlap → 여기선 unified-memory라 submit 병합으로 재해석, 2 chunked staging, 4 upload budget의 기반이 되는 persistent staging). Platform note: this machine (Apple M4 Max) is unified-memory, so there is no discrete copy engine to overlap and no real PCIe transfer; the measured cost is per-submit queue-flush overhead, and the win here is **fewer submits**, which helps on any GPU.

## Global Constraints

- Phase 1~4 plan의 Global Constraints 유지. **기존 27개 Directional 테스트는 매 Task 후 전부 그대로 통과해야 한다** (동작 불변, 최적화만).
- `Engine::Core` 변경은 **순수 additive만** 허용 (새 public 메서드 + private helper 리팩터; 기존 시그니처/동작 불변). 머지된 `Engine::Render`가 쓰는 `ComputePipeline` API를 절대 깨지 않는다.
- `Engine::Compute`는 `Engine::Core`에만 링크한다 (`Engine::Render`/`vkRender`/`vkCommon` 의존 금지).
- Barrier 정책: 초기 구현은 보수적 global `VkMemoryBarrier` (src `SHADER_WRITE|TRANSFER_WRITE`, dst `SHADER_READ|SHADER_WRITE|TRANSFER_READ|TRANSFER_WRITE`, stage `COMPUTE|TRANSFER` 양방향). 정밀 per-buffer barrier는 YAGNI.
- `CommandBatch` 사용 제약: **동일한 `ComputePipeline` 객체를 한 배치 안에서 두 번 dispatch하지 않는다** (파이프라인이 디스크립터셋 1개를 공유하므로, submit 전에 재바인딩하면 첫 dispatch가 참조하는 셋을 덮어씀). 서로 다른 바인딩이 필요하면 별도 파이프라인 객체이거나 별도 배치여야 한다.
- 테스트 baseline: 현재 전체 스위트 통과 개수는 머지 직후 상태 기준(빌드 확인 완료). `WideBVHTest.RadiusMatchesCpuReference`는 기존 무관 실패.
- 빌드:
  ```bash
  export VULKAN_SDK=/Users/sjy/VulkanSDK/1.4.328.1/macOS
  cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
  ```

---

### Task 1: `ComputePipeline::RecordDispatch` (additive to Engine::Core)

**Files:**
- Modify: `src/Engine/Core/ComputePipeline.h`
- Modify: `src/Engine/Core/ComputePipeline.cpp`
- Modify: `test/test_engineCore.cpp`

**Interfaces:**
- Produces (used by Task 2's `CommandBatch`):
  ```cpp
  // Records pipeline bind + push constants + dispatch into a caller-owned command
  // buffer. Ensures the pipeline/descriptors are built/updated first, but does NOT
  // submit. The caller owns submission (see Engine::Compute::CommandBatch).
  void RecordDispatch(VkCommandBuffer cmd, uint32_t gridX, uint32_t gridY = 1, uint32_t gridZ = 1);
  ```

- [ ] **Step 1: Write the failing test**

Append to `test/test_engineCore.cpp` (it already includes `Buffer.h`/`ComputePipeline.h`/`Context.h`/`OneShotCommands.h`):

```cpp
TEST(ComputePipelineTest, RecordDispatchIntoExternalCommandBufferProducesSameResult) {
    Context ctx;

    constexpr uint32_t N = 4096u;
    std::vector<uint32_t> input(N, 3u);

    Buffer inputBuffer(ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    inputBuffer.Allocate(N * sizeof(uint32_t));
    inputBuffer.Upload(input.data(), N * sizeof(uint32_t));

    static const char *kShader = R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer Buf { uint v[]; } b;
        layout(push_constant) uniform PC { uint count; } pc;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= pc.count) return;
            b.v[i] = b.v[i] * 2u;
        }
    )";
    struct PC { uint32_t count; };

    ComputePipeline pipe(ctx);
    pipe.Build(kShader, ShaderInput::GlslSrc).Bind(0, inputBuffer).Args(PC{N});

    // Drive submission ourselves via a one-shot command buffer + RecordDispatch.
    const uint32_t gridX = (N + pipe.GetLocalSize().width - 1) / pipe.GetLocalSize().width;
    SubmitOneShot(ctx, QueueRole::Compute, [&](VkCommandBuffer cmd) {
        pipe.RecordDispatch(cmd, gridX);
    });

    std::vector<uint32_t> result(N, 0u);
    inputBuffer.Download(result.data(), N * sizeof(uint32_t));
    for (uint32_t i = 0; i < N; ++i) EXPECT_EQ(result[i], 6u) << "index " << i;
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)" 2>&1 | grep -m1 "RecordDispatch"
```
Expected: compile error — `no member named 'RecordDispatch' in 'ComputePipeline'`.

- [ ] **Step 3: Declare `RecordDispatch` and a private `recordInto` in the header**

In `src/Engine/Core/ComputePipeline.h`, after the `Sync();` declaration (line ~55), add the public method:

```cpp
        // Records pipeline bind + push constants + dispatch into a caller-owned command
        // buffer without submitting. Ensures the pipeline and descriptors are built first.
        void RecordDispatch(VkCommandBuffer cmd, uint32_t gridX, uint32_t gridY = 1, uint32_t gridZ = 1);
```

In the private section, next to `void submit(uint32_t gridX, uint32_t gridY, uint32_t gridZ);`, add:

```cpp
        void recordInto(VkCommandBuffer cmd, uint32_t gridX, uint32_t gridY, uint32_t gridZ);
```

- [ ] **Step 4: Implement in the .cpp (DRY refactor of `submit`)**

In `src/Engine/Core/ComputePipeline.cpp`, replace the whole `submit(...)` function (lines ~349-363) with:

```cpp
    void ComputePipeline::recordInto(VkCommandBuffer cmd, uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);
        if (!m_pushData.empty())
            vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               static_cast<uint32_t>(m_pushData.size()), m_pushData.data());
        vkCmdDispatch(cmd, gridX, gridY, gridZ);
    }

    void ComputePipeline::submit(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        SubmitOneShot(m_context, QueueRole::Compute, [&](VkCommandBuffer cmd) {
            recordInto(cmd, gridX, gridY, gridZ);
        });
    }

    void ComputePipeline::RecordDispatch(VkCommandBuffer cmd, uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        ensurePipeline();
        if (m_dirty) updateDescriptors();
        recordInto(cmd, gridX, gridY, gridZ);
    }
```

- [ ] **Step 5: Build and run — the new test plus the whole Engine::Core suite**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="ComputePipelineTest.*:BufferTest.*:ContextTest.*:OneShotCommandsTest.*:ImageTest.*"
```
Expected: all PASS (existing Dispatch path is unchanged because `submit` now delegates to `recordInto`).

- [ ] **Step 6: Commit**

```bash
git add src/Engine/Core/ComputePipeline.h src/Engine/Core/ComputePipeline.cpp test/test_engineCore.cpp
git commit -m "Add ComputePipeline::RecordDispatch for external command-buffer recording"
```

---

### Task 2: `Engine::Compute` library — `StagingBuffer` + `CommandBatch`

**Files:**
- Create: `src/Engine/Compute/StagingBuffer.h`
- Create: `src/Engine/Compute/StagingBuffer.cpp`
- Create: `src/Engine/Compute/CommandBatch.h`
- Create: `src/Engine/Compute/CommandBatch.cpp`
- Modify: `src/Engine/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`
- Create: `test/test_engineCompute.cpp`

**Interfaces:**
- Consumes: `Engine::Core::Context`, `Engine::Core::ComputePipeline::RecordDispatch` (Task 1), `Engine::Core::QueueRole`.
- Produces (used by Tasks 3-5):
  ```cpp
  namespace Engine::Compute {
      class StagingBuffer {
      public:
          StagingBuffer(Engine::Core::Context &ctx, VkDeviceSize bytes, VkBufferUsageFlags usage);
          ~StagingBuffer();
          void *Mapped() const;          // persistently-mapped host pointer
          VkBuffer Handle() const;
          VkDeviceSize Size() const;
      };
      class CommandBatch {
      public:
          explicit CommandBatch(Engine::Core::Context &ctx,
                                Engine::Core::QueueRole role = Engine::Core::QueueRole::Compute);
          ~CommandBatch();
          CommandBatch &Dispatch(Engine::Core::ComputePipeline &pipe, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);
          CommandBatch &DispatchElements(Engine::Core::ComputePipeline &pipe, uint32_t numElements);
          CommandBatch &CopyBuffer(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy> &regions);
          CommandBatch &FillBuffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, uint32_t data);
          CommandBatch &Barrier();
          void Submit();  // end + submit + waitIdle + free; single-use
      };
  }
  ```

- [ ] **Step 1: Write `StagingBuffer.h`**

```cpp
#pragma once

#include "Engine/Core/Context.h"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace Engine::Compute {

    // A persistently-mapped host-visible VMA buffer for staging transfers. Unlike
    // Engine::Core::Buffer (device-local, transient staging created per Upload/Download),
    // this is allocated once and its Mapped() pointer stays valid, so it can back many
    // batched copies without re-allocating each frame.
    class StagingBuffer {
    public:
        // usage must include the transfer direction(s) you need
        // (VK_BUFFER_USAGE_TRANSFER_SRC_BIT and/or VK_BUFFER_USAGE_TRANSFER_DST_BIT).
        StagingBuffer(Engine::Core::Context &context, VkDeviceSize bytes, VkBufferUsageFlags usage);
        ~StagingBuffer();

        StagingBuffer(const StagingBuffer &) = delete;
        StagingBuffer &operator=(const StagingBuffer &) = delete;

        void *Mapped() const { return m_mapped; }
        VkBuffer Handle() const { return m_buffer; }
        VkDeviceSize Size() const { return m_size; }

    private:
        Engine::Core::Context &m_context;
        VkBuffer m_buffer = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;
        void *m_mapped = nullptr;
        VkDeviceSize m_size = 0;
    };

} // namespace Engine::Compute
```

- [ ] **Step 2: Write `StagingBuffer.cpp`**

```cpp
#include "Engine/Compute/StagingBuffer.h"

#include <stdexcept>

namespace Engine::Compute {

    StagingBuffer::StagingBuffer(Engine::Core::Context &context, VkDeviceSize bytes,
                                 VkBufferUsageFlags usage)
        : m_context(context), m_size(bytes) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bytes;
        bufferInfo.usage = usage;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo allocationInfo{};
        if (vmaCreateBuffer(m_context.allocator, &bufferInfo, &allocInfo,
                            &m_buffer, &m_allocation, &allocationInfo) != VK_SUCCESS)
            throw std::runtime_error("StagingBuffer: failed to allocate");
        m_mapped = allocationInfo.pMappedData;
    }

    StagingBuffer::~StagingBuffer() {
        if (m_buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(m_context.allocator, m_buffer, m_allocation);
    }

} // namespace Engine::Compute
```

Note: `VMA_MEMORY_USAGE_AUTO` with the mapped+host-access flags yields host-visible memory that on this platform (and any integrated/unified device) is coherent; the mirror of `Engine::Core::Buffer::Download` in the repo does not flush/invalidate for these, and the existing tests pass — we follow that same working convention.

- [ ] **Step 3: Write `CommandBatch.h`**

```cpp
#pragma once

#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/OneShotCommands.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Compute {

    // Records a run of compute dispatches and buffer transfers into ONE command buffer
    // and submits it once (a single vkQueueWaitIdle), instead of one submit per op.
    //
    // Constraint: do NOT Dispatch the SAME ComputePipeline object twice within one batch.
    // Each pipeline owns a single descriptor set; re-binding it before Submit would rewrite
    // the set the first dispatch still references. Use separate pipeline objects or batches.
    class CommandBatch {
    public:
        explicit CommandBatch(Engine::Core::Context &context,
                              Engine::Core::QueueRole role = Engine::Core::QueueRole::Compute);
        ~CommandBatch();

        CommandBatch(const CommandBatch &) = delete;
        CommandBatch &operator=(const CommandBatch &) = delete;

        CommandBatch &Dispatch(Engine::Core::ComputePipeline &pipe,
                               uint32_t gridX, uint32_t gridY = 1, uint32_t gridZ = 1);
        CommandBatch &DispatchElements(Engine::Core::ComputePipeline &pipe, uint32_t numElements);
        CommandBatch &CopyBuffer(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy> &regions);
        CommandBatch &FillBuffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, uint32_t data);
        // Conservative global compute+transfer read/write barrier between dependent ops.
        CommandBatch &Barrier();

        // End recording, submit on the batch's queue, wait idle, free the command buffer.
        // Single-use: a batch cannot be submitted twice.
        void Submit();

    private:
        Engine::Core::Context &m_context;
        VkQueue m_queue = VK_NULL_HANDLE;
        VkCommandPool m_pool = VK_NULL_HANDLE;
        VkCommandBuffer m_cmd = VK_NULL_HANDLE;
        bool m_submitted = false;

        void ensureBegun();
    };

} // namespace Engine::Compute
```

- [ ] **Step 4: Write `CommandBatch.cpp`**

```cpp
#include "Engine/Compute/CommandBatch.h"

#include <stdexcept>

namespace Engine::Compute {

    CommandBatch::CommandBatch(Engine::Core::Context &context, Engine::Core::QueueRole role)
        : m_context(context) {
        if (role == Engine::Core::QueueRole::Compute) {
            m_queue = context.computeQueue;
            m_pool = context.cmdPool;
        } else {
            m_queue = context.graphicsQueue;
            m_pool = context.graphicsCmdPool;
        }
        if (m_queue == VK_NULL_HANDLE || m_pool == VK_NULL_HANDLE)
            throw std::runtime_error("CommandBatch: requested QueueRole is unavailable on this Context");
    }

    CommandBatch::~CommandBatch() {
        if (m_cmd != VK_NULL_HANDLE && !m_submitted)
            vkFreeCommandBuffers(m_context.device, m_pool, 1, &m_cmd);
    }

    void CommandBatch::ensureBegun() {
        if (m_cmd != VK_NULL_HANDLE) return;

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(m_context.device, &allocInfo, &m_cmd) != VK_SUCCESS)
            throw std::runtime_error("CommandBatch: failed to allocate command buffer");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_cmd, &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("CommandBatch: failed to begin command buffer");
    }

    CommandBatch &CommandBatch::Dispatch(Engine::Core::ComputePipeline &pipe,
                                         uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        ensureBegun();
        pipe.RecordDispatch(m_cmd, gridX, gridY, gridZ);
        return *this;
    }

    CommandBatch &CommandBatch::DispatchElements(Engine::Core::ComputePipeline &pipe,
                                                 uint32_t numElements) {
        const uint32_t local = pipe.GetLocalSize().width;
        if (local == 0)
            throw std::runtime_error("CommandBatch::DispatchElements: pipeline local_size.x=0");
        return Dispatch(pipe, (numElements + local - 1) / local, 1, 1);
    }

    CommandBatch &CommandBatch::CopyBuffer(VkBuffer src, VkBuffer dst,
                                           const std::vector<VkBufferCopy> &regions) {
        if (regions.empty()) return *this;
        ensureBegun();
        vkCmdCopyBuffer(m_cmd, src, dst, static_cast<uint32_t>(regions.size()), regions.data());
        return *this;
    }

    CommandBatch &CommandBatch::FillBuffer(VkBuffer buffer, VkDeviceSize offset,
                                           VkDeviceSize size, uint32_t data) {
        ensureBegun();
        vkCmdFillBuffer(m_cmd, buffer, offset, size, data);
        return *this;
    }

    CommandBatch &CommandBatch::Barrier() {
        ensureBegun();
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        const VkPipelineStageFlags stages =
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        vkCmdPipelineBarrier(m_cmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        return *this;
    }

    void CommandBatch::Submit() {
        if (m_submitted)
            throw std::runtime_error("CommandBatch: already submitted");
        if (m_cmd == VK_NULL_HANDLE) { // nothing recorded
            m_submitted = true;
            return;
        }
        if (vkEndCommandBuffer(m_cmd) != VK_SUCCESS)
            throw std::runtime_error("CommandBatch: failed to end command buffer");

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_cmd;
        if (vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
            throw std::runtime_error("CommandBatch: failed to submit");
        vkQueueWaitIdle(m_queue);

        vkFreeCommandBuffers(m_context.device, m_pool, 1, &m_cmd);
        m_cmd = VK_NULL_HANDLE;
        m_submitted = true;
    }

} // namespace Engine::Compute
```

- [ ] **Step 5: Add the `EngineCompute` CMake target**

In `src/Engine/CMakeLists.txt`, after the `EngineSpatial` block (end of file), append:

```cmake
file(GLOB_RECURSE ENGINE_COMPUTE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Compute/*.cpp")

add_library(EngineCompute STATIC ${ENGINE_COMPUTE_SOURCES})
add_library(Engine::Compute ALIAS EngineCompute)

target_link_libraries(EngineCompute
        PUBLIC Engine::Core Vulkan::Vulkan)

target_include_directories(EngineCompute
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/vma>
        PRIVATE
            "${VULKAN_SDK}/include")
```

And make `EngineSpatial` link it (DirectionalTSDF will use it in Task 3). In the existing `target_link_libraries(EngineSpatial ...)` line, add `Engine::Compute`:

```cmake
target_link_libraries(EngineSpatial
        PUBLIC Engine::Core Engine::Compute Vulkan::Vulkan)
```
(If the current EngineSpatial link line differs, keep its existing entries and add `Engine::Compute`.)

- [ ] **Step 6: Link `Engine::Compute` into the test target**

In `test/CMakeLists.txt`, add `Engine::Compute` to `target_link_libraries(vkspatial_tests ...)` right after `Engine::Spatial`:

```cmake
        Engine::Spatial
        Engine::Compute
```

- [ ] **Step 7: Write `test/test_engineCompute.cpp`**

```cpp
#include <gtest/gtest.h>

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Compute/StagingBuffer.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include <cstring>
#include <vector>

using namespace Engine::Core;
using namespace Engine::Compute;

TEST(StagingBufferTest, MappedPointerRoundTrips) {
    Context ctx;
    StagingBuffer staging(ctx, 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ASSERT_NE(staging.Mapped(), nullptr);
    EXPECT_EQ(staging.Size(), 256u);

    auto *p = static_cast<uint32_t *>(staging.Mapped());
    for (int i = 0; i < 64; ++i) p[i] = uint32_t(i * 7);
    for (int i = 0; i < 64; ++i) EXPECT_EQ(p[i], uint32_t(i * 7));
}

TEST(CommandBatchTest, TwoDispatchesWithBarrierRunInOneSubmit) {
    Context ctx;
    constexpr uint32_t N = 4096u;

    Buffer buf(ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    buf.Allocate(N * sizeof(uint32_t));
    std::vector<uint32_t> zeros(N, 0u);
    buf.Upload(zeros.data(), N * sizeof(uint32_t));

    // Program A: v[i] = i.  Program B: v[i] = v[i] * 2.  Separate pipeline objects.
    static const char *kFill = R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer Buf { uint v[]; } b;
        layout(push_constant) uniform PC { uint n; } pc;
        void main(){ uint i=gl_GlobalInvocationID.x; if(i<pc.n) b.v[i]=i; })";
    static const char *kDouble = R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer Buf { uint v[]; } b;
        layout(push_constant) uniform PC { uint n; } pc;
        void main(){ uint i=gl_GlobalInvocationID.x; if(i<pc.n) b.v[i]=b.v[i]*2u; })";
    struct PC { uint32_t n; };

    ComputePipeline fill(ctx);
    fill.Build(kFill, ShaderInput::GlslSrc).Bind(0, buf).Args(PC{N});
    ComputePipeline dbl(ctx);
    dbl.Build(kDouble, ShaderInput::GlslSrc).Bind(0, buf).Args(PC{N});

    CommandBatch batch(ctx);
    batch.DispatchElements(fill, N).Barrier().DispatchElements(dbl, N).Submit();

    std::vector<uint32_t> result(N, 0xFFFFFFFFu);
    buf.Download(result.data(), N * sizeof(uint32_t));
    for (uint32_t i = 0; i < N; ++i) EXPECT_EQ(result[i], i * 2u) << "index " << i;
}

TEST(CommandBatchTest, FillAndCopyRecordInOneSubmit) {
    Context ctx;
    constexpr uint32_t N = 64u;

    Buffer src(ctx);
    Buffer dst(ctx);
    src.Allocate(N * sizeof(uint32_t));
    dst.Allocate(N * sizeof(uint32_t));

    CommandBatch batch(ctx);
    batch.FillBuffer(src.Handle(), 0, VK_WHOLE_SIZE, 0xABABABABu).Barrier();
    std::vector<VkBufferCopy> regions(1);
    regions[0].srcOffset = 0;
    regions[0].dstOffset = 0;
    regions[0].size = N * sizeof(uint32_t);
    batch.CopyBuffer(src.Handle(), dst.Handle(), regions).Submit();

    std::vector<uint32_t> out(N, 0u);
    dst.Download(out.data(), N * sizeof(uint32_t));
    for (uint32_t v : out) EXPECT_EQ(v, 0xABABABABu);
}
```

- [ ] **Step 8: Configure, build, run**

```bash
export VULKAN_SDK=/Users/sjy/VulkanSDK/1.4.328.1/macOS
cmake -S . -B build -DVULKAN_SDK="$VULKAN_SDK"
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="StagingBufferTest.*:CommandBatchTest.*"
```
Expected: 3 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/Engine/Compute src/Engine/CMakeLists.txt test/CMakeLists.txt test/test_engineCompute.cpp
git commit -m "Add Engine::Compute: StagingBuffer + CommandBatch (batched compute submission)"
```

---

### Task 3: `Stats.gpuSubmits` + batched `BeginFrame` (classify + readback + write-back)

> **IMPLEMENTATION CORRECTIONS (supersede the messy sub-steps below):**
> - Add a dedicated `m_stageCleanFree` StagingBuffer (DST, `poolCapacity*u32`) alongside `m_stageLists` so Batch 1 copies counts + reusable + cleanFree back in ONE submit → a bare `BeginFrame` (no write-back) = **exactly 1 submit**. Do NOT use the separate cleanFree copy batch shown in Step 6.
> - IGNORE Step 5's placeholder body entirely; implement the final `BeginFrame` from Step 6, but read `reusable` from `m_stageLists->Mapped()` and `cleanFree` from `m_stageCleanFree->Mapped()` (both filled in Batch 1), removing the standalone cleanFree copy batch and its `++m_stats.gpuSubmits`.
> - Skip Step 7's "Reconciliation" and Step 8; the corrections above already fold cleanFree into Batch 1. The test expectation is `gpuSubmits == 1` for a bare BeginFrame.

**Files:**
- Modify: `src/Engine/Spatial/DirectionalTSDF.h`
- Modify: `src/Engine/Spatial/DirectionalTSDF.cpp`
- Modify: `test/test_directionalTSDF.cpp`

**Interfaces:**
- Produces: `Stats.gpuSubmits` (queue submissions counted this frame, reset at BeginFrame start); a `BeginFrame` that performs fill+classify+all list readbacks in ONE batch, plus at most one write-back batch. `DebugDownloadIndexGrid`/`DebugDownloadGroupVoxels`/`DebugQueryPoolIndex` unchanged (still use `Engine::Core::Buffer`, not counted — they are test-only, called outside the counted window).

- [ ] **Step 1: Write the failing test**

Append to `test/test_directionalTSDF.cpp`:

```cpp
TEST(DirectionalTSDFPhase5Test, BareBeginFrameUsesFewSubmits) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 256);

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    // fill+classify+readback fold into one batch; no dirty groups → no write-back batch.
    EXPECT_EQ(tsdf.LastFrameStats().gpuSubmits, 1u);
}
```

- [ ] **Step 2: Run to verify it fails** (`gpuSubmits` not a member yet → compile error, or 0).

- [ ] **Step 3: Add `gpuSubmits` to `Stats` and includes**

In `src/Engine/Spatial/DirectionalTSDF.h`, add to `Stats` (after `mergeMs`):

```cpp
            uint32_t gpuSubmits = 0; // queue submissions this frame (Phase 5 batching metric)
```

Add persistent staging members. In the private section, after `m_candidateCounter`, add:

```cpp
        // Persistent host-visible staging (Phase 5): allocated once in Build, reused every
        // frame to back batched copies instead of per-call transient staging.
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageCounts;      // DST, 3*u32
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageLists;       // DST, poolCapacity*u32 (reusable|cleanFree|writeBack read one at a time)
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageGroups;      // SRC|DST, kStageGroupCap*groupBytes
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageMeta;        // SRC|DST, poolCapacity*sizeof(ActiveGroupMeta)
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stagePoints;      // SRC, maxPoints*6*f32
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageSlotList;    // SRC, poolCapacity*u32
        std::unique_ptr<Engine::Compute::StagingBuffer> m_stageCandidates;  // DST, maxCandidates*sizeof(DirectionalCandidate)
```

At the top include block add:

```cpp
#include "Engine/Compute/CommandBatch.h"
#include "Engine/Compute/StagingBuffer.h"
```

- [ ] **Step 4: In `Build`, allocate the persistent staging buffers**

In `src/Engine/Spatial/DirectionalTSDF.cpp`, add a group-cap constant to the anonymous namespace next to `kGroupBytes`:

```cpp
        constexpr uint32_t kStageGroupCap = 4096; // max groups staged per H2D/D2H batch (chunked if exceeded)
```

In `Build()`, after `m_candidateCounter->Allocate(sizeof(uint32_t));`, add:

```cpp
        using Engine::Compute::StagingBuffer;
        const VkBufferUsageFlags kSrc = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        const VkBufferUsageFlags kDst = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const VkBufferUsageFlags kSrcDst = kSrc | kDst;
        m_stageCounts = std::make_unique<StagingBuffer>(ctx, 3u * sizeof(uint32_t), kDst);
        m_stageLists = std::make_unique<StagingBuffer>(ctx, poolCapacity * sizeof(uint32_t), kDst);
        m_stageGroups = std::make_unique<StagingBuffer>(ctx, VkDeviceSize(kStageGroupCap) * kGroupBytes, kSrcDst);
        m_stageMeta = std::make_unique<StagingBuffer>(ctx, poolCapacity * sizeof(ActiveGroupMeta), kSrcDst);
        m_stagePoints = std::make_unique<StagingBuffer>(ctx, VkDeviceSize(maxPoints) * 6u * sizeof(float), kSrc);
        m_stageSlotList = std::make_unique<StagingBuffer>(ctx, poolCapacity * sizeof(uint32_t), kSrc);
        m_stageCandidates = std::make_unique<StagingBuffer>(ctx, VkDeviceSize(maxCandidates) * sizeof(DirectionalCandidate), kDst);
```

- [ ] **Step 5: Rewrite `BeginFrame` to batch fill+classify+readback and (if needed) write-back**

Replace the entire `BeginFrame` body with:

```cpp
    void DirectionalTSDF::BeginFrame(const Eigen::Vector3f &aabbCenterHint) {
        if (!m_ctx)
            throw std::runtime_error("DirectionalTSDF: Build() must be called first");

        m_localBase = quantizeLocalBase(aabbCenterHint);
        m_stats = {};
        m_requiredThisFrame.clear();

        // Batch 1: reset indexGrid + counts, classify, then copy counts + all three lists
        // back — all in a single submit.
        {
            Engine::Compute::CommandBatch batch(*m_ctx);
            batch.FillBuffer(m_indexGrid->Handle(), 0, VK_WHOLE_SIZE, kInvalidPoolIndex);
            batch.FillBuffer(m_countsBuffer->Handle(), 0, VK_WHOLE_SIZE, 0u);
            batch.Barrier();
            ClassifyPC cpc{m_poolCapacity, m_localBase.x(), m_localBase.y(), m_localBase.z()};
            batch.DispatchElements(*m_classifyKernel, m_poolCapacity);
            batch.Barrier();
            (void) cpc; // classify push constants set below via Args before Dispatch
            // NOTE: push constants must be set on the pipeline before recording; see Step 6.
            batch.Submit();
            ++m_stats.gpuSubmits;
        }
        // (Args wiring corrected in Step 6 — this placeholder body is replaced there.)
    }
```

This body is intentionally incomplete — it is finalized in Step 6 (push-constant ordering matters). Do not build against this Step-5 version.

- [ ] **Step 6: Finalize `BeginFrame` with correct push-constant ordering and readback**

`ComputePipeline` records the push constants captured by the last `.Args(...)` call at `RecordDispatch` time, so call `.Args(...)` on the kernel BEFORE `batch.DispatchElements(kernel, ...)`. Replace the whole `BeginFrame` body with the final version:

```cpp
    void DirectionalTSDF::BeginFrame(const Eigen::Vector3f &aabbCenterHint) {
        if (!m_ctx)
            throw std::runtime_error("DirectionalTSDF: Build() must be called first");

        m_localBase = quantizeLocalBase(aabbCenterHint);
        m_stats = {};
        m_requiredThisFrame.clear();

        // Batch 1: reset indexGrid + counts → classify → copy counts + lists back. One submit.
        {
            ClassifyPC cpc{m_poolCapacity, m_localBase.x(), m_localBase.y(), m_localBase.z()};
            m_classifyKernel->Args(cpc);

            std::vector<VkBufferCopy> countsRegion(1);
            countsRegion[0] = {0, 0, 3u * sizeof(uint32_t)};
            std::vector<VkBufferCopy> listRegion(1);
            listRegion[0] = {0, 0, VkDeviceSize(m_poolCapacity) * sizeof(uint32_t)};

            Engine::Compute::CommandBatch batch(*m_ctx);
            batch.FillBuffer(m_indexGrid->Handle(), 0, VK_WHOLE_SIZE, kInvalidPoolIndex);
            batch.FillBuffer(m_countsBuffer->Handle(), 0, VK_WHOLE_SIZE, 0u);
            batch.Barrier();
            batch.DispatchElements(*m_classifyKernel, m_poolCapacity);
            batch.Barrier();
            batch.CopyBuffer(m_countsBuffer->Handle(), m_stageCounts->Handle(), countsRegion);
            batch.CopyBuffer(m_reusableList->Handle(), m_stageLists->Handle(), listRegion);
            batch.Submit();
            ++m_stats.gpuSubmits;
        }

        uint32_t counts[3];
        std::memcpy(counts, m_stageCounts->Mapped(), sizeof(counts));
        m_lastCounts = {counts[0], counts[1], counts[2]};

        std::vector<uint32_t> reusable(counts[0]);
        if (counts[0] > 0)
            std::memcpy(reusable.data(), m_stageLists->Mapped(), counts[0] * sizeof(uint32_t));

        // cleanFree list: copy it in its own tiny batch (it shares m_stageLists, so read
        // reusable first, above, before overwriting the staging buffer).
        {
            std::vector<VkBufferCopy> listRegion(1);
            listRegion[0] = {0, 0, VkDeviceSize(m_poolCapacity) * sizeof(uint32_t)};
            Engine::Compute::CommandBatch batch(*m_ctx);
            batch.CopyBuffer(m_cleanFreeList->Handle(), m_stageLists->Handle(), listRegion);
            batch.Submit();
            ++m_stats.gpuSubmits;
        }
        std::vector<uint32_t> cleanFree(counts[1]);
        if (counts[1] > 0)
            std::memcpy(cleanFree.data(), m_stageLists->Mapped(), counts[1] * sizeof(uint32_t));
        m_freeSlots.assign(cleanFree.begin(), cleanFree.end());

        // Write-back (design doc §12): dirty groups that left the window → host store,
        // clear their meta, free their slots. Processed in chunks of kStageGroupCap.
        m_stats.writeBackCount = counts[2];
        if (counts[2] > 0) {
            std::vector<uint32_t> writeBack(counts[2]);
            {
                std::vector<VkBufferCopy> listRegion(1);
                listRegion[0] = {0, 0, VkDeviceSize(m_poolCapacity) * sizeof(uint32_t)};
                Engine::Compute::CommandBatch batch(*m_ctx);
                batch.CopyBuffer(m_writeBackList->Handle(), m_stageLists->Handle(), listRegion);
                batch.Submit();
                ++m_stats.gpuSubmits;
            }
            std::memcpy(writeBack.data(), m_stageLists->Mapped(), counts[2] * sizeof(uint32_t));

            const uint32_t groupBytes = kVoxelsPerGroup * uint32_t(sizeof(GpuTsdfVoxel));
            std::vector<ActiveGroupMeta> clearMeta(kStageGroupCap); // zero = invalid/Free

            for (uint32_t base = 0; base < counts[2]; base += kStageGroupCap) {
                const uint32_t chunk = std::min(kStageGroupCap, counts[2] - base);

                // D2H: pool → staging (voxels), and upload zeroed meta into the staging.
                std::vector<VkBufferCopy> voxRegions(chunk), metaRegions(chunk);
                for (uint32_t i = 0; i < chunk; ++i) {
                    const uint32_t slot = writeBack[base + i];
                    voxRegions[i] = {VkDeviceSize(slot) * groupBytes, VkDeviceSize(i) * groupBytes, groupBytes};
                    metaRegions[i] = {VkDeviceSize(i) * sizeof(ActiveGroupMeta),
                                      VkDeviceSize(slot) * sizeof(ActiveGroupMeta), sizeof(ActiveGroupMeta)};
                }
                std::memcpy(m_stageMeta->Mapped(), clearMeta.data(), chunk * sizeof(ActiveGroupMeta));

                Engine::Compute::CommandBatch batch(*m_ctx);
                batch.CopyBuffer(m_poolVoxels->Handle(), m_stageGroups->Handle(), voxRegions);
                batch.CopyBuffer(m_stageMeta->Handle(), m_metaBuffer->Handle(), metaRegions);
                batch.Submit();
                ++m_stats.gpuSubmits;

                const auto *raw = static_cast<const GpuTsdfVoxel *>(m_stageGroups->Mapped());
                for (uint32_t i = 0; i < chunk; ++i) {
                    const uint32_t slot = writeBack[base + i];
                    DirectionalHostStore::Group group{};
                    for (uint32_t v = 0; v < kVoxelsPerGroup; ++v) {
                        const GpuTsdfVoxel &g = raw[size_t(i) * kVoxelsPerGroup + v];
                        group[v].weight = float(g.sumW) / float(kTsdfFixedScale);
                        group[v].value = g.sumW > 0 ? float(double(g.sumDW) / double(g.sumW)) : 0.0f;
                    }
                    m_hostStore.Put(m_slotKeys[slot], group);
                    m_freeSlots.push_back(slot);
                }
                m_stats.d2hBytes += chunk * groupBytes;
            }
        }

        // Rebuild the CPU resident mirror from the reusable set (indexGrid registration for
        // reusable slots is folded into the EnsureResident batch — invariant #6 still holds
        // because missing detection uses this CPU mirror, not the GPU indexGrid).
        m_residentIndex.clear();
        for (uint32_t slot : reusable)
            m_residentIndex.emplace(m_slotKeys[slot], slot);
        m_stats.residentCount = uint32_t(m_residentIndex.size());
        m_reusableSlots.assign(reusable.begin(), reusable.end());
    }
```

Add `std::vector<uint32_t> m_reusableSlots;` as a private member in the header (carries the reusable slots into `EnsureResident` so its register dispatch can register reusable + missing together — see Task 4). Also add `#include <cstring>` to the .cpp if not present.

**Design note (invariant #6):** In Phases 2–4 `BeginFrame` re-registered reusable slots into the GPU indexGrid immediately. Missing-group detection, however, reads the CPU `m_residentIndex` mirror (rebuilt above), never the GPU indexGrid — so deferring reusable indexGrid registration to the EnsureResident batch changes nothing about which groups are detected missing. The GPU indexGrid only needs to be correct before the integrate/extract kernels read it, and those run inside/after the EnsureResident batch. Invariant #6 (missing判정 after reusable재등록) holds at the CPU-mirror level.

- [ ] **Step 7: Build and run the Phase 5 BeginFrame test + full Directional regression**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="DirectionalTSDFPhase5Test.BareBeginFrameUsesFewSubmits"
./build/test/vkspatial_tests --gtest_filter="Directional*"
```
Expected: the Phase 5 test PASS (`gpuSubmits == 1` for a clean BeginFrame — Batch 1 only; the cleanFree copy batch is counted too, so this must be reconciled).

**Reconciliation:** the code above submits Batch 1 (fill+classify+counts+reusable copy) AND a separate cleanFree copy batch → that is 2 submits for a bare BeginFrame, not 1. Fold the cleanFree copy into Batch 1 by copying counts, reusable, AND cleanFree into three DISTINCT staging buffers in one batch. Change `m_stageLists` usage: give cleanFree its own staging buffer `m_stageCleanFree` (add a second `StagingBuffer` member sized `poolCapacity*u32`, allocated in Build), and in Batch 1 add `batch.CopyBuffer(m_cleanFreeList->Handle(), m_stageCleanFree->Handle(), listRegion);`. Then read reusable from `m_stageLists` and cleanFree from `m_stageCleanFree` without a second submit. Update the test expectation to `gpuSubmits == 1`. Re-run.

- [ ] **Step 8: Apply the reconciliation, rebuild, confirm `gpuSubmits == 1` and all 27 Directional tests pass**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="Directional*" 2>&1 | tail -4
```
Expected: 28 tests (27 + BareBeginFrame) PASS.

- [ ] **Step 9: Commit**

```bash
git add src/Engine/Spatial/DirectionalTSDF.h src/Engine/Spatial/DirectionalTSDF.cpp test/test_directionalTSDF.cpp
git commit -m "Batch DirectionalTSDF BeginFrame (classify + readback + write-back) via CommandBatch"
```

---

### Task 4: Batched `EnsureResident` + integrate + extract; steady-state submit assertion

> **IMPLEMENTATION CORRECTIONS (supersede the messy sub-steps below):**
> - IGNORE Step 3's broken `EnsureResident` (with the half-baked `m_frameBatch*` hooks) entirely. Implement ONLY the final design from Step 4/5: a private `recordResidency(required, batch)` that stages + records upload copies and the combined (reusable+missing) register dispatch WITHOUT submitting, a thin `EnsureResident` wrapper that submits it standalone, and an `Integrate` that folds `recordResidency` into Batch 2 with integration. There are NO `m_frameBatchActive`/`m_frameBatchRegCount`/`m_frameBatch` members — do not add them.
> - `maxCandidates` default becomes `1u << 16` (header declaration only; the .cpp definition carries no default). This keeps the full-candidate readback in Batch 3 at ~2.6MB.
> - Timing: since residency+integrate share one batch, report `ensureResidentMs = 0.0f` and `integrateMs = ms(t1, t3)` (combined). Do not try to split them.

**Files:**
- Modify: `src/Engine/Spatial/DirectionalTSDF.h`
- Modify: `src/Engine/Spatial/DirectionalTSDF.cpp`
- Modify: `test/test_directionalTSDF.cpp`

**Interfaces:**
- Consumes: `m_reusableSlots` from Task 3, all staging buffers, `CommandBatch`.
- Produces: `EnsureResident` that stages voxel/meta/slot data into persistent staging and records upload copies + combined register dispatch in ONE batch (Batch 2, together with integration when driven by `Integrate`); `Integrate` that runs Batch 2 (upload+register+integrate) and Batch 3 (extract+readback). Steady-state `Integrate` → `gpuSubmits == 3`.

- [ ] **Step 1: Write the failing steady-state submit test**

Append to `test/test_directionalTSDF.cpp`:

```cpp
TEST(DirectionalTSDFPhase5Test, SteadyStateIntegrateUsesThreeSubmits) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 4096);

    std::vector<Eigen::Vector3f> points, normals;
    makePlane(0.0f, 0.4f, 0.05f, Eigen::Vector3f(1, 0, 0), points, normals);

    // Frame 1 (cold): uploads everything.
    tsdf.Integrate(points, normals, Eigen::Vector3f(2, 0, 0), Eigen::Vector3f::Zero());
    // Frame 2 (identical window): no missing, no write-back → the minimal batch count.
    tsdf.Integrate(points, normals, Eigen::Vector3f(2, 0, 0), Eigen::Vector3f::Zero());

    // Batch 1 (BeginFrame classify+readback) + Batch 2 (register+integrate) +
    // Batch 3 (extract+readback) = 3 submits, no write-back this frame.
    EXPECT_EQ(tsdf.LastFrameStats().gpuSubmits, 3u);
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 0u);
}
```

- [ ] **Step 2: Run to verify it fails** (current `EnsureResident`/`Integrate` still use `Buffer::Upload`/`Dispatch` → many uncounted submits; `gpuSubmits` will be 1 from BeginFrame only, so `== 3` fails).

- [ ] **Step 3: Rewrite `EnsureResident` to stage + record into a member batch**

The register dispatch must cover reusable + missing slots together (one dispatch, avoids re-binding the register pipeline twice in one batch). Replace the entire `EnsureResident` body with:

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
                throw std::runtime_error("DirectionalTSDF: required group is outside the local window");

            m_requiredThisFrame.insert(key);
            if (m_residentIndex.find(key) != m_residentIndex.end())
                continue;
            if (m_freeSlots.empty())
                throw std::runtime_error("DirectionalTSDF: active pool exhausted");

            const uint32_t slot = m_freeSlots.back();
            m_freeSlots.pop_back();
            m_residentIndex.emplace(key, slot);
            m_slotKeys[slot] = key;
            pending.push_back({key, slot});
        }
        m_stats.residentCount = uint32_t(m_residentIndex.size());

        // Encode missing groups' voxels + meta into persistent staging (CPU side).
        // Chunk by kStageGroupCap so the group staging buffer is bounded.
        const uint32_t groupBytes = kVoxelsPerGroup * uint32_t(sizeof(GpuTsdfVoxel));
        auto *voxStage = static_cast<GpuTsdfVoxel *>(m_stageGroups->Mapped());
        auto *metaStage = static_cast<ActiveGroupMeta *>(m_stageMeta->Mapped());

        // Build the combined register list (reusable slots + missing slots) in slot staging.
        auto *slotStage = static_cast<uint32_t *>(m_stageSlotList->Mapped());
        uint32_t regCount = 0;
        for (uint32_t s : m_reusableSlots) slotStage[regCount++] = s;

        // Open the frame's compute batch. Missing uploads are chunked; if there is more
        // than one chunk each chunk is its own submit (rare — steady state has ≤ a handful).
        Engine::Compute::CommandBatch batch(*m_ctx);
        bool anyRecorded = false;

        for (uint32_t base = 0; base < pending.size(); base += kStageGroupCap) {
            const uint32_t chunk = std::min<uint32_t>(kStageGroupCap, uint32_t(pending.size()) - base);
            std::vector<VkBufferCopy> voxRegions(chunk), metaRegions(chunk);
            for (uint32_t i = 0; i < chunk; ++i) {
                const auto &pk = pending[base + i];
                const auto &group = m_hostStore.GetOrCreate(pk.key);
                for (uint32_t v = 0; v < kVoxelsPerGroup; ++v) {
                    const HostTsdfVoxel &h = group[v];
                    GpuTsdfVoxel &g = voxStage[size_t(i) * kVoxelsPerGroup + v];
                    g.sumW = uint32_t(std::lround(double(h.weight) * kTsdfFixedScale));
                    g.sumDW = int32_t(std::lround(double(h.value) * double(h.weight) * kTsdfFixedScale));
                }
                metaStage[i] = ActiveGroupMeta{pk.key.gx, pk.key.gy, pk.key.gz,
                                               PackMeta(pk.key.direction, SlotState::ResidentClean, false, true)};
                voxRegions[i] = {VkDeviceSize(i) * groupBytes, VkDeviceSize(pk.slot) * groupBytes, groupBytes};
                metaRegions[i] = {VkDeviceSize(i) * sizeof(ActiveGroupMeta),
                                  VkDeviceSize(pk.slot) * sizeof(ActiveGroupMeta), sizeof(ActiveGroupMeta)};
                slotStage[regCount++] = pk.slot;
            }
            // Each chunk after the first needs its own submit (staging is reused).
            if (base + kStageGroupCap < pending.size()) {
                Engine::Compute::CommandBatch chunkBatch(*m_ctx);
                chunkBatch.CopyBuffer(m_stageGroups->Handle(), m_poolVoxels->Handle(), voxRegions);
                chunkBatch.CopyBuffer(m_stageMeta->Handle(), m_metaBuffer->Handle(), metaRegions);
                chunkBatch.Submit();
                ++m_stats.gpuSubmits;
            } else {
                batch.CopyBuffer(m_stageGroups->Handle(), m_poolVoxels->Handle(), voxRegions);
                batch.CopyBuffer(m_stageMeta->Handle(), m_metaBuffer->Handle(), metaRegions);
                anyRecorded = true;
            }
            m_stats.h2dBytes += chunk * groupBytes;
        }

        // Register combined slot list → indexGrid (reusable + last missing chunk).
        std::vector<VkBufferCopy> slotRegion(1);
        slotRegion[0] = {0, 0, VkDeviceSize(regCount) * sizeof(uint32_t)};
        if (regCount > 0) {
            batch.CopyBuffer(m_stageSlotList->Handle(), m_slotListBuffer->Handle(), slotRegion);
            batch.Barrier();
            RegisterPC rpc{regCount, m_localBase.x(), m_localBase.y(), m_localBase.z()};
            m_registerKernel->Args(rpc).Bind(0, *m_slotListBuffer);
            batch.DispatchElements(*m_registerKernel, regCount);
            anyRecorded = true;
        }

        // Stash the open batch for Integrate to append integration into, or submit now if
        // EnsureResident was called standalone (tests). We finalize in Integrate via
        // m_frameBatch; for standalone use, submit here.
        if (m_frameBatchActive) {
            m_frameBatchRegCount = regCount;
            // Ownership: move the recorded work — but CommandBatch is single-use and not
            // movable here, so we instead re-record in Integrate. Standalone path submits.
        }
        if (anyRecorded) {
            batch.Submit();
            ++m_stats.gpuSubmits;
        }
        updateOverlapRatio();
    }
```

**Problem:** the above keeps register+upload and integrate in SEPARATE submits (EnsureResident submits its batch, then Integrate would submit integrate separately) → 4 submits, not 3, and the `m_frameBatch*` hooks are half-baked. Fix in Step 4 by giving `Integrate` full control and making `EnsureResident` a thin wrapper.

- [ ] **Step 4: Refactor so `Integrate` owns Batch 2 (upload+register+integrate); make `EnsureResident` delegate**

Replace the just-written `EnsureResident` and the existing `Integrate` with a shared private helper `recordResidency(...)` that only STAGES data and RECORDS into a caller-provided `CommandBatch` (no submit), plus public wrappers.

Add to the header private section:

```cpp
        // Stages missing groups into persistent staging and records upload copies +
        // combined (reusable+missing) register dispatch into `batch`. Does NOT submit.
        // Returns the number of slots registered.
        uint32_t recordResidency(const std::vector<DirectionalGroupKey> &required,
                                 Engine::Compute::CommandBatch &batch);
```

Implement `recordResidency` (replaces the slot-allocation + staging + register-record logic; no chunk-submit — for simplicity Phase 5 assumes required ≤ kStageGroupCap per frame, which holds for the 40mm window: max ≈ 20K groups but our demo/tests are far under 4096; if exceeded, throw a clear error to be addressed by a future chunking pass):

```cpp
    uint32_t DirectionalTSDF::recordResidency(const std::vector<DirectionalGroupKey> &required,
                                              Engine::Compute::CommandBatch &batch) {
        struct Pending { DirectionalGroupKey key; uint32_t slot; };
        std::vector<Pending> pending;
        pending.reserve(required.size());

        for (const auto &key : required) {
            const int lx = key.gx - m_localBase.x();
            const int ly = key.gy - m_localBase.y();
            const int lz = key.gz - m_localBase.z();
            if (lx < 0 || ly < 0 || lz < 0 ||
                lx >= int(kLocalGroupGrid) || ly >= int(kLocalGroupGrid) ||
                lz >= int(kLocalGroupGrid))
                throw std::runtime_error("DirectionalTSDF: required group is outside the local window");
            m_requiredThisFrame.insert(key);
            if (m_residentIndex.find(key) != m_residentIndex.end())
                continue;
            if (m_freeSlots.empty())
                throw std::runtime_error("DirectionalTSDF: active pool exhausted");
            const uint32_t slot = m_freeSlots.back();
            m_freeSlots.pop_back();
            m_residentIndex.emplace(key, slot);
            m_slotKeys[slot] = key;
            pending.push_back({key, slot});
        }
        m_stats.residentCount = uint32_t(m_residentIndex.size());

        if (pending.size() > kStageGroupCap)
            throw std::runtime_error("DirectionalTSDF: missing groups exceed staging cap (chunking TODO)");

        const uint32_t groupBytes = kVoxelsPerGroup * uint32_t(sizeof(GpuTsdfVoxel));
        auto *voxStage = static_cast<GpuTsdfVoxel *>(m_stageGroups->Mapped());
        auto *metaStage = static_cast<ActiveGroupMeta *>(m_stageMeta->Mapped());
        auto *slotStage = static_cast<uint32_t *>(m_stageSlotList->Mapped());

        uint32_t regCount = 0;
        for (uint32_t s : m_reusableSlots) slotStage[regCount++] = s;

        std::vector<VkBufferCopy> voxRegions(pending.size()), metaRegions(pending.size());
        for (size_t i = 0; i < pending.size(); ++i) {
            const auto &pk = pending[i];
            const auto &group = m_hostStore.GetOrCreate(pk.key);
            for (uint32_t v = 0; v < kVoxelsPerGroup; ++v) {
                const HostTsdfVoxel &h = group[v];
                GpuTsdfVoxel &g = voxStage[i * kVoxelsPerGroup + v];
                g.sumW = uint32_t(std::lround(double(h.weight) * kTsdfFixedScale));
                g.sumDW = int32_t(std::lround(double(h.value) * double(h.weight) * kTsdfFixedScale));
            }
            metaStage[i] = ActiveGroupMeta{pk.key.gx, pk.key.gy, pk.key.gz,
                                           PackMeta(pk.key.direction, SlotState::ResidentClean, false, true)};
            voxRegions[i] = {VkDeviceSize(i) * groupBytes, VkDeviceSize(pk.slot) * groupBytes, groupBytes};
            metaRegions[i] = {VkDeviceSize(i) * sizeof(ActiveGroupMeta),
                              VkDeviceSize(pk.slot) * sizeof(ActiveGroupMeta), sizeof(ActiveGroupMeta)};
            slotStage[regCount++] = pk.slot;
        }
        if (!pending.empty()) {
            batch.CopyBuffer(m_stageGroups->Handle(), m_poolVoxels->Handle(), voxRegions);
            batch.CopyBuffer(m_stageMeta->Handle(), m_metaBuffer->Handle(), metaRegions);
            m_stats.h2dBytes += uint32_t(pending.size()) * groupBytes;
        }
        if (regCount > 0) {
            std::vector<VkBufferCopy> slotRegion(1);
            slotRegion[0] = {0, 0, VkDeviceSize(regCount) * sizeof(uint32_t)};
            batch.CopyBuffer(m_stageSlotList->Handle(), m_slotListBuffer->Handle(), slotRegion);
            batch.Barrier();
            RegisterPC rpc{regCount, m_localBase.x(), m_localBase.y(), m_localBase.z()};
            m_registerKernel->Args(rpc).Bind(0, *m_slotListBuffer);
            batch.DispatchElements(*m_registerKernel, regCount);
        }
        return regCount;
    }

    void DirectionalTSDF::EnsureResident(const std::vector<DirectionalGroupKey> &required) {
        if (!m_ctx)
            throw std::runtime_error("DirectionalTSDF: Build() must be called first");
        Engine::Compute::CommandBatch batch(*m_ctx);
        const uint32_t reg = recordResidency(required, batch);
        if (reg > 0) {
            batch.Submit();
            ++m_stats.gpuSubmits;
        }
        updateOverlapRatio();
    }
```

- [ ] **Step 5: Rewrite `Integrate` to run Batch 2 (residency+integrate) and Batch 3 (extract+readback)**

Replace the `Integrate` body's residency+integrate+extract sections. Keep the write-set/halo computation (CPU) as-is; change the GPU-driving part:

```cpp
    void DirectionalTSDF::Integrate(const std::vector<Eigen::Vector3f> &points,
                                    const std::vector<Eigen::Vector3f> &normals,
                                    const Eigen::Vector3f &cameraPos,
                                    const Eigen::Vector3f &aabbCenterHint) {
        if (points.size() != normals.size())
            throw std::runtime_error("DirectionalTSDF: points/normals size mismatch");
        if (points.empty()) return;

        using Clock = std::chrono::steady_clock;
        auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<float, std::milli>(b - a).count();
        };

        const auto t0 = Clock::now();
        BeginFrame(aabbCenterHint);
        const auto t1 = Clock::now();

        const uint32_t N = std::min(uint32_t(points.size()), m_maxPoints);

        // --- write-set + halo (CPU, unchanged) ---
        std::unordered_set<DirectionalGroupKey, DirectionalGroupKeyHash> writeSet;
        for (uint32_t i = 0; i < N; ++i) {
            Eigen::Vector3f diff = points[i] - cameraPos;
            float depth = diff.norm();
            if (depth < 1e-6f) continue;
            Eigen::Vector3f dir = diff / depth;
            const uint8_t d = dominantAxisOf(normals[i]);
            const float band = m_truncation + m_voxelSize;
            Eigen::Vector3f a = points[i] - dir * band;
            Eigen::Vector3f b = points[i] + dir * band;
            Eigen::Vector3i vmin, vmax;
            for (int c = 0; c < 3; ++c) {
                const float lo = std::min(a[c], b[c]);
                const float hi = std::max(a[c], b[c]);
                vmin[c] = int(std::floor(lo / m_voxelSize)) - 1;
                vmax[c] = int(std::floor(hi / m_voxelSize)) + 1;
            }
            for (int gz = vmin.z() >> 3; gz <= (vmax.z() >> 3); ++gz)
                for (int gy = vmin.y() >> 3; gy <= (vmax.y() >> 3); ++gy)
                    for (int gx = vmin.x() >> 3; gx <= (vmax.x() >> 3); ++gx)
                        writeSet.insert({gx, gy, gz, d});
        }
        std::vector<DirectionalGroupKey> required;
        {
            std::unordered_set<DirectionalGroupKey, DirectionalGroupKeyHash> requiredSet;
            for (const auto &k : writeSet)
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                            requiredSet.insert({k.gx + dx, k.gy + dy, k.gz + dz, k.direction});
            required.assign(requiredSet.begin(), requiredSet.end());
        }

        // Stage point samples (CPU → persistent staging).
        auto *ptStage = static_cast<float *>(m_stagePoints->Mapped());
        for (uint32_t i = 0; i < N; ++i) {
            ptStage[size_t(i) * 6 + 0] = points[i].x();
            ptStage[size_t(i) * 6 + 1] = points[i].y();
            ptStage[size_t(i) * 6 + 2] = points[i].z();
            ptStage[size_t(i) * 6 + 3] = normals[i].x();
            ptStage[size_t(i) * 6 + 4] = normals[i].y();
            ptStage[size_t(i) * 6 + 5] = normals[i].z();
        }

        // --- Batch 2: residency (upload + register) + integrate, one submit ---
        {
            Engine::Compute::CommandBatch batch(*m_ctx);
            recordResidency(required, batch);
            std::vector<VkBufferCopy> ptRegion(1);
            ptRegion[0] = {0, 0, VkDeviceSize(N) * 6u * sizeof(float)};
            batch.CopyBuffer(m_stagePoints->Handle(), m_pointBuffer->Handle(), ptRegion);
            batch.Barrier();
            IntegratePC ipc{N, m_voxelSize, m_truncation, m_localBase.x(), m_localBase.y(),
                            m_localBase.z(), cameraPos.x(), cameraPos.y(), cameraPos.z()};
            m_integrateKernel->Args(ipc);
            batch.DispatchElements(*m_integrateKernel, N);
            batch.Submit();
            ++m_stats.gpuSubmits;
        }
        const auto t3 = Clock::now();

        // --- recompute mask (spatial) + Batch 3: extract + candidate readback ---
        std::unordered_set<uint64_t> recomputeSpatial;
        for (const auto &k : writeSet)
            recomputeSpatial.insert(spatialKey(k.gx, k.gy, k.gz));
        std::vector<uint32_t> groupSlots;
        for (const auto &entry : m_residentIndex)
            if (recomputeSpatial.count(spatialKey(entry.first.gx, entry.first.gy, entry.first.gz)) > 0)
                groupSlots.push_back(entry.second);

        uint32_t candidateCount = 0;
        if (!groupSlots.empty()) {
            auto *slotStage = static_cast<uint32_t *>(m_stageSlotList->Mapped());
            std::memcpy(slotStage, groupSlots.data(), groupSlots.size() * sizeof(uint32_t));
            std::vector<VkBufferCopy> slotRegion(1);
            slotRegion[0] = {0, 0, VkDeviceSize(groupSlots.size()) * sizeof(uint32_t)};
            std::vector<VkBufferCopy> candCopy(1);
            candCopy[0] = {0, 0, VkDeviceSize(m_maxCandidates) * sizeof(DirectionalCandidate)};
            std::vector<VkBufferCopy> cntCopy(1);
            cntCopy[0] = {0, 0, sizeof(uint32_t)};

            ExtractPC epc{uint32_t(groupSlots.size()), m_voxelSize, m_maxCandidates,
                          m_localBase.x(), m_localBase.y(), m_localBase.z()};
            m_extractKernel->Args(epc);

            Engine::Compute::CommandBatch batch(*m_ctx);
            batch.CopyBuffer(m_stageSlotList->Handle(), m_slotListBuffer->Handle(), slotRegion);
            batch.FillBuffer(m_candidateCounter->Handle(), 0, VK_WHOLE_SIZE, 0u);
            batch.Barrier();
            batch.DispatchElements(*m_extractKernel, uint32_t(groupSlots.size()) * kVoxelsPerGroup);
            batch.Barrier();
            batch.CopyBuffer(m_candidateCounter->Handle(), m_stageCounts->Handle(), cntCopy);
            batch.CopyBuffer(m_candidateBuffer->Handle(), m_stageCandidates->Handle(), candCopy);
            batch.Submit();
            ++m_stats.gpuSubmits;

            std::memcpy(&candidateCount, m_stageCounts->Mapped(), sizeof(uint32_t));
            candidateCount = std::min(candidateCount, m_maxCandidates);
        }
        std::vector<DirectionalCandidate> candidates(candidateCount);
        if (candidateCount > 0)
            std::memcpy(candidates.data(), m_stageCandidates->Mapped(),
                        candidateCount * sizeof(DirectionalCandidate));
        const auto t4 = Clock::now();

        // --- merge + old-point replacement (CPU, unchanged) ---
        std::vector<ExtractedPoint> fresh = mergeCandidates(candidates);
        m_pointCloud.erase(
                std::remove_if(m_pointCloud.begin(), m_pointCloud.end(),
                               [&](const ExtractedPoint &pt) {
                                   return recomputeSpatial.count(spatialKey(pt.ownerGx, pt.ownerGy, pt.ownerGz)) > 0;
                               }),
                m_pointCloud.end());
        m_pointCloud.insert(m_pointCloud.end(), fresh.begin(), fresh.end());
        const auto t5 = Clock::now();

        m_stats.beginFrameMs = ms(t0, t1);
        m_stats.ensureResidentMs = ms(t1, t3); // residency now folded into the integrate batch
        m_stats.integrateMs = ms(t1, t3);
        m_stats.extractMs = ms(t3, t4);
        m_stats.mergeMs = ms(t4, t5);
    }
```

Note the extract phase copies the FULL candidate buffer back (`m_maxCandidates`) in one submit to avoid a second readback submit. Lower the `maxCandidates` default from `1u << 18` to `1u << 16` in the `Build` signature (header + .cpp) so this readback is ~2.6MB, not ~10MB. Confirm the header default and .cpp default both read `uint32_t maxCandidates = 1u << 16`.

- [ ] **Step 6: Build and run the steady-state submit test + full Directional regression**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="DirectionalTSDFPhase5Test.*"
./build/test/vkspatial_tests --gtest_filter="Directional*" 2>&1 | tail -4
```
Expected: `SteadyStateIntegrateUsesThreeSubmits` PASS (`gpuSubmits == 3`), all Directional tests PASS.

- [ ] **Step 7: Full-suite regression**

```bash
./build/test/vkspatial_tests 2>&1 | tail -6
```
Expected: only `WideBVHTest.RadiusMatchesCpuReference` fails.

- [ ] **Step 8: Commit**

```bash
git add src/Engine/Spatial/DirectionalTSDF.h src/Engine/Spatial/DirectionalTSDF.cpp test/test_directionalTSDF.cpp
git commit -m "Batch DirectionalTSDF residency+integrate+extract to 3 submits/frame (Phase 5)"
```

---

### Task 5: Demo `gpuSubmits`/`d2hBytes` in CSV + before/after timing observation

**Files:**
- Modify: `example2/directional_tsdf_demo.cpp`

**Interfaces:**
- Consumes: `Stats.gpuSubmits`, `Stats.d2hBytes` (already in Stats).

- [ ] **Step 1: Add the columns to the CSV header and rows**

In `example2/directional_tsdf_demo.cpp`, change the header line:

```cpp
    csv << "frame,points,resident,missing,overlapPct,h2dKB,d2hKB,gpuSubmits,writeBack,"
           "beginMs,ensureMs,integrateMs,extractMs,mergeMs,cloudPoints\n";
```

And the per-frame row (insert `d2hKB` and `gpuSubmits` after `h2dKB`):

```cpp
        csv << f << ',' << points.size() << ',' << st.residentCount << ','
            << st.missingCount << ',' << st.overlapRatio * 100.0f << ','
            << st.h2dBytes / 1024 << ',' << st.d2hBytes / 1024 << ',' << st.gpuSubmits << ','
            << st.writeBackCount << ',' << st.beginFrameMs << ',' << st.ensureResidentMs << ','
            << st.integrateMs << ',' << st.extractMs << ',' << st.mergeMs << ','
            << tsdf.PointCloud().size() << '\n';
```

Also add `gpuSubmits` to the stdout line:

```cpp
        std::cout << "frame " << f << ": pts=" << points.size()
                  << " missing=" << st.missingCount
                  << " overlap=" << st.overlapRatio * 100.0f << "%"
                  << " submits=" << st.gpuSubmits
                  << " cloud=" << tsdf.PointCloud().size() << "\n";
```

- [ ] **Step 2: Build, run, confirm steady-state submits and surviving separation**

```bash
export VULKAN_SDK=/Users/sjy/VulkanSDK/1.4.328.1/macOS
cmake --build build --target directional_tsdf_demo -j"$(sysctl -n hw.ncpu)"
cd build/example2 && ./directional_tsdf_demo 2>&1 | tail -6
awk -F, 'NR>1 && $4==0 {print "steady frame", $1, "submits", $8}' directional_tsdf_stats.csv | head -3
awk 'NR>10 { if ($1 < 0) neg++; else pos++ } END { print "x<0:", neg, " x>0:", pos }' directional_result.ply
```
Expected: steady-state frames (missing==0) show `submits 3`; the two opposing surfaces still both present in the PLY (both counts > 0). Compare `integrateMs`/`ensureMs` against the Phase 3 run's CSV (kept in git history) to observe the drop.

- [ ] **Step 3: Commit**

```bash
git add example2/directional_tsdf_demo.cpp
git commit -m "Report gpuSubmits/d2hKB in DirectionalTSDF demo CSV"
```

---

## Phase 5 completion checklist

- [ ] `ComputePipeline::RecordDispatch` additive; all Engine::Core tests green (existing Dispatch path unchanged).
- [ ] `Engine::Compute` (StagingBuffer + CommandBatch) builds as its own target, links only Engine::Core, has unit tests.
- [ ] Bare `BeginFrame` = 1 submit; steady-state `Integrate` = 3 submits (deterministic test).
- [ ] All 27 pre-Phase-5 Directional tests still pass unchanged (correctness preserved).
- [ ] Demo CSV shows steady-state `submits 3` and both opposing surfaces preserved.
- [ ] Full suite keeps the known baseline (only `WideBVHTest.RadiusMatchesCpuReference` fails).
- [ ] Report results; note remaining deferred items (upload-budget cap / large-jump budget = spec §20 Phase 5 items 4/7; async copy-queue overlap is moot on unified memory).

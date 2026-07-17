# DirectionalTSDF Phase 2 (local base 이동과 reuse) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Phase 1's full-reload frame semantics with device-side active-pool classification (design doc §9) and reusable-slot re-registration (§10), so that groups already resident on the GPU are reused across frames and only genuinely missing groups are uploaded — verified by analytic reuse-count tests and an invariant sweep.

**Architecture:** One new compute shader (`directional_tsdf_classify.comp`) classifies every pool slot against the new local base into ReusableList / CleanFreeList / WriteBackList via atomic counters. `BeginFrame` becomes: reset indexGrid → classify → download lists → re-register ReusableList (reusing the existing register kernel with a different binding) → rebuild the CPU resident mirror. `EnsureResident` allocates slots from the downloaded CleanFreeList instead of a sequential counter. `Stats.overlapRatio` is computed from the per-frame required set vs missing count. WriteBackList is produced but not consumed (nothing is dirty until Phase 3 integration; write-back processing is Phase 4).

**Tech Stack:** C++17, Vulkan 1.3 via `Engine::Core`, GLSL compute, Eigen, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-07-17-directional-tsdf-design.md` (아키텍처 §4.1, §5 파이프라인 1~8단계, 예제/테스트 계획의 테스트 #4/#5). Streaming 성능 검증 방침: 이 Phase는 하드웨어 무관한 **비율 지표(overlapRatio/missing count/h2dBytes)와 해석적 재사용 검증**을 책임진다. 프레임별 CSV 로그와 스테이지별 타이밍은 Phase 3 데모에서 추가한다 (unified-memory 머신에서는 절대 GB/s가 문서의 discrete-GPU 전제를 대표하지 못하므로, 검증의 무게는 bytes/비율에 둔다).

## Global Constraints

- Phase 1 plan의 Global Constraints 전부 유지 (namespace/예외/std430 packing/동기 제출/staging 멀티 region 패턴/CMake GLOB).
- Phase 1의 기존 테스트 14개는 전부 그대로 통과해야 한다 (public API 시그니처 불변 — `Build`/`BeginFrame`/`EnsureResident`/debug 헬퍼는 동작 의미만 바뀜).
- 새 `.cpp` 파일은 없다 (셰이더 1개 + 기존 파일 수정) — CMake 재구성 불필요, 단 셰이더는 런타임 컴파일이므로 빌드 시스템 변경 없음.
- Invariant #6 (스펙 §21): missing 판정은 반드시 reusable 재등록 **이후**에 이뤄져야 한다 — `BeginFrame`이 재등록까지 끝내고, `EnsureResident`는 그 결과(`m_residentIndex`)만 본다.
- 테스트 baseline: 65 tests, 64 pass (`WideBVHTest.RadiusMatchesCpuReference`는 기존 무관 실패).
- 빌드 명령:
  ```bash
  export VULKAN_SDK=/Users/sjy/VulkanSDK/1.4.328.1/macOS
  cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
  ```

---

### Task 1: classify 커널 + reuse 기반 BeginFrame/EnsureResident

**Files:**
- Create: `src/shader/directional_tsdf_classify.comp`
- Modify: `src/Engine/Spatial/DirectionalTSDF.h`
- Modify: `src/Engine/Spatial/DirectionalTSDF.cpp`
- Modify: `test/test_directionalTSDF.cpp` (append tests)

**Interfaces:**
- Consumes: Phase 1의 모든 타입/버퍼/register 커널.
- Produces:
  - `struct DirectionalTSDF::ClassifyCounts { uint32_t reusable, cleanFree, writeBack; }` + `ClassifyCounts DebugLastClassifyCounts() const`
  - `Stats.overlapRatio` = 1 − (이번 프레임 missingCount / 이번 프레임 required 고유 키 수)
  - Shader contract (`directional_tsdf_classify.comp`): binding 0 = `ActiveGroupMeta meta[]` (readonly), binding 1/2/3 = `uint reusable[]`/`cleanFree[]`/`writeBack[]`, binding 4 = `uint counts[3]` (dispatch 전 0으로 리셋); push constants `{uint poolCapacity; int baseX, baseY, baseZ;}`. 슬롯당 1스레드: PendingUpload/PendingWriteBack은 어느 리스트에도 안 들어감(invariant #3/#4), invalid/Free → cleanFree, valid+inside window → reusable, valid+outside+dirty → writeBack, valid+outside+clean → cleanFree.

- [ ] **Step 1: classify 셰이더 작성**

Create `src/shader/directional_tsdf_classify.comp`:

```glsl
#version 460
layout(local_size_x = 256) in;

// Classifies every active-pool slot against the new local base (design doc §9).
// Atomic-counter list building — POOL_CAPACITY ≈ 32K, contention is acceptable
// for the initial implementation (the doc says the same).

#define LOCAL_GRID 50

struct ActiveGroupMeta {
    int  gx;
    int  gy;
    int  gz;
    uint packed; // bits 0-7 direction, 8-15 state, 16-23 dirty, 24-31 valid
};

layout(push_constant) uniform PC {
    uint g_poolCapacity;
    int  g_baseX;
    int  g_baseY;
    int  g_baseZ;
};

layout(std430, set = 0, binding = 0) readonly buffer Meta { ActiveGroupMeta g_meta[]; };
layout(std430, set = 0, binding = 1) buffer ReusableList { uint g_reusable[]; };
layout(std430, set = 0, binding = 2) buffer CleanFreeList { uint g_cleanFree[]; };
layout(std430, set = 0, binding = 3) buffer WriteBackList { uint g_writeBack[]; };
layout(std430, set = 0, binding = 4) buffer Counts { uint g_counts[3]; }; // reusable, cleanFree, writeBack

void main() {
    uint poolIndex = gl_GlobalInvocationID.x;
    if (poolIndex >= g_poolCapacity) return;

    ActiveGroupMeta m = g_meta[poolIndex];
    uint state = (m.packed >> 8) & 0xFFu;
    uint dirty = (m.packed >> 16) & 0xFFu;
    uint valid = (m.packed >> 24) & 0xFFu;

    // In-flight slots never enter any list (invariants #3/#4).
    if (state == 3u || state == 4u) return; // PendingUpload / PendingWriteBack

    if (valid == 0u || state == 0u) { // never used, or explicitly freed
        uint i = atomicAdd(g_counts[1], 1u);
        g_cleanFree[i] = poolIndex;
        return;
    }

    int lx = m.gx - g_baseX;
    int ly = m.gy - g_baseY;
    int lz = m.gz - g_baseZ;
    bool inside = lx >= 0 && ly >= 0 && lz >= 0 &&
                  lx < LOCAL_GRID && ly < LOCAL_GRID && lz < LOCAL_GRID;

    if (inside) {
        uint i = atomicAdd(g_counts[0], 1u);
        g_reusable[i] = poolIndex;
    } else if (dirty != 0u) {
        uint i = atomicAdd(g_counts[2], 1u);
        g_writeBack[i] = poolIndex;
    } else {
        uint i = atomicAdd(g_counts[1], 1u);
        g_cleanFree[i] = poolIndex;
    }
}
```

- [ ] **Step 2: 헤더 수정**

`src/Engine/Spatial/DirectionalTSDF.h`에서 다음을 변경한다.

(a) `Stats` 정의 바로 아래(public 영역)에 추가:

```cpp
        struct ClassifyCounts {
            uint32_t reusable = 0;
            uint32_t cleanFree = 0;
            uint32_t writeBack = 0;
        };
```

(b) `LastFrameStats()` 선언 아래에 추가:

```cpp
        // Result of the most recent BeginFrame classification (test/debug).
        ClassifyCounts DebugLastClassifyCounts() const { return m_lastCounts; }
```

(c) private 멤버에서

```cpp
        std::unique_ptr<Engine::Core::ComputePipeline> m_registerKernel;
```

를 다음으로 교체:

```cpp
        std::unique_ptr<Engine::Core::Buffer> m_reusableList;   // uint32[poolCapacity]
        std::unique_ptr<Engine::Core::Buffer> m_cleanFreeList;  // uint32[poolCapacity]
        std::unique_ptr<Engine::Core::Buffer> m_writeBackList;  // uint32[poolCapacity]
        std::unique_ptr<Engine::Core::Buffer> m_countsBuffer;   // uint32[3]
        std::unique_ptr<Engine::Core::ComputePipeline> m_registerKernel;
        std::unique_ptr<Engine::Core::ComputePipeline> m_classifyKernel;
```

(d) private 멤버에서

```cpp
        std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> m_residentIndex;
        std::vector<DirectionalGroupKey> m_slotKeys;
        uint32_t m_nextFreeSlot = 0;

        Stats m_stats;
```

를 다음으로 교체 (`m_nextFreeSlot` 제거, free-slot 리스트/required 집합/counts 추가):

```cpp
        std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> m_residentIndex;
        std::vector<DirectionalGroupKey> m_slotKeys;
        std::vector<uint32_t> m_freeSlots; // rebuilt from CleanFreeList every BeginFrame
        std::unordered_set<DirectionalGroupKey, DirectionalGroupKeyHash> m_requiredThisFrame;

        Stats m_stats;
        ClassifyCounts m_lastCounts;
```

(e) private 메서드 선언에 추가:

```cpp
        void updateOverlapRatio();
```

(f) 파일 상단 include에 `#include <unordered_set>` 추가.

(g) 클래스 주석의 "Phase 1 scope" 문단을 다음으로 교체:

```cpp
    // Phase 2 scope: BeginFrame classifies the active pool against the new local base on
    // the GPU (ReusableList / CleanFreeList / WriteBackList), re-registers reusable slots
    // into the indexGrid, and EnsureResident uploads only genuinely missing groups using
    // CleanFreeList slots. Integration/extraction (Phase 3) and dirty write-back (Phase 4)
    // come later; WriteBackList is produced but not yet consumed.
```

- [ ] **Step 3: 실패하는 테스트 작성**

`test/test_directionalTSDF.cpp` 끝에 추가:

```cpp
TEST(DirectionalTSDFPhase2Test, RepeatedFrameSameAABBReusesEverything) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/256);

    std::vector<DirectionalGroupKey> required = {
            {0, 0, 0, 0}, {1, 0, 0, 2}, {-2, 3, 1, 5}};

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident(required);
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 3u);

    std::vector<uint32_t> slotsBefore;
    for (const auto &key : required)
        slotsBefore.push_back(tsdf.DebugQueryPoolIndex(key));

    // Same AABB again: everything must be reused, nothing uploaded.
    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident(required);

    auto stats = tsdf.LastFrameStats();
    EXPECT_EQ(stats.missingCount, 0u);
    EXPECT_EQ(stats.h2dBytes, 0u);
    EXPECT_EQ(stats.residentCount, 3u);
    EXPECT_FLOAT_EQ(stats.overlapRatio, 1.0f);
    EXPECT_EQ(tsdf.DebugLastClassifyCounts().reusable, 3u);

    for (size_t i = 0; i < required.size(); ++i)
        EXPECT_EQ(tsdf.DebugQueryPoolIndex(required[i]), slotsBefore[i]);
}

TEST(DirectionalTSDFPhase2Test, OverlapRatioReflectsPartialReuse) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 256);

    // Frame 1: groups x ∈ [0, 9]. Frame 2: x ∈ [5, 14] → 5 reused, 5 missing.
    std::vector<DirectionalGroupKey> frame1, frame2;
    for (int x = 0; x <= 9; ++x) frame1.push_back({x, 0, 0, 0});
    for (int x = 5; x <= 14; ++x) frame2.push_back({x, 0, 0, 0});

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident(frame1);
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 10u);

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident(frame2);

    auto stats = tsdf.LastFrameStats();
    EXPECT_EQ(stats.missingCount, 5u);
    EXPECT_EQ(stats.h2dBytes, uint32_t(5u * kVoxelsPerGroup * sizeof(GpuTsdfVoxel)));
    EXPECT_FLOAT_EQ(stats.overlapRatio, 0.5f);
    // All 10 frame-1 groups are still inside the window → all reusable.
    EXPECT_EQ(tsdf.DebugLastClassifyCounts().reusable, 10u);
    // Resident = 10 reused + 5 newly uploaded.
    EXPECT_EQ(stats.residentCount, 15u);
}
```

- [ ] **Step 4: 실행해서 실패 확인**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
```
Expected: 컴파일 에러 (`DebugLastClassifyCounts` 미정의) — 헤더 수정(Step 2)까지 끝냈다면 링크/논리 실패로 진행. Step 2를 먼저 완료한 상태라면:
```bash
./build/test/vkspatial_tests --gtest_filter="DirectionalTSDFPhase2Test.*"
```
Expected: FAIL (Phase 1 semantics에서는 두 번째 프레임도 전부 missing → missingCount 3/10이 나와 0/5 기대와 불일치).

- [ ] **Step 5: 구현 — Build 확장**

`src/Engine/Spatial/DirectionalTSDF.cpp`에서:

(a) 익명 namespace의 `RegisterPC` 아래에 추가:

```cpp
        struct ClassifyPC {
            uint32_t poolCapacity;
            int32_t baseX;
            int32_t baseY;
            int32_t baseZ;
        };
```

(b) `Build()`에서 `m_slotListBuffer->Allocate(...)` 다음에 추가:

```cpp
        m_reusableList = std::make_unique<Engine::Core::Buffer>(ctx);
        m_cleanFreeList = std::make_unique<Engine::Core::Buffer>(ctx);
        m_writeBackList = std::make_unique<Engine::Core::Buffer>(ctx);
        m_countsBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_reusableList->Allocate(poolCapacity * sizeof(uint32_t));
        m_cleanFreeList->Allocate(poolCapacity * sizeof(uint32_t));
        m_writeBackList->Allocate(poolCapacity * sizeof(uint32_t));
        m_countsBuffer->Allocate(3u * sizeof(uint32_t));
```

(c) `Build()`의 register 커널 생성 다음에 classify 커널 생성 추가:

```cpp
        m_classifyKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_classifyKernel->Build("directional_tsdf_classify.comp")
                .Bind(0, *m_metaBuffer)
                .Bind(1, *m_reusableList)
                .Bind(2, *m_cleanFreeList)
                .Bind(3, *m_writeBackList)
                .Bind(4, *m_countsBuffer);
```

(d) `Build()`에서 `m_slotKeys.assign(...)` 앞에 meta 버퍼 zero-init 추가 (classify가 모든 슬롯의 meta를 읽으므로 미초기화 메모리를 valid로 오판하지 않도록):

```cpp
        // classify reads every slot's meta, so never-used slots must read as invalid.
        VkBuffer meta = m_metaBuffer->Handle();
        Engine::Core::SubmitOneShot(ctx, Engine::Core::QueueRole::Compute,
                                    [&](VkCommandBuffer cmd) {
                                        vkCmdFillBuffer(cmd, meta, 0, VK_WHOLE_SIZE, 0u);
                                    });
```

(e) `Build()`의 마지막 상태 초기화를 다음으로 교체 (`m_nextFreeSlot` 제거):

```cpp
        m_slotKeys.assign(poolCapacity, {});
        m_residentIndex.clear();
        m_freeSlots.clear();
        m_requiredThisFrame.clear();
        m_stats = {};
        m_lastCounts = {};

        fillIndexGridInvalid();
```

- [ ] **Step 6: 구현 — BeginFrame 재작성**

`BeginFrame` 본문 전체를 다음으로 교체:

```cpp
    void DirectionalTSDF::BeginFrame(const Eigen::Vector3f &aabbCenterHint) {
        if (!m_ctx)
            throw std::runtime_error("DirectionalTSDF: Build() must be called first");

        m_localBase = quantizeLocalBase(aabbCenterHint);
        m_stats = {};
        m_requiredThisFrame.clear();

        fillIndexGridInvalid();

        // 1. Classify every pool slot against the new local base (design doc §9).
        const uint32_t zeros[3] = {0, 0, 0};
        m_countsBuffer->Upload(zeros, sizeof(zeros));
        ClassifyPC cpc{m_poolCapacity, m_localBase.x(), m_localBase.y(), m_localBase.z()};
        m_classifyKernel->Args(cpc).DispatchElements(m_poolCapacity);

        // 2. Download the three lists (synchronous; the pool is small).
        uint32_t counts[3] = {0, 0, 0};
        m_countsBuffer->Download(counts, sizeof(counts));
        m_lastCounts = {counts[0], counts[1], counts[2]};

        std::vector<uint32_t> reusable(counts[0]);
        if (counts[0] > 0)
            m_reusableList->Download(reusable.data(), counts[0] * sizeof(uint32_t));

        std::vector<uint32_t> cleanFree(counts[1]);
        if (counts[1] > 0)
            m_cleanFreeList->Download(cleanFree.data(), counts[1] * sizeof(uint32_t));
        m_freeSlots.assign(cleanFree.begin(), cleanFree.end());

        // WriteBackList is produced for Phase 4; nothing marks dirty until integration.
        m_stats.writeBackCount = counts[2];

        // 3. Re-register reusable slots into the freshly-reset indexGrid. Invariant #6:
        //    this must complete before any missing-group decision is made.
        if (counts[0] > 0) {
            m_registerKernel->Bind(0, *m_reusableList);
            RegisterPC rpc{counts[0], m_localBase.x(), m_localBase.y(), m_localBase.z()};
            m_registerKernel->Args(rpc).DispatchElements(counts[0]);
        }

        // 4. Rebuild the CPU resident mirror from the reusable set.
        m_residentIndex.clear();
        for (uint32_t slot : reusable)
            m_residentIndex.emplace(m_slotKeys[slot], slot);
        m_stats.residentCount = uint32_t(m_residentIndex.size());
    }
```

- [ ] **Step 7: 구현 — EnsureResident 슬롯 할당 교체 + overlapRatio**

`EnsureResident`에서:

(a) 키 검사 루프를 다음으로 교체 (`m_requiredThisFrame` 추적 + free-list 할당):

```cpp
        for (const auto &key : required) {
            const int lx = key.gx - m_localBase.x();
            const int ly = key.gy - m_localBase.y();
            const int lz = key.gz - m_localBase.z();
            if (lx < 0 || ly < 0 || lz < 0 ||
                lx >= int(kLocalGroupGrid) || ly >= int(kLocalGroupGrid) ||
                lz >= int(kLocalGroupGrid))
                throw std::runtime_error(
                        "DirectionalTSDF: required group is outside the local window");

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
        if (pending.empty()) {
            updateOverlapRatio();
            return;
        }
```

(b) register 커널 dispatch 직전, binding 0을 slot-list 버퍼로 되돌리는 rebind 추가 (BeginFrame이 reusableList로 바꿔놨을 수 있으므로):

```cpp
        m_slotListBuffer->Upload(slots.data(), uint32_t(slots.size() * sizeof(uint32_t)));

        m_registerKernel->Bind(0, *m_slotListBuffer);
        RegisterPC pc{uint32_t(slots.size()), m_localBase.x(), m_localBase.y(),
                      m_localBase.z()};
        m_registerKernel->Args(pc).DispatchElements(uint32_t(slots.size()));

        m_stats.missingCount += uint32_t(pending.size());
        updateOverlapRatio();
```

(c) 파일 끝(`fillIndexGridInvalid` 앞 아무 곳)에 private 헬퍼 정의 추가:

```cpp
    void DirectionalTSDF::updateOverlapRatio() {
        const size_t requiredUnique = m_requiredThisFrame.size();
        m_stats.overlapRatio =
                requiredUnique == 0
                        ? 0.0f
                        : 1.0f - float(m_stats.missingCount) / float(requiredUnique);
    }
```

- [ ] **Step 8: 빌드 + Phase 2 테스트 통과 확인**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="DirectionalTSDFPhase2Test.*"
```
Expected: 2 tests PASS.

- [ ] **Step 9: Phase 1 테스트 회귀 확인**

```bash
./build/test/vkspatial_tests --gtest_filter="Directional*"
```
Expected: 16 tests 전부 PASS (기존 14 + 신규 2). Phase 1 테스트는 의미가 그대로 유지된다: 첫 프레임은 여전히 전부 missing이고, `PoolExhaustionThrows`는 free-list 고갈로 동일하게 throw.

- [ ] **Step 10: Commit**

```bash
git add src/shader/directional_tsdf_classify.comp src/Engine/Spatial/DirectionalTSDF.h src/Engine/Spatial/DirectionalTSDF.cpp test/test_directionalTSDF.cpp
git commit -m "Add device-side classification and slot reuse to DirectionalTSDF (Phase 2)"
```

---

### Task 2: 재사용/invariant 검증 테스트 배터리

이 Task는 순수 테스트 추가다 — Task 1 구현이 올바르면 전부 즉시 통과해야 하고, 실패하면 Task 1 코드의 버그를 인라인으로 수정한다. 스트리밍 검증 계획(스펙 예제/테스트 계획 #4~#6의 clean 경로 + invariant #1)의 Phase 2 몫을 여기서 완결한다.

**Files:**
- Modify: `test/test_directionalTSDF.cpp` (append tests)
- (버그 발견 시) Modify: `src/Engine/Spatial/DirectionalTSDF.cpp`

**Interfaces:**
- Consumes: Task 1의 `DebugLastClassifyCounts`/`Stats.overlapRatio` 포함 전체 API.
- Produces: 없음 (검증만).

- [ ] **Step 1: 검증 테스트 4개 작성**

`test/test_directionalTSDF.cpp` 끝에 추가:

```cpp
// Window shift eviction → return: a clean group whose slot was NOT reused must be
// revived from the still-intact pool data (no re-upload) when the window returns.
TEST(DirectionalTSDFPhase2Test, EvictedCleanGroupRevivesWhenWindowReturns) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/8);

    DirectionalGroupKey key{0, 0, 0, 0};
    DirectionalHostStore::Group group{};
    group[42] = {0.25f, 3.0f};
    tsdf.HostStore().Put(key, group);

    // Frame 1: window at origin, upload the group.
    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident({key});
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 1u);

    // Frame 2: window far away (base.x = floor(80/0.8)-25 = 75) → key outside → clean-freed.
    tsdf.BeginFrame(Eigen::Vector3f(80.0f, 0.0f, 0.0f));
    EXPECT_EQ(tsdf.DebugLastClassifyCounts().reusable, 0u);
    EXPECT_EQ(tsdf.DebugQueryPoolIndex(key), kInvalidPoolIndex); // outside window

    // Frame 3: window returns; slot was never reused, so classify revives it.
    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident({key});

    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 0u); // no re-upload needed
    EXPECT_EQ(tsdf.DebugLastClassifyCounts().reusable, 1u);
    auto out = tsdf.DebugDownloadGroupVoxels(key);
    EXPECT_NEAR(out[42].value, 0.25f, 1e-3f);
    EXPECT_NEAR(out[42].weight, 3.0f, 1e-3f);
}

// Tiny pool forces the evicted slot to be stolen; the group must then reload from the
// host store when the window returns (clean groups need no write-back — invariant #8).
TEST(DirectionalTSDFPhase2Test, StolenSlotGroupReloadsFromHostStore) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/2);

    DirectionalGroupKey k1{0, 0, 0, 0};
    DirectionalHostStore::Group group{};
    group[7] = {-0.5f, 2.0f};
    tsdf.HostStore().Put(k1, group);

    // Frame 1: upload k1.
    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident({k1});

    // Frame 2: window far away; two new groups consume both pool slots (k1's included).
    tsdf.BeginFrame(Eigen::Vector3f(80.0f, 0.0f, 0.0f));
    tsdf.EnsureResident({DirectionalGroupKey{100, 0, 0, 0},
                         DirectionalGroupKey{101, 0, 0, 0}});
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 2u);

    // Frame 3: window back at origin; k1's slot was stolen → must reload from host store.
    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident({k1});

    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 1u);
    auto out = tsdf.DebugDownloadGroupVoxels(k1);
    EXPECT_NEAR(out[7].value, -0.5f, 1e-3f);
    EXPECT_NEAR(out[7].weight, 2.0f, 1e-3f);
}

// Invariant #1 sweep: a window sliding one group per frame must never produce a
// duplicate pool slot in the indexGrid, the per-frame missing count must match the
// analytic expectation, and non-invalid cell count must equal residentCount.
TEST(DirectionalTSDFPhase2Test, DuplicateActiveSlotNeverOccurs) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/512);

    for (int f = 0; f < 6; ++f) {
        // Shift the window by exactly one group (0.8mm) per frame.
        tsdf.BeginFrame(Eigen::Vector3f(0.8f * float(f), 0.0f, 0.0f));

        // Required: groups gx ∈ [f-2, f+2] × directions {0, 1}.
        std::vector<DirectionalGroupKey> required;
        for (int dx = -2; dx <= 2; ++dx)
            for (uint8_t d = 0; d < 2; ++d)
                required.push_back({f + dx, 0, 0, d});
        tsdf.EnsureResident(required);

        // Analytic expectation: frame 0 uploads all 10; every later frame reuses
        // gx ∈ [f-2, f+1] (8 keys) and uploads only gx = f+2 (2 directions).
        if (f == 0) {
            EXPECT_EQ(tsdf.LastFrameStats().missingCount, 10u);
        } else {
            EXPECT_EQ(tsdf.LastFrameStats().missingCount, 2u);
            EXPECT_FLOAT_EQ(tsdf.LastFrameStats().overlapRatio, 0.8f);
        }

        auto grid = tsdf.DebugDownloadIndexGrid();
        std::unordered_set<uint32_t> seen;
        uint32_t nonInvalid = 0;
        for (uint32_t v : grid) {
            if (v == kInvalidPoolIndex) continue;
            ++nonInvalid;
            EXPECT_TRUE(seen.insert(v).second) << "duplicate pool slot " << v
                                               << " at frame " << f;
        }
        EXPECT_EQ(nonInvalid, tsdf.LastFrameStats().residentCount) << "frame " << f;
    }
}

// The full-reload H2D baseline vs steady-state: after the first frame, per-frame upload
// bytes must be a small fraction of the initial full upload (the doc's core claim,
// hardware-independent form).
TEST(DirectionalTSDFPhase2Test, SteadyStateUploadIsSmallFractionOfFullReload) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 512);

    auto requiredAt = [](int f) {
        std::vector<DirectionalGroupKey> keys;
        for (int dx = -4; dx <= 4; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                keys.push_back({f + dx, dy, 0, 0});
        return keys; // 27 groups, sliding one group per frame
    };

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident(requiredAt(0));
    const uint32_t fullReloadBytes = tsdf.LastFrameStats().h2dBytes;
    ASSERT_GT(fullReloadBytes, 0u);

    uint32_t steadyBytesMax = 0;
    for (int f = 1; f <= 5; ++f) {
        tsdf.BeginFrame(Eigen::Vector3f(0.8f * float(f), 0.0f, 0.0f));
        tsdf.EnsureResident(requiredAt(f));
        steadyBytesMax = std::max(steadyBytesMax, tsdf.LastFrameStats().h2dBytes);
    }

    // 27-group window sliding 1 column/frame → 3 new groups/frame = 1/9 of full reload.
    EXPECT_EQ(steadyBytesMax, uint32_t(3u * kVoxelsPerGroup * sizeof(GpuTsdfVoxel)));
    EXPECT_LT(float(steadyBytesMax) / float(fullReloadBytes), 0.15f);
}
```

파일 상단 include에 `#include <algorithm>`을 추가한다 (`std::max`).

- [ ] **Step 2: 빌드 + 실행**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="DirectionalTSDFPhase2Test.*"
```
Expected: 6 tests PASS (Task 1의 2개 + 신규 4개). 실패 시 Task 1 구현의 버그 — 원인 규명 후 인라인 수정 (테스트 기대값을 바꾸지 말 것: 기대값은 전부 해석적으로 유도된 것).

- [ ] **Step 3: 전체 스위트 회귀**

```bash
./build/test/vkspatial_tests 2>&1 | tail -6
```
Expected: 71 tests, 70 pass, 실패는 `WideBVHTest.RadiusMatchesCpuReference` 1건뿐.

- [ ] **Step 4: Commit**

```bash
git add test/test_directionalTSDF.cpp
git commit -m "Add reuse/invariant verification battery for DirectionalTSDF Phase 2"
```

---

## Phase 2 completion checklist

- [ ] 같은 AABB 반복 시 missingCount==0, overlapRatio==1.0 (스펙 테스트 #4)
- [ ] 슬라이딩 윈도우에서 missing 수가 해석적 기대치(프레임당 2/10)와 일치
- [ ] indexGrid에 중복 pool slot 없음 + 셀 수==residentCount (invariant #1)
- [ ] evict→복귀 시 clean 그룹의 revive(슬롯 미탈취)/host store reload(슬롯 탈취) 모두 데이터 보존
- [ ] steady-state h2dBytes가 full-reload 대비 소분율(테스트 시나리오에선 1/9)로 유지
- [ ] 전체 스위트 baseline 유지
- [ ] Phase 3 계획(integration/extraction + 데모 CSV 로깅/타이밍) 착수 전 리뷰

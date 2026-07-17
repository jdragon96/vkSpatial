# DirectionalTSDF Phase 4 (dirty write-back / eviction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process the WriteBackList that classification already produces — download dirty groups that left the local window, write them back into the host authoritative store, free their slots — so that integrated data survives eviction and the running average continues correctly when the window returns (design doc §12, spec 테스트 #6).

**Architecture:** `BeginFrame`이 리스트 다운로드 직후 WriteBackList를 동기 처리한다: pool→staging 멀티 region D2H 복사 → CPU에서 `GpuTsdfVoxel`→`HostTsdfVoxel` 변환 → `DirectionalHostStore::Put` → 해당 슬롯 meta를 0으로 클리어(scattered region copy; 클리어하지 않으면 다음 frame classify가 stale meta로 **이중 write-back**을 유발) → 슬롯을 free 리스트에 추가. 동기 구현이므로 `PendingWriteBack` 상태는 등장하지 않으며(스펙 "범위 밖" 명시), invariant #5(write-back 완료 전 재사용 금지)는 BeginFrame 내 순차 처리로 자동 보장된다. Pool 부족의 dirty-eviction 케이스는 outside-dirty가 매 프레임 즉시 회수되므로 자연 해소되고, required set 자체가 용량을 넘는 경우만 기존대로 throw한다.

**Tech Stack:** C++17, Vulkan 1.3 via `Engine::Core`, GoogleTest. (신규 셰이더 없음 — classify가 이미 리스트를 만든다.)

**Spec:** `docs/superpowers/specs/2026-07-17-directional-tsdf-design.md` §5(17단계)/§12, invariant #5/#8.

## Global Constraints

- Phase 1~3 plan의 Global Constraints 유지. 기존 Directional 테스트 25개 통과 유지.
- Write-back 변환은 Phase 1 업로드 변환의 정확한 역: `weight = sumW/kTsdfFixedScale`, `value = sumW>0 ? sumDW/sumW : 0` (이미 `DebugDownloadGroupVoxels`가 쓰는 공식과 동일).
- Meta 클리어는 zero-fill 값(`{0,0,0,0}` = invalid/Free/clean)이어야 다음 classify에서 cleanFree로 분류된다.
- 테스트 baseline: 75 tests, 74 pass.

---

### Task 1: BeginFrame write-back 처리 + 보존/복원 테스트

**Files:**
- Modify: `src/Engine/Spatial/DirectionalTSDF.cpp` (BeginFrame)
- Modify: `test/test_directionalTSDF.cpp`

**Interfaces:**
- Produces: `Stats.d2hBytes`가 실제 write-back 바이트를 반영; write-back된 그룹은 host store에서 조회 가능; 슬롯은 재사용 풀로 환원.

- [ ] **Step 1: 실패하는 테스트 작성** (`DirectionalTSDFPhase4Test` 스위트)

```cpp
TEST(DirectionalTSDFPhase4Test, DirtyGroupWritesBackToHostStore) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 4096);

    std::vector<Eigen::Vector3f> points, normals;
    makePlane(0.0f, 0.4f, 0.05f, Eigen::Vector3f(1, 0, 0), points, normals);
    tsdf.Integrate(points, normals, Eigen::Vector3f(2, 0, 0), Eigen::Vector3f::Zero());

    const DirectionalGroupKey key{0, 0, 0, 0};
    auto pre = tsdf.DebugDownloadGroupVoxels(key);
    ASSERT_GT(pre[0].weight, 0.0f);

    // Window leaves: dirty groups must be written back and their slots freed.
    tsdf.BeginFrame(Eigen::Vector3f(80.0f, 0.0f, 0.0f));
    auto st = tsdf.LastFrameStats();
    EXPECT_GT(st.writeBackCount, 0u);
    EXPECT_GT(st.d2hBytes, 0u);

    ASSERT_TRUE(tsdf.HostStore().Contains(key));
    const auto &stored = tsdf.HostStore().Get(key);
    EXPECT_NEAR(stored[0].value, pre[0].value, 1e-3f);
    EXPECT_NEAR(stored[0].weight, pre[0].weight, 1e-3f);

    // Slots were cleared, not revived: returning finds nothing reusable.
    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    EXPECT_EQ(tsdf.DebugLastClassifyCounts().reusable, 0u);
}

TEST(DirectionalTSDFPhase4Test, EvictedDirtyGroupReloadsFromHostStore) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 4096);

    std::vector<Eigen::Vector3f> points, normals;
    makePlane(0.0f, 0.4f, 0.05f, Eigen::Vector3f(1, 0, 0), points, normals);
    tsdf.Integrate(points, normals, Eigen::Vector3f(2, 0, 0), Eigen::Vector3f::Zero());

    const DirectionalGroupKey key{0, 0, 0, 0};
    auto pre = tsdf.DebugDownloadGroupVoxels(key);

    tsdf.BeginFrame(Eigen::Vector3f(80.0f, 0.0f, 0.0f)); // write-back + free

    tsdf.BeginFrame(Eigen::Vector3f::Zero());
    tsdf.EnsureResident({key});
    EXPECT_EQ(tsdf.LastFrameStats().missingCount, 1u); // freed → reload, not revive

    auto post = tsdf.DebugDownloadGroupVoxels(key);
    EXPECT_NEAR(post[0].value, pre[0].value, 1e-3f);
    EXPECT_NEAR(post[0].weight, pre[0].weight, 1e-3f);
}
```

- [ ] **Step 2: 실행해서 실패 확인** — `DirtyGroupWritesBackToHostStore`는 `Contains(key)`가 true여도(적분 전 GetOrCreate로 생성된 **빈** 그룹) `stored[0].weight ≈ 0 ≠ pre` 로 실패해야 정상.

- [ ] **Step 3: BeginFrame에 write-back 처리 구현**

`BeginFrame`에서 `m_freeSlots.assign(...)` 다음, `m_stats.writeBackCount = counts[2];` 를 다음 블록으로 교체:

```cpp
        // Write-back: dirty groups that left the window go to the host store, their
        // meta is cleared (otherwise next frame's classify would see stale-valid meta
        // and write them back twice), and their slots are freed (design doc §12).
        // Synchronous processing inside BeginFrame guarantees invariant #5.
        m_stats.writeBackCount = counts[2];
        if (counts[2] > 0) {
            std::vector<uint32_t> writeBack(counts[2]);
            m_writeBackList->Download(writeBack.data(), counts[2] * sizeof(uint32_t));

            const uint32_t kGroupBytesLocal = kVoxelsPerGroup * uint32_t(sizeof(GpuTsdfVoxel));
            const uint32_t bytes = counts[2] * kGroupBytesLocal;
            Engine::Core::Buffer staging(*m_ctx);
            staging.Allocate(bytes);
            {
                std::vector<VkBufferCopy> regions(counts[2]);
                for (uint32_t i = 0; i < counts[2]; ++i) {
                    regions[i].srcOffset = VkDeviceSize(writeBack[i]) * kGroupBytesLocal;
                    regions[i].dstOffset = VkDeviceSize(i) * kGroupBytesLocal;
                    regions[i].size = kGroupBytesLocal;
                }
                VkBuffer src = m_poolVoxels->Handle();
                VkBuffer dst = staging.Handle();
                Engine::Core::SubmitOneShot(*m_ctx, Engine::Core::QueueRole::Compute,
                                            [&](VkCommandBuffer cmd) {
                                                vkCmdCopyBuffer(cmd, src, dst,
                                                                uint32_t(regions.size()),
                                                                regions.data());
                                            });
            }
            std::vector<GpuTsdfVoxel> raw(size_t(counts[2]) * kVoxelsPerGroup);
            staging.Download(raw.data(), bytes);

            for (uint32_t i = 0; i < counts[2]; ++i) {
                const uint32_t slot = writeBack[i];
                DirectionalHostStore::Group group{};
                for (uint32_t v = 0; v < kVoxelsPerGroup; ++v) {
                    const GpuTsdfVoxel &g = raw[size_t(i) * kVoxelsPerGroup + v];
                    group[v].weight = float(g.sumW) / float(kTsdfFixedScale);
                    group[v].value = g.sumW > 0
                                             ? float(double(g.sumDW) / double(g.sumW))
                                             : 0.0f;
                }
                m_hostStore.Put(m_slotKeys[slot], group);
            }

            // Clear the written-back slots' meta so they classify as free next frame.
            {
                std::vector<ActiveGroupMeta> clearMeta(counts[2]); // zero = invalid/Free
                Engine::Core::Buffer metaStaging(*m_ctx);
                const uint32_t metaBytes = counts[2] * uint32_t(sizeof(ActiveGroupMeta));
                metaStaging.Allocate(metaBytes);
                metaStaging.Upload(clearMeta.data(), metaBytes);
                std::vector<VkBufferCopy> regions(counts[2]);
                for (uint32_t i = 0; i < counts[2]; ++i) {
                    regions[i].srcOffset = VkDeviceSize(i) * sizeof(ActiveGroupMeta);
                    regions[i].dstOffset = VkDeviceSize(writeBack[i]) * sizeof(ActiveGroupMeta);
                    regions[i].size = sizeof(ActiveGroupMeta);
                }
                VkBuffer src = metaStaging.Handle();
                VkBuffer dst = m_metaBuffer->Handle();
                Engine::Core::SubmitOneShot(*m_ctx, Engine::Core::QueueRole::Compute,
                                            [&](VkCommandBuffer cmd) {
                                                vkCmdCopyBuffer(cmd, src, dst,
                                                                uint32_t(regions.size()),
                                                                regions.data());
                                            });
            }

            for (uint32_t slot : writeBack)
                m_freeSlots.push_back(slot);
            m_stats.d2hBytes += bytes;
        }
```

- [ ] **Step 4: 빌드 + Phase 4 테스트 통과 + Directional 전체 회귀**
- [ ] **Step 5: Commit** — `"Add dirty write-back to host store on window exit (Phase 4)"`

---

### Task 2: 축적 왕복 검증 + 전체 회귀

**Files:**
- Modify: `test/test_directionalTSDF.cpp`

- [ ] **Step 1: 테스트 작성**

```cpp
// End-to-end: integration → eviction (write-back) → return (reload) → re-integration.
// The running weighted average must continue seamlessly across the host-store round
// trip: same data integrated twice → double the weight, same value.
TEST(DirectionalTSDFPhase4Test, IntegrationAccumulatesAcrossEviction) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 4096);

    std::vector<Eigen::Vector3f> points, normals;
    makePlane(0.0f, 0.4f, 0.05f, Eigen::Vector3f(1, 0, 0), points, normals);
    const Eigen::Vector3f cam(2, 0, 0);

    tsdf.Integrate(points, normals, cam, Eigen::Vector3f::Zero());
    const DirectionalGroupKey key{0, 0, 0, 0};
    auto pass1 = tsdf.DebugDownloadGroupVoxels(key);
    ASSERT_GT(pass1[0].weight, 0.0f);

    tsdf.BeginFrame(Eigen::Vector3f(80.0f, 0.0f, 0.0f)); // evict + write back

    tsdf.Integrate(points, normals, cam, Eigen::Vector3f::Zero()); // reload + accumulate
    auto pass2 = tsdf.DebugDownloadGroupVoxels(key);

    EXPECT_NEAR(pass2[0].weight, 2.0f * pass1[0].weight, pass1[0].weight * 0.1f);
    EXPECT_NEAR(pass2[0].value, pass1[0].value, 0.05f);
}
```

- [ ] **Step 2: 실행 → PASS 확인 (Task 1이 올바르면 즉시 통과; 실패 시 Task 1 버그 인라인 수정)**
- [ ] **Step 3: 전체 스위트** — 78 tests, 77 pass 기대 (기존 실패 1건).
- [ ] **Step 4: Commit** — `"Verify integration accumulates across eviction round trips (Phase 4)"`

## Phase 4 completion checklist

- [ ] dirty-outside 그룹이 host store로 정확히 write-back (값/weight 보존)
- [ ] write-back된 슬롯은 meta 클리어로 free 환원 (이중 write-back/revive 없음)
- [ ] evict→복귀 시 host store에서 reload되어 running average가 이어짐 (spec 테스트 #6)
- [ ] 전체 스위트 baseline 유지 → Phase 1~4 완료, 스펙 범위 종결 리뷰

# 크로스플랫폼 Directional TSDF 복원 아키텍처 (UMA + Discrete) 설계

> 선행 스펙: [`2026-07-17-directional-tsdf-design.md`](./2026-07-17-directional-tsdf-design.md)
> 원본 분석: `example2/DirectionalTSDF.md` (SP1팀 이재필, CUDA 기반 제품 설계)

## 배경 / 목적

`example2/DirectionalTSDF.md`(및 이를 구현한 `2026-07-17` 스펙)의 핵심 전제는 **discrete GPU + PCIe** 환경이다: direction이 6배로 늘어난 TSDF는 GPU VRAM에 다 담을 수 없으니, local AABB(40mm³) 안에서 실제로 필요한 group만 GPU에 상주시키고 나머지는 host(CPU) sparse store에 두고 H2D/D2H로 스트리밍한다. 문서 절반(§8~12, §16~19: active pool 캐시 관리, MissingGroupList, staging, PendingUpload/WriteBack 상태기계, 대역폭·latency 예산)이 이 "PCIe 버스가 있다"는 전제 위에서만 의미가 있다.

MacBook 같은 **통합 메모리(UMA)** 환경에서는 CPU/GPU가 같은 물리 DRAM을 공유하고 그 사이에 PCIe가 없다. 따라서:

- **스트리밍 기계 절반은 무의미하거나 역효과**다 — UMA에서 "복사"는 없어도 될 intra-DRAM copy + 포맷 변환 + eviction 오버헤드가 되어 오히려 느리고 메모리를 더 먹는다.
- 반면 **sparse 표현과 directional TSDF의 품질 이점은 아키텍처 무관하게 유효**하다. 12GB는 dense 기준 허수이고, sparse + 압축 인코딩이면 100μm에서도 전체 모델이 수백 MB~1GB 수준이라 **전체 모델을 통째로 GPU-addressable 통합 메모리에 상주**시킬 수 있다.

**최종 타깃은 Windows(discrete/PCIe)와 macOS(UMA) 둘 다**이다. 그러므로 목표는 "UMA 전용 설계"가 아니라 **두 메모리 토폴로지를 하나의 Vulkan-compute 코어로 커버하는 이식성 아키텍처**다. 이 스펙은 `2026-07-17` 스펙이 정의한 `DirectionalTSDF`(현재는 사실상 discrete streaming 단일 구현)를 **residency 백엔드로 추상화된 형태**로 진화시키고, UMA 백엔드와 자동 선택, 그리고 UMA가 열어주는 품질 업그레이드를 추가한다.

## 범위

- `src/Engine/Spatial/`의 `DirectionalTSDF` 코어를 **백엔드 무관(backend-agnostic)** 하게 리팩터: 코어는 Vulkan 메모리 복사/할당 API를 직접 부르지 않고 `IResidencyBackend` 인터페이스만 호출한다.
- `IResidencyBackend` 인터페이스 + 두 구현:
  - `StreamingResidencyBackend` — `2026-07-17` 스펙의 host store + active pool + staging + eviction 메커니즘을 인터페이스 뒤로 이동(로직 재사용, 껍데기만 교체).
  - `UnifiedResidencyBackend` — UMA coherent 힙에 전체 모델 상주, 복사·eviction·staging 없음.
- init 시 `VkPhysicalDeviceMemoryProperties` probe로 백엔드 **자동 선택** + config override(테스트용 강제).
- 단일 **fixed-point 8B 저장 포맷**으로 통일(host-float / GPU-fixed 이중 포맷 붕괴 — 아래 §2).
- 품질 코어 업그레이드: **multi-direction soft integration**, **confidence-weighted fusion**, directional extraction의 **candidate merge/split** 정착.
- UMA 품질 보너스(capability-gated): **CPU/GPU zero-copy co-refinement**, **touched 영역 전역 재추출**. discrete에서는 recomputeMask 부분추출로 graceful degrade.
- `test/`에 **cross-backend 정합성 테스트**(동일 입력 → 두 백엔드 ε 이내 동일 포인트클라우드) + 메모리 상한 테스트 추가.

### 범위 밖

- **실제 ICP/tracking 파이프라인** — `2026-07-17` 스펙과 동일하게 호출자가 매 프레임 `aabbCenterHint`를 넘긴다.
- **discrete 백엔드의 비동기 최적화(Phase 5)** — H2D/D2H overlap, chunked staging, upload budget 등. `2026-07-17` 스펙처럼 동기식으로 두되 인터페이스는 나중에 async로 바꿀 수 있게 잡는다.
- **CUDA / Metal 네이티브 경로** — compute는 Vulkan compute로 통일한다(맥은 MoltenVK). CUB/Thrust는 Vulkan subgroup 연산으로 대체.
- **Marching cubes mesh** — extraction은 point cloud만(원문 §15).
- **`SimpleTSDF` 변경** — 손대지 않는다.
- **원문 §16~19의 정확한 bandwidth/latency 수치** — 20B 포맷 전제라 우리 8B 구현엔 그대로 적용 안 됨. "reuse로 missing 비율이 낮게 유지된다"는 정성적 결론만 재현.

## 아키텍처

### 0. 기존 스펙과의 관계

`2026-07-17` 스펙이 정의한 자료구조·커널·invariant는 대부분 그대로 유효하다. 이 스펙이 바꾸는 것은 **"group 데이터가 어디에 살고 어떻게 device-addressable해지는가"** 딱 한 축이다. 구체적으로:

| `2026-07-17`에서                                                                                              | 이 스펙에서                                                               |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `DirectionalHostStore` + active pool + `Buffer::Upload/Download` staging이 `DirectionalTSDF`에 직접 박혀 있음 | 그 전부가 `StreamingResidencyBackend` 안으로 이동. 코어는 인터페이스만 봄 |
| `HostTsdfVoxel`(float) ↔ `GpuTsdfVoxel`(fixed-point) 변환                                                     | 단일 fixed-point 8B로 통일, 변환 소멸(§2)                                 |
| single-direction `dominantAxis` integration                                                                   | multi-direction soft(K≤2) 기본, single은 K=1 특수케이스(§7)               |
| candidate 무조건 append                                                                                       | position+normal merge/split(§7)                                           |
| 단일(discrete) 경로                                                                                           | UMA/discrete 자동 선택 + cross-backend 테스트                             |

### 1. 레이어 구조

의존성은 아래로만 흐른다. 코어는 백엔드 타입을 모른다(인터페이스 포인터만 보유).

```
┌ Reconstruction Core (백엔드 무관, Vulkan-compute) ──────────────┐
│  DirectionalTSDF · IntegrationPass · ExtractionPass            │
│  DirectionalVolumeCodec · IndexGrid · LocalWindow             │
└───────────────▲────────────────────────────────────────────────┘
                │ IResidencyBackend (순수 가상 인터페이스)
        ┌───────┴────────────────┐
 UnifiedResidencyBackend   StreamingResidencyBackend
 (macOS / UMA)             (Windows / discrete = 2026-07-17 메커니즘)
 - coherent 공유 힙        - VRAM active pool + host store
 - 전체 모델 상주          - classify/staging/H2D·D2H/eviction
 - acquire=O(1), 복사 없음  - acquire=ensureResident 경유 복사
```

**핵심 규칙**: 코어의 어떤 코드도 `vkCmdCopyBuffer`/`vkCmdFillBuffer`/메모리 할당을 직접 호출하지 않는다. 코어는 백엔드에 다음만 요청한다: "이 required set을 이번 프레임에 device-addressable하게 만들어라", "이 group을 dirty로 표시하라", "프레임 끝났다(정리해라)". 그게 공짜(UMA)인지 복사(discrete)인지는 백엔드가 결정한다.

### 2. 단일 fixed-point 8B 저장 포맷

`2026-07-17` 스펙은 host는 `{float value; float weight;}`(8B), GPU는 `{int32 sumDW; uint32 sumW;}`(8B, GLSL std430에 `atomicAdd(float)`가 없어 고정소수점 누적)로 **두 포맷**을 두고 up/download 직전·직후 CPU에서 변환했다. UMA에는 전송이 없으므로 이 변환은 순수 낭비다. **누적 accumulator를 canonical 저장 포맷으로 승격**한다:

```cpp
// 단일 저장/작업/전송 포맷. value는 저장하지 않고 읽을 때 파생한다.
struct DirVoxel {
    int32_t  sumDW;  // atomicAdd 대상, signedDistance * weight * kTsdfFixedScale
    uint32_t sumW;   // atomicAdd 대상, weight * kTsdfFixedScale
}; // 8B
// value(읽기) = float(sumDW) / float(sumW);  weight = float(sumW) / kTsdfFixedScale
```

- **UMA**: host store 없이 이 8B pool 하나가 authoritative. CPU/GPU가 같은 버퍼를 coherent하게 접근 → **변환·복사 0**.
- **discrete**: 전송도 이 8B 그대로(변환 없이 `vkCmdCopyBuffer`). 원문 20B 대비 PCIe 대역폭 60% 절감(순개선).
- normal은 저장하지 않는다(`2026-07-17` 결정 유지) — extraction 시 SDF gradient(같은 direction layer 내 유한차분)로 계산. direction 선택에 쓰는 normal은 **integration 입력 sample의 normal**이지 저장값이 아니므로 충돌 없음.
- overflow 안전성: `maxCnt`로 `sumW` 상한을 두면(원문 §13) int32 범위 안에 머문다. `kTsdfFixedScale`/`maxCnt` 조합은 `static_assert`로 상한 검증.

메모리 추정(8B 기준): 40mm 라이브 윈도우(~20K directional groups) ≈ 82MB, 전체 아치 누적(~40K groups) ≈ 160MB, 풀마우스 정밀 worst(~187K groups) ≈ 0.5~0.8GB → **16GB+ Mac이면 전체 모델 상주 가능**.

### 3. `IResidencyBackend` 인터페이스

```cpp
namespace Engine::Spatial {

// 코어가 group을 다루는 유일한 창구. Vulkan 세부는 전부 이 뒤에 숨는다.
class IResidencyBackend {
public:
    virtual ~IResidencyBackend() = default;

    // required set(IntegrationWriteSet + halo)을 이번 프레임에 device-addressable하게.
    // discrete: classify → missing → H2D → register. UMA: no-op(이미 상주) + indexGrid relabel.
    virtual void EnsureResident(std::span<const DirectionalGroupKey> required,
                                const LocalWindow &window) = 0;

    // group을 dirty로 표시(재적분 발생). discrete: writeback 후보. UMA: co-refine/persist 후보.
    virtual void MarkDirty(const DirectionalGroupKey &key) = 0;

    // 프레임 종료 정리. discrete: dirty write-back + eviction. UMA: soft-cap 초과 시에만 cold compact.
    virtual void EndFrame(const LocalWindow &next) = 0;

    // 셰이더 바인딩 대상 — 코어가 push constant/descriptor로 넘긴다.
    virtual VkBuffer      PoolBuffer()      const = 0;  // DirVoxel[capacity*512]
    virtual VkBuffer      IndexGridBuffer() const = 0;  // uint32[kIndexGridCells]
    virtual uint32_t      PoolCapacity()    const = 0;

    // capability — 코어가 품질 보너스 분기에 사용.
    virtual bool IsUnified()                 const = 0;
    virtual bool SupportsZeroCopyCpuAccess() const = 0;
};

} // namespace Engine::Spatial
```

코어(`DirectionalTSDF::Integrate`)는 이 인터페이스만 호출한다. `2026-07-17` 스펙의 프레임 파이프라인(§5)에서 "indexGrid reset → classify → register_reusable → missing upload → integrate → extract → writeback" 중 **indexGrid/classify/missing/writeback 부분이 backend.EnsureResident/EndFrame 안으로 흡수**되고, 코어에는 integrate/extract 디스패치와 required-set 계산만 남는다.

### 4. `UnifiedResidencyBackend` (macOS / UMA)

- `PoolBuffer`를 **`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`** 힙에 하나 할당(전체 모델 용량, chunk 단위로 성장). Vulkan에서 UMA는 이 조합을 만족하는 힙을 노출한다.
- `IndexGridBuffer`도 같은 coherent 힙. `persistently mapped` 포인터를 들고 있어 CPU가 group 슬롯을 **복사 없이 직접** 읽고 쓴다.
- **슬롯 할당**: `(gx,gy,gz,direction)` → slot을 CPU측 `unordered_map`으로 O(1) 관리. 새 키면 pool 끝에 슬롯을 하나 append(필요 시 pool grow). authoritative store가 pool 자체이므로 별도 host store 없음.
- `EnsureResident`: 전부 이미 상주 → **복사 없음**. 하는 일은 (a) required set 중 처음 보는 키에 슬롯 append, (b) 새 window base 기준 `indexGrid` relabel(coherent 힙이라 CPU가 직접 채우거나 `vkCmdFillBuffer`+가벼운 register 커널). eviction/staging/missing 개념 없음.
- `MarkDirty`: dirty 비트만 세팅(co-refine/persist 큐용). write-back 대상 아님.
- `EndFrame`: 기본 no-op. **soft cap**(예: 예산의 90%) 초과 시에만 가장 오래된(lastUsedFrame) cold group을 disk-backed mmap으로 compact(정밀 풀마우스가 저사양 Mac 예산을 넘길 때의 안전장치, 평상시엔 안 돎).
- `IsUnified()=true`, `SupportsZeroCopyCpuAccess()=true`.

### 5. `StreamingResidencyBackend` (Windows / discrete)

`2026-07-17` 스펙의 메커니즘을 **그대로** 인터페이스 뒤로 옮긴 것(로직 신규 발명 없음):

- `PoolBuffer`는 `DEVICE_LOCAL` VRAM(POOL_CAPACITY=32,768 슬롯) + 별도 `DirectionalHostStore`(이제 8B `DirVoxel` 저장) + staging 버퍼.
- `EnsureResident`: `directional_tsdf_classify.comp`(Reusable/CleanFree/WriteBack) → `register_reusable.comp` → MissingGroupList 추림 → staging에 채워 `vkCmdCopyBuffer` region 배치 H2D → register. (`2026-07-17` §5의 4~11단계.)
- `EndFrame`: dirty group D2H write-back → host store 반영 → slot free. pool shortage 시 WriteBackList eviction(원문 §11).
- 변경점: 전송 포맷이 8B `DirVoxel` 단일(변환 커널 없음), classify의 atomic compaction은 유지하되 **CUB/Thrust가 아니라 Vulkan subgroup ballot/prefix-sum**(원문 §9의 대안 3 "block-local aggregation 후 global atomic"에 해당)로 구현 — 이래야 맥에서도 같은 셰이더가 돈다.
- `IsUnified()=false`, `SupportsZeroCopyCpuAccess()=false`.

### 6. 백엔드 자동 선택

```
init 시 vkGetPhysicalDeviceMemoryProperties:
  DEVICE_LOCAL 이면서 HOST_VISIBLE 인 힙이 존재하고 그 크기 ≥ 모델 예산
    → UnifiedResidencyBackend
  아니면
    → StreamingResidencyBackend
config(env/flag)로 강제 override 가능 (UMA에서 Streaming 강제 → cross-backend 정합성 테스트)
```

저장소는 이미 "pluggable backend + per-backend Context + cross-backend 정합성 테스트"(Engine::Spatial BVH) 패턴을 쓰므로 동일한 결에 얹는다.

### 7. 품질 코어 업그레이드 (양 플랫폼 공통 baseline)

**7.1 Multi-direction soft integration** (원문 §13의 "향후" 항목을 baseline으로)

- 각 sample normal을 가장 가까운 **K≤2 canonical 방향**에 `w_d = max(0, dot(n, axis_d))^p`(p≈2~4) 가중으로 분배. 방향 경계(예: normal이 45°)에서 한 layer만 갱신되어 생기는 seam을 제거 → 얇은 벽/인접면 매끄러움 향상.
- `K=1`(dominant만)이 `2026-07-17`의 기존 동작 = 특수케이스. push constant `maxDirections`로 전환.

**7.2 Confidence-weighted running average** (원문 §13 공식 승격)

- `weight = frontendConfidence · viewAngleWeight · directionConfidence` (`viewAngleWeight = max(0, dot(n, viewDir))`, `directionConfidence = w_d` from 7.1). `sampleWeight=1` 상수 대신 이 값을 `kTsdfFixedScale` 곱해 `atomicAdd`. 관측각이 나쁜 샘플의 기여를 자연히 줄여 경계 왜곡(원문 §2) 완화.

**7.3 Directional extraction merge/split** (원문 §15 정착)

- center-voxel 3×3×3, **같은 direction layer끼리만 sign crossing**(invariant 유지). 한 spatial voxel의 최대 6 candidate를 position+normal로 클러스터:
  - position 가깝 + normal angle < `normalMergeThreshold`(30°) → weighted merge
  - normal angle > `strongSplitThreshold`(60°) → 별도 surface
  - 그 사이 → 별도 cluster
- 이로써 원문이 지적한 "candidate 무조건 허용 → 표면 거칢"을 해결. normal orientation은 view direction 기준 일관화(초기엔 `abs(dot)` 허용).

### 8. UMA 품질 보너스 (capability-gated, graceful degrade)

`backend.SupportsZeroCopyCpuAccess()` / `backend.IsUnified()`로 분기. discrete에서는 자동으로 대체 경로.

**8.1 전역 재추출 (UMA)**: 전체 모델이 상주하므로 touched 영역을 ghosting 없이 전역 재추출 가능.

- discrete 대체: recomputeMask 기반 부분 재추출 + kept old points(원문 §14).

**8.2 CPU/GPU zero-copy co-refinement (UMA 전용)**: CPU 스레드가 coherent pool 버퍼를 **복사 없이** 직접 읽어, GPU integration과 오버랩하여 cold group에 edge-aware TSDF regularization / outlier 제거를 수행. GPU 파이프라인을 막지 않는 백그라운드 품질 향상.

- discrete 대체: skip 또는 옵션 GPU 패스(복사 비용 때문에 기본 off).

### 9. Invariant

원문 §21 + `2026-07-17`의 invariant 유지. 백엔드별 성립 양상:

| Invariant                                               | Streaming                         | Unified                         |
| ------------------------------------------------------- | --------------------------------- | ------------------------------- |
| 키당 pool slot 중복 금지                                | classify/register로 강제          | pool 하나·키당 슬롯 하나 → 자명 |
| indexGrid는 local base 종속(변경 시 reset+재등록)       | 매 프레임 reset+classify+register | 매 프레임 relabel(값싼)         |
| dirty group은 write-back 완료 전 free 재사용 금지       | 유지(PendingWriteBack)            | write-back 없음 → 무해          |
| MissingGroupList는 reusable 재등록 이후 생성            | 유지                              | missing 개념 없음               |
| integration은 IntegrationWriteSet에만 write             | 양쪽 공통(코어 로직)              | 동일                            |
| directional extraction은 같은 layer끼리만 sign crossing | 양쪽 공통                         | 동일                            |

**추가 invariant(이 스펙)**: 동일 입력·동일 파라미터에서 두 백엔드는 ε 이내 동일한 extraction 결과를 낸다(cross-backend 결정성). 이게 추상화 건전성의 계약이다.

### 10. 테스트 전략

- **Cross-backend 정합성 (핵심)**: 합성 interproximal 씬 + 스크립트 카메라 경로를 `UnifiedResidencyBackend`와 (강제) `StreamingResidencyBackend` 양쪽에 통과 → 포인트클라우드가 ε 이내 일치 assert. 저장소의 기존 cross-backend BVH 테스트와 동일 패턴.
- **메모리 상한**: 풀마우스 스캔 시퀀스에서 UMA pool 총량이 목표(예: <1GB) 이하 유지 assert.
- **품질 회귀**: 얇은 벽/인접면 테스트 씬에서 `SimpleTSDF`(baseline) vs `DirectionalTSDF` vs multi-direction 결과를 PLY로 export, 구멍/섞임 정성 비교(원문 §2/§23 실패 케이스).
- **invariant 단위 테스트**: 키당 슬롯 중복 금지, indexGrid local-base 종속 등(GTest).
- **주의**: 저장소 `Engine::Core`에 large-N(≳1000) GPU compute 비결정성 버그가 문서화돼 있음(`docs/KNOWN_ISSUES_engine_core_large_n.md`) → 여기 커널은 그 경로를 피하거나, cross-backend 테스트를 그 임계 이하 N에서 먼저 통과시킨 뒤 확장.

### 11. 단계 (Phasing)

1. **`DirVoxel` 8B 통일 + `IResidencyBackend` 인터페이스 추출** — `2026-07-17` 코드의 host/GPU 이중 포맷을 fixed-point 단일로, streaming 로직을 `StreamingResidencyBackend`로 이동(동작 동일, 리팩터만). 기존 테스트 그대로 통과가 게이트.
2. **`UnifiedResidencyBackend`** — coherent 힙 pool + 전체 상주 + relabel. macOS에서 복원 동작.
3. **백엔드 자동 선택 + cross-backend 정합성 테스트** — 두 백엔드 ε 일치.
4. **품질 코어**: multi-direction soft integration + confidence fusion + directional merge/split(§7).
5. **UMA 보너스**: 전역 재추출 / CPU-GPU co-refinement, discrete recomputeMask 부분추출(§8).
6. **성능**: subgroup classification, chunked staging(discrete only), H2D/D2H overlap(discrete only), UMA cold compact.

## 문서/기존 스펙과의 의도적 차이점

- **20B/32B 이중 포맷 → fixed-point 8B 단일**: UMA 변환 제거 + discrete 대역폭 절감. 원문 §5/§11/§12의 decode/encode 커널 삭제.
- **스트리밍을 "지우는" 게 아니라 "백엔드로 격리"**: discrete에선 여전히 필요하므로 인터페이스 뒤에 보존. UMA에선 그 구현을 no-op에 가깝게 대체.
- **single-direction → multi-direction 기본**: 원문이 "향후"로 미룬 품질 항목을 baseline으로 당김(UMA에서 남는 compute 예산을 품질에 재투자한다는 이 재설계의 논지).
- **CUB/Thrust → Vulkan subgroup**: 크로스플랫폼 단일 셰이더.

## 리스크

| 리스크                                       | 대응                                                                                      |
| -------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 인터페이스가 UMA에 불필요한 추상화 비용 전가 | `EnsureResident`/`EndFrame`를 UMA에서 near-no-op으로; 슬롯 접근은 인라인 가능한 얇은 경로 |
| coherent 힙 write가 UMA에서 캐시 일관성 함정 | `HOST_COHERENT` 강제 + 필요 시 명시적 `vkFlush/MakeVisibleToCPUMemoryRanges` 경로 준비    |
| MoltenVK가 `DEVICE_LOCAL                     | HOST_VISIBLE` 큰 힙을 기대대로 노출 안 함                                                 | init probe에서 실패 시 StreamingBackend로 폴백(자동 선택이 이미 이 폴백을 포함) |
| cross-backend ε 불일치(부동소수·atomic 순서) | fixed-point 누적으로 순서 무관성 확보, ε는 정량 threshold로 명문화                        |
| Engine::Core large-N 버그                    | 임계 이하에서 먼저 검증, 별도 이슈로 추적(`docs/KNOWN_ISSUES_engine_core_large_n.md`)     |
| **Unified 시딩 경로 미구현** (아래 참고)     | 현재 `Build` 기본값 Streaming으로 완전히 gated. `Auto` 기본값 전환 전 반드시 해소         |

## 알려진 제약 (Phase 1~3 구현 후 최종 리뷰에서 확인)

**Unified 백엔드의 host-store 시딩 경로가 Streaming과 갈린다.** `UnifiedResidencyBackend`는 authoritative store가 pool 자체이므로(§4.1), first-touch에서 slot을 **zero-fill**하고 `HostStore()`가 반환하는 placeholder(`m_storeView`)를 읽지 않는다. 반면 `StreamingResidencyBackend`는 `RecordResidency`에서 `m_hostStore.GetOrCreate(key)`로 시딩된 voxel을 업로드한다.

- **영향**: `HostStore().Put(key, group)` → `EnsureResident({key})` → 시딩 값 기대, 이 워크플로우는 **Streaming에서만** 정확하다. 기존 `test_directionalTSDF.cpp`의 여러 테스트가 이 경로를 쓰며, `Build` 기본값이 `ResidencyMode::Streaming`이라 통과한다.
- **cross-backend 등가 증명의 범위**: `CrossBackendReconstructionMatches`는 **`Integrate` 경로**(GPU가 voxel을 직접 write하므로 시딩 무관)만 커버한다. 시딩 경로의 등가는 증명되지 않았다.
- **전제 조건**: 향후 계획에서 `Build` 기본값을 `ResidencyMode::Auto`로 뒤집으려면(= UMA 머신에서 자동으로 Unified 선택) 먼저 이 갈림을 해소해야 한다. 선택지: (a) Unified가 first-touch 시 `m_storeView` 내용을 pool로 복사, (b) 시딩 경로에 대한 cross-backend 테스트 추가, (c) Unified `HostStore()`/시딩을 명시적 unsupported로 만들고 `IResidencyBackend::HostStore()` doc 코멘트 수정. 이 항목은 8B fixed-point 통일과 함께 후속 계획으로 이관한다.

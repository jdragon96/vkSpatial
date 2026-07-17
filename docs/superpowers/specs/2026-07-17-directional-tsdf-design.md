# DirectionalTSDF 설계 (`docs/DirectionalTSDF.pdf` / `example2/DirectionalTSDF.md` 기반)

## 배경 / 목적

`example2/DirectionalTSDF.md`는 intraoral scanner reconstruction용 "Directional TSDF GPU streaming 구조" 설계 문서(SP1팀 이재필, CUDA 기반 실제 제품 설계)를 분석·정리한 것이다. 핵심 아이디어는 두 가지다.

1. **Directional TSDF 알고리즘**: 하나의 voxel 위치에 하나의 SDF만 저장하면 얇은 구조·interproximal 영역에서 서로 다른 표면이 섞이는 문제가 생긴다. 위치마다 6개의 canonical direction(±X,±Y,±Z)을 두고, normal이 가장 가까운 방향의 layer에 독립적으로 SDF를 적분하면 이 문제가 완화된다.
2. **Host/GPU 하이브리드 스트리밍 캐시**: direction이 6개로 늘면 dense grid로는 GPU 메모리가 감당 안 되므로(문서 §3: 12GB), local AABB(40mm³) 안에서 실제로 필요한 group만 GPU에 상주시키고 나머지는 host(CPU) sparse store에 두는 캐시 구조(indexGrid + active pool + H2D/D2H streaming + dirty write-back/eviction).

이 스펙은 이 설계 전체(문서 §1~§15, §21의 핵심 알고리즘·자료구조·invariant)를 `Engine::Spatial`에 `DirectionalTSDF`(vk 접두사 없음, `SimpleTSDF`와 동일한 명명 관례)로 구현하기 위한 것이다. 문서의 Phase 5(비동기 H2D/D2H overlap, chunked staging, upload budget 등 성능 최적화)는 이번 스펙 범위 밖이며, Phase 1~4(기본 streaming cache → local base 이동/reuse → integration/extraction 연동 → write-back/eviction)까지 동기식으로 구현한다.

기존 `Engine::Spatial::SimpleTSDF`(open-addressing sparse hash, `voxel_tsdf_integrate.comp`)와 나란히 존재하는 두 번째 TSDF 구현이 된다 — `SimpleTSDF`를 대체하지 않는다. `SimpleTSDF`는 카메라 방향과 무관하게 하나의 SDF만 유지하는 단순 버전으로 계속 남고, `DirectionalTSDF`는 방향별 독립 SDF + host/GPU 스트리밍 캐시를 갖는 확장 버전이다.

## 범위

- `src/Engine/Spatial/`에 `DirectionalTSDF`(및 보조 타입)를 신규 추가.
- indexGrid, active pool(`ActiveGroupMeta`/`SlotState`), host authoritative sparse store, missing-group H2D upload, dirty group D2H write-back/eviction, dominant-direction integration, direction-layer-aware extraction(+ candidate merge/split), recomputeMask 기반 old point 병합까지 전부 구현.
- `example2/`에 interproximal 합성 시나리오 + 스크립트된 카메라 경로로 여러 프레임을 진행시키는 데모 실행 파일 추가, `SimpleTSDF`(baseline) vs `DirectionalTSDF` 결과를 PLY로 비교 export.
- `test/`에 핵심 invariant(중복 slot 금지, indexGrid가 local base에 종속적 등)를 검증하는 GTest 케이스 추가.

### 범위 밖

- **Phase 5(비동기 H2D/D2H, chunked staging, upload budget, large tracking jump 대응 등 성능 최적화)** — 상태머신(`PendingUpload`/`PendingWriteBack`)은 나중에 비동기로 바꿀 수 있게 구조만 잡아두되, 이번 구현은 전부 `Engine::Core::Buffer::Upload/Download`(각각 blocking)로 동작한다.
- **실제 ICP/tracking 파이프라인** — 이 저장소에 카메라 트래킹이 없으므로 `Integrate()`는 문서의 `ICP/Tracking → Patch AABB` 단계를 스킵하고, 호출자가 매 프레임 local AABB 중심(`aabbCenterHint`)을 직접 넘긴다. `example2` 데모는 스크립트된 카메라 경로로 이 값을 생성한다.
- **문서의 정확한 byte-packed voxel format(20B/32B, oct16 normal, colorMap, patchID, segmentation 등)** — 아래 "문서와의 의도적 차이점"에서 설명하듯 우리 시스템엔 color/segmentation/patch 개념이 없으므로 필요한 필드만 남기고 대폭 축소한다. 문서 §16~§19의 정확한 bandwidth/latency 수치(0.32GB/s 등)는 그 byte 크기를 전제로 계산된 것이라 우리 구현에는 그대로 적용되지 않는다 — 대신 "reuse 덕분에 매 프레임 missing 비율이 낮게 유지된다"는 정성적 결론만 재현 목표로 삼는다.
- `SimpleTSDF`의 API/구현 변경 — 손대지 않는다.
- Marching cubes 기반 mesh export(`ExportMC`류) — 문서 §15가 명시하듯 Directional TSDF의 extraction은 triangle이 아니라 point cloud이므로 `DirectionalTSDF`는 `ExportPointCloud`만 제공한다.

## 아키텍처

### 1. 데이터 구조 (`DirectionalTSDFTypes.h`)

파라미터는 문서 §3 그대로 기본값을 두되(voxelSize/truncation은 `SimpleTSDF`와 동일하게 world-unit 무관 float으로 파라미터화), 정수형 상수는 `constexpr`로 둔다.

```cpp
namespace Engine::Spatial {

constexpr uint32_t kGroupDim         = 8;                              // 8x8x8 voxels
constexpr uint32_t kVoxelsPerGroup   = kGroupDim * kGroupDim * kGroupDim; // 512
constexpr uint32_t kLocalGroupGrid   = 50;                             // 50x50x50 groups
constexpr uint32_t kNumDirections    = 6;                              // ±X,±Y,±Z
constexpr uint32_t kIndexGridCells   = kLocalGroupGrid * kLocalGroupGrid * kLocalGroupGrid * kNumDirections;
constexpr uint32_t kInvalidPoolIndex = 0xFFFFFFFFu;
constexpr int32_t  kTsdfFixedScale   = 10000; // SimpleTSDF와 동일한 고정소수점 스케일

enum class SlotState : uint8_t {
    Free = 0, ResidentClean = 1, ResidentDirty = 2, PendingUpload = 3, PendingWriteBack = 4,
};

struct DirectionalGroupKey {
    int32_t gx, gy, gz;
    uint8_t direction; // 0..5 = +X,-X,+Y,-Y,+Z,-Z
    bool operator==(const DirectionalGroupKey &o) const;
};
struct DirectionalGroupKeyHash { size_t operator()(const DirectionalGroupKey &k) const; };

// Host authoritative sparse store의 저장 단위이자 H2D/D2H wire format.
// 문서 §5의 20B 포맷에서 이 프로젝트엔 없는 필드(colorMap/segmentation/patchID)를 뺐다.
// normal은 저장하지 않는다 — extraction 시 SDF gradient로 계산한다(아래 §4.4 참고).
struct HostTsdfVoxel {
    float value;   // 4B, [-1, 1] truncated signed distance
    float weight;  // 4B, 누적 weight (재적분 시 running-average를 이어가기 위함)
}; // 8B

// GPU active pool의 작업 포맷. GLSL std430에는 atomicAdd(float)가 없으므로
// SimpleTSDF의 TSDFEntry(sumDW/sumW 고정소수점 누적)와 동일한 패턴을 재사용한다.
struct GpuTsdfVoxel {
    int32_t  sumDW; // atomicAdd, signedDistance * kTsdfFixedScale
    uint32_t sumW;  // atomicAdd, weight * kTsdfFixedScale
}; // 8B

struct ActiveGroupMeta {
    int32_t gx, gy, gz;
    uint8_t direction;
    uint8_t state;   // SlotState
    uint8_t dirty;
    uint8_t valid;
}; // 16B (std430 정렬)

struct DirectionalCandidate {
    float px, py, pz;      // extraction position
    float nx, ny, nz;      // gradient 기반 normal
    int32_t gx, gy, gz;    // owner group (recomputeMask 판정용)
    uint8_t direction;
}; // GPU→CPU 다운로드용, atomic counter로 append

struct ExtractedPoint {
    Eigen::Vector3f position;
    Eigen::Vector3f normal;
    int32_t ownerGx, ownerGy, ownerGz;
    uint8_t dirMask; // merge된 direction들의 비트마스크 (시각화/디버깅용)
};

} // namespace Engine::Spatial
```

`static_assert(std::is_standard_layout_v<...>)` + `offsetof` 체크를 `HostTsdfVoxel`/`GpuTsdfVoxel`/`ActiveGroupMeta`/`DirectionalCandidate`에 추가한다 (vkSpatial 관례, `types.h:76-103` 참고).

### 2. `DirectionalHostStore` (`DirectionalHostStore.h/.cpp`)

문서의 "host authoritative sparse TSDF store"에 대응하는, GPU를 전혀 모르는 순수 CPU 클래스. 디스크 영속성은 없음(프로세스 메모리에만 존재) — 실제 제품과 달리 이 데모는 세션 내 캐시로 충분하다.

```cpp
class DirectionalHostStore {
public:
    using Group = std::array<HostTsdfVoxel, kVoxelsPerGroup>;

    bool Contains(const DirectionalGroupKey &key) const;
    const Group &Get(const DirectionalGroupKey &key) const;   // 없으면 throw
    Group &GetOrCreate(const DirectionalGroupKey &key);        // 없으면 전부 value=0,weight=0으로 초기화
    void Put(const DirectionalGroupKey &key, const Group &data);
    size_t Size() const;

private:
    std::unordered_map<DirectionalGroupKey, Group, DirectionalGroupKeyHash> m_groups;
};
```

`Group`이 8B×512=4KB이므로 `unordered_map`의 노드당 오버헤드는 무시할 만한 수준 — 최적화 없이 표준 라이브러리로 충분하다(문서 §9 결론과 동일한 태도: "초기 구현은 단순성 우선").

### 3. indexGrid

`Engine::Core::Buffer` 하나(`uint32_t[kIndexGridCells]`, ~3MB)로 `indexGrid[((gz*50+gy)*50+gx)*6+direction] = poolIndex`를 표현한다 (문서 §6, §10 offset 공식 그대로). Local base가 바뀌든 안 바뀌든 **매 프레임** 다음 순서로 리셋·재구성한다(문서 §4 파이프라인은 조건부 스킵 없이 매 프레임 동일 절차를 돈다 — local base가 그대로면 classify 단계에서 자연히 전부 ReusableList로 재등록되어 missing 개수가 0에 수렴할 뿐이다):

```
1. indexGrid 전체를 kInvalidPoolIndex로 채운다 — vkCmdFillBuffer(indexGridBuffer, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu)
   (0xFFFFFFFF는 4바이트 반복 패턴이라 컴퓨트 셰이더 없이 vkCmdFillBuffer 한 번으로 끝난다)
2. classify.comp 실행 (아래)
3. register_reusable.comp 실행 (아래)
```

### 4. GPU 커널 (신규 셰이더 4개, `src/shader/directional_tsdf_*.comp`)

문서 §5(20B/32B 포맷 변환), §11/§12(decode/encode)에 해당하는 GPU 디코드/인코드 커널은 만들지 않는다 — `HostTsdfVoxel`↔`GpuTsdfVoxel` 변환(고정소수점 스케일 곱/나눗셈)은 group당 512개 float 연산이라 CPU에서 `Upload`/`Download` 직전/직후에 처리해도 충분히 저렴하다(우리 규모의 missing/writeback 그룹 수는 데모 기준 수백 개 이하). 이 판단은 "문서와의 의도적 차이점"에 다시 정리한다.

**4.1 `directional_tsdf_classify.comp`** (문서 §9)

- 입력: `ActiveGroupMeta[POOL_CAPACITY]`, push constant로 `localBase(gx,gy,gz)` + `poolCapacity`
- 출력: `ReusableList[]`/`CleanFreeList[]`/`WriteBackList[]` (각각 `uint32_t[POOL_CAPACITY]`) + `ListCounts{reusableCount, cleanFreeCount, writeBackCount}` (매 dispatch 전 0으로 리셋)
- 로직: 문서 §9 표 그대로 — `PendingUpload`/`PendingWriteBack` 슬롯은 스킵, `Free`/`invalid`는 CleanFreeList, local AABB 안이면 ReusableList, 밖이고 dirty면 WriteBackList, 밖이고 clean이면 CleanFreeList. `atomicAdd` 기반 (문서가 권장하는 초기 구현 그대로, `POOL_CAPACITY`=32,768 규모에서 contention은 허용 가능하다고 명시됨).

**4.2 `directional_tsdf_register_reusable.comp`** (문서 §10)

- 입력: `PoolIndexList[]`(임의의 poolIndex 목록), `count`, `ActiveGroupMeta[]`(이미 gx/gy/gz/direction이 채워져 있어야 함), `localBase`
- 출력: `indexGrid`에 `indexGrid[offset(lx,ly,lz,dir)] = poolIndex` 기록 + `meta.resident=1`
- `count`개 스레드로 병렬 실행 — 각 스레드가 서로 다른 grid cell에 쓰므로 atomic 불필요.
- **이 커널은 두 곳에서 재사용된다**: (a) §5 파이프라인 7번 — classify가 만든 `ReusableList`를 그대로 넣어 기존 resident 그룹을 재등록, (b) §5 파이프라인 11번 — missing-group 업로드로 새로 채워진 슬롯 인덱스 목록을 넣어 신규 그룹을 등록. 별도의 "missing 전용 등록 커널"은 만들지 않는다 — 두 경우 모두 "poolIndex 목록 → 그 슬롯의 meta를 읽어 indexGrid에 쓴다"는 동일한 로직이기 때문.

**4.3 `directional_tsdf_integrate.comp`** (문서 §13)

- 입력: `Points[]`(position+normal, N개), `indexGrid`, `activePoolVoxels[POOL_CAPACITY*512]`, `ActiveGroupMeta[]`, push constant(`N, localBase, voxelSize, truncation`)
- 로직: `SimpleTSDF`의 `voxel_tsdf_integrate.comp`와 동일한 ray-march + truncation band 순회를 direction별로 확장한 것 — 각 point마다 `direction = dominantAxis(normal)`을 한 번만 계산(초기 구현은 single-direction, 문서 §13과 동일), truncation band 안의 각 voxel에 대해 `indexGrid`로 해당 (group, direction)의 `poolIndex`를 찾고, 없으면(=아직 resident 안 됨) 스킵 — **invariant #7: integration은 IntegrationWriteSet(=이번 프레임에 resident로 만든 그룹)에 대해서만 write**. 찾으면 `atomicAdd(sumDW, ...)`/`atomicAdd(sumW, ...)`로 누적하고 `meta.dirty=1`, `meta.state=ResidentDirty` 마킹.
- `dominantAxis`: normal의 |x|,|y|,|z| 중 최댓값의 부호로 6방향 중 하나 선택.

**4.4 `directional_tsdf_extract.comp`** (문서 §15)

- 입력: "이번 프레임에 touch된 그룹 목록"(=IntegrationWriteSet, recomputeMask 정의 그대로 §14) 각각에 대해, 그 그룹의 512 voxel + 3×3×3 이웃(같은 direction layer만!)을 검사
- 로직: 문서 §15 규칙 그대로 — `TSDF(center,dir)` vs `TSDF(neighbor,dir)`만 비교, 다른 direction 간 비교 금지. Sign crossing이 있으면 후보 position(zero-crossing 보간)과 normal(같은 direction layer 내에서 유한차분으로 gradient 추정)을 계산해 `DirectionalCandidate`를 atomic append.
- 이웃 조회에 `indexGrid`가 필요하므로, 이 그룹들의 3×3×3 이웃도 모두 resident해야 한다 — 그래서 `ResidentRequiredSet`은 IntegrationWriteSet에 **+1 group halo**를 더한 것이다(문서 §7 그대로, invariant #8).

**구현 메모 — 불연속 슬롯 전송**: `Engine::Core::Buffer::Upload`/`Download`는 항상 버퍼 offset 0부터 전체(또는 앞부분)를 옮기는 API라, activePoolVoxels 안의 서로 떨어진 슬롯 여러 개에 동시에 쓰거나 읽는 용도로는 그대로 쓸 수 없다. missing-group 업로드(파이프라인 11번)와 dirty write-back 다운로드(17번)는 `Buffer`의 편의 메서드를 우회해 직접 처리한다: 임시 staging `Engine::Core::Buffer`(또는 raw VMA 버퍼)를 만들어 CPU 데이터를 그 안에 연속으로 채운 뒤, `Engine::Core::SubmitOneShot`으로 감싼 커맨드버퍼 안에서 `vkCmdCopyBuffer(cmd, staging.Handle(), activePoolVoxels.Handle(), regionCount, regions)`를 슬롯 개수만큼의 region으로 한 번에 호출한다 (region 목록은 10번에서 이미 CPU가 들고 있는 슬롯 인덱스로 구성). `indexGrid` 전체 리셋(4번, `vkCmdFillBuffer`)도 같은 이유로 `Buffer`의 고수준 API가 아니라 `activePoolVoxels`/`indexGrid`의 `Handle()`을 직접 쓰는 raw Vulkan 호출이 필요하다.

### 5. Frame 파이프라인 (`DirectionalTSDF::Integrate`)

CPU/GPU 작업을 명시적으로 나눈다.

```
Integrate(points, normals, cameraPos, aabbCenterHint):
  (CPU) 1. localBase = quantize(aabbCenterHint)                     // group-grid 정수 좌표로 스냅
  (CPU) 2. IntegrationWriteSet 계산: 각 point의 world-space truncation-band AABB를
            group-grid 좌표 범위로 변환해 겹치는 모든 (gx,gy,gz,direction) 그룹 키를 모음
            (direction은 point의 dominantAxis(normal))
  (CPU) 3. ResidentRequiredSet = IntegrationWriteSet을 모든 축으로 1 group씩 팽창(halo)
  (GPU) 4. indexGrid 전체 리셋 (vkCmdFillBuffer)
  (GPU) 5. classify.comp 실행 → ReusableList/CleanFreeList/WriteBackList
  (CPU) 6. ListCounts 다운로드 (12B, 저렴)
  (GPU) 7. register_reusable.comp 실행 (ReusableList 전체 재등록)
  (CPU) 8. ResidentRequiredSet 중 indexGrid에 없는 키만 MissingGroupList로 추림
            — **주의(invariant #6): 6번(재등록) 이후에만 판정한다.**
  (CPU) 9. MissingGroupList 각각을 host store에서 조회(없으면 GetOrCreate로 빈 그룹) →
            HostTsdfVoxel[512] → GpuTsdfVoxel[512]로 변환해 하나의 staging Buffer에 패킹
  (CPU/GPU) 10. CleanFreeList 다운로드 → 앞에서부터 missing 개수만큼 슬롯 할당
            (부족하면 WriteBackList에서 마저 확보 — pool shortage 대응, 문서 §11)
  (GPU) 11. staging Buffer → activePoolVoxels의 (불연속) 할당 슬롯들로 vkCmdCopyBuffer
            (region 여러 개, 슬롯 인덱스는 10번에서 이미 CPU가 알고 있음) + ActiveGroupMeta
            갱신(state=ResidentClean, valid=1, gx/gy/gz/direction 기록) + 4.2의
            register_reusable.comp를 이번엔 "새로 채워진 슬롯 목록"으로 재실행해 indexGrid 등록
  (GPU) 12. integrate.comp 실행 (points 전체, N개 스레드)
  (CPU) 13. recomputeMask = IntegrationWriteSet (§14, 초기 구현은 group 단위 conservative)
  (GPU) 14. extract.comp 실행 (recomputeMask 그룹들 + halo)
  (CPU) 15. candidate 다운로드 → position/normal 각도 임계값 기반 merge/split (§6 참고,
            문서 §15의 candidate merge/split 정책 표 그대로) → 새 ExtractedPoint 목록 생성
  (CPU) 16. m_pointCloud에서 ownerGroup이 recomputeMask 안에 있는 기존 점을 제거하고
            15번의 새 점을 추가 (invariant #9)
  (CPU) 17. WriteBackList 중 아직 슬롯이 필요해 evict된 dirty 그룹들을 다운로드 →
            GpuTsdfVoxel[512] → HostTsdfVoxel[512]로 변환해 host store에 Put, slot=Free
```

전부 동기 호출(각 GPU 단계는 `Engine::Core::Buffer::Upload/Download`/`ComputePipeline::Dispatch`가 이미 blocking) — Phase 5로 미룬 비동기화 전까지는 이 순서를 그대로 따른다.

### 6. Candidate merge/split (CPU, `DirectionalTSDF` 내부 헬퍼)

문서 §15의 candidate merge/split 정책 표와 threshold 파라미터를 그대로 구현한다: `positionMergeThreshold`(기본 0.6 voxel), `normalMergeThreshold`(30°), `strongSplitThreshold`(60°), voxel당 cluster 최대 6개. 같은 spatial voxel(=같은 gx,gy,gz, 다른 direction)에서 나온 candidate들을 모아 인접 candidate끼리 position+normal 각도로 비교해 병합(weighted average) 또는 별도 유지 여부를 결정한다. GPU 병렬 클러스터링 대신 CPU에서 처리하는 이유: 후보 수는 표면 근처에서만 생기므로(spatial voxel당 최대 6개) 다운로드 비용이 작고, threshold 튜닝/디버깅이 CPU 코드에서 훨씬 쉽다.

### 7. Public API

```cpp
class DirectionalTSDF {
public:
    void Build(Engine::Core::Context &ctx,
               float voxelSize = 0.1f, float truncation = 0.3f,
               uint32_t poolCapacity = 32768);

    void Integrate(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    const Eigen::Vector3f &cameraPos,
                    const Eigen::Vector3f &aabbCenterHint);

    const std::vector<ExtractedPoint> &PointCloud() const;
    void ExportPointCloud(const std::string &path) const; // PLY, normal 포함

    struct Stats {
        uint32_t residentCount, missingCount, writeBackCount;
        uint32_t h2dBytes, d2hBytes;
        float overlapRatio; // reusableCount / residentRequiredSetSize
    };
    Stats LastFrameStats() const;
};
```

`confidence`(문서 `IntegrationSample.confidence`)는 넣지 않는다 — 문서 자신이 "초기 sampleWeight=1, confidence는 향후 확장"이라 명시했으므로 YAGNI로 제외.

## 문서와의 의도적 차이점

| 항목 | 문서 | 이번 구현 | 이유 |
|---|---|---|---|
| Voxel format | 20B(host)/32B(gpu), oct16 normal, colorMap, patchID | 8B(host)/8B(gpu), normal 없음 | color/segmentation/patch 개념이 이 프로젝트엔 없음(YAGNI). normal은 저장 대신 extraction 시 gradient로 계산(표준 TSDF 기법) |
| Voxel update 연산 | 순차 running-average 공식 | `SimpleTSDF`와 동일한 고정소수점 `sumDW`/`sumW` atomic 누적 | 같은 프레임 안에서 여러 point가 같은 voxel을 동시에 건드릴 수 있어 GPU에서는 atomic이 필요 — GLSL은 float atomicAdd가 없어 정수 고정소수점이 필요(이미 검증된 패턴 재사용) |
| Decode/encode kernel | GPU 전용 커널 | CPU에서 Upload/Download 전후 변환 | group당 512 float 연산, 우리 규모(missing/writeback 수백 개 이하)에선 GPU 커널을 새로 짤 이유가 없음 |
| ResidentRequiredSet 계산 | GPU 병렬 (또는 미명시) | CPU (points가 이미 host 메모리에 있으므로) | 점 개수가 수천~수만 개 수준이라 CPU 계산이 마이크로초 단위, GPU 동적 리스트 빌딩(prefix-sum 등)의 복잡도를 피함 |
| IntegrationWriteSet 모양 | 50×50×3 slab 가정(스캐너 특유의 평면형 표면) | 점별 truncation-band AABB → group 범위 (일반적인 도형에도 정확) | 우리 데모는 특정 스캐너 형상을 가정하지 않음 |
| Bandwidth/latency 목표 수치(§16~19) | 0.32GB/s 등 구체적 수치 | 수치 목표 없음, "reuse로 missing 비율이 낮게 유지된다"는 정성적 결론만 재현 | byte 크기가 다르므로 문서 수치가 그대로 적용되지 않음 |
| Extraction 산출물 | point cloud (marching cubes 아님) | 동일 | 문서 §15 그대로 |

## 파일 변경 목록

| 파일 | 변경 |
|---|---|
| `src/Engine/Spatial/DirectionalTSDFTypes.h` (신규) | 공유 타입/상수 |
| `src/Engine/Spatial/DirectionalHostStore.h`/`.cpp` (신규) | CPU sparse store |
| `src/Engine/Spatial/DirectionalTSDF.h`/`.cpp` (신규) | 메인 오케스트레이터 |
| `src/shader/directional_tsdf_classify.comp` (신규) | §9 active pool 분류 |
| `src/shader/directional_tsdf_register_reusable.comp` (신규) | §10 indexGrid 재등록 |
| `src/shader/directional_tsdf_integrate.comp` (신규) | §13 dominant-direction 적분 |
| `src/shader/directional_tsdf_extract.comp` (신규) | §15 방향별 candidate 추출 |
| `example2/directional_tsdf_demo.cpp` (신규) | interproximal 시나리오 + 스크립트 경로 데모 |
| `example2/CMakeLists.txt` | 데모 실행 파일 추가 |
| `test/test_directionalTSDF.cpp` (신규) | invariant/유닛 테스트 |
| `test/CMakeLists.txt` | 새 테스트 파일 반영(GLOB이라 보통 무변경) |

`SimpleTSDF`, `vkSpatial`, `vkCommon`, `vkRender`는 이 스펙에서 변경하지 않는다.

## CMake 통합

기존 `EngineSpatial` 타겟(`src/Engine/CMakeLists.txt`)이 이미 `${CMAKE_CURRENT_SOURCE_DIR}/Spatial/*.cpp`를 `GLOB_RECURSE`로 수집하므로 새 `.cpp`를 `src/Engine/Spatial/` 밑에 추가하기만 하면 빌드에 자동 포함된다 — CMake 파일 자체는 변경 불필요. `example2/CMakeLists.txt`에는 `directional_tsdf_demo` 실행 파일 블록을 `voxel_tsdf_mc`와 같은 패턴으로 추가한다:
```cmake
add_executable(directional_tsdf_demo directional_tsdf_demo.cpp)
target_link_libraries(directional_tsdf_demo PRIVATE Engine::Spatial)
target_compile_definitions(directional_tsdf_demo PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")
```

## 예제/테스트 계획

**`example2/directional_tsdf_demo.cpp`**: 서로 마주보는 normal을 가진 두 개의 얇은 평행면(치아 사이 간격 모사)을 합성하고, 카메라가 스크립트된 경로(연속 프레임 간 이동량을 작게 유지)를 따라 이동하며 점진적으로 스캔. `aabbCenterHint`는 각 프레임에서 보이는 점들의 중심. 매 프레임 `LastFrameStats()`를 로그로 남기고(overlap ratio, missing count, h2d/d2h bytes), 마지막에 `DirectionalTSDF::ExportPointCloud`와 `SimpleTSDF::ExportMC`(또는 export 가능한 형태) 결과를 각각 PLY로 저장해 육안 비교 가능하게 한다.

**`test/test_directionalTSDF.cpp`** (GTest): 실제 GPU 컨텍스트 필요, 헤드리스로 동작.
1. `DirectionalGroupKeyTest.*` — 키 해시/동등성
2. `DirectionalHostStoreTest.*` — Contains/Get/Put 왕복 (GPU 불필요, 가장 빠른 유닛 테스트)
3. `DirectionalTSDFTest.SingleFrameUploadRegistersIndexGrid` — 점 하나만 있는 프레임 이후 indexGrid 조회가 올바른 poolIndex를 반환하는지
4. `DirectionalTSDFTest.RepeatedFrameSameAABBReusesEverything` — 같은 aabbCenterHint로 두 번 Integrate 호출 시 두 번째 프레임의 missingCount==0, overlapRatio==1.0
5. `DirectionalTSDFTest.DuplicateActiveSlotNeverOccurs` (invariant #1) — 여러 프레임 진행 후 동일 `DirectionalGroupKey`를 가진 active slot이 중복 없는지 전수 검사
6. `DirectionalTSDFTest.EvictedGroupReloadsFromHostStore` (Phase 4 완료 후) — AABB가 멀리 이동했다 되돌아왔을 때 이전에 적분한 값이 host store를 거쳐 정확히 복원되는지

## 구현 순서 (Phase 1~4)

문서의 Phase 구분을 그대로 구현 단계로 쓰고, 각 Phase를 별도 구현 계획 문서(`docs/superpowers/plans/`)로 분리해 단계마다 리뷰 체크포인트를 둔다.

1. **Phase 1 — 기본 streaming cache**: 데이터 구조(아키텍처 §1)/HostStore(§2)/indexGrid(§3) 정의, active pool 할당, missing-group H2D 업로드(CPU 디코드) + `register_reusable.comp`(§4.2)로 indexGrid 등록까지 (local base 이동에 따른 재사용 분류, integration, extraction은 아직 없음 — 매 프레임 전부 missing으로 취급해도 무방). 검증: 단일 프레임에서 업로드된 그룹이 indexGrid로 정확히 조회되는지.
2. **Phase 2 — local base 이동과 reuse**: `classify.comp`(§4.1) 추가, Reusable/CleanFree/WriteBackList, 같은 AABB 반복 시 `register_reusable.comp`가 신규 슬롯 대신 재사용 슬롯을 등록하는지 확인. 검증: AABB 이동 시 재사용 카운트가 기대치와 일치.
3. **Phase 3 — integration/extraction 연동**: integrate.comp(dominant direction), dirty mark, recomputeMask, extract.comp, candidate merge/split, old point 병합. 검증: interproximal 데모로 정성적(PLY 육안)+정량적(포인트 수, 두 표면이 분리 유지되는지) 확인.
4. **Phase 4 — write-back/eviction**: dirty write-back, pool shortage 대응. 검증: evict된 영역으로 AABB가 돌아왔을 때 host store에서 정확히 복원.

각 Phase 종료 후 다음 Phase로 넘어가기 전에 리뷰를 받는다.

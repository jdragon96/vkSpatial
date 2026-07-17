# Directional TSDF Streaming 설계 문서 분석

> 원본: `docs/DirectionalTSDF.pdf` (SP1팀 이재필, intraoral scanner reconstruction 설계 문서)
> 이 문서는 원본 PDF(총 23개 섹션)의 내용을 마크다운으로 정리·분석한 것이다.

## 1. 배경과 목적

Intraoral scanner reconstruction을 위한 **Directional TSDF의 GPU streaming 구조**를 정의하는 문서다.

기존 구현은 GPU-only storage 방식으로, 전체 hash data를 GPU 메모리에 상주시켰다. 문제는 Directional TSDF가 기존 TSDF 대비 훨씬 큰 메모리를 요구한다는 점이다 (§2, §3 참고). 실측 그래프(ActiveBlocks over Frames)에서도 Directional TSDF의 active block 수가 Original TSDF보다 최대 약 1.7배 많게 나타난다. GPU 메모리만으로는 사실상 구현이 불가능한 수준이다.

**해결 방향**: GPU + CPU hybrid storage. Active region의 SDF block만 GPU에 유지하고, 나머지는 CPU(host)에 hash data로 보관한다.

### 설계 목표 (7가지)

1. 40mm × 40mm × 40mm local reconstruction window를 GPU에서 처리
2. TSDF voxel data는 sparse active pool로 유지
3. dense 구조는 voxel data가 아니라 indexGrid에만 사용
4. frame 간 AABB overlap을 이용해 이미 GPU에 resident한 group을 재사용
5. Host→Device transfer는 MissingGroupList에 대해서만 수행
6. Device→Host write-back은 dirty group 중 eviction 대상에 대해서만 수행
7. Integration과 Extraction이 필요한 group만 resident 유지

### 핵심 수식

```
Actual H2D upload = ResidentRequiredSet - AlreadyResidentGroups
```

매 frame 전체를 업로드하지 않고, 이미 GPU active pool에 있는 group을 최대한 재사용한다는 것이 이 설계의 핵심 아이디어다.

## 2. Directional TSDF를 도입하는 이유

**기존 single SDF의 한계**: 하나의 voxel 위치에는 하나의 SDF 값만 저장된다. 치아처럼 얇은 구조나 인접면(interproximal)에서는 서로 다른 surface가 같은 voxel 영역에 반영되어, 두 표면이 섞이거나 붙어버리는 현상이 발생한다.

**핵심 아이디어**: 같은 위치라도 normal direction이 다르면 별도의 SDF로 관리한다. 즉 하나의 3D 위치에 대해 **6개의 canonical direction(±X, ±Y, ±Z)** 을 정의하고, normal과 가장 가까운 방향의 voxel에 SDF를 저장한다. 인접한 표면을 독립적으로 integration함으로써 surface interference를 완화한다.

문서는 관측 방향과 무관하게 TSDF value를 업데이트할 때 경계 부분에서 형상 왜곡이 발생하는 예시를 표(2D grid의 signed distance 값)로 보여준다. 노말 방향이 서로 상쇄되는 voxel 위치에서는 값이 상쇄되어 관측 결과가 사라지는 현상이 나타나고, 이는 interproximal이나 얇은 면에서 구멍 등의 문제를 일으킨다. 이는 술자(시술자)에 따른 결과 편차, 치아 위치에 따른 왜곡 정도 차이의 원인이며, 근본 시스템(단일 SDF field) 변경 없이는 해결 불가하다고 결론짓는다.

## 3. 기본 파라미터


| 항목                          | 값                         | 설명                                |
| --------------------------- | ------------------------- | --------------------------------- |
| Local AABB 크기               | 40mm × 40mm × 40mm        | GPU에서 한 번에 처리하는 local TSDF window |
| Voxel 크기                    | 0.1mm (100μm)             | Reconstruction spatial resolution |
| Voxel group 크기              | 8×8×8 voxels (512개)       | 최소 관리 단위                          |
| Voxel group 실제 크기           | 0.8mm × 0.8mm × 0.8mm     | 8 voxels × 0.1mm                  |
| Local group grid            | 50×50×50                  | 40mm / 0.8mm                      |
| Direction 수                 | 6                         | ±X, ±Y, ±Z                        |
| Logical directional group 수 | 750,000                   | 50×50×50×6                        |
| GPU indexGrid 크기            | 약 3MB                     | 750,000 × 4B                      |
| Active pool capacity        | 32,768 directional groups | 초기 권장값                            |
| GPU active pool memory      | 약 537MB                   | 32,768 × 512 × 32B                |


**Dense voxel 방식이 불가능한 이유**를 정량적으로 보여준다:

- 40mm / 0.1mm = 400 voxels → 400³ = 64,000,000 voxels
- Directional 6 layers = 384,000,000 directional voxels
- GPU active voxel이 32B라면: 384,000,000 × 32B = **12.288GB**

→ 부가 buffer까지 고려하면 dense voxel data 방식은 불가능하거나 매우 비효율적. 따라서:

```
Dense:  indexGrid[50][50][50][6] → poolIndex   (약 3MB, 위치 인덱스만)
Sparse: activePool[POOL_CAPACITY] → actual directional TSDF groups (실제 데이터)
```

## 4. 전체 구조 요약

핵심 개념: local TSDF window가 ICP 이후 조금씩 이동하며, t와 t+1의 local AABB는 대부분 overlap된다. Non-overlap 영역만 missing groups로 취급한다.

**Directional TSDF frame pipeline** (device side):

```
ICP/R,t → Patch AABB → Required Set → Resident Check → Missing Upload → Integration → Extraction → Point Cloud
```

Host store는 MissingGroupList에 대해서만 packed 20B/voxel group을 준비한다.

전체 frame pipeline (상세):

```
Input depth/surface → ICP/Tracking → Current frame patch 생성 → Patch AABB 계산
  → IntegrationWriteSet 생성 → ExtractionReadHaloSet 생성
  → ResidentRequiredSet = IntegrationWriteSet ∪ ExtractionReadHaloSet
  → indexGrid reset → Active pool classification → Reusable group re-registration
  → MissingGroupList 생성 → Host sparse store에서 missing group upload
  → GPU decode + active pool registration → TSDF integration
  → dirty mark + recomputeMask 생성 → Extraction → Old point merge
  → Final point cloud 생성 → Dirty group write-back / eviction
```

## 5. Voxel format 설계

Host↔Device transfer는 bandwidth 절감을 위해 packed format을 사용한다. 세 가지 포맷이 존재한다: 기존 voxel(약 40B), Host transfer voxel(20B), GPU active voxel(32B).

```cpp
// Host transfer voxel (20B) — bandwidth 최적화
struct HostTsdfVoxel {
    float value;              // 4B
    uint32_t packedNormal;    // 4B, oct16x2
    uint32_t flagAndLabel;    // 4B
    uint16_t valueCnt;        // 2B
    uint16_t startPatchID;    // 2B
    uint8_t colorMap[4];      // 4B
}; // 20B

// GPU active voxel (32B) — 연산 최적화 (normal을 float3으로 unpack)
struct GpuTsdfVoxel {
    float value;               // 4B
    float normalX, normalY, normalZ; // 12B
    uint32_t flagAndLabel;     // 4B
    uint16_t valueCnt;         // 2B
    uint16_t startPatchID;     // 2B
    uint8_t colorMap[4];       // 4B
    uint32_t reserved;         // 4B
}; // 32B
```

기존 voxel 대비 color RGB(3B), segmentation(1B), dlClass(1B), materialID(1B)는 streaming path에서 제외된다(reconstruction에 불필요).

**Bandwidth 절감 효과**: Host packed format을 20B로 줄이면 기존 aligned 40B 대비 **50% 절감**된다.


| 기준                | 기존 aligned 40B | Host packed 20B | 절감률 |
| ----------------- | -------------- | --------------- | --- |
| voxel당 size       | 40B            | 20B             | 50% |
| group당 size       | 20,480B        | 10,240B         | 50% |
| 20K groups/update | 409.6MB        | 204.8MB         | 50% |
| 20K groups @30Hz  | 12.288GB/s     | 6.144GB/s       | 50% |


## 6. Directional group과 indexGrid

```cpp
struct DirectionalGroupKey {
    int32_t gx, gy, gz;
    uint8_t direction; // +X,-X,+Y,-Y,+Z,-Z
};
```

GPU는 global group key를 local grid 좌표로 변환해 indexGrid에 접근한다 (`localG = globalG - localBase`). `indexGrid[lx][ly][lz][direction] = poolIndex`이며, 없는 group은 `INVALID_POOL_INDEX (0xFFFFFFFF)`로 표시한다.

indexGrid는 **local base에 종속적**이므로, local base가 바뀌면 반드시 다음 순서로 재구성해야 한다:

```
new local base 결정 → indexGrid 전체 INVALID로 reset → active pool classification
  → ReusableList를 indexGrid에 재등록
```

## 7. IntegrationWriteSet과 ResidentRequiredSet

TSDF update는 surface 주변 truncation band에 대해서만 수행한다.

- truncation distance = ±1mm
- total integration band thickness = 2mm
- group size = 0.8mm → integration group layers = ceil(2mm/0.8mm) = **3**
- Extraction은 3×3×3 neighborhood를 참조하므로 halo가 필요 → extraction halo = +1 group layer
- **resident band = integration 3 layers + halo 1 layer = 4 layers**

서강대학교 구현 실험 결과, 평균 활성 direction 수는 **2 미만**으로 나타난다 (그래프 참고, 약 1.7~1.9 수준으로 수렴).

평균 활성 direction 수를 2로 가정한 계산:


| Set                       | 계산식           | Directional group 수 | 용도                                     |
| ------------------------- | ------------- | ------------------- | -------------------------------------- |
| IntegrationWriteSet       | 50×50×3×2     | 15,000              | 실제 TSDF value 업데이트                     |
| ExtractionReadHaloSet 추가분 | 50×50×1×2     | 5,000               | extraction neighborhood read-only halo |
| ResidentRequiredSet       | 50×50×4×2     | 20,000              | integration+extraction 전 resident 필요   |
| Active pool capacity      | —             | 32,768              | resident set + reuse margin            |
| Reuse margin              | 32,768-20,000 | 12,768              | local AABB 이동 시 여유                     |


→ worst resident set은 20K directional groups, active pool은 32K 수준으로 잡는다.

## 8. Active pool metadata

```cpp
enum SlotState : uint8_t {
    Slot_Free = 0,
    Slot_ResidentClean = 1,
    Slot_ResidentDirty = 2,
    Slot_PendingUpload = 3,
    Slot_PendingWriteBack = 4,
};

struct ActiveGroupMeta {
    int32_t globalGx, globalGy, globalGz;
    uint8_t direction;
    uint8_t valid;
    uint8_t dirty;
    uint8_t resident;
    uint8_t state;
    uint32_t lastUsedFrame;
};
```

### 상태 전이표


| From                | Event                 | To                  | 설명                   |
| ------------------- | --------------------- | ------------------- | -------------------- |
| Free                | missing group slot 할당 | PendingUpload       | Host group upload 예정 |
| PendingUpload       | H2D + decode 완료       | ResidentClean       | indexGrid 등록 가능      |
| ResidentClean       | integration update 발생 | ResidentDirty       | dirty mark           |
| ResidentClean       | AABB 밖으로 나감           | Free                | write-back 없이 즉시 재사용 |
| ResidentDirty       | AABB 밖으로 나감           | PendingWriteBack    | Host store에 반영 필요    |
| PendingWriteBack    | encode + D2H 완료       | Free                | slot 재사용 가능          |
| ResidentClean/Dirty | 새 AABB 안에 유지          | ResidentClean/Dirty | reusable로 재등록        |


## 9. Device-side active group classification

Active pool은 GPU device memory에 있으므로, local base 변경 후 어떤 group이 재사용 가능한지 판단하는 작업도 device-side에서 수행한다.

```
Resident inside AABB → ReusableList → indexGrid re-register
Clean outside or invalid → CleanFreeList → Missing group upload slot
Dirty outside AABB → WriteBackList → Encode+D2H then Free
```

(PendingUpload/PendingWriteBack 슬롯은 CleanFreeList에서 제외됨)


| List          | 생성 조건                                  | Counter        | 용도                        |
| ------------- | -------------------------------------- | -------------- | ------------------------- |
| ReusableList  | valid &amp;&amp; inside new local AABB | reusableCount  | indexGrid 재등록             |
| CleanFreeList | invalid 또는 clean outside AABB          | cleanFreeCount | missing group upload slot |
| WriteBackList | dirty outside AABB                     | writeBackCount | D2H write-back 대상         |


초기 구현은 가장 단순한 **atomic counter 방식**을 사용한다 (`classifyActiveGroupsAtomicKernel`, `POOL_CAPACITY = 32,768` 수준이므로 atomic contention은 허용 가능한 비용으로 간주). 성능 문제가 profile에서 확인되면 다음으로 교체 가능:

1. CUB DeviceSelect
2. prefix-sum 기반 compaction
3. block-local aggregation 후 global atomic
4. Thrust `copy_if` 기반 prototype

## 10. indexGrid 재등록 순서

local base가 바뀐 frame에서 순서는 반드시 다음과 같아야 한다:

```
1. new local base 결정
2. indexGrid 전체 INVALID 초기화
3. activeGroupMeta classification
4. ReusableList를 indexGrid에 재등록
5. ResidentRequiredSet과 indexGrid를 비교
6. MissingGroupList 생성
7. CleanFreeList를 이용해 upload slot 할당
```

**중요**: ReusableList 재등록 전에 missing 판정을 하면 안 된다. 이미 GPU active pool에 resident한 group도 indexGrid가 INVALID 상태이므로 missing으로 오판될 수 있기 때문이다.

설계 invariant: 동일한 `(globalGx, globalGy, globalGz, direction)`에 대해 active pool slot이 중복 존재하면 안 된다.

## 11. MissingGroup upload

`ResidentRequiredSet`을 생성한 뒤 indexGrid를 조회하여 `poolIndex == INVALID_POOL_INDEX`이면 MissingGroupList에 추가한다. 이후 CleanFreeList에서 slot을 할당한다. 만약 missingCount가 cleanFreeCount를 초과하면, 부족한 slot 수만큼 WriteBackList에서 dirty eviction을 수행한다.

```cpp
struct UploadTask {
    DirectionalGroupKey key;
    uint32_t poolIndex;
    uint32_t stagingIndex;
};
```

H2D flow:

```
Host sparse TSDF store → Host packed staging buffer → cudaMemcpyAsync H2D
  → GPU packed staging buffer → decode kernel → GPU active pool 32B voxel format
  → ActiveGroupMeta update → indexGrid registration
```

chunk size 예시: 4096 groups × 10,240B ≈ 41.9MB → 20K group full upload는 약 5 chunks.

## 12. Dirty write-back

Dirty group은 Host authoritative sparse TSDF store로 write-back되어야 한다. write-back 대상:

1. dirty &amp;&amp; outside new local AABB
2. pool shortage로 eviction이 필요한 dirty group
3. forced flush 대상

```cpp
struct WriteBackTask {
    DirectionalGroupKey key;
    uint32_t poolIndex;
    uint32_t stagingIndex;
};
```

D2H flow:

```
GPU active pool 32B voxel format → encode kernel → GPU packed staging buffer 20B/voxel
  → cudaMemcpyAsync D2H → Host packed staging buffer
  → Host authoritative sparse TSDF store update → slot state = Free
```

중요 invariant: dirty group은 write-back 완료 전까지 free slot으로 재사용하면 안 되며, write-back 시작 시 slot state는 반드시 `Slot_PendingWriteBack`이어야 한다.

## 13. TSDF Integration

```cpp
struct IntegrationSample {
    float3 position;
    float3 normal;
    float confidence;
};
```

초기 direction selection은 sample normal의 dominant axis를 사용 (`direction = dominantAxis(sample.normal)`). 초기 구현은 single direction update만 사용하며, 이후 normal이 axis 경계에 가까운 경우 multi-direction update를 고려할 수 있다.

TSDF update 식 (running weighted average):

```
newValue = clamp(signedDistance / truncationDistance, -1.0, 1.0)
voxel.value = (voxel.value * voxel.valueCnt + newValue * sampleWeight) / (voxel.valueCnt + sampleWeight)
voxel.valueCnt = min(voxel.valueCnt + sampleWeight, maxValueCnt)
```

초기 `sampleWeight = 1`. 향후 확장에서는 `frontendConfidence × viewAngleWeight × directionConfidence × optionalDistanceWeight`를 quantize하여 반영한다.

Integration kernel 수행 항목:

1. IntegrationWriteSet 내부 group만 update
2. TSDF value update
3. valueCnt update
4. normal update
5. flag update
6. dirty mark
7. recomputeMask update

## 14. recomputeMask와 old point merge

Extraction은 매 frame 전체 point cloud를 재생성하지 않고, integration으로 영향을 받은 영역만 recompute한다.

```
recomputeMask = 이번 frame에서 point를 다시 계산해야 하는 spatial voxel 영역

if oldPoint.ownerVoxelCoord ∈ recomputeMask: old point drop
else: old point keep

finalPointCloud = keptOldPoints + newlyExtractedPoints
```

recomputeMask는 단순히 patch AABB 전체가 아니라, 실제 update된 voxel range를 기준으로 만들어야 한다 (다만 초기 구현은 conservative하게 group 단위 recomputeMask 사용 가능). 정확한 old point 제거를 위해 extracted point는 owner voxel coordinate를 저장한다.

```cpp
struct ExtractedPoint {
    float3 position;
    float3 normal;
    int3 ownerVoxelCoord;
    uint8_t clusterIndex;
    uint8_t dirMask;
    uint16_t flags;
    float score;
};
```

## 15. Directional TSDF Extraction

기존 extraction 방식은 marching cubes triangle extraction이 아니라 **center voxel 기준 point extraction**이다. 각 center voxel에 대해 3×3×3 neighborhood(center + 26 neighbors)를 검사한다.

핵심 규칙: **같은 direction layer끼리만 비교**한다.

- `TSDF(center, direction)` vs `TSDF(neighbor, direction)` — 허용
- `TSDF(center, dirA)` vs `TSDF(neighbor, dirB)` — **금지** (서로 다른 direction layer 간 sign crossing 비교 안 함)

```cpp
struct DirectionalCandidate {
    float3 position;
    float3 normal;
    float score;
    float confidence;
    uint8_t direction;
    uint8_t valid;
    uint16_t flags;
};
```

하나의 spatial voxel에서 최대 6개의 directional candidate가 나올 수 있다.

**품질 이슈**: 여러 direction의 candidate를 아무 조건 없이 모두 허용하면 표면 생성이 매끄럽지 않게 나올 수 있음을 실험(포인트클라우드 시각화)으로 확인했다. 적절한 기준으로 blending하면 더 매끄러운 표면을 생성함을 확인했다.

### Candidate merge/split 정책


| 조건                             | 판단                  | 동작                     |
| ------------------------------ | ------------------- | ---------------------- |
| valid candidate 없음             | surface 없음          | point 생성 안 함           |
| candidate 1개                   | 단일 surface          | point 1개 생성            |
| position 가까움 + normal angle 작음 | 같은 surface          | weighted merge         |
| position 가까움 + normal angle 큼  | 다른 surface 가능성      | 별도 cluster             |
| position 멀음                    | 다른 surface 또는 noise | 별도 cluster 또는 reject   |
| normal angle &gt; 60°          | strong split        | 다른 surface로 분리         |
| voxel 내 cluster 수 &gt; 6       | 제한                  | direction 수가 6이므로 최대 6 |


### 초기 threshold 값


| 파라미터                   | 초기값            | 설명                               |
| ---------------------- | -------------- | -------------------------------- |
| positionMergeThreshold | 0.5~0.75 voxel | 같은 surface candidate merge 거리    |
| normalMergeThreshold   | 30°            | 같은 surface로 볼 normal angle       |
| strongSplitThreshold   | 60°            | 다른 surface로 강하게 판단할 normal angle |
| thresholdWeight        | 1.25           | candidate valid 판단 threshold     |
| weakFloor              | 0.125          | lateral crossing의 최소 evidence    |


normal 방향은 TSDF sign convention 또는 view direction 기준으로 일관되게 맞춰야 한다. 초기 구현에서는 필요 시 `abs(dot(n1, n2))`를 임시로 사용할 수 있지만, 장기적으로는 normal orientation을 정리해야 한다.

## 16. AABB overlap 실험 결과

실제 `localRegistrationMC` 로그의 AABB min/max 값을 이용해 연속 frame 사이 local AABB overlap을 분석했다.

overlap ratio 계산식 (40mm fixed AABB volume overlap):

```
overlapRatio = max(0, 40-|dx|) × max(0, 40-|dy|) × max(0, 40-|dz|) / 40³
```

group-grid snap 기준 (50×50×50 groups):

```
overlapGroupRatio = (50-|Δgx|) × (50-|Δgy|) × (50-|Δgz|) / 50³
```

### 실험 결과


| 항목                        | 값        |
| ------------------------- | -------- |
| Group-grid 평균 overlap     | 약 94.8%  |
| Group-grid median overlap | 약 96.0%  |
| 평균 center displacement    | 약 1.55mm |
| p95 center displacement   | 약 4.32mm |


→ 실제 연속 frame에서는 40mm local AABB가 평균적으로 **약 95% 가까이 overlap**된다. 이는 active pool reuse 전략이 매우 효과적일 수 있음을 의미한다.

## 17. Bandwidth 추정

조건: Update rate = 30Hz, Target PCIe practical bandwidth = 12GB/s, ResidentRequiredSet = 20,000 groups, Host packed group size = 10,240B.

평균 overlap이 약 94.8%라면 non-overlap(missing)은 약 5.2%다.

```
missingGroupsMean = 20,000 × 0.052 ≈ 1,040 groups/update
streamInMean = 1,040 × 10,240B ≈ 10.6MB/update
H2D @30Hz = 10.6MB × 30 ≈ 318MB/s ≈ 0.32GB/s
```


| 기준          | Overlap | Missing ratio | Missing groups | H2D/update | H2D@30Hz   |
| ----------- | ------- | ------------- | -------------- | ---------- | ---------- |
| 평균          | 약 94.8% | 약 5.2%        | 약 1.0K         | 약 10.6MB   | 약 0.32GB/s |
| p05 수준 큰 이동 | 약 86.2% | 약 13.8%       | 약 2.8K         | 약 28MB     | 약 0.84GB/s |
| 로그 worst    | 약 78.8% | 약 21.2%       | 약 4.2K         | 약 43MB     | 약 1.3GB/s  |
| full reload | 0%      | 100%          | 20K            | 204.8MB    | 6.14GB/s   |


→ full reload와 비교하면, 평균 steady-state H2D는 full reload 대비 **약 5% 수준**(0.32/6.14 ≈ 5.2%)에 불과하다.

## 18. Stream out 크기 및 latency 추정

stream out은 local AABB 밖으로 나가면서 dirty인 group에 대한 D2H write-back이다. 가정: `stream out size = stream in size × 0.8` (AABB 밖으로 빠지는 group 중 약 20%는 clean/read-only라 버릴 수 있고, 80%만 write-back).


| 기준          | Stream In     | Stream Out (×0.8) | Total Transfer | Sequential latency @12GB/s |
| ----------- | ------------- | ----------------- | -------------- | -------------------------- |
| 평균          | 약 10.6~11.1MB | 약 8.5~8.9MB       | 약 19~20MB      | 약 1.6~1.7ms                |
| p05 큰 이동    | 약 28MB        | 약 22MB            | 약 50MB         | 약 4.2ms                    |
| 로그 worst    | 약 43MB        | 약 34MB            | 약 77MB         | 약 6.4ms                    |
| full reload | 204.8MB       | 163.8MB           | 368.6MB        | 약 30.7ms                   |


H2D/D2H를 별도 copy engine으로 overlap 가능하다면 latency는 합이 아니라 max에 가까워진다 (`overlappedLatency ≈ max(streamIn, streamOut) / bandwidth`, stream out이 stream in의 80%이므로 stream in이 지배적).


| 기준          | Ideal overlapped latency @12GB/s |
| ----------- | -------------------------------- |
| 평균          | 약 0.9~1.0ms                      |
| p05 큰 이동    | 약 2.3ms                          |
| 로그 worst    | 약 3.6ms                          |
| full reload | 약 17ms                           |


실제 구현에서 추가되는 overhead: (1) Host packed buffer 준비, (2) cudaMemcpyAsync submit overhead, (3) decode kernel(Host 20B→GPU 32B), (4) encode kernel(GPU 32B→Host 20B), (5) staging buffer chunk 관리, (6) synchronization/fence 비용.

목표 latency:


| 상황                         | 예상 latency         |
| -------------------------- | ------------------ |
| 평균 steady-state sequential | 약 1.5~2.0ms/update |
| 평균 steady-state overlapped | 약 1.0~1.5ms/update |
| 큰 이동                       | 약 4~7ms/update     |
| full reload spike          | 30ms 이상 가능         |


30Hz frame budget은 약 33.3ms이므로, 평균 streaming latency는 frame budget의 **약 5% 내외**로 예상된다.

## 19. 성능 목표


| 구분                             | 목표 bandwidth | 설명                        |
| ------------------------------ | ------------ | ------------------------- |
| Expected steady-state H2D      | 0.3~0.5GB/s  | 평균 AABB overlap 기반        |
| Expected H2D + D2H             | 0.6~1.0GB/s  | dirty write-back 포함 보수 추정 |
| Large movement case            | 1.5~3.0GB/s  | p05~worst overlap 이동      |
| Full reload H2D                | 6.14GB/s     | spike로만 허용                |
| Full reload + dirty write-back | 12GB/s 근처    | steady-state로 허용 불가       |
| Target PCIe bandwidth          | 약 12GB/s     | 타겟 노트북 실효 bandwidth       |


설계 목표:

- **Target**: 평균 streaming bandwidth는 1GB/s 이하로 유지
- **Soft limit**: 큰 이동이나 dirty eviction이 겹치는 경우에도 3GB/s 이하 목표
- **Hard worst**: full reload 6.14GB/s는 spike로만 허용하고 steady-state로는 허용하지 않음

## 20. 구현 우선순위

초기 구현은 단순성과 안정성을 우선한다.

**Phase 1: 기본 streaming cache**

1. HostTsdfVoxel 20B format 정의
2. GpuTsdfVoxel 32B format 정의
3. active pool allocation
4. indexGrid allocation
5. Host sparse TSDF store 연동
6. MissingGroupList 생성
7. H2D upload + decode
8. indexGrid registration

**Phase 2: local base 이동과 reuse**

1. local base 계산
2. indexGrid reset
3. activeGroupMeta classification
4. atomic 기반 ReusableList/CleanFreeList/WriteBackList 생성
5. reusable group re-registration
6. missing group upload

**Phase 3: integration/extraction 연동**

1. IntegrationWriteSet 생성
2. ExtractionReadHaloSet 생성
3. ResidentRequiredSet 생성
4. TSDF integration
5. dirty mark
6. recomputeMask 생성
7. extraction candidate 생성
8. candidate merge/split
9. old point merge

**Phase 4: write-back / eviction**

1. dirty outside AABB detection
2. WriteBackTask 생성
3. encode kernel
4. D2H write-back
5. Host sparse store update
6. slot free 처리
7. pool shortage 대응

**Phase 5: 성능 최적화**

1. H2D/D2H overlap
2. chunked staging buffer
3. write-back background stream
4. upload budget control
5. CUB 기반 classification 교체 검토
6. group upload ordering 최적화
7. large tracking jump 대응

### 20.1 전체 일정 요약 (6주)


| 주차  | 목표                                      | 주요 산출물                                                         |
| --- | --------------------------------------- | -------------------------------------------------------------- |
| 1주차 | 구조 확정 및 memory 기반 구현                    | data structure, Host/GPU voxel format, active pool, indexGrid  |
| 2주차 | Host↔Device streaming 기본 구현             | H2D upload, decode, D2H encode/write-back 기본 path              |
| 3주차 | local AABB reuse cache 구현               | classification, ReusableList, CleanFreeList, MissingGroupList  |
| 4주차 | Directional integration 구현              | dominant direction update, dirty mark, recomputeMask           |
| 5주차 | Directional extraction 및 point merge 구현 | directional candidate extraction, merge/split, old point merge |
| 6주차 | 통합 검증 및 결과 확인                           | 기본 reconstruction 결과, 성능 로그, 기존 TSDF 비교                        |


## 21. 주요 invariant

구현 중 반드시 지켜야 할 invariant:

1. 동일한 `DirectionalGroupKey`에 대해 active pool slot이 중복 존재하면 안 된다.
2. indexGrid는 local base에 종속적이다. local base가 바뀌면 반드시 reset 후 reusable group을 재등록해야 한다.
3. PendingUpload slot은 CleanFreeList에 들어가면 안 된다.
4. PendingWriteBack slot은 CleanFreeList에 들어가면 안 된다.
5. Dirty group은 write-back 완료 전까지 free slot으로 재사용하면 안 된다.
6. MissingGroupList는 reusable group 재등록 이후에 생성해야 한다.
7. Integration은 IntegrationWriteSet에 대해서만 write한다.
8. ExtractionReadHaloSet은 read-only일 수 있으며, 이 group이 AABB 밖으로 나갈 때 dirty가 아니면 write-back 없이 버릴 수 있다.
9. recomputeMask 내부 old point는 제거하고, recomputeMask 외부 old point는 유지해야 한다.
10. Directional extraction은 같은 direction layer끼리만 sign crossing을 비교한다.

## 22. 위험 요소와 대응


| 위험 요소                            | 설명                        | 대응                                           |
| -------------------------------- | ------------------------- | -------------------------------------------- |
| Tracking jump                    | AABB overlap 급감           | full reload spike 허용, upload budget 적용       |
| local reset                      | 기존 active pool reuse 어려움  | clean free 우선, dirty background write-back   |
| dirty eviction burst             | D2H write-back 집중         | PendingWriteBack queue, async D2H            |
| pool shortage                    | CleanFreeList 부족          | WriteBackList에서 eviction 후 재사용               |
| atomic contention                | classification counter 집중 | 초기 허용, 필요 시 CUB/prefix-sum 교체                |
| duplicate active group           | indexGrid 충돌 가능           | Host/device allocation invariant 강제          |
| extraction boundary miss         | halo 부족 시 candidate 누락    | +1 group halo 유지                             |
| old point ghosting               | recomputeMask 부정확         | ownerVoxelCoord 저장, conservative mask 사용     |
| normal orientation inconsistency | candidate merge 오류        | sign convention 정리, view direction alignment |


## 23. 결론

기존 TSDF 기반 reconstruction 구조는 일정 수준 이상의 결과를 제공해왔지만, 구조적으로 피하기 어려운 한계(얇은 면의 불안정한 생성, interproximal 영역 데이터 소실, 치아 모서리/sharp edge 부근의 부정확한 reconstruction)가 있다. 단순한 파라미터 조정이나 후처리 휴리스틱만으로는 근본적인 해결이 어렵다.

**Directional TSDF**는 방향별로 TSDF를 분리 관리함으로써 서로 다른 표면이 하나의 voxel field 안에서 섞이는 문제를 줄이고, 얇은 구조나 인접면, 모서리 영역에서 기존 TSDF보다 더 안정적인 surface representation을 기대할 수 있다.

다만 Directional TSDF는 방향 수(6배)만큼 voxel field가 확장되어 매우 큰 메모리 요구량을 가진다. 기존과 같은 GPU-only dense storage로는 구현이 불가능하거나 하이엔드 GPU에서만 동작 가능해, 실제 시장이 요구하는 "높은 reconstruction 품질"과 "낮은 디바이스 사양 지원"을 동시에 만족시키기 어렵다.

**본 설계의 해결책**: Host memory를 적극 활용하는 streaming cache 구조. GPU에는 전체 Directional TSDF volume을 유지하지 않고, 현재 frame의 integration/extraction에 필요한 active directional group만 resident하게 유지한다. ICP 이후 local AABB가 frame 간 대부분 overlap되는 특성을 이용해, 이미 GPU에 있는 group은 재사용하고 새로 필요한 missing group만 Host→Device로 streaming한다.

로그 기반 분석 결과 연속 frame 사이 40mm local AABB는 평균 약 95% overlap을 보였고, 이에 따라 평균적으로 새로 upload해야 하는 group 수는 전체 ResidentRequiredSet의 약 5% 수준으로 추정된다. 즉 Directional TSDF의 높은 메모리 요구량을 Host/GPU streaming 구조로 완화할 수 있으며, steady-state에서는 PCIe bandwidth가 병목이 될 가능성이 낮다.

결과적으로 본 설계는 Directional TSDF의 reconstruction 품질상 이점을 유지하면서도 GPU 메모리 사용량을 현실적인 수준으로 낮추기 위한 시스템적 접근이며, active group reuse / packed host transfer format / dirty write-back / recomputeMask 기반 partial extraction을 함께 적용함으로써 기존 구조보다 낮은 GPU 메모리 사용량과 경쟁력 있는 실행 성능을 기대할 수 있다. 단순 최적화가 아니라, 기존 TSDF 구조의 품질 한계를 넘어서기 위한 **reconstruction architecture의 전환**으로 볼 수 있다.
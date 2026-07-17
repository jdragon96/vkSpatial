# VkLBVH 분석 및 Wide BVH 개선안

> 기준 코드: [`vkBVH.cpp`](../src/vkSpatial/vkBVH.cpp) 및 관련 compute shader
>
> 조사 기준일: 2026-06-13

## 1. 결론

현재 구현은 Karras 방식의 GPU binary radix tree를 사용하는 전형적인 LBVH다.
구조는 단순하고 병렬 빌드가 가능하지만, 실제 병목은 binary BVH 자체보다 다음 항목에 더 크게 걸려 있다.

1. scene maximum 초기화 오류로 Morton code의 공간 분해가 사실상 무너질 수 있다.
2. radix sort의 prefix scan과 local rank 계산이 순차적이다.
3. 모든 dispatch가 별도 command buffer와 `vkQueueWaitIdle`을 사용한다.
4. 각 빌드와 쿼리에서 shader compile 및 compute pipeline 생성이 반복된다.
5. Radius/KNN 쿼리가 GPU invocation 하나에서만 실행된다.
6. 고정 크기 64 stack이 넘치면 오류 없이 subtree를 버려 false negative가 발생한다.

따라서 권장 순서는 다음과 같다.

```text
정확성 수정 및 계측
  -> command/pipeline 재사용
  -> batch query
  -> uncompressed BVH4/BVH8
  -> SAH 기반 wide collapse
  -> conservative 8-bit AABB quantization
  -> query별 stackless/subgroup 특화
```

Wide BVH의 첫 구현은 **binary LBVH를 유지한 채 후처리로 BVH8을 생성**하는 방식이 가장 적합하다.
빌더 전체를 한 번에 교체하면 hierarchy 품질, wide layout, traversal 변화가 섞여 회귀 원인을 찾기 어렵다.

## 2. 현재 구현

### 2.1 빌드 파이프라인

[`vkBVH::buildBVH`](../src/vkSpatial/vkBVH.cpp)은 다음 단계를 수행한다.

| 단계             | 구현                     | 출력                       |
| ---------------- | ------------------------ | -------------------------- |
| Primitive upload | CPU `Primitive[]` 업로드 | 28 B/primitive             |
| Morton encoding  | `bvh_mortonCode.comp`    | 30-bit Morton code + index |
| Radix sort       | 4-bit radix, 8 pass      | Morton 순서                |
| Binary hierarchy | `bvh_hierarchy.comp`     | Karras binary radix tree   |
| AABB propagation | `bvh_boundingBox.comp`   | 내부 노드 AABB             |

노드 배열은 다음 규칙을 사용한다.

- 총 노드 수: `2N - 1`
- 내부 노드: `[0, N - 2]`
- 리프 노드: `[N - 1, 2N - 2]`
- 루트: `0`
- 자식: absolute index
- 리프 판정: `left == 0`

현재 `Node`는 36 B이며 모든 리프도 완전한 노드 구조를 가진다.

```text
Primitive buffers           28N B
Morton ping/pong             16N B
Binary Node buffer       ~   72N B
Construction buffer      ~   16N B
-----------------------------------
합계                    ~  132N B + histogram
```

### 2.2 Morton hierarchy

Morton code는 각 축을 10 bit로 양자화해 총 30 bit를 사용한다.
동일한 Morton code가 발생하면 hierarchy shader가 primitive 순서 index를 tie-breaker로 사용하므로
트리 생성 자체는 가능하다. 다만 동일 code가 많아질수록 공간 locality와 hierarchy 품질은 낮아진다.

현재 코드는 Karras의 다음 절차와 직접 대응한다.

```text
centroid Morton encoding
  -> key sort
  -> longest common prefix로 binary radix tree 생성
  -> leaf에서 root 방향으로 AABB 병합
```

### 2.3 쿼리

`cmd_radiusSearch.comp`와 `cmd_knn.comp`는 모두 `local_size_x = 1`이며,
C++에서도 `Dispatch(1)`을 호출한다. 즉 GPU 전체를 사용하지 않고 단일 invocation이 전체 트리를 순회한다.

- Radius: sphere-AABB test 후 겹치는 모든 leaf를 반환한다.
- KNN: AABB 최소거리와 invocation-local max heap을 사용한다.
- 둘 다 `int stack[64]`를 사용한다.
- KNN 결과는 거리순으로 정렬되어 있지 않다.

현재 API 의미도 명확히 구분할 필요가 있다.

- `PointPrim`: leaf AABB가 점이므로 Radius/KNN 결과가 실제 점 거리와 일치한다.
- `TrianglePrim`: Radius는 sphere와 **triangle AABB**가 겹치는 후보를 반환한다.
- 임의 `Primitive`: KNN은 primitive geometry가 아니라 **AABB까지의 거리**를 기준으로 한다.

Triangle에 대한 exact sphere/triangle 또는 point/triangle 거리가 필요하면 leaf narrow phase를 추가해야 한다.

## 3. 즉시 수정할 문제

Wide BVH 작업 전에 아래 항목을 먼저 해결해야 올바른 기준 성능을 얻을 수 있다.

### 3.1 Scene maximum 초기화

[`MortonConstant`](../src/vkSpatial/types.h)의 `g_maxX/Y/Z`가
`std::numeric_limits<float>::max()`로 초기화되어 있다.

```cpp
float g_maxX = std::numeric_limits<float>::max();
```

이 값에 `std::max(g_maxX, primitiveMax)`를 적용하면 maximum은 갱신되지 않는다.
결과적으로 scene extent가 약 `3.4e38`이 되고 대부분의 centroid가 0 부근으로 정규화되어
Morton code가 같은 값에 몰릴 수 있다.

다음처럼 수정해야 한다.

```cpp
float g_maxX = std::numeric_limits<float>::lowest();
float g_maxY = std::numeric_limits<float>::lowest();
float g_maxZ = std::numeric_limits<float>::lowest();
```

이 문제는 결과 정확성 테스트만으로 놓치기 쉽다. 모든 code가 같아도 index tie-breaker로 유효한 트리가
생성되기 때문이다. Morton code 분포, tree depth, node visit count를 별도로 검사해야 한다.

### 3.2 Stack overflow의 무음 누락

현재 traversal은 stack 공간이 부족하면 자식을 push하지 않는다.

```glsl
if (top + 2 <= 64) {
    stack[top++] = node.right;
    stack[top++] = node.left;
}
```

이는 성능 저하가 아니라 false negative다. 최소한 다음 중 하나가 필요하다.

- overflow flag를 기록하고 쿼리를 실패 처리
- 충분한 크기의 global spill stack 사용
- escape index를 이용한 stackless traversal
- tree depth/최대 stack 사용량을 빌드 후 검증

BVH8은 깊이를 줄이지만 한 노드에서 push할 자식 수가 늘기 때문에 고정 stack 문제가 자동으로 사라지지는 않는다.

### 3.3 GPU 작업 직렬화

`vkComputeBase::submit()`은 각 dispatch마다 다음 작업을 수행한다.

```text
command buffer allocate
  -> record
  -> queue submit
  -> vkQueueWaitIdle
  -> command buffer free
```

그 뒤 호출부에서 `Sync()`가 다시 `vkQueueWaitIdle`을 호출한다.
현재 BVH 빌드는 Morton 1회, radix 24회, hierarchy 1회, AABB 1회로 최소 27 dispatch를 사용한다.

개선 방향:

- 전체 빌드 pass를 하나의 command buffer에 기록한다.
- pass 사이에는 `vkCmdPipelineBarrier2`의 compute write/read barrier를 둔다.
- host 대기는 빌드 끝에서 fence 또는 timeline semaphore로 한 번만 수행한다.
- shader module, pipeline, descriptor set layout을 캐시한다.
- query마다 shaderc compile과 pipeline 생성을 반복하지 않는다.

이 작업은 Wide BVH보다 먼저 수행해도 큰 개선 가능성이 있다.

### 3.4 Radix sort 확장성

현재 sort에는 두 개의 순차 구간이 있다.

- prefix scan: invocation 하나가 `16 * numWorkGroups` 항목을 순차 처리
- reorder: 각 lane이 앞선 최대 255개 원소를 다시 검사해 local rank 계산

데이터가 커지면 reorder는 workgroup당 사실상 `O(WG_SIZE^2)` 비교를 수행한다.
다음 구조로 교체하는 것이 좋다.

- subgroup ballot 또는 shared-memory scan으로 digit별 local rank 계산
- block scan + block sum scan + uniform add의 계층형 global prefix scan
- 동일 scan primitive를 wide-node allocation과 batch Radius output에도 재사용

### 3.5 Baseline 테스트 보강

추가해야 할 회귀 테스트:

- scene bound가 음수만 포함하는 경우
- 동일 위치 또는 동일 Morton code가 대량으로 발생하는 경우
- 선/평면처럼 한두 축 extent가 0인 경우
- 최대 깊이가 64를 넘는 adversarial key 분포
- `N = 0`, `N = 1`의 명시적 정책
- KNN distance tie
- Triangle/AABB 쿼리의 broad-phase 의미

## 4. Wide BVH가 주는 이점과 비용

Wide BVH는 내부 노드가 4개 또는 8개의 자식을 갖는다.

장점:

- tree depth 감소
- 내부 노드 수 감소
- 한 번의 node fetch로 여러 child AABB를 검사
- child test의 instruction-level parallelism 증가
- child bounds를 SoA 또는 quantized block으로 묶기 쉬움

비용:

- 방문한 노드마다 수행하는 AABB test 수 증가
- sparse wide node는 저장 공간을 낭비
- KNN은 near-first ordering 비용이 증가
- 단일 query에서 8 child를 검사해도 GPU occupancy 문제는 별도로 해결해야 함
- binary LBVH 품질이 나쁘면 wide collapse만으로는 overlap을 해결할 수 없음

따라서 `BVH8`을 고정 결론으로 두기보다 `BVH4`와 `BVH8`을 같은 benchmark에서 비교해야 한다.
모바일/Apple 계열과 데스크톱 GPU에서 최적 폭이 다를 수 있다.

## 5. 권장 빌드 구조

### 5.1 1단계: Binary LBVH 유지

기존 Morton, sort, binary hierarchy, AABB pass를 유지한다.
이 binary tree는 최종 traversal 구조가 아니라 wide tree 생성용 중간 표현이 된다.

```text
Primitive
  -> Binary LBVH
  -> Wide collapse
  -> Wide layout compaction
  -> Optional quantization
```

이 방식의 장점은 현재 구현과 테스트를 재사용하면서 wide 변환만 독립 검증할 수 있다는 점이다.

### 5.2 2단계: Fixed-depth BVH8 prototype

최초 prototype은 binary node 3 level을 한 wide node로 펼쳐 최대 8개 frontier child를 만든다.

```text
binary depth +3
  -> 최대 2^3 = 8 child
```

구현이 단순하고 correctness oracle을 만들기 좋지만 다음 단점이 있다.

- child AABB 면적과 primitive 수를 고려하지 않음
- 불균형 binary tree에서 빈 slot이 많아짐
- leaf 크기를 제어하기 어려움

따라서 이 방식은 layout과 traversal 검증용으로만 사용하고, 최종 builder는 SAH collapse로 교체한다.

### 5.3 3단계: SAH 기반 BVH4/BVH8 collapse

Ylitie, Karras, Laine의 CWBVH는 고품질 binary BVH를 입력으로 받아,
dynamic programming으로 최대 8개의 descendant를 선택하는 SAH-optimal wide 변환을 제시한다.

이 프로젝트에서는 다음 두 비용을 비교하면 된다.

```text
C(node) = min(
    node를 leaf range로 만드는 비용,
    최대 W개의 descendant child를 갖는 internal node 비용
)
```

초기 구현은 표준 surface-area ratio를 사용하되 비용 상수는 실제 shader 계측값으로 맞춘다.

- `Cnode`: wide node 한 개를 읽고 child AABB를 검사하는 비용
- `Cprim`: leaf primitive 한 개를 검사하는 비용
- `Pmax`: leaf 하나에 넣을 최대 primitive 수
- `W`: 4 또는 8

Ray tracing 논문의 SAH 상수와 leaf size를 그대로 사용하면 안 된다.
현재 workload는 point Radius/KNN이며 primitive test 비용과 방문 확률이 다르다.

추가 실험으로 query-aware cost를 고려할 수 있다.

- Radius: 대표 반지름으로 확장한 AABB의 volume ratio
- KNN: 샘플 query에서 측정한 node 방문 확률
- 혼합 workload: Radius/KNN 비율을 반영한 가중 비용

이는 표준 SAH보다 builder 비용이 늘 수 있으므로 offline/static point cloud에 우선 적용한다.

### 5.4 GPU 변환 pass

권장 GPU pass는 다음과 같다.

1. binary node별 subtree primitive count와 collapse DP 상태 계산
2. wide node가 될 binary root에 flag 기록
3. exclusive scan으로 compact wide index 할당
4. 각 wide node의 frontier child와 leaf range 생성
5. child wide index 및 primitive range 연결
6. traversal locality에 맞춰 wide node와 leaf index를 compact
7. 선택적으로 child bounds 양자화

현재 AABB pass처럼 child 완료 횟수를 atomic으로 집계하면 bottom-up DP를 구현할 수 있다.
다만 장기적으로는 depth/level 목록을 생성해 level-order pass로 처리하는 편이 디버깅과 재현성이 좋다.

CPU wide converter도 먼저 만들 가치가 있다. production 경로가 아니라 GPU 결과를 비교하는
reference builder와 layout 검사기로 사용한다.

## 6. 권장 메모리 레이아웃

### 6.1 Uncompressed BVH8

첫 구현은 float32 SoA layout을 권장한다.

```cpp
struct WideNode8F32 {
    float minX[8], minY[8], minZ[8];
    float maxX[8], maxY[8], maxZ[8];
    uint32_t child[8];
    uint32_t validMask;
    uint32_t internalMask;
};
```

실제 C++/GLSL 구조는 `std430` alignment를 반영하고 `static_assert`로 크기와 offset을 검증해야 한다.

SoA를 선택하는 이유:

- 8개 child의 같은 축을 연속 load 가능
- scalar loop와 subgroup path가 같은 layout을 공유
- quantized layout과 결과를 비교하기 쉬움

리프는 node 구조와 분리한다.

```cpp
struct LeafRange {
    uint32_t firstPrimitive;
    uint32_t primitiveCount;
};
```

Morton 순서의 primitive index 배열을 별도로 유지하면 하나의 leaf가 여러 primitive를 포함할 수 있다.
Point workload에서는 `Pmax = 4`와 `8`을 우선 비교한다.

### 6.2 Quantized BVH8

CWBVH의 핵심은 wide라는 점뿐 아니라 **child AABB와 index metadata를 한 cache-friendly block에 압축**하는 데 있다.
논문은 8 child 정보를 80 B internal node에 저장한다.

이 프로젝트의 목표 layout도 80-96 B/node로 두는 것이 적절하다.

```cpp
struct WideNode8Q8Concept {
    float origin[3];          // local quantization origin
    uint32_t expXYZAndFlags;  // 3 x signed exponent + flags
    uint32_t childBase;
    uint32_t primitiveBase;
    uint32_t meta[2];         // 8 x packed child metadata
    uint32_t qBounds[12];     // 6 planes x 8 uint8, packed in uint32
};
```

위 구조는 encoding 방향을 보여주는 개념안이며, 최종 bit 배치는 leaf range 한도와 buffer 크기를 기준으로 확정한다.

`uint8_t` storage feature에 의존하지 않고 `uint32_t`에 byte를 pack하면 Vulkan portability가 좋아진다.
지원 장치에서는 `storageBuffer8BitAccess` 특화 경로를 별도로 둘 수 있다.

### 6.3 보수적 양자화

child AABB는 parent-local grid에 다음처럼 저장한다.

```text
qMin = floor((childMin - origin) / scale)
qMax = ceil ((childMax - origin) / scale)
```

복원된 box는 원래 child box를 반드시 포함해야 한다.

```text
decodedMin <= exactMin
decodedMax >= exactMax
```

이 조건이면 양자화는 false positive를 늘릴 수 있지만 false negative를 만들지 않는다.
KNN에서도 quantized AABB 최소거리는 실제 최소거리보다 작거나 같아야 pruning 안전성이 유지된다.

각 축 scale을 power-of-two로 두면 exponent만 저장할 수 있고 shader에서 복원이 단순해진다.
NaN, infinity, 매우 큰 좌표, subnormal, 0 extent 축은 별도 테스트가 필요하다.

### 6.4 예상 메모리 효과

완전히 채워진 8-wide tree의 내부 노드 수는 대략 `(N - 1) / 7`이다.

```text
현재 binary node:       약 72 B/primitive
80 B BVH8 internal:     약 11.4 B/primitive
leaf index array:        약 4 B/primitive
```

실제 값은 sparse node, leaf size, alignment에 따라 달라진다.
그래도 리프마다 36 B node를 저장하는 현재 구조보다 hierarchy traffic을 크게 줄일 여지가 있다.

## 7. Wide traversal

### 7.1 공통 child test

wide node를 읽으면 valid child의 AABB를 한 번에 검사해 hit mask를 만든다.

```glsl
uint hitMask = 0u;
for (uint lane = 0u; lane < childCount; ++lane) {
    if (queryIntersects(childBounds[lane]))
        hitMask |= 1u << lane;
}
```

첫 portable 구현은 invocation 하나가 8 child를 scalar loop로 검사한다.
이후 선택적으로 한 subgroup이 query 하나를 담당하고 lane별 child test 후 ballot으로 mask를 합칠 수 있다.

subgroup path는 다음을 런타임 확인해야 한다.

- compute stage subgroup 지원
- subgroup size
- ballot 및 shuffle 지원
- required subgroup size 제어 가능 여부

현재 `VkContext`는 optional feature chain을 구성하지 않으므로 capability query와 feature enable 코드가 먼저 필요하다.

### 7.2 Radius Search

Radius는 intersect하는 모든 subtree를 방문하므로 child distance 정렬이 필요 없다.
hit mask의 set bit를 stack에 push하면 된다.

추천 경로:

1. BVH8 + local stack
2. batch query
3. stack overflow 계측
4. escape index 기반 stackless traversal 비교

2024년 Prokopenko와 Lebrun-Grandié는 Karras/Apetrei 계열 hierarchy에 range query용 escape connection을
추가하는 방법을 제시했다. Radius처럼 순서보다 누락 없는 전체 방문이 중요한 쿼리에 적합한 방향이다.

Batch Radius output은 가변 길이이므로 두 가지 API를 고려한다.

- fixed capacity/query: 빠르지만 overflow 정책 필요
- count -> global scan -> scatter의 2-pass compact output: 정확한 크기, 추가 pass 필요

공용 parallel scan을 구현하면 radix sort와 Radius output 모두에 사용할 수 있다.

### 7.3 KNN

KNN은 현재 worst distance보다 가까운 child를 먼저 방문할수록 pruning이 강해진다.
Ray CWBVH의 octant order는 point query에 직접 적용할 수 없다.

BVH8 KNN 권장 절차:

1. 8개 child AABB의 `minDist2` 계산
2. `minDist2 <= heapWorst()`인 child만 후보로 선택
3. 가까운 child부터 방문
4. leaf range의 primitive exact distance 계산
5. max heap의 worst distance 갱신

8개 전체 sorting network는 비용이 있으므로 다음을 비교한다.

- hit child만 작은 insertion sort
- 반복 `findMin`으로 필요한 순서만 추출
- `(distance, childIndex)` local priority queue
- subgroup min-reduction

KNN은 stackless 순회보다 작은 candidate priority queue가 유리할 가능성이 높다.
Radius와 KNN이 동일 traversal 구현을 공유하도록 강제하지 않는 것이 좋다.

### 7.4 Batch query가 우선

Wide BVH를 도입해도 query 하나를 invocation 하나로 실행하면 GPU utilization이 낮다.
API를 다음처럼 확장하는 것이 효과가 크다.

```cpp
RadiusSearchBatch(queries, offsets, results);
KNNBatch(queries, k, results, distances);
```

- 최소 구현: invocation 하나/query
- KNN heap: private/local memory 사용량을 계측하고 `k`별 shader specialization
- 큰 `k`: shared/global scratch 또는 별도 알고리즘
- 단일 query API: batch size 1 wrapper로 유지

`MAX_K = 64` 배열 두 개는 invocation당 최소 512 B의 논리 저장 공간을 요구한다.
compiler가 이를 register에 유지하지 못하면 local memory spill이 발생할 수 있으므로 `k <= 8`,
`k <= 32`, `k <= 64` specialization을 권장한다.

## 8. Hierarchy 품질 개선

Wide collapse는 node 수와 depth를 줄이지만, 낮은 품질의 LBVH overlap을 고치지는 않는다.

우선순위:

1. scene bounds 오류 수정
2. 30-bit Morton baseline 계측
3. 60/63-bit Morton key 실험
4. local treelet rotation 또는 reinsertion
5. PLOC 계열 builder와 비교

64-bit key는 `shaderInt64` 의존성과 radix pass 증가가 있으므로 두 개의 `uint32_t`로 key를 표현하는
portable 경로도 고려한다.

Static point cloud에서는 빌드 시간이 조금 늘더라도 SAH와 overlap이 개선된 tree가 유리할 수 있다.
매 frame rebuild가 필요한 동적 데이터에서는 빠른 LBVH + 제한된 treelet optimization이 현실적이다.

## 9. Refit과 동적 데이터

현재는 매번 전체 rebuild만 가능하다. primitive topology가 유지되고 AABB만 변하는 경우 refit 경로를 추가할 수 있다.

```text
leaf AABB update
  -> bottom-up wide-node AABB merge
  -> quantization frame 갱신
```

Quantized BVH에서 parent-local frame을 고정하면 새 child bound가 grid를 벗어날 수 있다.
다음 중 하나를 선택해야 한다.

- 매 refit마다 node quantization frame 재계산
- 여유 margin을 포함한 frame 유지
- overflow node만 float32 fallback
- 품질 또는 overflow 임계값을 넘으면 rebuild

Refit은 hierarchy overlap이 누적될 수 있으므로 build 시점의 SAH cost와 현재 cost 비율을 추적해 rebuild를 결정한다.

## 10. Vulkan RT 사용 여부

조사 기준 최신 Vulkan 1.4.353 specification(2026-06-04)의 acceleration structure와 ray query는
ray traversal용 opaque 구조다.
현재 핵심 API인 Radius와 KNN은 내부 node를 직접 방문하고 AABB 거리를 계산해야 하므로
`VK_KHR_ray_query`가 custom Wide BVH의 직접 대체가 되지는 않는다.

권장 정책:

- Radius/KNN: compute 기반 custom BVH 유지
- 실제 ray query API가 추가될 때만 Vulkan RT backend 검토
- RT extension이 없는 MoltenVK/portable 환경을 기본 경로로 유지

## 11. 구현 로드맵

### Phase 0: 정확성 및 측정

- `MortonConstant::g_max*` 초기화 수정
- stack overflow flag와 실패 정책 추가
- GPU timestamp query로 pass별 시간 측정
- node visit, child test, leaf test, stack high-water counter 추가
- query 의미를 Point exact / AABB broad-phase로 문서화

완료 조건:

- CPU brute force와 기존 query 결과 일치
- adversarial tree에서 overflow가 무음으로 지나가지 않음
- Morton histogram과 tree depth를 테스트에서 확인

### Phase 1: 실행 모델 개선

- shader/pipeline/descriptor cache
- 하나의 command buffer에 build pass 기록
- pass 사이 compute barrier
- 계층형 parallel scan
- radix local rank 병렬화
- Radius/KNN batch API

완료 조건:

- build당 host wait 1회
- shader compile/pipeline 생성이 hot query path에서 발생하지 않음
- `N`과 batch size 증가에 따라 GPU utilization이 증가

### Phase 2: Uncompressed BVH4/BVH8

- CPU reference wide converter
- fixed-depth GPU converter
- float32 SoA node
- leaf range 및 primitive index buffer
- Radius/KNN wide traversal

완료 조건:

- binary와 wide 결과가 동일
- BVH4/BVH8의 node visit, AABB test, memory traffic 비교
- sparse node 비율과 실제 bytes/primitive 보고

### Phase 3: SAH collapse

- subtree primitive count
- bottom-up DP
- compact wide index scan
- `Pmax`, `Cnode`, `Cprim` tuning
- 선택적 local treelet optimization

완료 조건:

- fixed-depth보다 SAH cost 및 query 시간이 개선
- build overhead와 query 이득의 crossover point 측정

### Phase 4: Quantized Wide BVH

- 8-bit conservative child bounds
- packed metadata
- float32 fallback/debug layout
- quantization error 및 false-positive 계측

완료 조건:

- CPU 및 float32 wide 결과 대비 false negative 0
- bytes/primitive와 cache traffic 감소
- KNN exact result 유지

### Phase 5: 특화 경로

- Radius stackless traversal
- subgroup cooperative child test
- `k`별 KNN specialization
- refit/rebuild 정책
- 선택적 BVH4/BVH8 runtime 선택

## 12. Benchmark 계획

### 데이터

| 분포             | 목적                             |
| ---------------- | -------------------------------- |
| Uniform 3D       | 기본 scaling                     |
| Clustered        | sibling overlap 및 leaf grouping |
| Plane/line       | degenerate extent                |
| Duplicate points | Morton collision                 |
| Mixed-size AABB  | centroid LBVH 약점               |
| Moving points    | refit 품질                       |

권장 크기: `1K`, `100K`, `1M`, 장치 메모리가 허용하면 `10M`.

### 쿼리

- Radius selectivity: 약 `0%`, `0.01%`, `1%`, `10%`, `100%`
- KNN: `k = 1, 8, 32, 64`
- batch size: `1`, `32`, `1K`, `64K`
- query 위치: scene 내부, 경계, scene 외부, cluster 중심

### 지표

- build total 및 pass별 GPU time
- query throughput와 latency
- bytes/primitive
- internal node occupancy
- tree depth
- nodes visited/query
- child AABB tests/query
- primitive tests/query
- stack high-water 및 overflow
- quantization으로 증가한 false-positive test
- pipeline compile/create 시간

단일 장면의 최고 수치보다 여러 분포의 geometric mean과 최악 회귀를 함께 봐야 한다.
Wide/quantized path는 binary baseline보다 메모리는 줄었지만 특정 workload에서 child test가 크게 늘 수 있다.

## 13. 권장 파일 구조

```text
src/vkSpatial/bvh/
  BinaryBuilder.cpp
  WideBuilder.cpp
  BVHQuery.cpp

src/shader/
  bvh_binary_hierarchy.comp
  bvh_binary_bounds.comp
  bvh_wide_dp.comp
  bvh_wide_scan.comp
  bvh_wide_emit.comp
  bvh_wide_quantize.comp
  cmd_radiusSearch_wide.comp
  cmd_knn_wide.comp
```

공통 CPU/GLSL layout에는 명시적인 이름을 사용한다.

```text
BinaryNodeF32
WideNode4F32
WideNode8F32
WideNode8Q8
LeafRange
```

`NODES * 36` 같은 literal allocation은 제거하고 CPU mirror struct의 `sizeof`와
`static_assert(offsetof(...))`를 사용해야 한다.

## 14. 참고 자료

1. Tero Karras, **Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees**, HPG 2012.
   현재 binary radix tree 생성 방식의 기반.
   <https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf>

2. Henri Ylitie, Tero Karras, Samuli Laine, **Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs**, HPG 2017.
   Binary-to-BVH8 SAH conversion, 8-bit conservative bounds, 80 B wide node.
   <https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf>

3. Michael P. Howard et al., **Quantized Bounding Volume Hierarchies for Neighbor Search in Molecular Simulations on Graphics Processing Units**, 2019.
   GPU neighbor search에서 quantized BVH의 직접적인 적용 사례.
   <https://arxiv.org/abs/1901.08088>

4. Toni Tan, Rene Weller, Gabriel Zachmann, **Compressed Bounding Volume Hierarchies for Collision Detection & Proximity Query**, 2020.
   ray 외 collision/proximity workload의 compressed treelet 접근.
   <https://arxiv.org/abs/2012.05348>

5. Andrey Prokopenko, Damien Lebrun-Grandié, **Revising Apetrei's Bounding Volume Hierarchy Construction Algorithm to Allow Stackless Traversal**, revised 2024.
   Karras 계열 hierarchy와 range query용 escape connection.
   <https://arxiv.org/abs/2402.00665>

6. Michael A. Kern et al., **DOBB-BVH: Efficient Ray Traversal by Transforming Wide BVHs into Oriented Bounding Box Trees using Discrete Rotations**, 2025.
   Wide BVH 후처리와 quantized orientation의 최신 연구 방향. 현재 Radius/KNN에는 우선순위가 낮지만,
   얇고 회전된 geometry의 AABB overlap이 병목일 때 실험할 수 있다.
   <https://arxiv.org/abs/2506.22849>

7. Khronos Group, **Vulkan 1.4.353 Specification**, 2026-06-04, 최신 등록 extension 포함.
   subgroup, 8-bit storage, synchronization, acceleration structure capability 확인 기준.
   <https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html>

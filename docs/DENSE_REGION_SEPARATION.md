# Dense 영역 분리 — `DenseRegionClassifier`

> 32³ 블록마다 **half-voxel detail 레벨을 만들 가치가 있는지** 판정하고, 한 프레임의 점을
> base/detail 두 인덱스 리스트로 쪼갠다. 전부 GPU에서, 프레임당 상수 크기 재사용 버퍼로.
>
> 코드: [`src/TSDF/Structure/DenseRegionClassifier.{h,cpp}`](../src/TSDF/Structure/) + 커널 4개
> 설계 근거: [`docs/superpowers/specs/2026-08-15-dense-region-separation-design.md`](superpowers/specs/2026-08-15-dense-region-separation-design.md)
> 관련: [`HASH_PROBING_AND_LOAD_FACTOR.md`](HASH_PROBING_AND_LOAD_FACTOR.md) §8 — 메모리를 지배하는 것은 해시가 아니라 레벨·타일의 개수와 용량

---

## 1. 역할

이 클래스는 **판정기**다. TSDF도 아니고 메모리 전략도 아니다. 자기 버퍼만 소유하고 어떤
memory strategy도 알지 못하므로, 단독으로 빌드하고 테스트할 수 있다.

```
점군 + 법선  ──►  DenseRegionClassifier  ──►  baseIndex[]   ──► base 레벨 TSDF
                                          └─► detailIndex[] ──► detail 레벨 TSDF (half voxel)
```

호출 예정지는 `SubmapStrategy::DividePoint`. **아직 배선되지 않았다** — `Memory/`→`Structure/`
이행 중이라 컴포넌트만 독립적으로 완성해 둔 상태다.

---

## 2. 무엇을 대체하는가

기존 판정은 `SubmapAdvancedTSDF`의 한 줄이었다.

```cpp
float(count) / float(occ.size()) >= m_detailK   // 기본값 m_detailK = 4.0
```

**점유 셀당 점 개수**, 즉 중복도다. 두 가지를 못 본다.

| 못 보는 것 | 왜 문제인가 |
|---|---|
| **간격** | 중복도가 높아도 샘플 간격이 detail 복셀보다 크면 refine해봐야 빈 복셀만 는다 |
| **디테일 유무** | 평평한 면을 아무리 촘촘히 스캔해도 refine할 이유가 없는데, 중복도는 **오히려 높다** |

그리고 판정이 CPU `unordered_map` 위에서 프레임마다 전체 점을 순회했다.

---

## 3. 핵심 아이디어 — kNN을 counting으로 대체한다

점별 국소 간격 `s(p) = r_k·√(π/k)`는 kNN이 필요하다. **하지만 여기서 내리는 결정은 블록당
하나뿐이다.** 그러면 처음부터 블록 통계를 세면 된다 — 정렬도, 트리도, prefix sum도 사라진다.

간격은 **세는 것만으로** 나온다. 표면은 2차원이므로 점유 셀 수는 면적을 간격의 제곱으로 나눈
값에 비례한다. 굵은 셀을 `v`(base voxel), 미세 셀을 `v/2`(detail voxel)라 하면:

| 샘플 간격 | occupiedFine | occupiedCoarse | 비율 |
|:---:|:---:|:---:|:---:|
| `s ≤ v/2` | `A/(v/2)²` | `A/v²` | **4** |
| `v/2 < s ≤ v` | `A/s²` | `A/v²` | `(v/s)²` ∈ [1,4) |
| `s > v` | `A/s²` | `A/s²` | **1** |

```
ratio = fineOccupied / coarseOccupied  ∈ [1, 4]
s     = v / √ratio
```

임계값 **3.2**(= 0.8×4)는 `s ≤ v/√3.2 ≈ 0.56 v`에 해당한다. 노이즈 때문에 정확히 4는 안 나온다.

**측정치**: `s = v/4`에서 4.047, `s = 2v`에서 1.000.

---

## 4. 자료구조

### `BlockRecord` — 4바이트 스칼라 10개, 40바이트

GPU 정의는 [`DenseRegionClassifier.common.glsl`](../src/TSDF/Structure/DenseRegionClassifier.common.glsl)
**한 곳**뿐이고 세 패스가 공유한다. C++ 쪽 복사본은 헤더의 struct 하나이고,
`static_assert(sizeof == 40)`이 드리프트를 잡는다.

| 필드 | 수명 | 용도 |
|---|---|---|
| `blockKey` | — | 패킹된 블록 좌표. `0xFFFFFFFF` = 빈 슬롯 |
| `pointCount` | **누적** | 조건 3의 분자 |
| `pointCountFrame` | **프레임** | 법선 응집도의 분모 |
| `coarseOccupied` | **누적** | 비율의 분모 |
| `fineOccupied` | **누적** | 비율의 분자 |
| `fineOccupiedFrame` | **프레임** | 조건 4의 절대 하한 |
| `fineOccupiedMax` | **프레임 최댓값** | detail 테이블 사이징 입력 |
| `sumNormalFrame{X,Y,Z}` | **프레임** | 고정소수 ×10000, 조건 2 |

### 버퍼

| 버퍼 | 크기 | 비고 |
|---|---|---|
| `blockRecords` | `kBlockCapacity` = **8192** | 씬 전체, host-visible readback |
| `blockIndex` | `maxPointPerFrame` | 점 → 블록 레코드 인덱스 |
| `fineCells` / `coarseCells` | `2 × maxPointPerFrame` | **키만 있는 해시, 매 프레임 비움** |
| `baseIndex` / `detailIndex` | `maxPointPerFrame` | 분할 결과 (atomic append) |
| `denseFlags` | `kBlockCapacity` | 래치. device-local |
| `totals` / `partitionCount` | 2 × u32 | 리드백 |

### 셀 해시를 프레임 간 유지하지 않는 이유

누적하면 "지금까지 본 서로 다른 미세 셀"을 기억해야 하고, 그건 **detail TSDF 자체와 같은 규모의
메모리**다. 판정기가 판정 대상만큼 커지는 것은 말이 안 된다. 대신 매 프레임 비우고
(그래서 `maxPointPerFrame`으로 크기가 잡힌다), 블록 레코드에 프레임별 개수를 누적한다.

### 키 패킹 — 비트 예산이 빡빡하다

```
blockKey   = ((x+512) & 0x3FF) << 20 | ((y+512) & 0x3FF) << 10 | ((z+512) & 0x3FF)
fineKey    = blockSlot << 18 | fz << 12 | fy << 6 | fx      // 축당 6비트, 슬롯 14비트
coarseKey  = blockSlot << 15 | cz << 10 | cy <<  5 | cx      // 축당 5비트
```

여기서 나오는 두 제약이 코드에 `static_assert`와 `throw`로 박혀 있다.

- **`blockVoxels ≤ 32`** — fine 키는 축당 6비트뿐이라 33 이상이면 서로 다른 셀이 한 키로
  앨리어싱된다. 그것도 오버플로가 슬롯 필드로 넘어가므로 **블록을 가로질러** 앨리어싱된다.
  `Build()`가 거부한다(클램프가 아니라 거부 — 클램프할 대상이 없다).
- **`kBlockCapacity ≤ (1<<14) − 1`** — 슬롯 16383 + fine 셀 (63,63,63)은 정확히
  `0xFFFFFFFF` = `EMPTY_KEY`로 패킹된다. 해시가 그 셀을 영원히 비어 있다고 읽어 모든 점에 대해
  다시 센다. **블록 테이블을 키울 사람은 둥근 수 `1<<14`가 아니라 `(1<<14)−1`에서 멈춰야 한다.**

---

## 5. 세 패스

세 패스 모두 **호출자의 `CommandBatch`에 기록만 하고 자체 제출하지 않는다.**

```
Record()    :  clear ──B──► accumulate
Classify()  :        ──B──► classify
Partition() :        ──B──► partition
                     ▲
                     └── batch.Barrier(). Vulkan은 연속 dispatch 간 순서를 보장하지 않는다.
```

### 패스 1 — accumulate (점당 1스레드)

```
1. 블록 좌표와, 그 안에서의 굵은/미세 셀 좌표를 구한다
2. 블록 레코드 확보:  범위 밖이거나 삽입 실패 → 카운터++, blockIndex[i] = EMPTY_KEY, 반환
3. pointCount, pointCountFrame, sumNormalFrame{X,Y,Z} 누적  (전부 atomic)
4. 셀 해시에 atomicCompSwap:  최초 삽입일 때만 occupancy++
```

셀 해시는 `atomicCompSwap`으로 "이 프레임에 이 셀을 처음 본 스레드"만 참을 돌려받는다.
`MAX_PROBE`(128) 안에 슬롯을 못 찾으면 `cellInsertFailureCount`를 올린다 — 세지 못한 것은
어쩔 수 없지만 **관측되지 않는 일은 없게** 한다.

### 패스 2 — classify (블록당 1스레드)

```glsl
atomicMax(fineOccupiedMax, fineOccupiedFrame);   // 래치 반환보다 먼저

if (g_dense[i] != 0u) {                          // 래치: 한번 dense면 되돌리지 않는다
    denseBlockCount++;  detailSlots += fineOccupiedMax;
    return;
}

bool resolvesFineGrid = fineOccupied      >= 3.2 * coarseOccupied;        // 누적
bool hasDetail        = |sumNormalFrame| / pointCountFrame < 0.9;         // 프레임
bool keepsSignal      = pointCount        >= 3.0 * fineOccupied;          // 누적
bool hasSurface       = fineOccupiedFrame >= 64;                          // 프레임

if (전부 참) { g_dense[i] = 1u; ... }
```

**AND인 것이 중요하다.** 어느 하나라도 빠지면 detail 레벨이 불필요하게 커지고, detail 레벨은
두 번째 타일 계층 전체 — 측정된 지배적 메모리 비용이다.

### 패스 3 — partition (점당 1스레드)

`blockIndex[i]`로 판정을 조회해 `baseIndex`/`detailIndex`에 **atomic append**. TSDF 적분은
순서 무관이므로 안정 분할이 필요 없고, 그래서 prefix sum이 필요 없다.

블록 레코드를 못 만든 점(`EMPTY_KEY`)도 **base로 간다.** 모든 점은 어딘가에 착지해야 하므로
`baseCount + detailCount`는 **항상** 입력 점 수와 같다.

---

## 6. 누적이냐 프레임이냐 — 조건마다 다르다

이 컴포넌트에서 가장 헷갈리는 지점이고, 둘 다 의도된 것이다.

**비율 조건(1, 3)은 누적이다.** 스펙 §5의 결정이다. `Σ fineOccupied_f / Σ coarseOccupied_f`는
프레임별 비율의 커버리지 가중 평균, 즉 **단일 시점의 샘플 간격**을 잰다. 여러 시점을 겹쳐
실효 간격이 줄어드는 효과는 **일부러 반영하지 않는다** — 50µm급 detail 복셀에서는 프레임 간
정합 오차가 그 복셀 크기를 넘고, 그때 시점을 겹쳐 얻는 것은 서브복셀 디테일이 아니라 노이즈다.

**법선·하한 조건(2, 4)은 프레임별이다.** 두 가지 이유가 있다.

1. **오버플로.** 법선 합을 누적하면 int32가 넘친다. 그리고 **응집된 법선 — 이 설계가 거부하려는
   바로 그 평면 — 이 합을 최대화하므로 평면이 가장 먼저 넘친다.** 랩어라운드가 응집도를 0.9
   아래로 끌어내려 `hasDetail`이 참이 되고, 평면에서 나머지 세 조건은 이미 참이므로 블록이
   **영구 래치**된다. 4096점 픽스처에서 56번째 프레임, 실측 밀도(0.32m 블록면 ~61k점/프레임)면
   ~4프레임이다.
2. **의미.** 시점을 가로질러 법선을 평균내는 것은 더 확신 있는 추정이 아니라 **서로 다르게
   어긋난 것들의 혼합**이다. 점유에 대해 §5가 이미 내린 판단과 같다.

조건 4가 `fineOccupiedMax`가 아니라 `fineOccupiedFrame`인 것도 여기서 나온다. 조건 2와 4가
**같은 프레임**을 서술해야 한다 — 블록은 크기와 곡률을 **한 시점에서 함께 본** 대가로 detail
레벨을 얻어야지, 크기는 예전 프레임에서 기억하고 곡률만 이번 프레임에서 재서는 안 된다.

남은 int32 한계는 **한 프레임, 한 블록에 214,748점**(2³¹/10000)이다. 위에서 계산한 최악 실측
밀도의 약 3.5배라 카운터를 두지 않았다.

---

## 7. 관측 가능한 천장

> **프레임은 절대 조용히 잘리거나 버려지지 않는다. 모든 용량 한계는 카운터로 관측 가능해야 한다.**

이 저장소에서 한 번 터진 적 있는 실패 모드라 이 컴포넌트의 최우선 제약이다.

| 천장 | 값 | 대응 |
|---|---|---|
| 점 버퍼 | `maxPointPerFrame` | **성장**(1.5배). 클램프하지 않는다 |
| 셀 해시 | `2 × maxPointPerFrame` | 점 버퍼와 **한 몸으로** 성장 |
| 블록 테이블 | 8192 | `BlockInsertFailureCount()`, 점은 base로 |
| 블록 좌표 범위 | ±512 블록 | 같은 카운터. 100µm base voxel이면 **±1.638m뿐** |
| 셀 해시 프로빙 | `MAX_PROBE` = 128 | `CellInsertFailureCount()` |

`growPointBuffers()`가 `m_maxPointPerFrame`이 바뀌는 **유일한 지점**이고, 점·법선·`blockIndex`·
`baseIndex`·`detailIndex`·`fineCells`·`coarseCells` 일곱 개를 한 블록에서 함께 키우고 커널 세 개를
전부 재바인딩한다. 갈라놓으면 `cellCapacity`가 실제 할당을 넘어서는 드리프트가 생긴다 —
실제로 한 번 그랬다.

---

## 8. 계약

**프레임당 하나의 배치.** `Record` / `Classify` / `Partition`은 한 프레임의 시퀀스이고,
셋은 **하나의 `CommandBatch`**에 들어가 다음 프레임 전에 제출되어야 한다. 스타일 취향이 아니다 —
`CommandBatch`는 한 배치에 같은 파이프라인을 두 번 dispatch하는 것을 금지하고, 이 클래스는 그
보장에 기대어 `totals`/`partitionCount`를 호스트에서 리셋한다. 두 프레임을 한 배치에 묶으면
합계가 두 배가 되고, partition이 인덱스 리스트를 넘어 쓰고, 두 번째 프레임이 버퍼를 키우면
첫 프레임의 이미 기록된 dispatch가 파괴된 버퍼를 읽는다.

**법선은 단위 벡터여야 한다.** 곡률 판정 전체가 `|Σn|/N`을 1 근처 임계값과 비교하는 것이므로,
길이가 1이 아닌 법선은 노이즈를 더하는 게 아니라 **측정을 리스케일한다** — 짧은 법선은 없는
곡률로 읽혀 평면을 refine한다.

**`normalCoherence`는 호출자의 법선 추정기에 맞춰 보정해야 한다.** 이 조건은 넘겨받은 법선만큼만
좋고, 여기서 법선 노이즈는 곡률과 구별되지 않는다. 축당 RMS 각오차 σ에 대해 `|Σn|/N ≈ 1 − σ²`
이므로 기본값 0.9는 **σ ≈ 18°**에서 걸린다. 실패한 추정은 노이즈보다 나쁘다 — 영 법선은 합에
0을, 개수에 1을 더하므로 **완벽한 평면도 법선 10%가 실패하면 정확히 0.90을 읽고 refine된다.**
판정은 래치라 한 프레임이면 영구적이다.

**크기 불일치와 `blockVoxels` 범위 위반은 `throw`한다.** `assert`가 아닌 이유는 프로젝트가
`-O3 -DNDEBUG`로 빌드되기 때문 — 정작 배포되는 구성에서 no-op가 된다.

**`Build()`와 `Reset()`은 큐 제출과 전체 대기를 유발한다** (래치가 device-local이라 호스트에서
memset할 포인터가 없다). 둘 다 씬 단위 경로이지 프레임 단위 경로가 아니다.

---

## 9. 측정치

설계의 중심 주장 — 이것이 중복도 휴리스틱보다 낫다 — 은 **두 조건을 가르는 케이스에서** 측정됐다.

### 조건 2가 하는 일 (평면 vs 곡면)

| 표면 | 법선 응집도 | 판정 |
|---|:---:|---|
| 촘촘한 평면 | **1.0000** | 거부 — 기존 휴리스틱은 이걸 못 본다 |
| 촘촘한 구 | 0.854 – 0.859 | refine |

### 조건 1이 하는 일 (비균일 샘플링)

**균일** 샘플링에서는 `keepsSignal ⟹ resolvesFineGrid`가 성립한다
(`P ≥ 3F`가 `s < v/2`를 강제 → `F = 4A/v²`, `C = A/v²` → 비율 = 4). 즉 균일 표면에서 두 조건은
합쳐서 정확히 `pointCount/coarseOccupied ≥ 12`, **대체하려던 중복도 휴리스틱 그 자체**다.

조건 1이 힘을 쓰는 곳은 **비균일** 샘플링 — 작은 초고밀도 패치 + 넓은 희소 영역, 즉 기존
휴리스틱이 정확히 틀리는 경우다. 한 블록, 한 프레임에서 측정:

| 조건 | 값 | 임계 | 판정 |
|---|---:|---:|---|
| 1. `resolvesFineGrid` | **1.949** | ≥ 3.2 | **거부 — 유일한 거부권** |
| 2. `hasDetail` | 0.047 | < 0.9 | 통과 |
| 3. `keepsSignal` | 4539 ≥ 3219 | | 통과 |
| 4. `hasSurface` | 345 | ≥ 64 | 통과 |
| — 기존 휴리스틱 `P/C` | **25.64** | ≥ 12 | **refine했을 것** |

### 조건 4가 하는 일 (예전 프레임의 크기를 빌려오지 못하게)

프레임 1에 촘촘한 평면(`fineOccupiedMax` 1024, 응집도 1.0 → 래치 안 됨), 프레임 2에 같은
블록을 성긴 노이즈로 흘긋 봄:

| | 값 | 임계 | |
|---|---:|---:|---|
| `fineOccupiedFrame` | **49** | ≥ 64 | **거부 — 유일한 거부권** |
| `fineOccupiedMax` | 1024 | ≥ 64 | 최댓값을 읽었다면 빌려왔을 값 |

---

## 10. 아직 하지 않은 것

| 항목 | 이유 |
|---|---|
| `SubmapStrategy::DividePoint` 배선 | `Memory/`→`Structure/` 이행이 끝난 쪽에 붙인다 |
| detail 테이블 실제 사이징 적용 | `DetailSlotEstimate()`는 **점유 수이지 테이블 크기가 아니다.** load-factor 여유를 곱하는 것은 호출자 몫 — 정확히 그 수로 잡으면 load factor 1.0이고, 이 저장소에서 그 실패 모드는 "느림"이 아니라 **구멍**이다 |
| 블록 경계 apron | 경계 점의 통계가 양쪽으로 갈려 각각 과소 추정된다. 정밀도 문제이지 정확성 문제가 아니고, 블록이 32³이라 경계 비중이 작다 (스펙 §7-3) |
| 블록 테이블 성장 | 지금은 8192 고정이고 넘치면 그 점들이 base로 간다(분할은 여전히 전수). **키울 때 `(1<<14)−1`에서 멈출 것** |
| 3레벨 이상 LOD | 지금은 base/detail 2레벨 |

`DetailSlotEstimate()`에는 두 번째 하한 성격도 있다. `fineOccupiedMax`는 **프레임별 최댓값**인데
detail 테이블은 **프레임의 합집합**을 담아야 한다. 스펙 §5는 max ≈ 합집합이라고 논하지만 그건 매
프레임이 블록 전체를 덮을 때만 성립한다 — 블록을 **가로질러 쓸고 지나가는** 스캐너는 매 프레임
다른 1/3을 보므로, 합집합은 최댓값의 몇 배가 된다.

---

## 11. 테스트

22개. 이 브랜치는 판정에 관여하는 모든 단언을 **뮤테이션으로 검증**했다 — 단언을 통과시키는
버그를 일부러 넣고 그 테스트가 정말 빨개지는지 확인했다.

| 무엇을 | 어떻게 |
|---|---|
| 비율이 간격을 재현한다 | `s = v/4, v/2, v, 2v`에서 4, 4, 1, 1 |
| 평면은 refine하지 않는다 | 조건 2에서 탈락 (`|Σn|/N ≈ 1`) |
| 곡면은 refine한다 | 촘촘한 구가 네 조건 통과 |
| **비균일 곡면은 refine하지 않는다** | 조건 1만으로 거부 — 기존 휴리스틱과 갈리는 케이스 |
| **법선 합이 넘쳐도 평면은 안 뒤집힌다** | 60프레임 재생 |
| **흘긋 본 프레임은 예전 크기를 못 빌린다** | 조건 4만으로 거부 |
| 분할이 점을 잃지 않는다 | 인덱스별 seen-map — 개수 합이 아니라 **각 인덱스가 정확히 한 번** |
| 블록 테이블이 넘쳐도 점은 안 잃는다 | 25³ 격자로 8192 초과, 그래도 전수 보존 |
| 래치 | dense가 된 블록은 이후 프레임에서 뒤집히지 않고, 점도 계속 detail로 간다 |
| 큰 프레임 | 힌트 초과 시 버퍼가 성장, 셀 해시도 **함께** |
| 먼 블록이 충돌하지 않는다 | `packBlockKey` 회귀 |
| 빈 프레임이 이전 프레임을 재생하지 않는다 | `Record`가 카운트를 맨 앞에서 무효화 |

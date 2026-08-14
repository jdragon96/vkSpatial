# Dense 영역 분리 (GPU) — 설계

> 목적: Tile TSDF에서 **detail 레벨을 만들 가치가 있는 블록만** 골라낸다. 판정과 점군 분할을
> 전부 GPU에서, 프레임당 상수 크기 재사용 버퍼로 처리한다.
>
> 대상: `src/TSDF/Structure/SubmapStrategy.cpp`의 `DividePoint`
> 작성일: 2026-08-15 · 상태: 리뷰 대기
> 관련: [`HASH_PROBING_AND_LOAD_FACTOR.md`](../../HASH_PROBING_AND_LOAD_FACTOR.md) §8 (타일당 용량이 메모리를 지배)

---

## 1. 지금 무엇이 문제인가

현행 `SubmapAdvancedTSDF`는 CPU에서 블록당 `count / occ.size() >= detailK` 하나로 판정한다.
이것은 **점유 셀당 점 개수**, 즉 중복도다. 두 가지를 못 본다:

- **간격**: 중복도가 높아도 샘플 간격이 detail 복셀보다 크면 refine해봐야 빈 복셀만 는다.
- **디테일 유무**: 평평한 면을 아무리 촘촘히 스캔해도 refine할 이유가 없는데, 중복도는 오히려 높다.

그리고 판정이 CPU `unordered_map` 위에서 프레임마다 전체 점을 순회한다.

---

## 2. 핵심 결정: kNN을 counting으로 대체한다

점별 국소 간격 `s(p) = r_k·√(π/k)`는 kNN이 필요하다. **하지만 여기서 내리는 결정은 블록당
하나뿐이고**, 점별 통계는 어차피 블록으로 집계된다. 그러면 처음부터 블록 통계를 세면 된다 —
정렬도, 트리도, prefix sum도 사라진다.

간격은 세는 것만으로 나온다. 표면은 2차원이므로, 점유 셀 수는 면적을 간격의 제곱으로 나눈 값에
비례한다. 굵은 셀 크기를 `v`(base voxel), 미세 셀을 `v/2`(detail voxel)라 하면:

| 샘플 간격 | occupiedFine | occupiedCoarse | 비율 |
|---|---|---|---|
| `s ≤ v/2` | `A/(v/2)²` | `A/v²` | **4** |
| `v/2 < s ≤ v` | `A/s²` | `A/v²` | `(v/s)²` ∈ [1,4) |
| `s > v` | `A/s²` | `A/s²` | **1** |

따라서

```
ratio = occupiedFine / occupiedCoarse  ∈ [1, 4]
s     = v / sqrt(ratio)                (ratio < 4일 때)
```

브리프의 조건 `s < 0.5·v`는 곧 `ratio ≥ 4`다. 노이즈 때문에 정확히 4는 안 나오므로
**3.2**(=0.8×4)를 쓴다 — `s ≤ v/√3.2 ≈ 0.56 v`에 해당한다.

---

## 3. 블록당 4개 통계, 한 패스

전부 atomic. 점당 한 번씩만 만진다.

| 통계 | 수집 방법 | 무엇을 판정하나 |
|---|---|---|
| `pointCount` | `atomicAdd(1)` | 조건 3 (refine 후 SNR) |
| `coarseOccupied` | 굵은 셀 해시에 **첫 삽입일 때만** `atomicAdd(1)` | 비율의 분모 |
| `fineOccupied` | 미세 셀 해시에 **첫 삽입일 때만** `atomicAdd(1)` | 조건 1 (간격) + detail 해시 사이징 |
| `sumNormal` | `atomicAdd`(고정소수 int3) | 조건 2 (디테일 유무) |

`|sumNormal| / pointCount`는 법선 응집도다. 평면이면 ~1, 곡률·에지에서 낮아진다.
법선은 단위벡터이므로 TSDF 누적기와 같은 고정소수 방식(`×10000` 후 int)으로 누적한다.

### 판정식

```
refine(block) =
      fineOccupied      >= 3.2 * coarseOccupied     // 1. 간격이 미세 격자를 해상한다
   && |sumNormal| / pointCount  <  0.9              // 2. 표현하지 못한 디테일이 있다
   && pointCount     >= 3 * fineOccupied            // 3. refine 후에도 복셀당 3샘플
   && fineOccupied   >= 64                          // 4. 블록에 실제 표면이 있다 (아래 §7-1)
```

AND인 것이 중요하다. 어느 하나라도 빠지면 detail 레벨이 불필요하게 커지고, **직전 브랜치 측정에
따르면 메모리를 지배하는 것은 해시 알고리즘이 아니라 레벨·타일의 개수와 용량이다.**

---

## 4. 부수 효과: detail 해시 사이징이 공짜로 나온다

알려진 결함이 하나 있다 — detail 해시가 작으면 오버플로로 표면에 구멍이 난다. 지금은 호출자가
숫자를 찍어 넣는다.

`fineOccupied`가 **그 블록이 detail 레벨에서 실제로 차지할 슬롯 수**다. 판정 패스가 이미 세고
있으므로, dense 블록들의 `fineOccupied` 합에 load-factor 여유를 곱하면 detail 테이블 용량이
그대로 결정된다. 판정과 사이징을 한 번에 끝낸다.

---

## 5. 자료구조 — 라이브러리를 쓰지 않는다

전부 counting과 atomic append다. 정렬·prefix sum·kd-tree가 없으므로 외부 의존을 추가할 이유가
없다. 셀 해시는 이미 있는 `findOrInsert` 계약([`TSDF/Memory/Hash/`](../../../src/TSDF/Memory/Hash/))을
재사용한다 — 저장소의 해싱 관용구가 하나로 유지되고, 삽입 실패 계측도 그대로 따라온다.

### 재사용 버퍼 (`maxPointPerFrame` 기준)

| 버퍼 | 크기 | 용도 |
|---|---|---|
| `blockIndex` | `maxPointPerFrame` × u32 | 점 → 블록 레코드 인덱스 |
| `baseIndex` | `maxPointPerFrame` × u32 | 분할 결과 (atomic append) |
| `detailIndex` | `maxPointPerFrame` × u32 | 분할 결과 (atomic append) |
| `partitionCount` | 2 × u32 | 두 파티션 크기 |

블록 레코드는 스캔 전체의 접촉 블록 수에 비례하고, **두 셀 해시는 프레임마다 비운다** — 이유는
바로 아래. 셋 다 `maxPointPerFrame`을 넘는 프레임이 오면 **clamp가 아니라 성장시킨다** — clamp는
조용히 점을 버리고, 그 실패 모드는 직전 브랜치에서 이미 한 번 겪었다.

### 셀 해시를 프레임 간 유지하지 않는 이유

누적하면 "지금까지 본 서로 다른 미세 셀"을 기억해야 하고, 그것은 **detail TSDF 자체와 같은
규모의 메모리**다. 판정기가 판정 대상만큼 커지는 것은 말이 안 된다.

대신 셀 해시는 프레임마다 비우고(그래서 `maxPointPerFrame`으로 크기가 잡힌다), 블록 레코드에는
프레임별 개수를 **누적**한다:

```
ratio = Σ_frames fineOccupied_f  /  Σ_frames coarseOccupied_f
```

이것은 프레임별 비율의 커버리지 가중 평균, 즉 **단일 시점의 샘플 간격**을 잰다. 여러 시점을
합쳐서 실효 간격이 줄어드는 효과는 일부러 반영하지 않는다 — **50µm급 detail 복셀에서는 프레임
간 정합 오차가 그 복셀 크기를 넘을 가능성이 높고, 그때 시점을 겹쳐 얻는 것은 서브복셀 디테일이
아니라 노이즈다.** 중복 관측을 밀집으로 세던 현행 판정(`count/occ.size()`)을 대체하는 이유가
정확히 이것이다.

**분할은 atomic append로 한다.** TSDF 적분은 순서 무관이므로 안정 분할이 필요 없고, 그래서
prefix sum이 필요 없다.

---

## 6. 패스 구성 (dispatch 3개)

| # | 스레드 | 하는 일 |
|---|---|---|
| 1 | 점당 | 블록 해시 삽입 → `blockIndex[i]` 기록; 굵은/미세 셀 해시 삽입(신규만 카운트); `pointCount`·`sumNormal` 누적 |
| 2 | 블록당 | §3 판정 → `isDense` 플래그. dense 블록의 `fineOccupied`를 합산해 detail 용량 산출 |
| 3 | 점당 | `blockIndex[i]`로 플래그 조회 → `baseIndex`/`detailIndex`에 atomic append |

세 패스 모두 호출자의 `CommandBatch`에 기록만 하고 자체 제출하지 않는다. 블록 레코드의 통계는
프레임 간 누적되고, 두 셀 해시는 1번 패스 시작 시 비워진다(§5).

`fineOccupied`가 프레임별 합이므로 §4의 detail 용량 산출은 합이 아니라 **프레임당 최댓값**을
써야 한다 — 같은 셀을 여러 프레임에서 다시 세기 때문이다. 블록 레코드에 `fineOccupiedMax`를
하나 더 두어 1번 패스에서 `atomicMax`로 갱신한다.

---

## 7. 결정된 사항

### 7-1. 판정은 래치하되, 최소 관측량을 요구한다

레벨이 스캔 도중 오가면 seam이 생기므로 **한번 dense면 되돌리지 않는다** (현행 동작과 같다).
다만 현행은 첫 프레임의 부분 데이터로도 확정될 수 있다. 통계를 프레임 간 누적하고, 판정식에
`fineOccupied >= 64`를 넣어 절대 하한을 둔다 — 32³ 블록에서 미세 셀 64개는 표면이 실제로 지나가야
나오는 수치다. 별도의 프레임 카운터는 두지 않는다.

### 7-2. 임계값은 판정기와 한 몸으로 둔다

세 임계값(3.2 / 0.9 / 3)과 하한 64를 `DensityCriteria` 구조체 하나에 모아
`SubmapStrategy` 헤더에 둔다. `VolumeParams`에는 올리지 않는다 — 아직 쓸어볼 축이 아니다(YAGNI).
해시 임계값에서 얻은 교훈과 같다: 임계값이 판정 로직과 떨어지면 서로 안 맞는 조합이 조용히 돈다.

### 7-3. 블록 경계 apron은 이번에 넣지 않는다

경계의 점은 통계가 양쪽 블록으로 갈려 각각 과소 추정된다. 1복셀 apron으로 겹쳐 세면 해결되지만
1번 패스의 쓰기가 배로 는다. 통계의 정밀도 문제이지 정확성 문제가 아니고, 블록이 32³이라 경계
비중이 작다. 측정 후 필요하면 넣는다.

---

## 8. 테스트 전략

| 무엇을 | 어떻게 |
|---|---|
| 비율이 간격을 재현한다 | 간격을 알고 만든 평면(`s = v/4`, `v/2`, `v`, `2v`)에서 `ratio`가 4, 4, 1, 1에 근접 |
| 평면은 refine하지 않는다 | 아무리 촘촘한 평면도 조건 2에서 탈락 (`|Σn|/count ≈ 1`) |
| 곡면은 refine한다 | 촘촘히 스캔한 구/에지가 세 조건을 모두 통과 |
| 희소 곡면은 refine하지 않는다 | 간격이 큰 곡면이 조건 1에서 탈락 |
| 분할이 점을 잃지 않는다 | `baseCount + detailCount == 입력 점 수` |
| 래치 | 한번 dense가 된 블록은 이후 프레임에서 뒤집히지 않는다 |
| 큰 프레임 | `maxPointPerFrame` 초과 프레임에서 버퍼가 성장하고 드롭이 0 |

마지막 두 줄이 회귀 방지의 핵심이다 — 각각 seam과 조용한 점 유실이라는, 증상 없이 결과만
망가지는 실패 모드에 대응한다.

---

## 9. 비목표

- 점별 `s(p)` 산출 — 블록 판정에 필요 없다. 다른 용도가 생기면 그때.
- 3레벨 이상 LOD — 지금은 base/detail 2레벨이다.
- crack/seam 메싱 처리 — 추출 단계의 문제이고 별도 사안이다.
- `Memory/` → `Structure/` 이행 자체 — 이 스펙은 `Structure/` 배치를 전제할 뿐 그 이행을 다루지 않는다.

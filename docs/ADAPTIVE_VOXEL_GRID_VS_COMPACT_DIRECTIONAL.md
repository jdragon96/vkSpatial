# AdaptiveVoxelGrid vs CompactDirectionalTSDF — 수식 비교

> 한 줄 요약: **둘은 서로 다른 축을 적응시킨다.** CompactDirectional은 **방향(orientation)** 을,
> AdaptiveVoxelGrid는 **해상도(resolution)** 를 적응시킨다. 두 축은 **직교(orthogonal)** 라 곱해서
> 합칠 수 있다("direction × resolution").

수식은 GitHub / 마크다운 뷰어에서 렌더됩니다.

---

## 0. 표기
- $h$: fine voxel 크기. $\tau$: truncation. $v$: voxel(정수 좌표). $d$: 방향 레이어(0..5).
- $A$: 표면적. 표면은 얇은 2D 껍질 → 점유 fine voxel 수 $N \sim A/h^2$.

---

## 1. 필드 표현 (무엇을 저장하나)

| | CompactDirectionalTSDF | AdaptiveVoxelGrid |
|---|---|---|
| 필드 | **방향별** $\Psi(v,d)$, 6개 부호축 레이어 | **단일** $D(v)$, 방향 없음 |
| 해상도 | **고정** $h$ | **가변** $h$(fine) 또는 $2h$(coarse) |
| 적응 축 | **방향(orientation)** | **해상도(resolution)** |
| 엔트리 | `DirEntry{key, ΣDW, ΣW, pad}` 16B | fine `{ΣDW, ΣW, ΣD²}` + coarse `{D̄, W}` |

**CompactDirectional** — 관측을 법선으로 여러 방향 레이어에 배분:
$$\Psi(v,d) = \frac{\sum \psi\,\varphi\,\rho_d}{\sum \varphi\,\rho_d},\qquad \rho_d = \Big(\tfrac{a_d}{a_{\max}}\Big)^{p_e}$$
(방향 선택 TopK; $a_d=|n\cdot e_d|$). 모서리를 방향 분리로 보존.

**AdaptiveVoxelGrid** — 단일 필드지만 voxel별 분산으로 해상도 결정:
$$D(v) = \frac{\sum_k d_k}{N},\qquad \sigma^2(v) = \frac{\sum_k d_k^2}{N} - D(v)^2$$
$$\bar\sigma^2_c = \frac{1}{|F_c|}\sum_{i\in F_c}\sigma^2_i,\qquad \bar\sigma^2_c < \theta \;\Rightarrow\; \text{coarse}$$
평탄부(저변이)를 $2h$로 합쳐 메모리 절감.

---

## 2. 저장 비용 (메모리 모델)

**CompactDirectional** — 단일 해상도 $h$, 방향 오버헤드:
$$M_{\text{cmp}} = N_{\text{dir}}\cdot 16\text{B},\qquad N_{\text{dir}} = \kappa\,N,\ \ \kappa \approx 1.1\text{–}1.4$$
($\kappa$ = voxel당 평균 방향 수; 대부분 단일 방향, 모서리에서 >1.)

**AdaptiveVoxelGrid** — 단일 방향, 가변 해상도. 점유 fine voxel을 고변이 $N_{\text{fine}}$ 와 저변이(coarse화) $N_{\text{flat}}$ 로 나누면, coarse는 $2^3=8$개 fine을 1개로:
$$M_{\text{adp}} = \big(N_{\text{fine}} + \tfrac{N_{\text{flat}}}{8}\big)\cdot s$$
평탄 비율 $f = N_{\text{flat}}/N$ 이면 유효 절감:
$$\frac{M_{\text{adp}}}{s\,N} = (1-f) + \frac{f}{8} = 1 - \frac{7f}{8}$$
→ 완전 평탄($f\to1$)이면 **8× 절감**, 논문 측정 RGB-D 1.95–4.07× (실제 장면 $f$).

**핵심 대비:**
$$\text{Compact: } M \propto \kappa\,N \ (\text{방향으로 } \uparrow),\qquad \text{Adaptive: } M \propto (1-\tfrac{7f}{8})N \ (\text{해상도로 } \downarrow)$$
**정반대 방향** — Compact은 모서리에 메모리를 더 쓰고, Adaptive는 평탄부에서 덜 쓴다.

---

## 3. 무엇을 최적화하나 (정확도 특성)

| | CompactDirectional | AdaptiveVoxelGrid |
|---|---|---|
| 강점 | **날카로운 모서리/얇은 구조** (방향 분리) | **평탄/유기적 영역 메모리** (해상도 병합) |
| 모서리 | 방향 레이어로 보존 | 단일 필드 → **고변이라 fine 유지되어** 보존(간접), 방향 분리는 없음 |
| 평탄부 | SimpleTSDF 대비 방향으로 정확 | fine면 정확, coarse면 $2h$ 해상도(평탄이라 손실 미미) |
| 법선 | 방향/gradient 기반 | MC gradient (단일 필드) |

- Compact의 모서리 보존은 **표현(6 레이어)에서 직접** 나온다.
- Adaptive의 모서리 보존은 **간접** — 모서리=고변이라 coarse로 안 내려가서 fine으로 남을 뿐, 필드는 여전히 단일(방향 뭉갬은 SimpleTSDF와 동일). 따라서 **Adaptive는 메모리를, Compact은 피처 정확도를 얻는다.**

---

## 4. 추출 (extraction)

| | CompactDirectional | AdaptiveVoxelGrid |
|---|---|---|
| 방식 | 방향별 zero-crossing 포인트(옵션 merge) | **멀티해상도 Marching Cubes** (메시) |
| 해상도 경계 | 없음(단일 해상도) | **transitional voxel 처리 필요**(fine↔coarse 균열) |
| 난점 | 방향 간 중복 병합 | finer-우선 SDF 보간 + coarse truncate + vertex collapse |

Adaptive는 단일 해상도가 아니라서 **경계 처리**(논문 Fig 5/6)가 추가 난제 — Compact에는 없는 비용.

---

## 5. 하드웨어 제약

| | CompactDirectional | AdaptiveVoxelGrid |
|---|---|---|
| 키 | 32bit 팩(9b/축) → **512³ 이동창** | fine 해시는 SimpleTSDF와 동일 제약(512³ 창) |
| 원인 | int64 atomics 없음 | 동일(fine GPU 해시); coarse는 CPU측이라 무관 |
| 대안 | Tiled로 확장 | 논문은 풀레인지 버킷 해시(우리 첫 구현은 창 내) |

둘 다 GPU fine 해시는 같은 512³ 창 제약을 공유(같은 HW 이유). Adaptive의 coarse 레벨은 CPU측 mixed grid라 제약 밖.

---

## 6. 직교성 — 합칠 수 있다

Compact은 **방향** $d$ 를, Adaptive는 **해상도** $\ell$ 를 적응시킨다. 두 인덱스는 독립:
$$\text{저장 단위} = (v,\ d,\ \ell)\quad\Rightarrow\quad \text{"방향 × 해상도"}$$
- 고변이 모서리: 여러 방향 레이어 + fine 해상도.
- 저변이 평탄: 단일 방향 + coarse 해상도.

→ 모서리 보존(Compact)과 평탄부 절감(Adaptive)을 **동시에**. 이 결합은 이미 오프라인으로 프로토타입됨(`tsdf_benchmark`의 `RunCompactVarianceAdaptive` — Directional 정확도 @ Simple 이하 메모리, [[VARIANCE_ADAPTIVE_VOXEL_GRID]] 참고). `AdaptiveVoxelGrid`는 그 **해상도 축을 런타임 자료구조로** 만드는 것이고, CompactDirectional은 **방향 축의 런타임 자료구조**다.

---

## 7. 요약

| 관점 | CompactDirectionalTSDF | AdaptiveVoxelGrid |
|---|---|---|
| 적응 축 | 방향(orientation) | 해상도(resolution) |
| 메모리 | $\kappa N$ (모서리 ↑) | $(1-\tfrac{7f}{8})N$ (평탄 ↓) |
| 노림수 | 날카로운 피처 정확도 | 평탄부 메모리 효율 |
| 필드 | 6 방향 레이어 | 단일 필드, 가변 크기 |
| 추출 | 방향별 zero-crossing | 멀티해상도 MC(경계 처리) |
| 결합 | — | 직교 → "방향 × 해상도"로 합침 |

**한 줄 결론:** 같은 hash-TSDF 계열이지만 **CompactDirectional = 방향 적응(피처 정확도), AdaptiveVoxelGrid = 해상도 적응(메모리 효율)**. 상충이 아니라 **직교**라, 궁극적으로 곱해서 둘 다 얻는 게 목표다.

---

## 참조
- 논문: `docs/Variance-Adaptive Voxel Grids.pdf` (De Rebotti et al. 2025, MrHash)
- 관련: `docs/VARIANCE_ADAPTIVE_VOXEL_GRID.md`, `docs/COMPACT_VS_DIRECTIONAL_TSDF.md`, `docs/DIRECTIONAL_TSDF_INTEGRATION.md`
- 설계: `docs/superpowers/specs/2026-07-26-adaptive-voxel-grid-design.md`
- 구현: `src/Engine/Spatial/CompactDirectionalTSDF.{h,cpp}`, `src/Engine/Spatial/AdaptiveVoxelGrid.{h,cpp}` (예정)

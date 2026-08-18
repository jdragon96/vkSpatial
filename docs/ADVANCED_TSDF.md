# AdvancedTSDF — 적용 기법 정리

> **한 줄 요약:** `AdvancedTSDF`는 이 저장소의 여러 TSDF 실험에서 **측정으로 검증된 최적 조합**만 뽑아 하나로 합친 정본(canonical) 클래스다. 핵심은 넷 — **① compact per-(voxel,direction) flat-hash 저장**(block 대비 ~11× 저메모리), **② 6축 방향 레이어링**(모서리·얇은 구조 보존), **③ point-to-plane 적분**(평면 거의 완벽, 모서리 개선), **④ stored-gradient mode-3 추출**(denoised 법선 + 서브복셀 위치). 결과: *block-DirectionalTSDF 급 정확도를 ~11× 적은 메모리로*.

> 대상: [`src/TSDF/Backends/AdvancedTSDF.{h,cpp}`](../src/TSDF/Backends/AdvancedTSDF.h), 셰이더 [`src/shader/advanced_tsdf_{integrate,extract}.vert.glsl`](../src/shader/advanced_tsdf_integrate.vert.glsl).
> 수식은 GitHub/마크다운 뷰어에서 렌더됩니다.

---

## 0. 파이프라인 개요

```
point cloud (+normals, camera)
        │  Integrate  (advanced_tsdf_integrate.vert.glsl, 1 thread / point)
        ▼
  GPU flat hash:  DirEntry{ key, sumDW, sumW, sumNx, sumNy, sumNz }  (24B, open-addressing)
        │  Extract  (advanced_tsdf_extract.vert.glsl, 1 thread / hash slot)
        ▼
  oriented points {position, normal}  →  (opt.) MergeCandidates  →  OrientedPointCloud
```

각 (voxel, 방향) 칸은 **가중 평균 TSDF**($\text{sumDW}/\text{sumW}$)와 **누적 관측 법선**($\text{sumN}$)을 fixed-point로 담는다. 적분·추출 모두 GPU 컴퓨트 셰이더이고, 셰이더는 런타임에 shaderc로 컴파일된다(파일명 `.vert.glsl`이라도 스테이지는 compute로 강제).

---

## 1. 저장 — compact per-(voxel,direction) flat hash

**무엇:** 표면이 지나가는 **(voxel, 방향) 칸만** open-addressing 해시(`wangHash` + linear probing)에 낱개로 저장. voxel은 이동식 $512^3$ 창의 로컬 좌표(축 9-bit) + 방향 3-bit로 32-bit 키에 팩킹([`packDirKey`](../src/shader/advanced_tsdf_integrate.vert.glsl)).

**왜:** 대안인 **block(8³=512 voxel 그룹)** 저장은 얇은 표면이 뚱뚱한 블록을 지날 때 대부분의 칸이 비어 **~56× 낭비**(실측). flat-hash는 빈 칸을 저장하지 않아 낭비 ≈ 0.

$$
M_{\text{block}} = B\cdot512\cdot s,\qquad M_{\text{compact}} = N_{occ}\cdot s'
$$

실측(합성 cube/cylinder): block DirectionalTSDF **6144/6784 KB** vs AdvancedTSDF **566/622 KB** = **~10.9× 감소**, 정확도는 동등 이상. 자세한 유도는 [`COMPACT_VS_DIRECTIONAL_TSDF.md`](COMPACT_VS_DIRECTIONAL_TSDF.md).

**대가:** 32-bit 키 → 한 번에 $512^3$ voxel **창**만 다룸(원점 이동식). 초과 장면은 타일링([`TiledCompactDirectionalTSDF`](../src/TSDF/Backends/TiledCompactDirectionalTSDF.h)).

### 24-byte 엔트리 설계
```cpp
struct AdvDirEntry { uint32_t key; int32_t sumDW; uint32_t sumW; int32_t sumNx, sumNy, sumNz; }; // 24B
```
- `sumDW = Σ tsdf·w·SCALE`(분자), `sumW = Σ w·SCALE`(분모) → 값 = `sumDW/sumW`. **둘을 나눠 저장**하는 이유: `atomicAdd` 두 번(정수 fixed-point)으로 **lock-free 병렬 적분**이 되기 때문(부동소수 atomic·CAS 루프 불필요). `sumNx/y/z`도 동일 원리.
- 모두 4-byte 스칼라 → std430 stride 24 = C++ `sizeof` (숨은 패딩 없음), `static_assert(sizeof==24)`로 셰이더 미러와 정합 강제.

---

## 2. 방향 레이어링 (6 signed axes)

**무엇:** 각 점의 법선을 6개 **부호축**($\pm X,\pm Y,\pm Z$) 중 정렬 강한 **최대 3개** 레이어로 나눠, 같은 voxel이라도 방향이 다르면 **다른 칸**에 적분([`selectDirections`](../src/shader/advanced_tsdf_integrate.vert.glsl); Splietker & Behnke 2019).

**왜:** 서로 **반대/수직인 표면**(얇은 벽, 날카로운 모서리)이 한 voxel에서 부호가 상쇄되어 사라지는 것을 막는다(anti-aliasing). $\rho_i = (|n_i|/|n_{\max}|)^{p_e}$ 가 5% 미만인 약한 축은 버려 낭비를 줄인다. 실측상 voxel당 평균 **1.1–1.4 방향**만 실제로 채워진다.

---

## 3. Integrate — point-to-plane 부호거리

**무엇:** voxel 중심 $x_v$의 부호거리를, 레이 투영이 아니라 **표면 접평면까지의 거리**로 계산([`Integrate`](../src/TSDF/Backends/kernel_AdvancedTSDF.integrate.comp.glsl), `g_pointToPlane` 토글, 기본 on):

$$
\psi_{\text{p2p}} = \operatorname{clamp}\!\Big(\frac{(x_v - p)\cdot \hat n}{\tau},\,-1,1\Big)
\quad\text{vs}\quad
\psi_{\text{proj}} = \operatorname{clamp}\!\Big(\frac{d - (x_v-\text{cam})\cdot \hat r}{\tau},\,-1,1\Big)
$$

$p$=관측점, $\hat n$=점 법선(단위), $\hat r$=레이, $d$=depth, $\tau$=truncation. 부호는 둘 다 **카메라 쪽 = +**.

**왜:** 투영식 $\psi_{\text{proj}}$은 표면을 **비스듬히(grazing) 볼수록 등가면이 실제 표면에서 밀린다**(KinectFusion식 편향). point-to-plane은 평면에 대해 **정확**하고 grazing 편향이 없다.

**측정(2026-07, A/B):** 투영 → point-to-plane 전환 시
- **flat(평면) 오차 거의 0**: cube 0.011 → **0.00001** mm, cyl 0.012 → 0.0016
- **edge −15~35%**, **전 영역 mean −45~62%**, 적분 시간 비용 **0**.

> ⚠️ **위 수치의 조건**: 전부 **합성·무노이즈·정확한 법선** fixture이고, 카메라가 각 면을 **거의 수직으로**
> 보는 배치다(cube/cyl을 축 방향에서 관측). 즉 "grazing 편향 제거"는 **수식의 성질**로는 참이지만,
> 2026-08까지의 **구현**에서는 grazing에서 성립하지 않았다 — 아래 참조.

### grazing에서 밴드가 잘리던 버그 (2026-08 수정)

밴드를 **레이 방향으로 행진**하면서 소속 판정은 **법선 방향으로** 재고 있었다. 두 축은 수직 입사에서만
일치하고, 입사각 $\theta$에서 법선 방향 도달 거리는 $\text{steps}\cdot v\cdot\cos\theta$ 뿐이므로
$\pm\tau$ 밴드가 $\cos\theta$ 배로 잘렸다. voxel 0.05 / $\tau$ 0.15에서 손익분기는 $\theta=41.4°$.

`AdvancedTSDF.PointToPlaneBandFillsTheFullTruncationDepthAtEveryIncidence` 실측(밴드 층 6개 기준):

| 입사각 | 0° | 30° | 45° | 60° | 75° |
|---|---|---|---|---|---|
| 수정 전 coverage | 100% | 100% | 100% | **83.3%** | **50%** |
| 수정 전 밴드깊이 | 0.99 | 0.99 | 0.99 | **0.68** | **0.35** |
| 수정 후 (양쪽) | 100% | 100% | 100% | 100% | 100% |

표면 **위치**는 잘려도 정확했다(밴드가 $t$에 대해 대칭이라 zero-crossing이 제자리) — 손실은
**완전성**과 **시점 불변성**이다. p2p 맵은 정의상 카메라와 무관해야 하는데(값 $ (x_v-p)\cdot\hat n$에
카메라가 없다), 실제로는 같은 평면을 0°와 75°에서 본 맵이 1045 vs 573 엔트리로 달랐다.
`AdvancedTSDF.PointToPlaneMapDoesNotDependOnTheViewpoint`가 이것을 byte-identical로 고정한다.

**수정:** p2p일 때 `samplePos = point + unitNormal * (t*voxelSize)` (projective는 레이 행진 유지).
샘플 수 불변 → **비용 0**. `steps`만 늘리는 대안은 손익분기 각도만 밀어내고 시점 의존성은 남으므로
기각(뮤테이션으로 확인).

**실데이터 GT 측정** (`tsdf_folder_eval --dir scan_out --voxel 0.5`, GT `scan_out/ground_truth.ply`):

| | 추출점 | occupied | accuracy mean | precision@1vox | recall@1vox |
|---|---|---|---|---|---|
| 수정 전 p2p | 23368 | 329371 | 0.32503 | 0.849 | 0.126 |
| projective | 21311 | 252808 | 0.30467 | 0.886 | 0.125 |
| **수정 후 p2p** | 7664 | 109602 | **0.17886** | **0.998** | 0.104 |

**판정이 뒤집혔다**: 수정 전 p2p는 projective보다 나빴고(0.325 vs 0.305 — 문서가 주장하는 것과 반대),
수정 후 41% 낫다. precision 0.998은 추출점이 거의 전부 GT 1복셀 이내라는 뜻이다.
**정직하게: recall은 0.126→0.104로 떨어진다.** 없어진 것은 밴드 깊이가 아니라(위 표대로 100%)
레이 행진이 만들던 **접선 방향 smear**다 — 남의 접평면 값으로 옆 복셀을 채우던 것이라 기하적 근거는
없지만 커버리지에는 기여했다. 정확도 우선으로 채택.

projective 경로는 수정 전후 **byte-identical**(수정이 분기 밖으로 새지 않았다는 회귀 검사).

> 부수 기법: **degenerate 법선 가드** — 입력 법선이 0이면 그 점을 skip(`normalize(0)`의 NaN·해시 오염 방지). 방향 선택·뷰가중·point-to-plane이 모두 단위 법선을 전제하므로 main 진입부에서 한 번 걸러낸다.

### 가중치
관측당 가중치 $w = \varphi\cdot\rho_{d}$ — **시야각 신뢰도** $\varphi=\max(0,\hat n\cdot(-\hat r))$(back-facing/grazing은 0 → 칸 자체를 안 만듦) × **방향 상대가중** $\rho_d$. 가중 평균이라 다중 관측이 자연히 융합된다.

### stored gradient 누적
같은 루프에서 관측 법선을 **가중 누적**: $\text{sumN}\mathrel{+}=\hat n\cdot w\cdot\text{SCALE}$ (raw 누적, 정규화는 추출에서). 이게 §4의 denoised 법선을 만든다.

---

## 4. Extract — stored-gradient mode-3 hybrid

추출은 두 부분을 **각각 최적 소스**에서 취한다([`main`](../src/shader/advanced_tsdf_extract.vert.glsl)):

**위치 = legacy 축 zero-crossing 보간** (`estimateCrossingPosition`). 같은 방향 레이어의 +축 이웃과 부호가 바뀌는 지점을 선형 보간 → 서브복셀 위치. (실험상 저장 gradient로 위치를 투영하는 mode-2는 위치를 ~10× 악화시켜 **위치는 legacy가 최적**.)

**법선 = 저장 gradient** $\hat g = \operatorname{normalize}(\text{sumN})$. 퇴화 시에만 중앙차분 gradient로 fallback(`estimateNormalFallback`).

$$
\text{position} = \text{zeroCrossing}(v,d),\qquad
\text{normal} = \frac{\text{sumN}}{\lVert\text{sumN}\rVert}
$$

**왜:** 중앙차분 법선은 이산 SDF장에서 뽑아 **양자화·노이즈**가 크고 이웃 의존적이다. 저장 gradient는 **관측 법선의 가중 평균**이라 denoised·정확하고, voxel-local(이웃 불필요)이다. FPFH/정합이 실제로 소비하는 게 법선이라 이 항목의 실익이 크다.

**측정:** cube 평면 법선각 **1.88° → 0.00°**(block-path), 곡면(cylinder) 법선 오차 **0.0104 vs 0.0121**(중앙차분 대비 개선). 단위 테스트 `CurvedCylinderNormalsAreRadial`가 곡면에서 저장-gradient 경로를 판별.

> **코너 보존 병합**(옵션, `merge=true`, [`MergeCandidates`](../src/TSDF/Backends/AdvancedTSDF.h)): 같은 voxel 버킷의 후보를 위치 0.6·voxel + 법선 30° 내에서 클러스터링하되, 60° 초과(코너)는 절대 병합 안 함.

---

## 5. 이동식 창 — 중심 정렬 기본값 (footgun 수정)

**무엇:** 명시 창을 안 주면 $512^3$ 창을 **원점 중심**(originVoxel = −256)에 놓는다 — **voxelSize와 무관**([`Build`](../src/TSDF/Backends/AdvancedTSDF.cpp)).

**왜:** 선행 `CompactDirectionalTSDF`의 기본 corner `(-25.6,…)`는 `floor(-25.6/voxel)`이라 **voxel 0.1에서만** 창이 원점을 덮고, 0.05에서는 창이 $[-25.6,0)$로 치우쳐 **원점을 지나는 표면이 조용히 잘렸다**. AdvancedTSDF는 origin voxel을 −256로 고정해 어떤 해상도든 $[-256v,\,256v)$로 원점을 중심에 둔다. 단위 테스트 `DefaultWindowCentredAtAnyVoxelSize`가 검증.

---

## 6. 측정 결과 (tsdf_benchmark, 합성 analytic GT)

| shape | method | mem_KB | acc_rmse | edge | flat | curved |
|---|---|---:|---:|---:|---:|---:|
| cube | Directional (block) | 6144 | 0.02223 | 0.02672 | 0.01098 | — |
| cube | **Advanced** | **565.88** | **0.01893** | 0.02433 | **0.00001** | — |
| cyl | Directional (block) | 6784 | 0.02678 | 0.02988 | 0.01184 | 0.01212 |
| cyl | **Advanced** | **621.84** | **0.01141** | 0.01239 | **0.00001** | **0.00289** |

→ **block-DirectionalTSDF 급(이상) 정확도**를 **~10.9× 적은 메모리**로. `Advanced` 행은 머지된 `Compact-Directional` v1과 **byte-identical**(정본 클래스가 동일 레시피임을 확인). (`--p2p` 없이 실행하면 투영 fallback 측정.)

---

## 7. 넣지 않은 것 (measure-first 원칙 → 로드맵)

검증 없이 "개선"으로 넣으면 회귀 위험이 있어 **의도적으로 제외**하고 별도 sub-project로 남김:
- **robust/Huber weighting (A2)** — cylinder RMSE의 소수 outlier 꼬리(전 영역 mean은 급감했으나 제곱합 tail 유지)를 잡는 항목. 미측정.
- **variance-adaptive coarsening (v1.5)** — flat=coarse/dense=fine. 벤치마크상 41×↓ 가능성 확인됨(별도 프로덕션화 필요).
- **at-rest 양자화(oct16 법선 + fp16) + full streaming(host 권위 pool LRU)** — 추가 메모리·대규모 장면. host/at-rest 계층이 필요(compact-v1은 GPU-resident, 이동식 창이 이미 bounded-VRAM). [`2026-07-26-highprecision-submap-tsdf-design.md`](superpowers/specs/2026-07-26-highprecision-submap-tsdf-design.md).
- **실시간 ICP odometry** — 저장 gradient를 직접 트래킹에 쓰는 설계(spec 존재, 미구현): [`2026-07-27-direct-tsdf-gradient-odometry-design.md`](superpowers/specs/2026-07-27-direct-tsdf-gradient-odometry-design.md).

---

## 8. 요약 표

| 기법 | 위치 | 효과 | 측정 |
|---|---|---|---|
| compact flat-hash | 저장 | 메모리 ~11×↓ (block 대비) | ✅ 566 vs 6144 KB |
| 6축 방향 레이어 | integrate | 모서리·얇은 구조 상쇄 방지 | ✅ (구조적) |
| point-to-plane | integrate | flat 거의 완벽, edge↓, grazing 편향 제거 | ✅ flat −99.9% |
| stored-gradient mode-3 | extract | denoised 법선 + 서브복셀 위치 | ✅ 법선 1.88°→0° |
| 24B 분자/분모 엔트리 | 저장 | lock-free 정수 atomic 적분 | ✅ (설계) |
| 시야각·방향 가중 | integrate | 다중관측 가중융합·노이즈↓ | ✅ (구조적) |
| degenerate 법선 가드 | integrate | NaN/해시 오염 방지 | ✅ (가드) |
| 중심 정렬 창 | Build | voxelSize 무관 원점 커버(footgun) | ✅ 테스트 |
| 코너 보존 merge | extract | 중복 dedup·코너 유지 | ✅ (옵션) |

## 9. 참조
- 구현: [`AdvancedTSDF.{h,cpp}`](../src/TSDF/Backends/AdvancedTSDF.h), [`advanced_tsdf_{integrate,extract}.vert.glsl`](../src/shader/advanced_tsdf_integrate.vert.glsl), 테스트 [`test/test_advancedTsdf.cpp`](../test/test_advancedTsdf.cpp).
- 관련 문서: [`COMPACT_VS_DIRECTIONAL_TSDF.md`](COMPACT_VS_DIRECTIONAL_TSDF.md)(compact vs block 메모리), [`DIRECTIONAL_TSDF_INTEGRATION.md`](DIRECTIONAL_TSDF_INTEGRATION.md)(방향 적분 수식), [`DB_TSDF_VS_COMPACT_DIRECTIONAL.md`](DB_TSDF_VS_COMPACT_DIRECTIONAL.md)(투영거리 편향·occlusion 대조), 설계 [`superpowers/specs/2026-07-27-compact-best-tsdf-v1-design.md`](superpowers/specs/2026-07-27-compact-best-tsdf-v1-design.md).
- 문헌: Splietker & Behnke, *Directional TSDF* (2019); Sommer et al., *Gradient-SDF* (CVPR 2022); Newcombe et al., *KinectFusion* (ISMAR 2011).

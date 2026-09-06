# AdvancedTSDF 개선 A1·A2 — 근거 & 설계 (measure-first)

> **목표:** AdvancedTSDF의 extraction RMSE를 더 낮춘다. 최신 논문 근거로 두 저비용 항목을 **토글로 넣고 A/B 측정** 후 이득이 있으면 채택한다(검증 없이 기본 on 하지 않음).
> - **A1 — 불확실성 가중(uncertainty weighting):** 관측을 신뢰도로 가중해 다중관측·노이즈에서 표면을 sharpen.
> - **A2 — Hermite(gradient-augmented) 위치 보간:** 저장된 gradient로 zero-crossing을 선형→3차 Hermite로 올려 **곡면 서브복셀 위치** 개선.

> 대상: [`AdvancedTSDF`](../src/TSDF/Backends/AdvancedTSDF.h), 셰이더 [`kernel_AdvancedTSDF.{integrate,extract}.comp.glsl`](../src/TSDF/Backends/kernel_AdvancedTSDF.integrate.comp.glsl). 배경: [`ADVANCED_TSDF.md`](ADVANCED_TSDF.md).

---

## 0. 현재 상태(측정) — 어디가 약점인가

`tsdf_benchmark`(합성 GT) 실측:

| shape | acc_rmse | edge | flat | curved |
|---|---:|---:|---:|---:|
| cube | 0.01893 | 0.02433 | 0.00001 | — |
| cyl | 0.01141 | 0.01239 | 0.00001 | 0.00289 |

- **flat 거의 완벽**(point-to-plane), **curved도 양호**(0.00289).
- **RMSE를 지배하는 건 edge**(cube 0.024, cyl 0.012) — 방향 불연속·zero-crossing 한계.
- point-to-plane A/B에서 관측된 현상: 전 영역 mean은 급감했으나 **소수 outlier가 RMSE(제곱합)를 유지** → 관측 신뢰도/보간 정밀도로 공략할 여지.

> 정직한 기대치: 합성·무노이즈 fixture에서는 이득이 **작을 수 있다**(특히 A1은 다중관측 불일치가 있어야 효과가 큼). 실이득은 **실제 노이즈 스캔**에서 더 크다. 그래서 measure-first — 숫자로 판단한다.

---

## A1. 불확실성 가중 (uncertainty-weighted fusion)

### 현재
integrate 가중치는 `w = φ · ρ_d` — 시야각 신뢰도 $\varphi=\max(0,\hat n\cdot(-\hat r))$(입사각 cos) × 방향 상대가중 $\rho_d$. **표면으로부터의 거리(밴드 내 위치)는 가중에 반영 안 됨** → 밴드 가장자리(|tsdf|→1)의 불확실한 관측이 표면 근처(|tsdf|→0)의 확실한 관측과 **동일 가중**.

### 근거
확률적 TSDF 융합의 표준: 관측 신뢰도를 evidence/가중에 반영. 센서 깊이 노이즈 하에서 표면은 **측정점 근처에 있을 확률이 가장 높음** → 부호거리의 likelihood로 가중.
- [PSDF Fusion (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Wei_Dong_Probabilistic_Signed_Distance_ECCV_2018_paper.pdf) — probabilistic SDF + evidence counter.
- 2024–2025 uncertainty-aware TSDF 흐름(uncertainty quantification을 evidence/occupancy에 반영).

### 설계
관측당 **표면-근접 신뢰도**를 곱한다:
$$
w \;=\; \varphi \cdot \rho_d \cdot c(\psi),\qquad
c(\psi) = 1 - \lambda\,|\psi|,\quad \psi=\text{clamp}(\text{sdf}/\tau,-1,1),\ \lambda\in[0,1]
$$
$\lambda=0$ = 현재(균일), $\lambda=1$ = 완전 램프(밴드 가장자리 가중 0). 가중 평균 $\Psi=\sum\psi w/\sum w$이므로, 같은 voxel에 서로 다른 $\psi$의 관측이 쌓일 때 **표면 근처(확실한) 관측 쪽으로 값이 sharpen**된다 → zero-crossing이 날카로워져 outlier↓.
- push-const `g_confWeight`($=\lambda$), 호스트 `SetConfidenceWeight(float)`, 기본 0(off).
- (실데이터 확장) 거리 항 $1/\text{depth}^2$ 추가 가능 — fixture는 궤도 반경 고정이라 무영향이므로 이번엔 제외.

### 기대·판정
같은 voxel에 다중 관측이 겹치는 **모서리/곡면**에서 RMSE 소폭↓ 기대. 무노이즈 fixture에선 작을 수 있음 — **A/B로 $\lambda\in\{0,0.5,1\}$ 측정**해 개선 없으면 기각.

---

## A2. Hermite (gradient-augmented) 위치 보간

### 현재
extract 위치는 **선형 zero-crossing**: voxel $v$(값 $c_0$)와 +축 이웃(값 $c_1$) 사이에서 $t=c_0/(c_0-c_1)$로 보간 → SDF가 두 voxel 사이 **선형**이라 가정. **곡면에서는 이 가정이 깨진다**.

### 근거
저장된 gradient를 보간에 활용하면 SDF 복원이 더 정확. [∇-SDF / OREN (arXiv 2510.18999, 2025)](https://arxiv.org/abs/2510.18999)는 **gradient-augmented octree 보간**으로 연속 SDF를 정확히 복원. 우리는 이미 voxel마다 gradient(`sumN`)를 저장하므로, 값만 쓰는 선형 보간을 **양끝 미분을 쓰는 3차 Hermite**로 올릴 수 있다(고전 Hermite/QEF 계열).

### 설계
$\nabla\text{sdf}=\hat n$(단위)이므로 축 $a$ 방향 미분 $\partial(\text{sdf})/\partial a = \hat n_a$. 정규화 값 기준 파라미터(voxel 스텝) 미분:
$$
g_0 = \hat n^{(0)}_a\cdot \frac{\text{voxel}}{\tau},\qquad
g_1 = \hat n^{(1)}_a\cdot \frac{\text{voxel}}{\tau}
$$
3차 Hermite $p(t)=h_{00}c_0+h_{10}g_0+h_{01}c_1+h_{11}g_1$의 근 $t^\*\in[0,1]$을 **선형 추정 $t=c_0/(c_0-c_1)$에서 Newton 2–3회**로 구해 위치를 잡는다(선형 대비 곡면에서 정확). 축 방향 위치 = $(v+0.5)\text{voxel} + t^\*\,\text{voxel}$.
- 이웃의 gradient가 필요 → `fetchDirectionalValue`를 **값+법선** 반환으로 확장.
- extract PC에 `g_truncation` 추가($\tau$ 필요). push-const `g_hermite`(0/1), 호스트 `SetHermitePosition(bool)`, 기본 off.
- **모서리 주의:** 법선 불연속인 모서리에서는 Hermite가 이득이 없거나 해로울 수 있음 → 곡면 위주 개선. 밴드 안 성분만 사용(부호가 바뀌는 축만, 현재와 동일 게이트).

### 기대·판정
**curved 위치** 개선(이미 0.00289로 낮아 여지는 작음), 모서리는 중립 기대. **A/B로 cube/cyl 측정** — curved가 유의미하게 줄고 edge/flat 회귀 없으면 채택.

---

## 검증 계획 (A/B 매트릭스)

`tsdf_benchmark`에 `--conf <λ>`(A1)·`--hermite`(A2) 노출, Advanced 행을 4조합 측정:

| 구성 | 기대 |
|---|---|
| baseline (현재) | 기준 |
| +A1 (λ=0.5, 1.0) | 모서리/곡면 RMSE ≤ 기준, flat 무회귀 |
| +A2 | curved 위치 ≤ 기준, edge/flat 무회귀 |
| +A1+A2 | 둘의 합 |

- **게이트:** 어떤 조합도 flat(≈0)·기본 정확도를 회귀시키면 그 항목 기각. 개선이 측정 노이즈 이내면 기본 off로 유지(코드는 남기되 문서화).
- 회귀 테스트: `test_advancedTsdf` 전부 통과(토글 off 시 기존과 동일).
- 결과는 §"측정 결과"에 추가(측정 후 기입).

## 측정 결과 (2026-07, tsdf_benchmark --p2p — 도구는 이후 삭제됨; 현재 A/B는 `tsdf_folder_eval --no-p2p`와 `icp_quality_diag --no-p2p`로 한다)

| 구성 | cube rmse | cube edge | cube mean | cube nPts | cyl rmse | cyl curved |
|---|---:|---:|---:|---:|---:|---:|
| baseline (λ=0) | 0.01893 | 0.02433 | 0.00703 | 4773 | 0.01141 | 0.00289 |
| **+A1 λ=0.5** | **0.01352** (−29%) | 0.02211 (−9%) | 0.00348 (−51%) | 3693 | 0.01140 | 0.00301 |
| +A1 λ=1.0 | 0.01311 (−31%) | 0.02152 (−12%) | 0.00342 | 3144 | 0.01129 | 0.00303 |
| +A2 hermite | 0.01893 (=) | 0.02433 (=) | 0.00703 | 4773 | 0.01142 | 0.00295 |
| +A1+A2 | 0.01311 | 0.02152 | 0.00342 | 3144 | 0.01130 | 0.00311 |

### 판정
- **A1 = 채택.** cube에서 **RMSE −29%(λ=0.5) / −31%(λ=1.0)**, edge −9~12%, mean −51%. cylinder는 이미 낮아 중립. λ↑일수록 RMSE↓지만 밴드 가장자리 관측이 0가중되어 점 수↓(λ=1.0은 −34% 점). **기본값 λ=0.5**(RMSE 이득과 점 보존 균형). `SetConfidenceWeight(0)`로 비활성.
- **A2 = 기각(기본 off).** 이 합성·무노이즈 fixture에서 **측정 이득 없음**(cube 불변, cyl 곡면 +2~7% 노이즈 이내). 곡면이 이미 0.003으로 정확하고 인접 voxel 선형 보간이 near-optimal이기 때문. 코드·토글은 유지 — 거친 voxel/고곡률/노이즈 실데이터에서 재평가할 여지(∇-SDF 근거). `SetHermitePosition(true)`로 opt-in.
- 회귀: `test_advancedTsdf` 5/5 통과(토글은 정확도/방향 불변, 점 수만 변화).

### 정직한 해석
A1의 이득은 부분적으로 **밴드 가장자리(부정확) 관측을 걷어내 표면을 sharpen**하는 것(점 수↓, RMSE·mean↓)이며, edge까지 개선된 것은 단순 점 감소가 아닌 실제 정밀화. 무노이즈 합성에서도 −29%가 나왔으니 **실노이즈 스캔에선 더 클 것으로 기대**. A2는 이론적 근거는 타당하나 현 조건에선 데이터가 이득을 지지하지 않음 → measure-first 원칙대로 기본 off.

## capture/ 실데이터 측정 (2026-08)

위 표까지의 모든 수치는 합성·무노이즈·정확법선 fixture다. 실제 D435 녹화
`capture/`(477프레임, 640×480, depth 0.25–6.8 m)에서 point-to-plane을 처음 측정한 결과와,
그 과정에서 고친 것들:

1. **밴드 행진 방향 버그 수정** — p2p 밴드를 레이로 행진하며 법선으로 측정 →
   입사각 60°에서 밴드 83 %, 75°에서 50 %만 채워짐(실측). capture/는 입사각 median 44°/p90 67°.
   수정 후 전 각도 100 %. GT 있는 scan_out에서 accuracy 0.325→0.179(−45 %), precision@1vox
   0.849→0.998, **판정 역전**(수정 전 p2p는 projective보다 나빴다). 상세: `ADVANCED_TSDF.md` §3.
2. **depth 프리필터** (`DepthFilterOptions::prefilterWindow`, 기본 0=off) — 원본 depth의 1픽셀
   차분 법선은 이웃 간 median 24° 흔들림(캡처 실측). 5×5 discontinuity-aware mean으로 2.9°.
   p2p는 SDF 값과(수정 후) 밴드 방향이 모두 법선에 걸리므로 이것이 지배 오차였다.
3. **재현성** — lock-step 없이 동일 명령 4회가 path 1.5~137 m로 발산(측정, `ICP_REGISTRATION_QUALITY.md` §9.5).
   `FrameHandshake` + 타깃 정규 정렬로 bit-identical 재현 확인. 아래 수치는 전부 결정론적이다.

`icp_quality_diag --replay capture --voxel 0.05 --trackers icp`, 2×2 (수정 A 적용 후):

| 구성 | entries | step avg | step max | turn max | path | align ms |
|---|---:|---:|---:|---:|---:|---:|
| p2p ON, 원본 depth | 1,088,202 | 0.0475 | 0.259 | 19.2° | 11.73 | 320 |
| p2p OFF, 원본 | 191,630 | **0.0083** | **0.063** | 5.5° | **2.05** | 307 |
| **p2p ON, `--prefilter 5`** | 387,950 | 0.0246 | 0.225 | **5.06°** | 6.08 | 170 |
| p2p OFF, `--prefilter 5` | 717,218 | 0.0490 | 0.614 | 21.3° | 12.12 | 118 |

### 정직한 판정

- **p2p는 프리필터와 함께 써야 한다.** 원본 depth의 p2p는 여전히 궤적이 비물리적이고
  (turn max 19°/frame), 프리필터가 그것을 5.06°로 끌어내리며 entries도 2.8× 줄인다.
  법선이 SDF의 입력이므로 당연한 결합이다.
- **프리필터는 projective를 오히려 해친다**(path 2.05→12.1) — projective는 값에 법선을 안 쓰고,
  스무딩은 projective가 의존하는 고주파 depth 정보를 깎는다. 그래서 **전역 기본값은 계속 0**이고,
  p2p 경로에서만 명시적으로 켠다(`--prefilter 5`).
- 이 표의 궤적 지표는 트래커("icp")를 통과한 값이라 순수 TSDF 품질이 아니다. TSDF만의 GT 증명은
  scan_out(위 1번)이 담당한다.
- **미해결**: 모든 구성에서 477프레임 중 229개가 `LowOverlap`(minFitness 0.4)으로 거부된다.
  재현성 수정 후 거부 원인이 이 하나로 수렴했으므로(전에는 4개 원인에 산개), 다음 작업은
  이 게이트/대응점 비율의 원인 규명이다 — ICP 트래커 쪽 작업으로, 이 문서 범위 밖.

## 참조
- [PSDF Fusion (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Wei_Dong_Probabilistic_Signed_Distance_ECCV_2018_paper.pdf) · [∇-SDF/OREN (arXiv 2510.18999, 2025)](https://arxiv.org/abs/2510.18999) · [NKSR (CVPR 2023)](https://arxiv.org/abs/2305.19590) · [PIN-SLAM (T-RO 2024)](https://arxiv.org/abs/2401.09101)
- 내부: [`ADVANCED_TSDF.md`](ADVANCED_TSDF.md), 구현 [`AdvancedTSDF.{h,cpp}`](../src/TSDF/Backends/AdvancedTSDF.h), 셰이더 `advanced_tsdf_{integrate,extract}.vert.glsl`.

# DirectionalTSDF Integration — 수학 정리

실제 구현(`src/shader/compact_directional_integrate.comp`)을 그대로 수식화한 것입니다.
기호는 코드 변수와 1:1 대응됩니다.

> 수식은 GitHub / 마크다운 뷰어에서 렌더됩니다. 터미널에서는 `$…$` 가 원문으로 보일 수 있습니다.

---

## 0. 표기 (Notation)

**한 관측(observation)** = 법선이 있는 표면 점 하나:

| 기호 | 코드 | 의미 |
|---|---|---|
| $p \in \mathbb{R}^3$ | `p` | 표면 점 |
| $n \in \mathbb{R}^3,\ \lVert n\rVert=1$ | `nrm` | 단위 표면 법선 |
| $c \in \mathbb{R}^3$ | `cam` | 카메라 위치 |
| $h$ | `g_voxelSize` | voxel 크기 |
| $\tau$ | `g_truncation` | 절단 거리 |
| $p_e$ | `g_dirExponent` | 방향 지수 (기본 4) |
| $K$ | `g_maxDirections` | 최대 방향 수 (기본 3) |
| $\tau_\rho = 0.05$ | `MIN_RELATIVE_WEIGHT` | 상대 가중치 게이트 |

6개 **부호축 레이어**: $+X{=}0,\ -X{=}1,\ +Y{=}2,\ -Y{=}3,\ +Z{=}4,\ -Z{=}5$.

---

## 1. 광선과 깊이 (Ray & depth)

$$
\text{diff} = p - c, \qquad \delta = \lVert p - c\rVert, \qquad r = \frac{p-c}{\delta}\ \ (\lVert r\rVert = 1)
$$

$\delta < 10^{-6}$ 이면 관측 폐기.

---

## 2. 시야각 신뢰도 (View-angle confidence)

$$
\varphi =
\begin{cases}
\max\!\big(0,\ n \cdot (-r)\big) & \text{if viewAngleWeight} \\
1 & \text{otherwise}
\end{cases}
$$

- $n\cdot(-r) = \cos\angle(n,\ \text{카메라 방향})$ — 정면일수록 $\to 1$, 빗각 $\to 0$, 뒷면 $<0$.
- **폐기 조건:** $\varphi \le 0$ 이면 이 관측은 **어느 레이어에도 기여 안 함** (zero-weight 엔트리 방지).

**왜:** 빗각 측정은 깊이 불확실성이 크고 투영 SDF 근사가 가장 부정확하므로 낮게 신뢰하고,
뒷면($<0$)은 물리적으로 불가능한 관측이라 완전히 거부한다.

---

## 3. 방향 선택 (Direction selection, TopK)

각 주축 $i \in \{x,y,z\}$ 에 대해:

$$
\underbrace{a_i = |n_i| = |n\cdot e_i| = \cos\angle(n,\text{axis}_i)}_{\text{정렬 강도}},
\qquad
\underbrace{d_i = 2i + [\,n_i < 0\,]}_{\text{부호축 인덱스}},
\qquad
\underbrace{g_i = a_i^{\,p_e}}_{\text{배정 가중치}}
$$

강도 내림차순 정렬 후 최강축을 $a_{(1)}$, $g_{(1)} = a_{(1)}^{p_e}$ 라 하면 **상대 가중치**:

$$
\rho_i = \frac{g_i}{g_{(1)}} = \left(\frac{a_i}{a_{(1)}}\right)^{p_e} \in [0,1]
$$

**선택 집합** $\mathcal{D}$ (내림차순, 최대 $K$개):

$$
\mathcal{D} = \Big\{\, d_{(1)} \,\Big\} \cup \Big\{\, d_{(j)} \ :\ \rho_{(j)} \ge \tau_\rho,\ \ j \le K \,\Big\}
$$

- 최강축은 **항상** 포함, $\rho_{(1)} = 1$.
- 나머지는 상대 가중치가 5% 이상일 때만 포함.
- $\tau_\rho=0.05,\ p_e=4$ ⇒ 컷오프 각 $\theta^\ast = \arctan\!\big(\tau_\rho^{1/p_e}\big) \approx 25^\circ$:
  **축에서 25° 이내면 단일 레이어(평면), 대각선 근처만 다중 레이어(모서리)**.

**왜 지수 $p_e$ 인가 (예: 축에서 20° 기운 법선, $a_x=\cos20°=0.94,\ a_y=\sin20°=0.34$):**

| $p_e$ | $\rho_y = (a_y/a_x)^{p_e}$ | 결과 |
|---|---|---|
| 1 | 0.36 | +Y 레이어로 36% 샘 → 모서리 다시 뭉개짐 |
| **4** | **0.018** | $<0.05$ → 버림 → 순수 +X 취급 |

$45°$(진짜 모서리, $a_x=a_y$)에서는 $p_e=4$ 여도 $\rho=1$ → 두 레이어에 꽉(분리를 *원함*).
즉 $p_e$ 가 "거의 하드, 대각선 근처에서만 소프트"를 만든다. $p_e{\uparrow}$ = 날카롭지만 seam 위험↑,
$p_e{\downarrow}$ = 매끈하지만 둥긂. $p_e=4$ 는 튜닝된 절충점.

---

## 4. 절단 밴드 레이 마치 (Truncation-band ray march)

표면 점 $p$ 주위로 광선을 따라 $\pm S$ 스텝:

$$
S = \Big\lceil \tfrac{\tau}{h} \Big\rceil + 1,
\qquad
t = -S,\dots,S
$$

$$
s_t = p + r\,(t\,h),
\qquad
v = \big\lfloor s_t / h \big\rfloor,
\qquad
x_v = \big(v + \tfrac12\big)\,h \quad(\text{voxel 중심})
$$

---

## 5. 투영 부호거리 (Projective signed distance)

$$
\mathrm{sdf}(v) = \delta - (x_v - c)\cdot r
$$

$(p-c)\cdot r = \delta$ 이므로 등가적으로:

$$
\boxed{\ \mathrm{sdf}(v) = (p - x_v)\cdot r\ }
$$

→ 광선 위로 투영한 voxel 중심–표면 부호거리. $>0$ 카메라 앞(빈 공간), $<0$ 표면 뒤.

**밴드 게이트:** $|\mathrm{sdf}| > \tau \Rightarrow$ skip. **정규화:**

$$
\psi(v) = \mathrm{clamp}\!\left(\frac{\mathrm{sdf}(v)}{\tau},\ -1,\ +1\right) \in [-1,1]
$$

정규화 $[-1,1]$ 로 두면 추출 커널의 부호 교차가 스케일 독립이 된다.

---

## 6. 방향별 가중 누적 (Per-direction weighted accumulation)

밴드 안의 각 voxel $v$, 각 방향 $d \in \mathcal{D}$ 에 대해 최종 가중치:

$$
w = \varphi \cdot \rho_d \qquad (\text{skip if } w \le 0)
$$

엔트리 $(v,d)$ 에 원자적 누적 (정수 아토믹, 스케일 $\text{TSDF\_SCALE}=10^4$):

$$
\mathrm{SumDW}(v,d)\ \mathrel{+}=\ \psi(v)\cdot w,
\qquad
\mathrm{SumW}(v,d)\ \mathrel{+}=\ w
$$

키 $(v,d)$ 는 origin-relative $512^3$ 윈도우 해시(범위 밖이면 skip).

---

## 7. 융합값 (Fused TSDF, 추출 시)

각 $(v,d)$ 엔트리의 최종 정규화 TSDF는 **모든 관측에 대한 신뢰도·소속 가중 평균**:

$$
\Psi(v,d) = \frac{\mathrm{SumDW}(v,d)}{\mathrm{SumW}(v,d)}
= \frac{\displaystyle\sum_{\text{obs}} \psi\cdot\varphi\cdot\rho_d}{\displaystyle\sum_{\text{obs}} \varphi\cdot\rho_d}
$$

레이어 $d$ 의 표면 = $\Psi(v,d)=0$ 인 zero-crossing.
추출 커널이 방향별로 이웃 voxel 부호교차에서 표면점을 생성한다.

---

## 마스터 식 (한 관측 → 여러 $(v,d)$ 기여)

관측 $(p,n,c)$ 는, 밴드 내 모든 voxel $v$ 와 모든 $d\in\mathcal{D}(n)$ 에 대해:

$$
\Delta\mathrm{SumDW}(v,d) = \psi(v)\,\varphi\,\rho_d,
\qquad
\Delta\mathrm{SumW}(v,d) = \varphi\,\rho_d
$$

$$
\psi(v)=\mathrm{clamp}\!\Big(\tfrac{(p-x_v)\cdot r}{\tau},-1,1\Big),\quad
\varphi=\max(0,\,n\cdot(-r)),\quad
\rho_d=\Big(\tfrac{a_d}{a_{\max}}\Big)^{p_e}\!,\ \ \rho_d\ge 0.05,\ |\mathcal{D}|\le K
$$

> **평범한 TSDF와의 차이:** $\mathcal{D}$ 가 항상 1개 $\Rightarrow$ 단일 필드(모서리 뭉갬).
> Directional은 법선으로 $\mathcal{D}$ 를 여러 부호축에 배분 $\Rightarrow$ 다르게 향한 면이 분리 저장
> $\Rightarrow$ 모서리 보존.

---

## 상수 요약 (튜닝된 기본값)

| 상수 | 값 | 역할 |
|---|---|---|
| $p_e$ (dirExponent) | 4 | 방향 배정 날카로움 (이방성) |
| $K$ (maxDirections) | 3 | 한 관측이 쓰는 최대 레이어 수 |
| $\tau_\rho$ | 0.05 | 약한 방향 컷오프 (≈축에서 25°) |
| $\tau$ (truncation) | $3h$ | 절단 밴드 폭 |
| TSDF\_SCALE | $10^4$ | 정수 아토믹 고정소수점 스케일 |

---

## 참조
- Splietker & Behnke, *Directional TSDF: Modeling Surface Orientation for Coherent Meshes* (2019)
- 구현: `src/shader/compact_directional_integrate.comp`, `src/TSDF/Backends/CompactDirectionalTSDF.{h,cpp}`
- 관련 문서: `docs/TSDF_IMPLEMENTATION.md`, `docs/MRHASH_VS_DIRECTIONAL_TSDF.md`

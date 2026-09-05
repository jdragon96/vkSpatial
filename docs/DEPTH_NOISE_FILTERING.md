# Depth 센서 필터링

> **상태 (2026-09-05).** 이 문서의 GPU 경로 서술 — `ValidationMask`의 `[H1]`..`[H6]` 게이트와
> `NormalEstimator`의 세 추정기 — 는 **삭제된 코드**를 설명한다. `src/Pipeline/Reconstruction/Algorithm/`은
> 파이프라인에 연결된 적이 없었고 `src/Realsense`가 그 역할을 대체했다. 링크가 걸린 파일은 더 이상 없다.
>
> **측정값은 여전히 유효하다.** 추정기 정확도 표(참값 대비 12.5°/32.8°/50.6°, `capture/` 477프레임의
> 인접 불일치 3.28°)는 새 구현이 재현했고, 그 검토는
> [`src/Realsense/Algorithm/NormalEstimation.md`](../src/Realsense/Algorithm/NormalEstimation.md)에 있다.
>
> CPU 경로(`PrefilterDepth`, `BackprojectDepth`)는 그대로 있고 여전히 파이프라인의 기본값이다.

depth 프레임 → 점 → 복셀 경로의 전체 필터.

기준: [`DepthCameraFrameSource.h`](../src/Pipeline/Acquisition/DepthCameraFrameSource.h), [`RealSenseDepthProvider.cpp`](../src/Pipeline/Realsense/RealSenseDepthProvider.cpp), [`kernel_AdvancedTSDF.integrate.comp.glsl`](../src/TSDF/Backends/kernel_AdvancedTSDF.integrate.comp.glsl).

## 1. 의사코드

```
DepthCameraFrameSource::Next(out Frame)
├─ IDepthProvider::Grab(out DepthFrame)                   RealSenseDepthProvider
│  ├─ try_wait_for_frames
│  ├─ ThresholdFilter                                     [D1] opt-in
│  └─ UnpackDepthRows                                     uint16 → metres
│
└─ BackprojectDepth(DepthFrame, CameraIntrinsics) → Frame
   ├─ PrefilterDepth                                      [H1] opt-in
   ├─ BuildVertexGrid                                     per pixel
   │  └─ RangeGate                                        [H2] opt-in
   └─ EmitPoints                                          per pixel
      ├─ EstimateSurfaceNormal                            [H3] always
      │  └─ forward | central | planefit                  전략, -D 매크로
      ├─ StraddlesADepthStep                              [H4] opt-in
      ├─ CountSameSurfaceNeighbours                       [H5] opt-in
      ├─ OrientTowardCamera
      └─ IncidenceGate                                    [H6] opt-in

TSDF::Integrate → kernel_AdvancedTSDF.integrate           per point
├─ SelectDirections                                       → dirCount, reliability[]
├─ ViewReliabilityFactor                                  [F1] on
├─ AdaptiveBandWidth                                      [F2] opt-in
│  └─ AxialNoiseSigma
└─ MarchBand                                              t ∈ [−T, T]
   ├─ SignedDistance                                      point-to-plane | projective
   ├─ BandMembership
   ├─ Confidence                                          [F3] symmetric | behind-dropoff
   └─ AccumulateWeighted                                  atomic
```

기본값

- on: `[H3]` `[F1]` `[F3]`=symmetric
- off: `[D1]` `[H2]` `[H4]` `[H5]` `[H6]` `[F2]`
- `[H1]`: `realsense_scan` 3, 그 외 0
- `[H3]` 추정기: `forward`(기존 동작)

## 2. 세부

기호: $z$ 깊이(m), $(u,v)$ 픽셀, $P$ 카메라 좌표 점, $\hat n$ 단위 법선, $\hat r$ 시선 단위벡터, $v_s$ 복셀 크기, $\delta$ truncation.

### UnpackDepthRows

$$z(u,v) = \text{raw}_{16}(u,v)\cdot s,\qquad s=\texttt{get\_depth\_scale()}$$

```cpp
const unsigned char *rowBase = base + std::size_t(row) * rowStrideBytes;
out[row * width + column] = float(raw) * metresPerUnit;
```

- 행 stride ≠ $2W$ 가능. 선형 순회 시 이미지 점진 전단.
- $s$는 장치에서 읽음. 비주얼 프리셋 적용 **후**. 프리셋이 `RS2_OPTION_DEPTH_UNITS` 변경 가능.

### 점프 허용치 $\tau$(tolerance)

$$tolerance = \tau(z)=\max\bigl(\text{minimumDepthJump},\ \text{relativeDepthJump}\cdot z\bigr)$$

| $z$ | $z^{2} / (fb)$ = 1px 오차당 깊이 오차 |
| :-: | :-----------------------------------: |
| 1m  |                 52mm                  |
| 2m  |                 209mm                 |
| 4m  |                 835mm                 |

- 위 값에 따라
  - minimumDepthJump: 0.005(m)
  - relativeDepthJump: 0.02(m)
- 스테레오 오차 $\varepsilon_z = \frac{z^2}{f B}\varepsilon_d$ (Keselman et al. eq. 2).
  - $z=\frac{fB}{d}$의 $d$ 미분.
    - $f$ 초점거리(px)
    - $B$ baseline(m)
    - $\varepsilon_d$ 매칭 불확실도, 거리가 멀면 커짐(px)

### [H1] PrefilterDepth

- 윈도우 내 이웃 점들의 깊이값을 평균낸다.
- 점프 허용치를 벗어나는 경우, 노이즈로 판단한다.
- 깊이 정보가 0 이하는 SNR이 깨진 경우로 간주한다.

$$\tilde z(u,v)=\frac{\sum_{W} z'\,\mathbb 1[\,z'>0 \wedge |z'-z|\le\tau(z)\,]}{\sum_{W}\mathbb 1[\cdots]}$$

```cpp
if (neighbour <= 0.0f || std::abs(neighbour - z) > tolerance) {
    continue;
}
sum += neighbour;
++count;
// 유효 이웃 없으면, 중심 깊이값 유지
out[centre] = count > 0 ? sum / float(count) : z;
```

### [H2] RangeGate + 백프로젝션

- 센서가 측정할 수 있는 거리를 이용해 필터링 한다.
  - minimumDepthMeters: 최소 거리
  - maximumDepthMeters: 최대 거리

$$\text{valid} \iff z>0 \wedge (z_{\min}=0 \vee z\ge z_{\min}) \wedge (z_{\max}=0 \vee z\le z_{\max})$$

$$P(u,v)=\Bigl(\tfrac{u-c_x}{f_x}z,\ \tfrac{v-c_y}{f_y}z,\ z\Bigr)$$

```cpp
if ((filter.minimumDepthMeters > 0.0f && z < filter.minimumDepthMeters) ||
    (filter.maximumDepthMeters > 0.0f && z > filter.maximumDepthMeters))
{
    if (stats) {
        stats->rejectedByRange.fetch_add(1, std::memory_order_relaxed);
    }
    continue;
}
```

### [H3] 스텐실 게이트

추정기가 **읽는 이웃**이 곧 **같은 표면이어야 하는 이웃**이다. 그래서 게이트와 계산은 한 함수(`EstimateSurfaceNormal`)에 있다 — 둘을 나누면 서로 어긋날 수 있다. `forward`의 경우:

$$|z(u{+}1,v)-z|\le\tau \ \wedge\ |z(u,v{+}1)-z|\le\tau$$

- 보장 범위: 법선 유효성. 점 신뢰도는 비보장(`[H4]`가 그 질문).
- `central`은 네 이웃 전부, `planefit`은 창 안 같은표면 표본 수 $\ge$ `minimumPlaneFitSamples`.
- **폴백 없음.** 스텐실이 모자라면 좁은 스텐실로 내려가지 않고 거부한다. 한 프레임 안에 두 추정기가 섞이면 "이 추정기가 나은가"를 측정할 수 없다.

거부는 `rejectedByNormalStencil`로 관측한다. 이전에는 이 경로가 **아무 카운터 없이 조용히** 버려졌다 — 프레임이 점 대부분을 깊이 스텝에 잃어도 통계에는 아무것도 안 나왔다.

테두리는 세지 않는다. 스텐실이 이미지 밖으로 나가는 것은 추정기의 정의역이 끝나는 것이지 측정값을 버리는 게 아니고, 둘을 합치면 숫자가 장면보다 창 크기를 따라간다.

### [H4] StraddlesADepthStep

$$\exists\,(du,dv)\in\{-1,0,1\}^2\setminus\{(0,0)\}:\ \text{valid}(u{+}du,v{+}dv)\wedge|z_{nb}-z|>\tau$$

```cpp
if (!valid[j]) continue;
if (std::abs(depth[j] - z) > tolerance) return true;
```

- 이미지 밖 이웃 = 부재, 스텝 아님. 미계수.
- `[H3]` 대비 차이: 스텝 뒤쪽 가장자리. 우/하가 자기 표면, 좌/상이 40 cm 뒤인 픽셀은 `[H3]` 통과.

### [H5] CountSameSurfaceNeighbours

$$S(u,v)=\sum_{(du,dv)\ne(0,0)}\mathbb 1[\,\text{valid}\wedge|z_{nb}-z|\le\tau\,]\ \ge\ \text{minimumValidNeighbours}$$

```cpp
if (!valid[j]) continue;
if (std::abs(depth[j] - z) > tolerance) continue;
++count;
```

- $\tau$ 조건 없이 유효성만 계수 시 2×2 근접 blob이 $S=8$로 통과. 자기 표면 이웃은 3.

### EstimateSurfaceNormal

GPU에서 세 가지 추정기 중 하나로 컴파일된다. 축이 셰이더 안에 있으므로 C++ 가상함수가 아니라 [`NormalStrategy.glsl`](../src/Pipeline/Reconstruction/Algorithm/NormalStrategy.glsl) 디스패처 + `-D` 매크로로 가른다. 이름은 [`NormalEstimator.h`](../src/Pipeline/Reconstruction/Algorithm/NormalEstimator.h)에 등록되고 `DepthFilterOptions::normalEstimator`로 고른다.

셋 다 **방향 없는** 단위 법선만 돌려준다. 부호 결정, `[H4]`, `[H5]`, `[H6]`, 모든 카운터는 커널이 갖는다 — 거부 원인 분류가 어느 조각이 컴파일됐는지에 따라 달라지면 안 된다.

**`forward`** (기본, 기존 동작)

$$\mathbf n = \bigl(P(u{+}1,v)-P\bigr)\times\bigl(P(u,v{+}1)-P\bigr)$$

**`central`** — 같은 3×3 발자국, 같은 비용

$$\mathbf n = \tfrac{1}{2}\bigl(P(u{+}1,v)-P(u{-}1,v)\bigr)\times\tfrac{1}{2}\bigl(P(u,v{+}1)-P(u,v{-}1)\bigr)$$

**`planefit`** — $k\times k$ 창의 같은표면 표본에 대한 전최소제곱 평면. 중심점 기준 공분산의 최소 고유벡터 (특성삼차식 닫힌해 + 영공간 외적).

$$C=\tfrac1N\sum(P_i-\bar P)(P_i-\bar P)^\top,\qquad \mathbf n=\arg\min_{\|x\|=1} x^\top C x$$

기울기 잡음 (표본당 축방향 잡음 $\sigma$ 대비, 등간격 최소제곱 기울기 $\sigma\sqrt{12/(k(k^2{-}1))}$):

| 추정기 | 스텐실 | 기울기 잡음 | 잃는 테두리 |
| --- | --- | --- | --- |
| `forward` | 2 표본 | $1.41\,\sigma$ | 마지막 행·열 |
| `central` | 4 표본 | $0.71\,\sigma$ | 1 px 액자 |
| `planefit` $k{=}5$ | ≤25 표본 | $0.32\,\sigma$ | 2 px 액자 |

`forward`는 코너에 물린 삼각형이라 실제로는 $(u{+}0.5,v{+}0.5)$의 법선을 $(u,v)$에 저장한다 — 곡면에서 법선장이 반 픽셀 밀린다. `central`이 같은 비용으로 그 편이도 없앤다.

`planefit`의 두 가지 구현 함정 (둘 다 뮤테이션으로 검증됨):

1. **원점 이동 필수.** 1.5 m에서 좌표는 $O(1\,\mathrm m)$인데 창 안 퍼짐은 $O(1\,\mathrm{mm})$이라, $E[p^2]-E[p]^2$를 float32로 계산하면 7자리가 상쇄돼 1자리만 남는다. 중심점을 원점으로 잡아 모멘트를 누적한다.
2. **트레이스 정규화 필수.** 공분산 성분이 $O(10^{-6})$이라 그 세제곱(행렬식·영공간 외적)이 float32 하한 근처로 내려간다. 정규화 없이는 절대 임계값이 실제 표면을 전부 거부한다.

표본이 공선이면(1 px 폭 표면) 최소 고유값이 중복근이라 영공간이 평면이 된다 — 세 쌍의 외적이 모두 붕괴하므로 거부한다. 개수 하한만으로는 못 막는다.

**정확도 (합성 평면, 참값 대비).** $\sigma_z$ = 2 mm, 횡방향 간격 3.9 mm — 잡음이 큰 영역:

| | `forward` | `central` | `planefit` |
| --- | --- | --- | --- |
| 평균 각오차 | 50.6° | 32.8° | 12.5° |
| $\tan$ 환산 | 1.22 | 0.64 | 0.22 |
| `forward` 대비 | 1× | 1.9× | 5.6× |

$\tan$ 환산이 이론 비($1.41/0.71/0.32$)와 맞는다. 각도 자체는 큰 잡음에서 $\arctan$ 압축을 받는다. **참값 대비 오차**이므로 매끄러움이 아니라 정확도를 재는 것이다 — 뭉개기만 하는 추정기는 여기서 못 이긴다.

**실측 (`capture/` 477프레임, D435 640×480).** [`normal_estimator_eval`](../example2/normal_estimator_eval.cpp), 인접 방출 픽셀 간 법선 불일치 중앙값:

| 추정기 | `[H1]` | 방출 점 | 중앙값 | p90 | `rejectedByNormalStencil` |
| --- | --- | --- | --- | --- | --- |
| `forward` | 0 | 123,453,636 | 22.57° | 51.23° | 3,063,945 |
| `forward` | 3 | 123,471,683 | 4.92° | 12.12° | 3,045,898 |
| `forward` | 5 | 123,449,748 | 3.22° | 8.75° | 3,067,833 |
| `central` | 0 | 121,185,841 | 11.88° | 26.13° | 5,120,579 |
| `central` | 5 | 121,156,487 | 2.77° | 7.36° | 5,149,933 |
| **`planefit`** | **0** | **125,411,710** | **3.28°** | **8.47°** | **240,756** |
| `planefit` | 5 | 125,433,354 | 2.30° | 6.38° | 219,112 |

- **`planefit`이 `[H1]`을 대체한다.** prefilter 없는 `planefit`(3.28°)이 5×5 prefilter를 건 `forward`(3.22°)와 동급인데, **점 좌표는 안 뭉갠다.**
- `planefit`은 점을 **더** 방출한다(+1.6%). 창에 구멍이 있어도 남은 표본으로 적합하므로, 차분 스텐실이 통째로 잃는 픽셀을 살린다. 스텐실 거부가 12.7배 줄어든다(306만 → 24만).
- `central`은 이론대로 1.9배 개선하지만(22.57° → 11.88°) 점을 **잃는다**(−1.8%, 거부 512만). 같은표면 이웃이 2개가 아니라 4개 필요하기 때문. 정확도를 점으로 산다.
- 비용은 전 구성 0.3–0.7 ms/frame이며 **제출 오버헤드가 지배해 추정기 차이가 측정되지 않는다**. 30 fps 예산 33 ms 대비 두 자릿수 여유.

한계: 이 지표는 인접 법선의 **일치도**라 "잡음이 줄었다"와 "디테일이 뭉개졌다"를 구분하지 못한다. 정확도는 위 합성 참값 표가 담당한다. TSDF RMSE로 확인하려면 depth 녹화에 대응하는 GT 메시가 필요한데 `capture/`에는 없고, `scan_out`/`scanData`는 이미 법선이 붙은 점군이라 이 경로를 안 탄다.

### OrientTowardCamera

$$\hat n \leftarrow -\hat n \quad\text{if}\quad \hat n\cdot P>0$$

- 카메라 = 원점. $P$ = 시선.
- $n_z$ 부호 판정은 광축 위에서만 등가. 87° 화각 가장자리에서 약 40° 입사부터 반전.

### [H6] IncidenceGate

$$-\hat n\cdot\hat P \ \ge\ \cos\theta_{\max},\qquad \hat P = P/\|P\|$$

```cpp
if (minimumIncidenceCosine > 0.0f &&
    -n.dot(grid[i].normalized()) < minimumIncidenceCosine)
```

- $\cos\theta_{\max}$는 루프 밖 1회 계산.
- `[F1]`이 동일 양을 가중치로 사용. 이중 적용. 비권장.

### [F1] ViewReliabilityFactor

$$w_{\text{view}}=\max\bigl(0,\ \hat n\cdot(-\hat r)\bigr)$$

```glsl
float viewReliabilityFactor = (g_viewAngleWeight != 0u)
    ? max(0.0, dot(unitNormal, -rayDirection))
    : 1.0;
if (viewReliabilityFactor <= 0.0) return;
```

### [F2] AxialNoiseSigma / AdaptiveBandWidth

$$\sigma_z(z,\theta)=a_0+a_1(z-z_0)^2+\frac{a_\theta}{\sqrt z}\cdot\frac{\theta^2}{(\pi/2-\theta)^2}$$

$$B=\operatorname{clamp}\bigl(N\sigma_z,\ m\,v_s,\ \delta\bigr),\qquad \theta=\arccos\bigl(\hat n\cdot(-\hat r)\bigr)$$

```glsl
float sigma = g_sigmaConstant + g_sigmaQuadratic * fromOffset * fromOffset;
float theta = clamp(incidenceRadians, 0.0, kHalfPi - 0.087266);
sigma += (g_sigmaAngular / sqrt(depth)) * (theta * theta) / (toGrazing * toGrazing);
return clamp(g_bandSigmaMultiplier * sigma, g_bandMinimumVoxels * voxelSize, truncateDistance);
```

- $(a_0,a_1,z_0,a_\theta)=(0.0012,\,0.0019,\,0.4,\,0.0001)$. Nguyen et al. 2012 eq. 4. Kinect v1 피팅.
- $\theta$ 상한 85°. eq. 4 분모가 90°에서 0.
- 하한 $m v_s$ ($m{=}2$) 필수: 근거리 $3\sigma_z\approx4\,\text{mm}<v_s$ → 밴드 < 격자 → 표면 소실.
- 상한 $\delta$ 필수: 정규화 계약(아래).

**클램프 구간 (voxel 0.01, $\delta$ 0.03, $N=3$)**

| $z$   | $3\sigma_z$ | 밴드    | 상태        |
| ----- | ----------- | ------- | ----------- |
| 1.0 m | 5.7 mm      | 20.0 mm | 하한 클램프 |
| 2.0 m | 18.2 mm     | 20.0 mm | 하한 클램프 |
| 2.5 m | 28.7 mm     | 28.7 mm | 모델 사용   |
| 3.0 m | 42.1 mm     | 30.0 mm | 상한 클램프 |
| 4.0 m | 77.5 mm     | 30.0 mm | 상한 클램프 |

- 모델이 실제로 참조되는 구간은 **2.10–2.55 m**뿐. 그 밖은 상수 밴드.
- 따라서 측정된 −22.3%는 σ 모델이 아니라 **하한**이 만든 것. `capture/`는 대부분 1.5 m 이내 → 밴드 30 → 20 mm.
- 현 설정의 실효 동작 = 2단 계단(근거리 2 voxel, 원거리 $\delta$). σ 모델은 전환점만 결정.
- 모델을 실제로 활용하려면 $\delta$를 원거리 요구에 맞춰 상향. 그래야 상한 클램프가 풀린다.
- 부수 효과: $\varepsilon_d$ 상수 가정이 깨지는 원거리는 상한 클램프 구간이므로 모델을 쓰지 않는다. 미검증 + 모델 사용 구간은 1.35–2.55 m로 한정되고 출력은 20–30 mm 유계.

### MarchBand / SignedDistance / BandMembership

$$x_t = P + \hat m\,(t\,v_s),\qquad t\in[-T,T],\quad T=\lceil B/v_s\rceil+1$$

$$d(x_t)=\begin{cases}(x_c-P)\cdot\hat n & \text{point-to-plane},\ \hat m=\hat n\\ z-(x_c-\text{cam})\cdot\hat r & \text{projective},\ \hat m=\hat r\end{cases}$$

$$|d|\le B,\qquad \text{tsdf}=\operatorname{clamp}(d/\delta,\,-1,\,1)$$

```glsl
vec3 marchDirection = usePointToPlane ? unitNormal : rayDirection;
float voxel2point = usePointToPlane
    ? dot(voxelCenter - point, unitNormal)
    : depth - dot(voxelCenter - camera, rayDirection);
if (abs(voxel2point) > bandWidth) continue;
float tsdf = clamp(voxel2point / truncateDistance, -1.0, 1.0);
```

- 행진 축 = 거리 측정 축. 불일치 시 밴드가 입사각 코사인만큼 절단(75°에서 1/3).
- 정규화는 $\delta$, $B$ 아님. 트래커 3곳과 extract가 $P_{\text{surf}}=x_c-\text{tsdf}\cdot\delta\cdot\hat n$로 미터 거리 복원. $B$ 사용 시 전부 표면 오배치.

### [F3] Confidence

$$w_{\text{conf}}^{\text{sym}} = 1-\lambda|\text{tsdf}|$$

$$w_{\text{conf}}^{\text{drop}} = \begin{cases}1 & d>-\epsilon\\[4pt] \max\Bigl(0,\ \dfrac{d+\delta}{\delta-\epsilon}\Bigr) & d\le-\epsilon\end{cases}\qquad \epsilon=v_s$$

```glsl
confidence = (voxel2point > -epsilon)
    ? 1.0
    : max(0.0, (voxel2point + truncateDistance) / max(truncateDistance - epsilon, 1e-9));
```

- symmetric: $|\cdot|$ 대칭. 관측된 앞쪽과 미관측 뒤쪽을 동일 감쇠.
- drop-off: 앞쪽 무감쇠, 가림 쪽만 램프. Bylow et al. 2013; Voxblox eq. 5.

### AccumulateWeighted

$$w=w_{\text{view}}\cdot w_{\text{dir}}\cdot w_{\text{conf}}$$

$$\textstyle\sum DW \mathrel{+}= \text{tsdf}\,w K,\quad \sum W \mathrel{+}= wK,\quad \sum \mathbf N \mathrel{+}= \hat n\,wK$$

```glsl
float w = viewReliabilityFactor * reliability[di] * confidence;
atomicAdd(g_hash[slot].sumDW, int(tsdf * w * TSDF_SCALE));
atomicAdd(g_hash[slot].sumW,  uint(w * TSDF_SCALE));
atomicAdd(g_hash[slot].sumNx, int(unitNormal.x * w * TSDF_SCALE));
```

- $K$ = `TSDF_SCALE` 고정소수점.
- 추출: $\text{tsdf}=\sum DW/\sum W$, $\hat n=\sum\mathbf N/\|\sum\mathbf N\|$.

## 3. 측정

`capture/` 477프레임, `--no-submap`, icp, `[H1]`=3.

| 설정                            | 점 유지     | 맵 복셀   | 추적    | align ms |
| ------------------------------- | ----------- | --------- | ------- | -------- |
| 게이트 없음                     | 123,471,684 | 1,000,198 | 476/476 | 70.6     |
| `[H4]` + `[H5]`=6 + `[F3]`=drop | −1.19%      | −4.34%    | 476/476 | 69.7     |
| 위 + `[H5]`=8 + `[H2]` far 4.0  | −3.58%      | −6.73%    | 476/476 | 64.3     |
| `[F2]`=3σ 단독                  | —           | −22.3%    | 476/476 | −21.5%   |

원인 규명 (raw depth 직접 분석, 12프레임):

| 가설                | 판정 | 근거                                                    |
| ------------------- | ---- | ------------------------------------------------------- |
| 불연속 flying pixel | 원인 | 방출 점 0.17%, 가려진 이웃과 중앙값 403 mm              |
| 센서 축방향 노이즈  | 기각 | 평탄면 $\sigma_z$ 0.68–1.12 mm < 횡방향 간격 1.8–2.4 mm |
| 소프트 보간 램프    | 기각 | `[H4]` 후 7×7 중앙값 대비 50 mm 초과 점 0개             |

측정 함정: `submap = true`에서 entry 수는 입력 점 수에 비단조. [분류 커널](../src/TSDF/Memory/RegionClassifier/kernel_DenseRegionClassifier.classify.comp.glsl)의 `g_dense[i]`가 latch 후 불변 → 경로 의존. 블록 1개 latch당 부피 $32^3\to64^3$(약 8배). dense blocks 27→28 차이가 entry +8.3% 유발. 프런트엔드 A/B는 `--no-submap`.

## 4. 미구현

1. **자유공간 카빙**. integrate는 $\pm B$ 밴드만 기록. 카메라–표면 구간 미갱신 → 융합된 flyer 영구 잔존. 걸림돌: point-to-plane은 $\hat n$ 행진, 카빙은 $\hat r$ 행진. directional TSDF에서 자유공간의 방향 레이어 미정의.
2. **σ 가중 prefilter**. `[H1]` 하드 임계 → $w=\exp\bigl(-\tfrac{\Delta u^2}{2\sigma_L^2}-\tfrac{\Delta z^2}{2\sigma_z^2}\bigr)$.
3. **ICP 잔차 가중** $\sigma_z(z_{\min},0)/\sigma_z$.
4. **$\sigma_z$ 계수 D435 재피팅**. 현재 Kinect v1(구조광) 피팅.

## 참고

- Keselman et al. (Intel), _Intel RealSense Stereoscopic Depth Cameras_, arXiv:1705.05548 §2.1 — eq. 1 $z=fB/d$, eq. 2 $\varepsilon_z=z^2\varepsilon_d/(fB)$. D400 계열 1차 출처
- Nguyen, Izadi & Lovell, 3DIMPVT 2012 — $\sigma_z$ 모델, 필터·ICP 가중·트런케이션 적용
- Curless & Levoy, SIGGRAPH 1996 — space carving
- Oleynikova et al., _Voxblox_, IROS 2017 — eq. 5 뒤쪽 감쇠 ($\delta=4v$, $\epsilon=v$)
- Bylow et al. 2013 — 뒤쪽 감쇠 원출처
- Weder et al., _RoutedFusion_, CVPR 2020 — 학습 융합, thickening artifact

# Depth 센서 필터링

depth 프레임이 점으로, 점이 복셀로 가는 동안 걸리는 필터 전부. 기준 파일은 [`DepthCameraFrameSource.h`](../src/Pipeline/Reconstruction/DepthCameraFrameSource.h), [`RealSenseDepthProvider.cpp`](../src/Pipeline/Reconstruction/RealSenseDepthProvider.cpp), [`kernel_AdvancedTSDF.integrate.comp.glsl`](../src/TSDF/Backends/kernel_AdvancedTSDF.integrate.comp.glsl).

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
      ├─ ForwardJumpGuard                                 [H3] always
      ├─ StraddlesADepthStep                              [H4] opt-in
      ├─ CountSameSurfaceNeighbours                       [H5] opt-in
      ├─ EstimateNormal
      │  └─ OrientTowardCamera
      └─ IncidenceGate                                    [H6] opt-in

TSDF::Integrate → kernel_AdvancedTSDF.integrate           per point
├─ SelectDirections                                       → dirCount, reliability[]
├─ ViewReliabilityFactor                                  [F1] on by default
├─ AdaptiveBandWidth                                      [F2] opt-in
│  └─ AxialNoiseSigma
└─ MarchBand                                              t ∈ [−steps, steps]
   ├─ SignedDistance                                      point-to-plane | projective
   ├─ BandMembership
   ├─ Confidence                                          [F3] symmetric | behind-dropoff
   └─ AccumulateWeighted                                  atomic
```

기본값: `[D1] [H1] [H2] [H4] [H5] [H6] [F2]` off, `[H3] [F1] [F3=symmetric]` on. `realsense_scan`만 `[H1]`을 3으로 켠다.

---

## 2. 세부

기호: $z$ 깊이(m), $(u,v)$ 픽셀, $P$ 카메라 좌표 점, $\hat n$ 단위 법선, $\hat r$ 시선 단위벡터, $v_s$ 복셀 크기, $\delta$ truncation.

### UnpackDepthRows

$$z(u,v) = \text{raw}_{16}(u,v)\cdot s,\qquad s=\texttt{get\_depth\_scale()}$$

행 stride가 $2W$가 아닐 수 있어 행마다 재계산한다. 선형으로 걸으면 이미지가 점진적으로 전단된다.

```cpp
const unsigned char *rowBase = base + std::size_t(row) * rowStrideBytes;
out[row * width + column] = float(raw) * metresPerUnit;
```

$s$는 장치에서 읽는다. **비주얼 프리셋 적용 후에** 읽어야 한다 — 프리셋이 `RS2_OPTION_DEPTH_UNITS`를 바꿀 수 있다.

### 점프 허용치 $\tau$

이하 모든 게이트가 공유한다.

$$\tau(z)=\max\bigl(\text{minimumDepthJump},\ \text{relativeDepthJump}\cdot z\bigr)$$

기본 $(0.005,\ 0.02)$. 상대항인 이유는 스테레오 오차가 $\propto z^2/(f\cdot b)$로 자라기 때문 — 고정값은 근거리에서 과잉 거부, 원거리에서 과소 거부다.

### [H1] PrefilterDepth

$$\tilde z(u,v)=\frac{\sum_{(u',v')\in W} z'\,\mathbb 1[\,z'>0\,\wedge\,|z'-z|\le\tau(z)\,]}{\sum_{(u',v')\in W}\mathbb 1[\cdots]}$$

$W$는 한 변 `prefilterWindow`의 정사각. 유효 이웃이 없으면 $z$ 그대로.

```cpp
if (neighbour <= 0.0f || std::abs(neighbour - z) > tolerance) continue;
sum += neighbour; ++count;
out[centre] = count > 0 ? sum / float(count) : z;
```

$\mathbb 1[\cdot]$의 $\tau$ 조건이 없으면 스텝을 가로질러 평균되어 [H3]의 flying-pixel 방어가 무효화된다.

**측정**: 인접 법선 불일치 중앙값 raw 24° → 3×3 4.7° → 5×5 2.9°.

### [H2] RangeGate + 백프로젝션

$$\text{valid}(u,v) \iff z>0 \;\wedge\; (z_{\min}=0 \vee z\ge z_{\min}) \;\wedge\; (z_{\max}=0 \vee z\le z_{\max})$$

$$P(u,v)=\Bigl(\tfrac{u-c_x}{f_x}z,\ \tfrac{v-c_y}{f_y}z,\ z\Bigr)$$

```cpp
if ((filter.minimumDepthMeters > 0.0f && z < filter.minimumDepthMeters) ||
    (filter.maximumDepthMeters > 0.0f && z > filter.maximumDepthMeters)) {
    if (stats) stats->rejectedByRange.fetch_add(1, std::memory_order_relaxed);
    continue;
}
```

`continue`가 `valid[i]=0`을 남긴다. 방출만 막고 `valid`를 세우면 그 픽셀이 이웃 자격을 유지해 옆 픽셀의 법선을 계속 결정한다.

### [H3] ForwardJumpGuard

$$|z(u{+}1,v)-z|\le\tau \;\wedge\; |z(u,v{+}1)-z|\le\tau$$

```cpp
if (std::abs(grid[i + 1].z() - z) > maxJump) continue;
if (std::abs(grid[i + W].z() - z) > maxJump) continue;
```

법선이 그 두 이웃의 전방차분이므로 검사 대상도 그 둘이다. **법선의 유효성**을 보장하며, 점의 신뢰도는 보장하지 않는다.

### [H4] StraddlesADepthStep

$$\exists\,(du,dv)\in\{-1,0,1\}^2\setminus\{(0,0)\}\ :\ \text{valid}(u{+}du,v{+}dv)\ \wedge\ |z_{nb}-z|>\tau$$

```cpp
if (!valid[j]) continue;
if (std::abs(depth[j] - z) > tolerance) return true;
```

이미지 밖 이웃은 부재이지 스텝이 아니므로 세지 않는다. [H3]과의 차이는 **스텝의 뒤쪽 가장자리** — 오른쪽·아래가 자기 표면이고 왼쪽·위가 40 cm 뒤인 픽셀은 [H3]을 통과한다.

### [H5] CountSameSurfaceNeighbours

$$S(u,v)=\sum_{(du,dv)\ne(0,0)}\mathbb 1[\,\text{valid}\,\wedge\,|z_{nb}-z|\le\tau\,]\ \ \ge\ \text{minimumValidNeighbours}$$

```cpp
if (!valid[j]) continue;
if (std::abs(depth[j] - z) > tolerance) continue;
++count;
```

$\tau$ 조건 없이 유효성만 세면 2×2 근접 blob이 $S=8$로 통과한다(자기 표면 이웃은 3개).

### EstimateNormal

$$\mathbf n = \bigl(P(u{+}1,v)-P\bigr)\times\bigl(P(u,v{+}1)-P\bigr),\qquad \hat n=\frac{\mathbf n}{\|\mathbf n\|}$$

$$\hat n \leftarrow -\hat n \quad\text{if}\quad \hat n\cdot P>0$$

```cpp
Eigen::Vector3f n = (grid[i + 1] - grid[i]).cross(grid[i + W] - grid[i]);
if (n.norm() < 1e-9f) continue;
n.normalize();
if (n.dot(grid[i]) > 0.0f) n = -n;
```

카메라가 원점이라 $P$가 곧 시선이다. $n_z$ 부호만 보면 광축 위에서만 같고, 87° 화각의 가장자리에서는 약 40° 입사부터 반전된다.

### [H6] IncidenceGate

$$-\hat n\cdot\hat P \ \ge\ \cos\theta_{\max},\qquad \hat P = P/\|P\|$$

```cpp
if (minimumIncidenceCosine > 0.0f &&
    -n.dot(grid[i].normalized()) < minimumIncidenceCosine)
```

$\cos\theta_{\max}$는 루프 밖에서 한 번 계산한다. `viewAngleWeight`([F1])가 이미 같은 양을 가중치로 쓰므로 **이중 적용**이다 — 권하지 않는다.

---

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

계수 $(a_0,a_1,z_0,a_\theta)=(0.0012,\,0.0019,\,0.4,\,0.0001)$ — Nguyen et al. 2012 eq. 4, Kinect v1 피팅.

```glsl
float sigma = g_sigmaConstant + g_sigmaQuadratic * fromOffset * fromOffset;
float theta = clamp(incidenceRadians, 0.0, kHalfPi - 0.087266);
sigma += (g_sigmaAngular / sqrt(depth)) * (theta * theta) / (toGrazing * toGrazing);
return clamp(g_bandSigmaMultiplier * sigma, g_bandMinimumVoxels * voxelSize, truncateDistance);
```

하한 $m\,v_s$($m=2$)가 없으면 근거리에서 $3\sigma_z\approx$ 4 mm $< v_s$가 되어 밴드가 격자보다 좁아지고 **표면이 사라진다.** 상한 $\delta$는 정규화 계약 때문이다(아래).

### MarchBand / SignedDistance / BandMembership

$$x_t = P + \hat m\,(t\,v_s),\qquad t\in[-T,T],\quad T=\bigl\lceil B/v_s\bigr\rceil+1$$

$$d(x_t)=\begin{cases}(x_c-P)\cdot\hat n & \text{point-to-plane}\\ z-(x_c-\text{cam})\cdot\hat r & \text{projective}\end{cases}\qquad \hat m=\begin{cases}\hat n\\ \hat r\end{cases}$$

$$|d|\le B,\qquad \text{tsdf}=\operatorname{clamp}(d/\delta,\,-1,\,1)$$

```glsl
vec3 marchDirection = usePointToPlane ? unitNormal : rayDirection;
float voxel2point = usePointToPlane
    ? dot(voxelCenter - point, unitNormal)
    : depth - dot(voxelCenter - camera, rayDirection);
if (abs(voxel2point) > bandWidth) continue;
float tsdf = clamp(voxel2point / truncateDistance, -1.0, 1.0);
```

행진 축과 거리 측정 축이 같아야 한다. 섞으면 밴드가 입사각 코사인만큼 잘린다(75°에서 1/3).

**정규화는 $B$가 아니라 $\delta$다.** 세 트래커와 extract 커널이 $P_{\text{surf}}=x_c-\text{tsdf}\cdot\delta\cdot\hat n$로 미터 거리를 복원한다. $B$로 나누면 그 전부가 표면을 엉뚱한 곳에 놓는다.

### [F3] Confidence

$$w_{\text{conf}}^{\text{sym}} = 1-\lambda|\text{tsdf}|$$

$$w_{\text{conf}}^{\text{drop}} = \begin{cases}1 & d>-\epsilon\\[4pt] \max\Bigl(0,\ \dfrac{d+\delta}{\delta-\epsilon}\Bigr) & d\le-\epsilon\end{cases}\qquad \epsilon=v_s$$

```glsl
confidence = (voxel2point > -epsilon)
    ? 1.0
    : max(0.0, (voxel2point + truncateDistance) / max(truncateDistance - epsilon, 1e-9));
```

symmetric은 $|{\cdot}|$이라 센서가 통과해 관측한 앞쪽을 아무것도 본 적 없는 뒤쪽과 똑같이 깎는다. drop-off는 앞쪽을 온전히 두고 가림 쪽만 램프한다(Bylow et al. 2013; Voxblox eq. 5).

### AccumulateWeighted

$$w=w_{\text{view}}\cdot w_{\text{dir}}\cdot w_{\text{conf}}$$

$$\sum DW \mathrel{+}= \text{tsdf}\cdot w\cdot K,\quad \sum W \mathrel{+}= w\cdot K,\quad \sum \mathbf N \mathrel{+}= \hat n\,w\,K$$

```glsl
float w = viewReliabilityFactor * reliability[di] * confidence;
atomicAdd(g_hash[slot].sumDW, int(tsdf * w * TSDF_SCALE));
atomicAdd(g_hash[slot].sumW,  uint(w * TSDF_SCALE));
atomicAdd(g_hash[slot].sumNx, int(unitNormal.x * w * TSDF_SCALE));
```

$K=$ `TSDF_SCALE` 고정소수점. 추출 시 $\text{tsdf}=\sum DW/\sum W$, $\hat n=\sum\mathbf N/\|\sum\mathbf N\|$.

---

## 3. 측정

`capture/` 477프레임, `--no-submap`, icp, `[H1]`=3.

| 설정 | 점 유지 | 맵 복셀 | 추적 | align ms |
|---|---|---|---|---|
| 게이트 없음 | 123,471,684 | 1,000,198 | 476/476 | 70.6 |
| `[H4] [H5]=6 [F3]=drop` | −1.19% | −4.34% | 476/476 | 69.7 |
| 위 + `[H5]=8 [H2]`far 4.0 | −3.58% | −6.73% | 476/476 | 64.3 |
| `[F2]`=3σ 단독 | — | −22.3% | 476/476 | −21.5% |

원인 규명 (raw depth 직접 분석, 12프레임):

| 가설 | 판정 |
|---|---|
| 불연속 flying pixel | **원인** — 방출 점의 0.17%, 가려진 이웃과 중앙값 403 mm |
| 센서 축방향 노이즈 | 기각 — 평탄면 $\sigma_z$ 0.68–1.12 mm < 횡방향 간격 1.8–2.4 mm |
| 소프트 보간 램프 | 기각 — `[H4]` 후 7×7 중앙값 대비 50 mm 초과 점 **0개** |

**측정 함정**: `submap = true`에서 entry 수는 입력 점 수의 단조 함수가 아니다. [분류 커널](../src/TSDF/Memory/RegionClassifier/kernel_DenseRegionClassifier.classify.comp.glsl)의 `g_dense[i]`가 latch되고 되돌아가지 않아 경로 의존적이고, 한 블록이 latch될 때마다 부피가 $32^3\to64^3$(약 8배)이 된다. dense blocks 27→28 하나 차이가 entry +8.3%를 만들었다. 프런트엔드 A/B는 `--no-submap`으로 한다.

## 4. 없는 것

1. **자유공간 카빙** — integrate는 $\pm B$ 밴드만 쓴다. 카메라–표면 사이를 갱신하지 않으므로 **한 번 융합된 flyer는 영구적이다.** point-to-plane은 $\hat n$을 따라 행진하는데 카빙은 $\hat r$을 따라야 하고, directional TSDF에서 자유공간의 방향 레이어가 자명하지 않다
2. **σ 가중 prefilter** — `[H1]`을 하드 임계에서 $w=\exp\bigl(-\tfrac{\Delta u^2}{2\sigma_L^2}-\tfrac{\Delta z^2}{2\sigma_z^2}\bigr)$로
3. **ICP 잔차 가중** $\sigma_z(z_{\min},0)/\sigma_z$
4. **$\sigma_z$ 계수 D435 재피팅** — 현재 계수는 Kinect v1(구조광) 피팅

## 참고

- Nguyen, Izadi & Lovell, 3DIMPVT 2012 — $\sigma_z$ 모델, 필터·ICP 가중·트런케이션 적용
- Curless & Levoy, SIGGRAPH 1996 — space carving
- Oleynikova et al., *Voxblox*, IROS 2017 — eq. 5 뒤쪽 감쇠 ($\delta=4v$, $\epsilon=v$)
- Bylow et al. 2013 — 뒤쪽 감쇠 원출처
- Weder et al., *RoutedFusion*, CVPR 2020 — 학습 융합, thickening artifact

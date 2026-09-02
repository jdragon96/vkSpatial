# Depth 센서 필터링

depth 프레임 → 점 → 복셀 경로의 전체 필터.

기준: [`DepthCameraFrameSource.h`](../src/Pipeline/Reconstruction/DepthCameraFrameSource.h), [`RealSenseDepthProvider.cpp`](../src/Pipeline/Reconstruction/RealSenseDepthProvider.cpp), [`kernel_AdvancedTSDF.integrate.comp.glsl`](../src/TSDF/Backends/kernel_AdvancedTSDF.integrate.comp.glsl).

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
  - minimumDepthJump: 0.05(m)
  - relativeDepthJump: 0.2(m)
- 스테레오 오차 $\varepsilon_z = \frac{z^2}{f B}\varepsilon_d$ (Keselman et al. eq. 2).
  - $\varepsilon_d$ 상수 가정은 active 시스템의 원거리에서 깨진다. 프로젝터 밝기 $1/z^2$ 감쇠 → SNR 저하 → 실제 오차는 $z^2$보다 빠르게 증가 (Keselman et al. §2.1 각주).
  - $z=\frac{fB}{d}$의 $d$ 미분.
    - $f$ 초점거리(px)
    - $B$ baseline(m)
    - $\varepsilon_d$ 매칭 불확실도(px)

### [H1] PrefilterDepth

$$\tilde z(u,v)=\frac{\sum_{W} z'\,\mathbb 1[\,z'>0 \wedge |z'-z|\le\tau(z)\,]}{\sum_{W}\mathbb 1[\cdots]}$$

```cpp
if (neighbour <= 0.0f || std::abs(neighbour - z) > tolerance) {
    continue;
}
sum += neighbour;
++count;
out[centre] = count > 0 ? sum / float(count) : z;
```

- $W$ = 한 변 `prefilterWindow` 정사각. 유효 이웃 0개면 $z$ 유지.
- $\tau$ 조건 제거 시 스텝 가로질러 평균 → `[H3]` 무효화.
- 측정: 인접 법선 불일치 중앙값 raw 24° → 3×3 4.7° → 5×5 2.9°.

### [H2] RangeGate + 백프로젝션

$$\text{valid} \iff z>0 \wedge (z_{\min}=0 \vee z\ge z_{\min}) \wedge (z_{\max}=0 \vee z\le z_{\max})$$

$$P(u,v)=\Bigl(\tfrac{u-c_x}{f_x}z,\ \tfrac{v-c_y}{f_y}z,\ z\Bigr)$$

```cpp
if ((filter.minimumDepthMeters > 0.0f && z < filter.minimumDepthMeters) ||
    (filter.maximumDepthMeters > 0.0f && z > filter.maximumDepthMeters)) {
    if (stats) stats->rejectedByRange.fetch_add(1, std::memory_order_relaxed);
    continue;
}
```

- `continue` → `valid[i]=0`. 방출만 막고 `valid` 유지 시 이웃 자격 잔존 → 옆 픽셀 법선 결정.

### [H3] ForwardJumpGuard

$$|z(u{+}1,v)-z|\le\tau \ \wedge\ |z(u,v{+}1)-z|\le\tau$$

```cpp
if (std::abs(grid[i + 1].z() - z) > maxJump) continue;
if (std::abs(grid[i + W].z() - z) > maxJump) continue;
```

- 검사 대상 = 법선 전방차분의 두 이웃.
- 보장 범위: 법선 유효성. 점 신뢰도는 비보장.

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

### EstimateNormal

$$\mathbf n = \bigl(P(u{+}1,v)-P\bigr)\times\bigl(P(u,v{+}1)-P\bigr),\qquad \hat n=\mathbf n/\|\mathbf n\|$$

$$\hat n \leftarrow -\hat n \quad\text{if}\quad \hat n\cdot P>0$$

```cpp
Eigen::Vector3f n = (grid[i + 1] - grid[i]).cross(grid[i + W] - grid[i]);
if (n.norm() < 1e-9f) continue;
n.normalize();
if (n.dot(grid[i]) > 0.0f) n = -n;
```

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

| $z$ | $3\sigma_z$ | 밴드 | 상태 |
|---|---|---|---|
| 1.0 m | 5.7 mm | 20.0 mm | 하한 클램프 |
| 2.0 m | 18.2 mm | 20.0 mm | 하한 클램프 |
| 2.5 m | 28.7 mm | 28.7 mm | 모델 사용 |
| 3.0 m | 42.1 mm | 30.0 mm | 상한 클램프 |
| 4.0 m | 77.5 mm | 30.0 mm | 상한 클램프 |

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

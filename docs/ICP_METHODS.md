# ICP 방법론 정리 — SLAM 전환을 위한 기초

**목적:** point cloud registration(정합)의 핵심인 ICP 계열을 정리하고, 이 저장소(VkLBVH)의
기존 컴포넌트(GPU BVH `KNN`, `DirectionalTSDF`, `Engine::Eval`)와 연결해 **ICP → SLAM**
점진적 구현 경로를 제시한다. ICP는 대부분의 SLAM front-end(scan matching / odometry)의
심장이므로, 여기서 다지면 SLAM으로의 확장이 자연스럽다.

---

## 1. ICP의 기본 구조

두 점군 — source `P = {p_i}`, target `Q = {q_j}` — 사이의 rigid 변환 `T = (R, t)`
(`R ∈ SO(3)`, `t ∈ ℝ³`)를 추정한다. 목적함수(가장 단순한 형태):

```
T* = argmin_T  Σ_i  w_i · dist( T·p_i ,  Q )²
```

**반복 루프:**
1. **Correspondence** — 현재 `T`로 `p_i`에 대응하는 `q_{c(i)}`를 찾는다 (보통 nearest neighbor).
2. **Transform 추정** — 대응쌍으로 `T`를 갱신 (error metric에 따라 closed-form 또는 선형 least squares).
3. **적용 & 반복** — 수렴(변화량/에러 임계) 또는 최대 iteration까지 1–2 반복.

**근본 한계:** ICP는 **local** 최적화다. 초기 정렬이 나쁘면 잘못된 극소값에 빠진다 →
좋은 초기값(global registration / motion prior)이 선행되어야 한다 (§4).

**설계 축 (Rusinkiewicz & Levoy 2001, "Efficient Variants of the ICP Algorithm"):**
거의 모든 변형은 아래 6단계의 조합이다.

| 단계 | 선택지 |
|---|---|
| Selection | 전부 / uniform / random / normal-space sampling / gradient |
| Matching | nearest-neighbor(kd-tree, **BVH**) / projective(depth image) / normal-shooting |
| Weighting | 균일 / 거리 / normal 호환성 / color |
| Rejecting | 거리 threshold / normal 각도 / boundary·중복 제거 / trimming |
| Error metric | point-to-point / point-to-plane / symmetric / plane-to-plane |
| Minimization | SVD closed-form / 선형 least squares / robust IRLS |

---

## 2. 핵심 변형 (error metric 기준)

### 2.1 Point-to-Point ICP — Besl & McKay 1992
```
min_{R,t}  Σ_i || R p_i + t − q_{c(i)} ||²
```
- 대응쌍이 고정되면 **closed-form** (Horn quaternion / Arun SVD): 중심 정렬 후 `H = Σ p̃ q̃ᵀ`, SVD `H = UΣVᵀ`, `R = V Uᵀ`.
- 장점: 단순, normal 불필요, 구현 쉬움. 단점: **평면 위 슬라이딩(tangential)에 약해 수렴 느림.**
- 오늘날에도 기준선(baseline)이며, KISS-ICP(2023)는 이 단순 metric + 적응형 threshold로 SOTA LiDAR odometry를 낸다.

### 2.2 Point-to-Plane ICP — Chen & Medioni 1991
```
min_{R,t}  Σ_i (( R p_i + t − q_{c(i)} ) · n_{c(i)})²
```
- 점을 target **접평면**까지의 거리로 측정 → 접선 방향 자유도를 허용해 **훨씬 빠르게 수렴**.
- 소각도 선형화(`R ≈ I + [ω]×`)하면 6×6 선형 least squares. target **normal 필요**.
- **실시간 depth 정합의 표준** (KinectFusion 등). 이 저장소의 `DirectionalTSDF`는 이미 normal을 다루므로 자연스럽게 얹힌다.

### 2.3 Symmetric ICP — Rusinkiewicz 2019
- source·target normal을 **대칭적으로** 쓰는 metric. point-to-plane보다 수렴역이 넓고 더 정확. 곡면에서 유리.

### 2.4 Generalized ICP (GICP) — Segal, Haehnel, Thrun 2009
- **plane-to-plane / probabilistic.** 각 점을 국소 공분산 `C_i`(표면이면 납작한 Gaussian)로 모델링:
```
min  Σ_i  d_iᵀ ( C^Q_{c(i)} + R C^P_i Rᵀ )^{-1} d_i ,   d_i = q_{c(i)} − (R p_i + t)
```
- point-to-point(공분산=I)와 point-to-plane(공분산=평면)을 특수경우로 포함. LiDAR에 강함.
- **VGICP** (Koide 2021): voxel별 분포로 근사 → 대응탐색 없이 GPU 병렬화에 적합.

### 2.5 특징 결합 변형
- **NICP** (Serafin & Grisetti 2015): normal + 곡률을 error/대응에 사용.
- **Colored ICP** (Park, Zhou, Koltun 2017): geometric + photometric joint 목적함수 → RGB-D에서 텍스처가 슬라이딩을 잡아줌.

### 2.6 NDT (Normal Distributions Transform) — Biber & Straßer 2003
- 엄밀히는 ICP가 아니지만 같은 자리를 노리는 **대안 scan matcher.** target을 voxel별 Gaussian으로 표현하고 source 점의 우도를 최대화 → **명시적 correspondence 불필요.** LiDAR SLAM에서 ICP 대안으로 흔히 쓰임.

---

## 3. Correspondence & Robustness — 실전 성패를 가르는 부분

**Matching:**
- Nearest neighbor: **kd-tree**가 표준. 이 저장소는 GPU **BVH**(`SpatialIndex::KNN`)로 이 단계를 가속할 수 있다(§6).
- Projective association: depth image에서 현재 `T`로 재투영해 픽셀 대응 → O(1), 실시간 dense에 필수(KinectFusion).
- Normal-shooting: source normal 방향으로 target 표면과 교차.

**Rejecting / Robust (아웃라이어·부분겹침 대응):**
- 하드 필터: 거리 threshold, normal 각도, boundary·중복 대응 제거.
- **Trimmed ICP** (Chetverikov 2002): 오차 작은 상위 `ξ`(overlap 비율)만 사용 → partial overlap에 강함. `DirectionalTSDF`가 이미 overlap ratio를 추적한다.
- **Robust kernels**: Huber / Cauchy / Geman-McClure 손실 + IRLS(가중 재계산). 큰 잔차의 영향 억제.
- **Sparse ICP** (Bouaziz 2013): `L_p (p<1)` 정규화 → 강한 아웃라이어 내성.

**Weighting:** 거리, normal 호환성, color 유사도 기반.

**핵심 병목:** correspondence 탐색이 ICP 비용의 대부분. 그래서 공간 가속구조(kd-tree/BVH)가 필수다.

---

## 4. 초기화 / Global Registration — "ICP는 local이다"

ICP는 초기 오정렬이 크면 실패한다. 그래서 **coarse(global) → fine(ICP)** 파이프라인:

- Feature 기반: **FPFH + RANSAC** (Rusu 2009), **Fast Global Registration** (Zhou 2016), **TEASER++** (Yang 2020, certifiable/robust).
- 또는 **motion prior**: odometry/IMU/constant-velocity로 초기 `T` 제공.

**SLAM 맥락:** 연속 프레임에서는 직전 pose가 훌륭한 초기값이라 global registration이 대체로 불필요하다.
Global registration은 주로 **loop closure / relocalization**(관계가 끊긴 큰 변환)에서 쓰인다.

---

## 5. ICP → SLAM 경로

### 5.1 ICP의 역할 = front-end (odometry / scan matching)
- **Frame-to-frame:** 연속 scan 간 상대 pose. 단순하지만 drift가 빠르게 누적.
- **Frame-to-model (권장):** 새 scan을 누적 **map**(TSDF/surfel)에 정합 → drift 감소.
  - **KinectFusion** (Newcombe 2011): map = TSDF, model 예측 = TSDF **raycast**로 얻은 vertex/normal map, 정합 = **point-to-plane ICP**(projective association). 이 저장소의 `DirectionalTSDF` + BVH가 정확히 이 조합을 지향한다.

### 5.2 대표 시스템 (참고)
- Dense RGB-D: **KinectFusion**(TSDF + point-to-plane ICP), **ElasticFusion**(surfel + deformation graph), **BundleFusion**.
- LiDAR: **LOAM**(point-to-edge/point-to-plane feature), **LeGO-LOAM**, **KISS-ICP**(2023, point-to-point + adaptive threshold — 단순함의 힘).

### 5.3 SLAM = ICP + 그 이상
ICP odometry만으로는 drift가 남는다. **Full SLAM**이 추가하는 것:
- **Keyframe** 선정 + map 관리(메모리/일관성).
- **Loop closure**: place recognition으로 재방문 감지 → 그 순간 registration(ICP/global)으로 loop constraint 생성.
- **Back-end**: **pose graph optimization**(g2o / GTSAM / Ceres), 필요시 global bundle adjustment로 모든 pose를 전역 일관되게 재추정.

요약하면 **ICP는 SLAM의 "측정(constraint 생성)" 단계이고, SLAM은 그 constraint들의 전역 최적화**다.

---

## 6. 이 저장소(VkLBVH)에 매핑 — 점진적 구현 경로

**이미 있는 재료:**
- **Correspondence 가속** — `BVH`(`binary`/`wide` 백엔드)의 GPU `KNN(cx,cy,cz,k)` / `RadiusSearch`. ICP nearest-neighbor 탐색을 GPU로.
- **Normal 있는 point cloud + frame-to-model model** — `TSDF::DirectionalTSDF`:
  `Integrate(points, normals, cameraPos)`로 프레임을 누적, `PointCloud()`(normal 포함) / `ExportPointCloud()`로 표면 추출, overlap ratio 추적. point-to-plane frame-to-model의 "model" 쪽.
- **합성 스캔 + 정확도 평가 하네스** — `Engine::Eval`(`ScanSampler`, `SyntheticSurface`, `RmseMetrics`).
  ground-truth pose로 registration 정확도를 RMSE 회귀 테스트로 검증 가능.

**추천 단계 (각 단계는 독립적으로 테스트 가능):**
1. **CPU point-to-point ICP** — `SpatialIndex::KNN`로 correspondence, Arun/SVD로 transform. `Engine::Eval` 합성 데이터(알려진 변환)로 수렴·정확도 검증.
2. **Point-to-plane + robustness** — target normal(TSDF/scan에서) 사용, 거리·normal rejection + Huber 가중. 수렴 속도·정확도 개선.
3. **Frame-to-model odometry** — 각 프레임 scan을 누적 `DirectionalTSDF` 표면에 정합 → 6-DoF odometry. 직전 pose를 초기값으로.
4. **최소 SLAM** — keyframe + pose graph(GTSAM/Ceres 등) back-end + loop closure constraint. 여기서부터 "SLAM".

**주의 (선행 과제):** 대규모 correspondence를 GPU BVH로 돌리기 전에, 현재
[Engine::Core large-N 비결정 버그](KNOWN_ISSUES_engine_core_large_n.md)(N≳1000에서 잘못된/비결정적
결과)를 먼저 해결해야 GPU KNN을 신뢰할 수 있다. 그전까지는 CPU kd-tree/작은 N으로 프로토타이핑 권장.

---

## 7. 한눈에 보는 비교

| 방법 | Error metric | Correspondence | 강점 | SLAM 용도 |
|---|---|---|---|---|
| Point-to-Point (1992) | 점–점 | NN(kd-tree/BVH) | 단순·normal 불필요 | baseline, KISS-ICP LiDAR odom |
| Point-to-Plane (1991) | 점–접평면 | NN/projective | 빠른 수렴 | 실시간 depth, KinectFusion |
| Symmetric (2019) | 대칭 normal | NN | 넓은 수렴역·정확 | 정밀 정합 |
| GICP (2009) | plane-to-plane | NN | 확률적·LiDAR 강함 | LiDAR SLAM(+VGICP GPU) |
| Colored ICP (2017) | geo+photo | projective/NN | 텍스처 활용 | RGB-D SLAM |
| NDT (2003) | 분포 우도 | correspondence-free | 대응 불필요 | LiDAR scan matching |

**세 가지 결정만 기억하면 됨:** (1) error metric(속도·정밀) → (2) correspondence(정확·속도) →
(3) robustness(아웃라이어·부분겹침). 이 저장소는 (2)를 GPU BVH로, (1)의 normal을 TSDF로 이미 지원한다.

---

## 8. 참고문헌
- Besl & McKay, *A Method for Registration of 3-D Shapes*, PAMI 1992. (point-to-point)
- Chen & Medioni, *Object modelling by registration of multiple range images*, 1991/92. (point-to-plane)
- Rusinkiewicz & Levoy, *Efficient Variants of the ICP Algorithm*, 3DIM 2001. (6-stage taxonomy)
- Segal, Haehnel, Thrun, *Generalized-ICP*, RSS 2009.
- Rusinkiewicz, *A Symmetric Objective Function for ICP*, SIGGRAPH 2019.
- Park, Zhou, Koltun, *Colored Point Cloud Registration Revisited*, ICCV 2017.
- Biber & Straßer, *The Normal Distributions Transform*, IROS 2003.
- Chetverikov et al., *The Trimmed ICP (TrICP)*, ICPR 2002.
- Bouaziz, Tagliasacchi, Pauly, *Sparse Iterative Closest Point*, SGP 2013.
- Rusu et al., *FPFH*, ICRA 2009. · Zhou et al., *Fast Global Registration*, ECCV 2016. · Yang et al., *TEASER++*, T-RO 2020.
- Newcombe et al., *KinectFusion*, ISMAR 2011. · Whelan et al., *ElasticFusion*, RSS 2015.
- Zhang & Singh, *LOAM*, RSS 2014. · Vizzo et al., *KISS-ICP*, RA-L 2023. · Koide et al., *Voxelized GICP (VGICP)*, ICRA 2021.

# ICP 커리큘럼 — 정합에서 SLAM·파운데이션 모델까지

**목적:** point cloud registration(정합)을 **기초 → 클래식 ICP 구현 → 강건성/변형 →
전역 정합 → SLAM 통합 → 딥러닝·파운데이션 프론티어**의 순서로 학습하는 로드맵.
이 저장소(VkLBVH)의 `SpatialIndex::KNN`(GPU 정합 가속), `DirectionalTSDF`(frame-to-model 모델),
`Engine::Eval`(RMSE 평가)과 연결된 실습을 포함한다.

이 문서는 **학습 경로(로드맵)**다. 방법론의 상세 이론과 수식은 이미 정리된 참조 문서를 쓴다:

> **참조(먼저/함께 볼 것):** [ICP_METHODS](ICP_METHODS.md) — ICP 계열 방법론·수식·6단계 분류법.
> [ICP_LOCAL_VS_GLOBAL](ICP_LOCAL_VS_GLOBAL.md) — local(fine) vs global(coarse) 정합 분석.
> 이 커리큘럼은 그 위에 **무엇을·어떤 순서로·어떻게 실습하는가**를 얹는다.

**대상:** 선형대수·3D 기하 기본이 있고 정합→SLAM으로 나아가려는 사람.
**선수지식:** SO(3)/SE(3) 감각, C++/파이썬, 최소제곱. 재구성(TSDF) 부분은 [TSDF_CURRICULUM](TSDF_CURRICULUM.md)와 병행.

---

## 로드맵 한눈에

| 레벨 | 주제 | 핵심 산출물 | 최신도 |
|---|---|---|---|
| **L0** | 강체변환·대응·목적함수 기초 | 문제 정식화 이해 | 항상 유효 |
| **L1** | 클래식 ICP 구현 | 동작하는 point-to-point/plane ICP | 1991~ (정본) |
| **L2** | 강건성 & 변형 | 아웃라이어·부분겹침에 강한 ICP | 2003~2019 |
| **L3** | 전역 정합(초기화) | coarse→fine 파이프라인 | 2009~현재 |
| **L4** | SLAM 통합 | frame-to-model odometry + 백엔드 | 2011~현재 |
| **L5** | 딥러닝·파운데이션 프론티어 | 학습 기반 정합/SLAM | 2019~2026 |

각 레벨은 **목표 / 핵심 개념 / 자료 / VkLBVH 실습 / 완료 기준**으로 구성된다.

---

## L0 — 강체변환 · 대응 · 목적함수의 기초

**목표:** 정합 문제를 정확히 정식화하고 ICP가 왜 **local** 최적화인지 이해한다.

**핵심 개념**
- 강체변환 `T=(R,t)`, `R∈SO(3)`, `t∈ℝ³`; 회전 표현(행렬/쿼터니언/axis-angle/Lie algebra `so(3)`).
- **SE(3)·left/right perturbation**, 지수/로그 사상 — L1의 선형화·최적화의 기반.
- 정합 목적함수와 **ICP 반복 루프**: correspondence → transform 추정 → 적용 → 반복.
- **핵심 한계:** 초기 정렬이 나쁘면 잘못된 극소값 → 좋은 초기값(전역 정합/모션 prior)이 선행.
- **6단계 분류법**(Rusinkiewicz & Levoy 2001): Selection·Matching·Weighting·Rejecting·Error metric·Minimization.
  → 상세: [ICP_METHODS](ICP_METHODS.md) §1.

**자료:** [ICP_METHODS](ICP_METHODS.md) §1, *A Micro Lie Theory for State Estimation*(Solà 2018), Rusinkiewicz & Levoy 2001.

**완료 기준:** ICP 루프 3단계와 "왜 local인가"를 설명하고, 6단계 분류의 각 축이 무엇을 결정하는지 안다.

---

## L1 — 클래식 ICP 구현

**목표:** point-to-point / point-to-plane ICP를 직접 구현하고 합성 데이터로 수렴·정확도를 검증한다.

**핵심 개념**
- **Point-to-Point**(Besl & McKay 1992): 대응 고정 시 **closed-form**(Arun SVD / Horn quaternion).
  중심 정렬 → `H=Σ p̃q̃ᵀ` → SVD → `R=VUᵀ`. 단순하지만 접선 슬라이딩에 약해 수렴 느림.
- **Point-to-Plane**(Chen & Medioni 1991): 점–접평면 거리 → 접선 자유도 허용해 **훨씬 빠른 수렴**.
  소각도 선형화로 6×6 선형 최소제곱. target **노멀 필요**. (실시간 depth 정합의 표준.)
- **대응 탐색:** nearest neighbor(**kd-tree** 표준, 또는 **BVH**), projective association(depth image, O(1)).
- 수렴 판정, 종료 조건, 좌표계 관리. → 상세: [ICP_METHODS](ICP_METHODS.md) §2.1–2.2, §3.

**자료:** [ICP_METHODS](ICP_METHODS.md) §2–3, Besl & McKay 1992, Chen & Medioni 1991, Open3D `registration_icp` 예제.

**VkLBVH 실습 (핵심 트랙)**
1. **CPU point-to-point ICP:** 대응은 `BVH::KNN(cx,cy,cz,k=1)`(`src/BVH/BVH.h`),
   변환은 Arun/SVD. `Engine::Eval`로 **알려진 변환**을 준 합성 점군에서 수렴·정확도 검증(RMSE).
2. **Point-to-plane 확장:** target 노멀(스캔/TSDF에서) 사용 + 거리·노멀 rejection → 수렴 속도·정확도 개선.

**완료 기준:** 두 metric을 구현해 동일 데이터에서 반복수·최종 RMSE를 비교하고, point-to-plane이 왜 빨리 수렴하는지 설명한다.

---

## L2 — 강건성 & 변형

**목표:** 아웃라이어·부분겹침·센서 노이즈에 강한 ICP 변형을 익힌다(실전 성패를 가르는 부분).

**핵심 개념**
- **변형(variant):** **GICP**(plane-to-plane, 확률적; +**VGICP** voxel·GPU), **Symmetric ICP**(2019, 넓은 수렴역),
  **Colored ICP**(2017, geo+photo), **NDT**(분포 우도, correspondence-free).
- **강건화:** robust kernel(Huber/Cauchy/Geman-McClure) + **IRLS**, **Trimmed ICP**(상위 overlap만),
  **Sparse ICP**(`L_p, p<1`). → 상세: [ICP_METHODS](ICP_METHODS.md) §2.3–2.6, §3.
- **가중/거부:** 거리·노멀 호환성·color, boundary/중복 대응 제거.

**자료:** [ICP_METHODS](ICP_METHODS.md) §2–3, Segal et al. *GICP* 2009, Rusinkiewicz *Symmetric* 2019,
Park et al. *Colored ICP* 2017, Koide et al. *VGICP* 2021.

**VkLBVH 실습**
- L1 ICP에 **Huber IRLS + trimming**을 추가하고, 아웃라이어를 섞은 합성 데이터에서 강건성 향상을 RMSE로 측정.
- `DirectionalTSDF`가 추적하는 **overlap ratio**를 trimming 비율 `ξ`에 연결.
- (선택) GICP의 공분산 항을 구현해 LiDAR류(얇은 표면) 데이터에서 point-to-plane과 비교.

**완료 기준:** robust kernel이 큰 잔차의 영향을 어떻게 줄이는지, trimmed ICP가 부분겹침에 왜 강한지 설명한다.

---

## L3 — 전역 정합 (초기화)

**목표:** ICP의 local 한계를 넘는 **coarse→fine** 파이프라인 — 특징 기반/학습 기반 전역 정합 — 을 이해한다.

**핵심 개념**
- **손수 특징:** **FPFH + RANSAC**(Rusu 2009), **Fast Global Registration**(Zhou 2016),
  **TEASER++**(Yang 2020, certifiable/robust).
- **학습 특징:** **FCGF**(fully convolutional geometric features), **Predator**(낮은 overlap), **D3Feat**.
- **트랜스포머 정합:** **GeoTransformer**(2022) — 전역 정보 통합으로 inlier↑, local-to-global로 **RANSAC-free**.
  Point Tree Transformer 등 후속.
- 언제 전역 정합이 필요한가: 연속 프레임은 직전 pose가 좋은 초기값 → 대개 불필요;
  **loop closure/relocalization**(큰 변환)에서 필수. → 상세: [ICP_LOCAL_VS_GLOBAL](ICP_LOCAL_VS_GLOBAL.md).

**자료:** [ICP_LOCAL_VS_GLOBAL](ICP_LOCAL_VS_GLOBAL.md), Rusu 2009(FPFH), Zhou 2016(FGR), Yang 2020(TEASER++),
Qin et al. *GeoTransformer*(CVPR 2022), [DL 정합 서베이](https://link.springer.com/article/10.1007/s11263-025-02723-w).

**VkLBVH 실습**
- **FPFH + RANSAC**(Open3D)로 큰 오정렬 합성쌍의 초기 `T`를 구한 뒤 L1/L2 ICP로 정밀화 → coarse→fine 파이프라인 완성.
- 초기 오정렬 각도를 키워가며 **순수 ICP vs (전역+ICP)** 의 성공률 곡선을 그린다(local 한계 재현).

**완료 기준:** 왜 ICP 앞에 전역 정합이 필요한지, GeoTransformer가 RANSAC 없이 어떻게 정합하는지 설명한다.

---

## L4 — SLAM 통합

**목표:** ICP를 SLAM front-end(odometry)로 올리고 back-end(pose graph/loop closure)까지 잇는다.

**핵심 개념**
- **역할:** ICP = SLAM의 **측정(constraint 생성)**; SLAM = 그 constraint들의 전역 최적화.
- **frame-to-model odometry:** 새 scan을 누적 **map**(TSDF/surfel)에 정합 → drift↓.
  **KinectFusion**(TSDF + point-to-plane, projective) — 이 저장소가 지향하는 조합.
  → [TSDF_CURRICULUM](TSDF_CURRICULUM.md) L1과 직접 연결.
- **대표 시스템:** dense RGB-D(KinectFusion·ElasticFusion·BundleFusion),
  LiDAR(LOAM·LeGO-LOAM·**KISS-ICP** 2023·VGICP).
- **back-end:** keyframe·map 관리, **loop closure**(place recognition → 재정합으로 constraint 생성),
  **pose graph optimization**(g2o/**GTSAM**/Ceres), 필요시 global BA. → 상세: [ICP_METHODS](ICP_METHODS.md) §5.

**자료:** [ICP_METHODS](ICP_METHODS.md) §5, Newcombe *KinectFusion* 2011, Vizzo *KISS-ICP* 2023,
GTSAM/Ceres 튜토리얼, *Factor Graphs for Robot Perception*(Dellaert & Kaess).

**VkLBVH 실습 (핵심 트랙)**
3. **frame-to-model odometry:** 각 프레임 scan을 누적 `DirectionalTSDF` 표면에 point-to-plane 정합 →
   6-DoF odometry. 직전 pose를 초기값으로. `Engine::Eval`의 GT pose로 drift를 정량화.
4. **최소 SLAM:** keyframe + pose graph 백엔드(GTSAM/Ceres) + loop closure constraint. 여기서부터 "SLAM".

**완료 기준:** frame-to-frame vs frame-to-model의 drift 차이를 측정으로 보이고, pose graph가 loop closure를 어떻게 전역 일관성으로 바꾸는지 설명한다.

---

## L5 — 딥러닝 · 파운데이션 프론티어 (2019–2026)

**목표:** 학습 기반 정합과, 재구성/정합 프론트엔드를 재편 중인 3D 파운데이션 모델을 파악한다.

**핵심 개념**
- **학습 기반 정합 계보:** **PointNetLK**·**DCP**(2019) → **DGR**(deep global registration) →
  **Predator**·**GeoTransformer** → 트랜스포머·트리 기반. (서베이로 지형 파악.)
- **correspondence-free 파운데이션 모델:** **DUSt3R**(CVPR 2024) — 이미지 쌍에서 직접 dense 3D 포인트맵 예측
  (대응/보정 불필요) → **MASt3R**(매칭 강화)·**VGGT**(다중뷰). global motion averaging으로 여러 뷰 통합.
- **파운데이션-SLAM:** **MASt3R-SLAM**(CVPR 2025, 실시간 dense SLAM), **FoundationSLAM**(depth foundation model 기반),
  VGGT-SLAM. 전통 ICP front-end가 학습 prior로 대체·보완되는 흐름.
- 관점: 학습 특징/모델은 **초기화·저오버랩·텍스처 부족**에서 강하고, 클래식 ICP는 **정밀·검증가능·경량**.
  실전은 둘의 조합(학습 coarse → 기하 fine).

**자료:** [DL 정합 서베이(IJCV 2025)](https://link.springer.com/article/10.1007/s11263-025-02723-w),
Wang et al. **DUSt3R**(CVPR 2024)·**MASt3R**, **VGGT**(2025), **MASt3R-SLAM**(CVPR 2025),
[DUSt3R/MASt3R/VGGT 평가](https://arxiv.org/pdf/2507.14798), [PCR 방법 모음](https://github.com/yxzhang15/PCR).

**VkLBVH 실습**
- **DUSt3R/MASt3R**(파이썬)로 이 저장소 스캔 이미지에서 초기 정합·포인트맵을 얻고,
  이를 초기값으로 L2 기하 ICP를 얹어 **학습 coarse → 기하 fine** 하이브리드를 만든다.
- 클래식(FPFH+ICP) vs 학습(GeoTransformer/DUSt3R)을 같은 합성 GT로 **성공률·정확도·시간** 비교(벤치마크 하네스).

**완료 기준:** 학습 기반 정합과 클래식 ICP의 상보성을 설명하고, DUSt3R류가 왜 "correspondence-free"로 불리는지 안다.

---

## 도구 · 프레임워크

- **정합/SLAM:** Open3D, PCL, small_gicp, **KISS-ICP**, **TEASER++**, GTSAM/g2o/Ceres(백엔드).
- **학습:** PyTorch, **GeoTransformer**·**Predator**·FCGF 공개 구현, **DUSt3R/MASt3R/VGGT** 리포.
- **평가/데이터:** 이 저장소 `Engine::Eval`(GT pose·RMSE), 3DMatch/KITTI/ETH(정합 벤치마크).

## VkLBVH 실습 트랙 (한눈에)

기존 [ICP_METHODS §6](ICP_METHODS.md)의 4단계를 커리큘럼 레벨에 매핑:

| 트랙 | 레벨 | 저장소 재료 |
|---|---|---|
| ① CPU point-to-point ICP | L1 | `SpatialIndex::KNN`, `Engine::Eval` |
| ② point-to-plane + robust | L1–L2 | target 노멀(TSDF/scan), IRLS |
| ③ frame-to-model odometry | L4 | `DirectionalTSDF`, GT pose |
| ④ 최소 SLAM(pose graph+loop) | L4 | GTSAM/Ceres + ③ |

**주의(선행 과제):** 대규모 correspondence를 GPU BVH로 돌리기 전에
[Engine::Core large-N 비결정 버그](KNOWN_ISSUES_engine_core_large_n.md)(N≳1000)를 먼저 해결해야
GPU KNN을 신뢰할 수 있다. 그전까지는 CPU kd-tree/작은 N 프로토타이핑 권장.

## 참고문헌 (핵심)
- Besl & McKay, *A Method for Registration of 3-D Shapes*, PAMI 1992. · Chen & Medioni, *point-to-plane*, 1991.
- Rusinkiewicz & Levoy, *Efficient Variants of the ICP Algorithm*, 3DIM 2001. · Rusinkiewicz, *Symmetric ICP*, SIGGRAPH 2019.
- Segal et al., *Generalized-ICP*, RSS 2009. · Koide et al., *VGICP*, ICRA 2021.
- Rusu et al., *FPFH*, ICRA 2009. · Zhou et al., *Fast Global Registration*, ECCV 2016. · Yang et al., *TEASER++*, T-RO 2020.
- Qin et al., *GeoTransformer*, CVPR 2022. · *DL-Based Point Cloud Registration Survey*, IJCV 2025.
- Newcombe et al., *KinectFusion*, ISMAR 2011. · Vizzo et al., *KISS-ICP*, RA-L 2023.
- Wang et al., *DUSt3R*, CVPR 2024. · *MASt3R-SLAM*, CVPR 2025. · *VGGT*, 2025.

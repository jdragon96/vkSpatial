# TSDF 커리큘럼 — 볼류메트릭 재구성부터 뉴럴·가우시안 매핑까지

**목적:** TSDF(Truncated Signed Distance Function) 기반 3D 재구성을 **기초 → KinectFusion
파이프라인 → 확장성(해싱/VDB/옥트리) → 표면 추출 → 뉴럴 암시적 → 가우시안/파운데이션 프론티어**로
학습하는 로드맵. 각 레벨은 이 저장소(VkLBVH)의 `Engine::Spatial` TSDF(`SimpleTSDF`/`DirectionalTSDF`),
marching cubes 추출(`SimpleTSDF::ExportMC`), `Engine::Eval` 스캔/평가 하네스와 연결된 실습을 포함한다. 2025 최신
(variance-adaptive grid, GS+SDF 하이브리드, 뉴럴 SLAM 매핑)까지 이어진다.

**대상:** 3D 기하/카메라 기본이 있고, dense 재구성 파이프라인을 밑바닥부터 이해하려는 사람.
**선수지식:** 선형대수, 카메라 모델(intrinsics/extrinsics), C++/GPU 기본. 정합(ICP) 부분은
[ICP_CURRICULUM](ICP_CURRICULUM.md)와 병행 권장.
**사용법:** 레벨 순차 진행. 실습은 `Engine::Spatial`/`Engine::Eval`을 뼈대로 삼는다.

> 관련 문서: [BVH](BVH.md) · [ENGINE_CORE_RENDER](ENGINE_CORE_RENDER.md) ·
> [ICP_METHODS](ICP_METHODS.md) · [ICP_LOCAL_VS_GLOBAL](ICP_LOCAL_VS_GLOBAL.md) ·
> [KNOWN_ISSUES_engine_core_large_n](KNOWN_ISSUES_engine_core_large_n.md)

---

## 로드맵 한눈에

| 레벨 | 주제 | 핵심 산출물 | 최신도 |
|---|---|---|---|
| **L0** | SDF/TSDF·복셀·통합 기초 | 단일 뷰 TSDF 볼륨 | 항상 유효 |
| **L1** | KinectFusion 파이프라인 | depth→integrate→raycast→mesh | 2011~ (정본) |
| **L2** | 확장성: 해싱/VDB/옥트리 | 대규모 씬 실시간 매핑 | 2013~2025 |
| **L3** | 표면 추출 & 방향/색/의미 | 고품질 메시, DirectionalTSDF | 표준~최신 |
| **L4** | 뉴럴 암시적 표현 | 학습된 SDF/occupancy | 2019~ |
| **L5** | 가우시안·파운데이션 프론티어 | GS+SDF, 뉴럴 SLAM 매핑 | 2023~2026 |

각 레벨은 **목표 / 핵심 개념 / 자료 / VkLBVH 실습 / 완료 기준**으로 구성된다.

---

## L0 — SDF/TSDF · 복셀 · 통합의 기초

**목표:** SDF와 TSDF의 정의, 복셀 그리드, truncation, 가중 이동평균 통합을 이해하고 단일 depth 뷰로 TSDF 볼륨을 만든다.

**핵심 개념**
- **SDF:** 공간의 각 점에서 표면까지의 부호 있는 거리(안=음, 밖=양, 표면=0 등위면).
- **TSDF:** SDF를 표면 근처 `±μ`(truncation)로 자른 것. 표면 근방만 저장·업데이트해 효율↑.
- **복셀 그리드:** 균일 격자에 `(tsdf, weight)`(+선택적 color) 저장. 해상도 vs 메모리 트레이드오프.
- **통합(integration):** 각 복셀을 카메라에 투영해 측정 depth와 비교, SDF 부호거리 계산 후
  **가중 이동평균**으로 누적: `tsdf ← (w·tsdf + wᵢ·sdfᵢ)/(w+wᵢ)`, `w ← min(w+wᵢ, w_max)`.
- 좌표/카메라: intrinsics(K), extrinsics(T_cw), 복셀 중심의 world→camera→pixel 투영.

**자료:** Curless & Levoy, *A Volumetric Method for Building Complex Models from Range Images*, SIGGRAPH 1996 (TSDF 원전),
[TSDF 개관](https://www.emergentmind.com/topics/truncated-signed-distance-field-tsdf), Open3D TSDF 튜토리얼.

**VkLBVH 실습**
- `Engine::Spatial::DirectionalTSDF`의 `Integrate(points, normals, cameraPos, aabbCenterHint)` 시그니처와
  내부 복셀 업데이트를 읽는다. 단일 프레임만 통합해 볼륨을 관찰.
- `Engine::Eval`의 `SyntheticSurface`(sphere/plane/box/torus)로 **정확한 depth**를 만들고 한 뷰를 통합.

**완료 기준:** truncation `μ`와 `w_max`가 각각 무엇을 조절하는지 설명하고, 가중 이동평균이 왜 센서 노이즈를 줄이는지 안다.

---

## L1 — KinectFusion 파이프라인

**목표:** 실시간 dense 재구성의 정본 루프 — **depth → TSDF integrate → raycast → ICP tracking** — 을 이해한다.

**핵심 개념**
- **KinectFusion**(Newcombe 2011)의 4단계 루프:
  1. **Surface measurement:** depth → 정점/노멀 맵(bilateral filter).
  2. **Pose estimation:** 예측 모델 vs 현재 프레임 **point-to-plane ICP**(projective association).
     → [ICP_CURRICULUM](ICP_CURRICULUM.md) L1/L4와 직접 연결.
  3. **Volumetric integration:** 위 L0의 가중 통합.
  4. **Raycasting:** TSDF를 sphere-trace해 예측 정점/노멀 맵 생성(다음 프레임 tracking·렌더용).
- **frame-to-model**(누적 모델에 정합)이 frame-to-frame보다 drift가 적은 이유.
- 한계: **고정 해상도 균일 그리드** → 작은 bounded 씬에만 적합, 메모리 폭발.

**자료:** Newcombe et al., *KinectFusion: Real-Time Dense Surface Mapping and Tracking*, ISMAR 2011.
Izadi et al.(KinectFusion UIST 2011). Open3D / `kinfu` 구현.

**VkLBVH 실습**
- `Engine::Eval`로 **합성 멀티뷰 스캔**(궤도 카메라)을 생성 → `SimpleTSDF`/`DirectionalTSDF`의
  `Integrate`로 프레임마다 누적 → `SimpleTSDF::ExportMC`(마칭큐브 메시) 또는
  `DirectionalTSDF::ExportPointCloud`(노멀 포인트클라우드)로 표면 추출.
- 이 저장소의 [BVH](BVH.md)/`SpatialIndex::KNN`이 raycast/association 가속에 어떻게 쓰일 수 있는지 스케치.

**완료 기준:** KinectFusion 4단계를 순서대로 설명하고, raycast가 왜 tracking·통합 양쪽에 필요한지 안다.

---

## L2 — 확장성: 해싱 · VDB · 옥트리

**목표:** 고정 그리드의 메모리 한계를 넘어 대규모 씬을 실시간 매핑하는 자료구조를 익힌다.

**핵심 개념**
- **Voxel Hashing**(Nießner 2013): 표면 근처 복셀 블록만 **공간 해시**로 희소 저장 → 무제한 크기 씬.
  이 저장소의 공간 해시/블록 개념과 연결.
- **VoxBlox**(Oleynikova 2017): adaptive weight + grouped raycasting, ESDF까지 실시간 생성(로봇 경로계획용).
- **OpenVDB / NanoVDB**: 영화·시뮬레이션의 표준 희소 볼륨 자료구조. **NanoVDB**는 GPU 친화적 read-only 트리.
  **VDBFusion**(Vizzo 2022): OpenVDB로 LiDAR 포인트를 CPU 단일 코어로도 20fps 통합.
- **옥트리 / 멀티해상도**: **Supereight2**(옥트리 기반 multi-res), octree TSDF.
- **최신(2025):** **variance-adaptive voxel grid** — 국소 TSDF 분산에 따라 해상도를 조절 +
  해상도 경계를 잇는 확장 marching cubes([Resolution Where It Counts, TOG 2025](https://dl.acm.org/doi/10.1145/3777909)).
  **DB-TSDF**(2025): directional bitmask 통합의 CPU 전용 LiDAR 매핑.

**자료:** Nießner et al., *Real-time 3D Reconstruction at Scale using Voxel Hashing*, SIGGRAPH Asia 2013.
Oleynikova et al., *Voxblox*, IROS 2017. Vizzo et al., *VDBFusion*, Sensors 2022. Museth, *OpenVDB/NanoVDB*.

**VkLBVH 실습**
- `DirectionalTSDF`의 저장 구조가 균일 그리드인지 해시/블록인지 확인하고,
  **표면 근처 블록만 할당**하도록(또는 그 설계를 문서화) 확장 방향을 잡는다.
- 큰 합성 씬(여러 오브젝트)에서 메모리 사용을 측정해 균일 그리드의 한계를 재현.

**완료 기준:** voxel hashing이 왜 "무제한 크기"를 가능케 하는지, NanoVDB가 왜 GPU에 유리한지 설명한다.

---

## L3 — 표면 추출 & 방향/색/의미 통합

**목표:** TSDF에서 고품질 메시를 뽑고(추출 알고리즘), 방향·색·의미 정보를 통합한다.

**핵심 개념**
- **등위면 추출:**
  - **Marching Cubes**(Lorensen 1987): 복셀 코너 부호로 삼각형 생성(256 케이스). 이 저장소의 `SimpleTSDF::ExportMC`.
  - 한계: 날카로운 특징 뭉개짐 → **Dual Contouring**(Hermite data, sharp features), **Dual Marching Cubes**.
- **DirectionalTSDF**(이 저장소): 방향별(예: 6-축) TSDF로 얇은 구조/반대면 간섭·접선 슬라이딩을 완화.
  단일 TSDF 대비 얇은 물체·마주보는 면에서 유리. (cf. [ICP_METHODS](ICP_METHODS.md)의 normal 활용과 시너지.)
- **속성 통합:** color TSDF, 시맨틱/instance 라벨 통합(semantic mapping), TSDF weight 설계(각도/거리/센서 모델).
- 품질 이슈: truncation·해상도·정합 정확도가 메시 품질에 미치는 영향, 홀 채우기.

**자료:** Lorensen & Cline, *Marching Cubes*, SIGGRAPH 1987. Ju et al., *Dual Contouring of Hermite Data*, SIGGRAPH 2002.
이 저장소 소스: `Engine::Spatial::SimpleTSDF`(`ExportMC` 마칭큐브 메시) · `DirectionalTSDF`(`ExportPointCloud` 노멀 포인트).

**VkLBVH 실습**
- `ObjectScanner`/`scan_dataset_gen`으로 스캔한 메시(예: `data/chair.ply`)를 `SimpleTSDF`로 재구성 →
  `ExportMC` 메시를 원본과 비교(방향 정보가 필요하면 `DirectionalTSDF::ExportPointCloud`의 노멀 포인트와 대조).
- `Engine::Eval`의 RMSE 지표(`AccuracyRMSE`/`CompletenessRMSE`, `RmseMetrics.h`)로 **재구성 정확도·완전성**을 정량화 —
  truncation/해상도를 바꿔 품질-비용 곡선을 그린다. (기존 RMSE 회귀 테스트 활용.)
- (심화) marching cubes vs dual contouring를 같은 볼륨에서 비교(날카로운 모서리 보존).

**완료 기준:** marching cubes가 날카로운 특징을 왜 뭉개는지, DirectionalTSDF가 단일 TSDF 대비 무엇을 개선하는지 설명한다.

---

## L4 — 뉴럴 암시적 표현

**목표:** 이산 복셀 대신 신경망으로 SDF/occupancy를 표현하는 계열을 이해한다.

**핵심 개념**
- **DeepSDF**(Park 2019): 좌표 → SDF를 MLP로. latent code로 형상 공간 표현.
- **Occupancy Networks**(Mescheder 2019): 점유 확률 필드.
- **Instant-NGP**(Müller 2022): **멀티해상도 해시 인코딩**으로 학습·추론을 실시간 급으로 가속(재구성/NeRF 공용).
- **NeuS / VolSDF**(2021): SDF를 볼륨 렌더링으로 학습 → 멀티뷰 이미지에서 표면 재구성(노멀 정합 좋음).
- **Neuralangelo**(2023): 해시 인코딩 + 수치 그래디언트로 고디테일 표면.
- 트레이드오프: 연속·컴팩트·홀에 강함 vs 학습 필요·실시간 통합의 어려움. (전통 TSDF의 **증분 통합**과 대비.)

**자료:** Park et al., *DeepSDF*, CVPR 2019. Wang et al., *NeuS*, NeurIPS 2021. Müller et al., *Instant-NGP*, SIGGRAPH 2022.
Li et al., *Neuralangelo*, CVPR 2023.

**VkLBVH 실습**
- 이 저장소의 합성 스캔(GT 포함)으로 작은 **DeepSDF/NeuS 재구성**을 학습(파이썬)해,
  전통 `DirectionalTSDF` 재구성과 **같은 GT로 RMSE 비교**(정확도·완전성·메모리·시간).
- 학습된 SDF를 복셀 그리드로 샘플링 → marching cubes(`SimpleTSDF::ExportMC`의 추출 경로 재사용)로 메시화.

**완료 기준:** 뉴럴 암시적 표현의 장단점을 전통 TSDF와 대비해 설명하고, 해시 인코딩이 왜 속도를 바꾸는지 안다.

---

## L5 — 가우시안 · 파운데이션 프론티어 (2023–2026)

**목표:** 표면 재구성의 최신 흐름 — 가우시안 스플래팅 기반 표면화, GS+SDF 하이브리드, 뉴럴 SLAM 매핑,
파운데이션 모델 — 을 파악하고 이 저장소 파이프라인과의 접점을 잡는다.

**핵심 개념**
- **가우시안 스플래팅 표면화:** 3DGS는 뷰 합성엔 좋지만 표면이 지저분 → **2D Gaussian Splatting(2DGS)**,
  **GOF(Gaussian Opacity Fields)**, unbiased-depth 2DGS(2025)로 **정확한 메시** 추출.
  → [RENDERING_CURRICULUM](RENDERING_CURRICULUM.md) L5와 교차.
- **GS + SDF 하이브리드(2025):** **PINGS**(GS + distance field 점기반 뉴럴 맵, RSS 2025),
  **GS-SDF**(LiDAR 보강), **GSDF/SplatSDF/GaussianRoom** — 렌더 품질과 기하 일관성을 동시에.
- **뉴럴/GS SLAM 매핑:** **iMAP**(2021), **NICE-SLAM**(뉴럴 그리드), **Co-SLAM**, **Point-SLAM**,
  **Gaussian-SLAM / SplaTAM**(3DGS를 맵으로) — TSDF 매핑의 뉴럴 후계.
- **파운데이션 모델:** **DUSt3R/MASt3R/VGGT** — 이미지에서 직접 dense 3D를 예측(대응/보정 불필요),
  **MASt3R-SLAM**(CVPR 2025), **FoundationSLAM** — 재구성의 프론트엔드를 재편 중.
- 큰 그림: **TSDF(증분·기하 정확·검증 쉬움) ↔ GS/뉴럴(고품질 렌더·연속) ↔ 파운데이션(데이터 구동)** 의 수렴.

**자료:** Huang et al. **2DGS**(SIGGRAPH 2024). PINGS(RSS 2025), GS-SDF(2025), GSDF(NeurIPS 2025).
Zhu et al. **NICE-SLAM**(CVPR 2022), **SplaTAM**(CVPR 2024). Wang et al. **DUSt3R**(CVPR 2024), **MASt3R-SLAM**(CVPR 2025).
[DUSt3R/MASt3R/VGGT 평가](https://arxiv.org/pdf/2507.14798).

**VkLBVH 실습**
- 이 저장소의 스캔 데이터셋을 입력으로 **2DGS 표면 재구성**을 돌려 `SimpleTSDF::ExportMC` TSDF 메시와 정량 비교(RMSE·시간·품질).
- **TSDF vs GS vs 뉴럴**을 같은 GT/데이터로 비교하는 **벤치마크 하네스**를 `Engine::Eval` 위에 확장(정확도·완전성·메모리·속도).

**완료 기준:** 2DGS/GS-SDF가 순수 TSDF나 순수 3DGS 대비 무엇을 개선하는지, 파운데이션 모델(DUSt3R류)이
전통 재구성 프론트엔드를 어떻게 바꾸는지 설명한다.

---

## 도구 · 프레임워크

- **재구성/볼륨:** Open3D, **OpenVDB/NanoVDB**, **VDBFusion**, Voxblox, Supereight2, `kinfu`.
- **뉴럴/GS:** PyTorch, tiny-cuda-nn, `nerfstudio`/`gsplat`, `graphdeco-inria/gaussian-splatting`, SDFStudio.
- **평가/데이터:** 이 저장소 `Engine::Eval`(RMSE·스캔 하네스), TUM RGB-D / ScanNet / Replica(벤치마크).

## VkLBVH 재료 매핑

| 커리큘럼 단계 | 저장소 컴포넌트 |
|---|---|
| L0–L1 통합/파이프라인 | `Engine::Spatial::DirectionalTSDF::Integrate` |
| L1 tracking(ICP) | `SpatialIndex::KNN` + [ICP_CURRICULUM](ICP_CURRICULUM.md) |
| L3 추출/평가 | `SimpleTSDF::ExportMC`(MC 메시) · `DirectionalTSDF::ExportPointCloud`, `RmseMetrics.h`(`AccuracyRMSE`/`CompletenessRMSE`) |
| 데이터 생성 | `ObjectScanner`, `scan_dataset_gen`, `SyntheticSurface` |

**주의(선행 과제):** 대규모 GPU 통합/정합 전에
[Engine::Core large-N 비결정 버그](KNOWN_ISSUES_engine_core_large_n.md)(N≳1000)를 먼저 해결해야
GPU 경로를 신뢰할 수 있다. 그전까지는 작은 N/CPU 프로토타이핑 권장.

## 참고문헌 (핵심)
- Curless & Levoy, *A Volumetric Method for Building Complex Models from Range Images*, SIGGRAPH 1996.
- Lorensen & Cline, *Marching Cubes*, SIGGRAPH 1987.
- Newcombe et al., *KinectFusion*, ISMAR 2011.
- Nießner et al., *Real-time 3D Reconstruction at Scale using Voxel Hashing*, SIGGRAPH Asia 2013.
- Oleynikova et al., *Voxblox*, IROS 2017. · Vizzo et al., *VDBFusion*, 2022.
- Park et al., *DeepSDF*, CVPR 2019. · Wang et al., *NeuS*, 2021. · Müller et al., *Instant-NGP*, 2022.
- Kerbl et al., *3D Gaussian Splatting*, 2023. · Huang et al., *2D Gaussian Splatting*, 2024.
- *Resolution Where It Counts (Variance-Adaptive Voxel Grids)*, ACM TOG 2025. · PINGS(RSS 2025) · MASt3R-SLAM(CVPR 2025).

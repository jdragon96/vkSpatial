# 3D Vision × Mesh 초특급 전문가 로드맵

당신은 이미 **GPU TSDF 재구성 엔진(VkLBVH)** 을 밑바닥부터 만들 수 있는 사람입니다. 여기서 "Mesh"로 더 깊이 파려면, **표면(surface)을 수학적으로 다루는 능력**을 층층이 쌓는 게 핵심입니다. 아래는 기초→전문→최전선 순서의 실전 로드맵입니다.

---

## 0. 큰 그림 — Mesh를 관통하는 3개 축
1. **표현(Representation):** 점군 ↔ SDF/TSDF ↔ 메시 ↔ 신경 임플리싯 ↔ Gaussian. 서로 변환하는 다리(Marching Cubes, Dual Contouring, Poisson, NKSR...)를 손에 쥐는 것.
2. **기하 처리(Geometry Processing):** 이미 있는 메시를 다루기 — 단순화·리메싱·평활화·파라미터화·곡률.
3. **미분가능(Differentiable):** 메시를 **학습·최적화 변수**로 다루기 — differentiable rendering, DMTet/FlexiCubes. ← 지금 최전선.

당신 강점(TSDF·GPU)은 축1의 왼쪽. **축2·축3로 확장**하는 게 "Mesh 전문가"의 길.

---

## 1. 수학 기초 (얕고 넓게, 필요할 때 깊게)
- **선형대수**: SVD·고유분해·최소자승·정규방정식 (이미 TSDF 융합에서 씀).
- **이산 미분기하(DDG)** ⭐가장 중요: 메시 위의 미적분.
  - 코탄젠트 라플라시안 $L$, 이산 평균곡률 $H\mathbf n=\tfrac12 L\mathbf x$, 가우스곡률(각결손), 이산 외미분(discrete exterior calculus).
  - **교재/코스:** Keenan Crane, *Discrete Differential Geometry: An Applied Introduction* (무료 PDF + CMU 강의영상). **이 한 코스가 메시 전문가의 척추.**
- **수치최적화**: Gauss–Newton·LM·ADMM (당신의 point-to-plane ICP·A1이 여기 속함).

## 2. 메시 기본기
- **자료구조**: half-edge / winged-edge (이웃 순회 O(1)). 직접 half-edge 구현 1회 = 평생 자산.
- **교재**: Botsch, Kobbelt, Pauly, *Polygon Mesh Processing* (메시 처리의 바이블). 챕터: 스무딩, 데시메이션, 리메싱, 파라미터화, 변형.
- **라이브러리로 손 풀기**: [libigl](https://libigl.github.io/) (C++/Python, 튜토리얼이 곧 교과서), Open3D, PyMeshLab, CGAL.

## 3. 표면 재구성 (점/SDF → 메시) — 당신 도메인의 연장
- **Marching Cubes** (이미 씀): 256 케이스, 모호성(topology) 처리, **Marching Tetrahedra**.
- **Dual Contouring** ⭐: **날카로운 모서리 보존**(MC의 약점 해결). 각 셀에서 Hermite 데이터(교차점+법선)로 QEF(quadratic error function) 최소화해 정점 1개 배치:
  $$\mathbf v^\star=\arg\min_{\mathbf v}\sum_i\big(\mathbf n_i\cdot(\mathbf v-\mathbf p_i)\big)^2.$$
  → 당신 TSDF는 이미 **stored-gradient(법선)** 를 가지므로 Dual Contouring에 완벽. **강력 추천 첫 프로젝트.**
- **Poisson Surface Reconstruction** (Kazhdan): 지시함수의 발산을 법선에 맞춤 $\nabla\cdot\nabla\chi=\nabla\cdot\vec V$. 노이즈에 강한 고전.
- **신경 재구성**: DeepSDF, Occupancy Networks, NeuS(→논문 06), **NKSR**(→논문 09). 당신 oriented cloud를 넣어 매끈한 메시 뽑기.

## 4. 기하 처리 (있는 메시 다루기)
- **단순화/데시메이션**: **QEM (Garland–Heckbert Quadric Error Metrics)** ⭐ — 각 정점에 오차 이차형식 $\mathbf Q$ 를 붙여 엣지 collapse 비용 최소화. 필수.
- **리메싱**: isotropic remeshing(균일 삼각형), 삼각형 품질, Delaunay.
- **스무딩**: 라플라시안 스무딩, **평균곡률 흐름** $\partial\mathbf x/\partial t=-H\mathbf n$ (implicit fairing, Desbrun).
- **파라미터화(UV)**: LSCM, ARAP(as-rigid-as-possible) — 텍스처·전개.
- **변형/디폼**: ARAP deformation, cage-based.

## 5. 최전선 — 미분가능 메시 & 신경 표면 ⭐지금 뜨는 곳
- **미분가능 렌더링**: [nvdiffrast](https://nvlabs.github.io/nvdiffrast/), PyTorch3D, Mitsuba 3. 래스터라이즈를 미분해 사진손실로 메시/텍스처 최적화.
- **미분가능 등위면 추출**:
  - **DMTet** (deformable tetrahedra + marching tets, 미분가능),
  - **FlexiCubes** (미분가능 dual MC, 그래디언트로 메시 정점 위치까지 최적화) ⭐ — 당신 TSDF/신경장에서 **학습가능 메시**를 뽑는 최신 도구.
- **3DGS → Mesh**: SuGaR, 2D Gaussian Splatting(2DGS), GOF — 가우시안에서 메시 추출(→논문 03의 3DGS와 연결).
- **차등 부호거리/OREN 계열**: SDF의 고차 성질(곡률·에르미트)로 서브복셀 정밀 추출 — 당신이 프로젝트에서 이미 실험한 방향.

## 6. 벤치마크·데이터셋 (실력 증명)
- 재구성: ShapeNet, Thingi10K, ScanNet, Matterport3D, Replica, **Tanks and Temples**, DTU, Newer College(LiDAR, →논문 02).
- 지표: Chamfer distance, F-score@τ, normal consistency, **precision/recall/completeness** (당신 tsdf_folder_eval이 이미 계산).

## 7. 추천 학습 순서 (12–20주 집중 코스)
1. **주1–3**: Crane DDG 강의 완주 + libigl 튜토리얼(코탄젠트 라플라시안·곡률 손코딩).
2. **주4–5**: half-edge 직접 구현 + QEM 데시메이션 구현(Botsch 책).
3. **주6–7**: **Dual Contouring를 당신 VkLBVH TSDF 위에 구현** (stored-gradient로 QEF) → MC와 모서리 품질 A/B. ← 당신만 할 수 있는 강력한 포트폴리오.
4. **주8–10**: Poisson + NKSR로 oriented cloud → 메시, Chamfer/F-score 비교.
5. **주11–14**: nvdiffrast/PyTorch3D로 미분가능 렌더 + **FlexiCubes**로 학습가능 메시(신경장 or 당신 TSDF).
6. **주15+**: 3DGS→mesh(2DGS/SuGaR), 논문 재현 1편(NeuS or NKSR), 자신의 데이터로 end-to-end.

## 8. 당신 엔진(VkLBVH)을 지렛대로 쓰는 킬러 프로젝트
- **A. Dual Contouring GPU 추출기** — stored-gradient QEF, 모서리 보존. (MC 대비 정량 우위 논문감)
- **B. 미분가능 등위면(FlexiCubes) on GPU TSDF** — TSDF를 학습변수로.
- **C. TSDF-gradient direct odometry** — 논문 10(PIN-SLAM) point-to-implicit을 당신 명시적 TSDF+$\nabla f$ 로 (spec만 있던 걸 완성).
- **D. Mesh 품질 벤치 통합** — tsdf_folder_eval에 Chamfer/F-score/normal-consistency + Dual Contouring/QEM 파이프라인.

## 9. 핵심 참고 (한 줄 링크 대신 이름)
- 📘 Crane, *Discrete Differential Geometry* (필수 1순위)
- 📘 Botsch et al., *Polygon Mesh Processing* (메시 처리 바이블)
- 📘 Hartley & Zisserman, *Multiple View Geometry* / Szeliski, *Computer Vision* (비전 기초)
- 🛠 libigl · Open3D · CGAL · PyMeshLab · nvdiffrast · PyTorch3D · Kaolin
- 🎓 논문: Marching Cubes, Dual Contouring(Ju 2002), QEM(Garland 1997), Poisson(Kazhdan 2006), DMTet, FlexiCubes, 그리고 이 폴더의 `../TSDF/` 11편.

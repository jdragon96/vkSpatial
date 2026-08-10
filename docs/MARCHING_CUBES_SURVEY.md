# Marching Cubes 계보 서베이 — 근본부터 최신까지 + 다중해상도(submap) 적용

Isosurface(등가면) 추출의 표준 알고리즘인 **Marching Cubes(MC)**를 **원조 → 위상 정확성 →
특징 보존/dual → 적응·다중해상도 → 고성능/GPU → 학습기반**의 순서로 정리하고, 마지막에
**서로 다른 voxel 크기(submap)에 MC를 적용할 수 있는가**(이 저장소의 실제 구현 포함)를 다룬다.

- **관련 코드:** `src/Engine/Spatial/AdaptiveVoxelGrid.{h,cpp}`(다중해상도 MC 메시),
  `src/Engine/Spatial/SubmapAdvancedTSDF.h`(2-레벨 detail submap, point-cloud 추출),
  `src/Engine/Spatial/MarchingCubesTables.h`
- **관련 문서:** [ADAPTIVE_VOXEL_GRID_VS_COMPACT_DIRECTIONAL.md](ADAPTIVE_VOXEL_GRID_VS_COMPACT_DIRECTIONAL.md),
  [VARIANCE_ADAPTIVE_VOXEL_GRID.md](VARIANCE_ADAPTIVE_VOXEL_GRID.md), [TSDF_IMPLEMENTATION.md](TSDF_IMPLEMENTATION.md)

---

## 0. 문제 정의

3D 스칼라장 `f: ℝ³ → ℝ`(예: TSDF, 밀도, occupancy)이 격자 위 샘플로 주어질 때, 등가면
`{x : f(x) = τ}`(보통 `τ=0`)를 **삼각형 메시**로 뽑는 것이 목표. MC는 격자를 정육면체(cube/cell)
단위로 훑으며(“march”), 각 큐브의 8개 꼭짓점 부호로 표면 통과 여부를 판정하고, 미리 계산된
**룩업 테이블**로 삼각형 토폴로지를 결정한 뒤, 모서리 위 교점을 **선형 보간**으로 배치한다.
8개 꼭짓점 → `2⁸ = 256` 구성, 회전·반전 대칭으로 **15개(원조) 기본 케이스**로 축약된다.

MC의 세 가지 근본 한계가 이후 모든 후속 연구의 축이 된다: **(A) 위상 모호성**(같은 부호
구성이 여러 표면 연결을 허용 → 구멍/불일치), **(B) 특징 손실**(선형 보간이라 날카로운
모서리·코너를 둥글게 뭉갬), **(C) 균일 격자 가정**(해상도가 다르면 경계에서 균열=crack).

---

## 1. 근본 (Foundations)

| 연도 | 논문 | 핵심 |
|---|---|---|
| 1986 | Wyvill, McPheeters, Wyvill — *Data structure for soft objects* (The Visual Computer) | MC의 사실상 선행 연구: 큐브를 훑어 soft object 등가면을 폴리곤화(당시 “marching”의 원형) |
| **1987** | **Lorensen & Cline — *Marching Cubes: A High Resolution 3D Surface Construction Algorithm*** (SIGGRAPH ’87, Computer Graphics 21(4):163–169) | **원조.** 256→15 케이스 룩업 테이블 + 모서리 선형 보간, divide-and-conquer로 슬라이스 간 연결. CG 역사상 최다 피인용. (특허 1985 출원, 현재 만료) |
| 1991 | Doi & Koide — *Marching Tetrahedra* (IEICE Trans.) | 큐브를 5~6개 사면체로 분할 후 각 사면체에서 등가면 추출. 사면체 안에서는 **위상 모호성이 원천적으로 없음**(당시 MC 특허 회피 목적도) — 대신 삼각형 수가 많고 격자 방향에 편향. |

**남는 문제:** Dürst(1988, *Additional reference to Marching Cubes*, 짧은 정정 노트)가
원조 테이블이 **구멍(holes)**을 낼 수 있음을 지적 → 위상 정확성 연구의 출발점.

---

## 2. 위상 정확성 / 모호성 해결 (Topological correctness)

문제 (A). 큐브의 **면(face) 모호성**(같은 면에서 두 대각 꼭짓점만 안쪽 → 연결/분리 둘 다 가능)과
**내부(interior) 모호성**을 trilinear 보간 기준으로 올바르게 결정한다.

| 연도 | 논문 | 핵심 |
|---|---|---|
| 1991 | **Nielson & Hamann — *The Asymptotic Decider*** (IEEE Visualization ’91) | 면 위 **bilinear** 보간의 쌍곡선 점근선으로 면 모호성을 일관되게 판정. 인접 큐브가 같은 결정을 공유 → 면 구멍 제거. |
| 1994 | Natarajan — *On generating topologically consistent isosurfaces* (The Visual Computer) | 큐브 **내부** 모호성까지 확장(면만으로 부족). |
| 1995 | **Chernyaev — *Marching Cubes 33*** (Tech. Report, CERN CN/95-17) | trilinear 보간이 큐브 안에서 취할 수 있는 **모든 위상 케이스를 최초로 완전 열거 → 33 케이스**. 원조 15개는 23개만 커버. 현대 “위상적으로 올바른 MC”의 표준 케이스표. |
| 2003 | **Lewiner, Lopes, Vieira, Tavares — *Efficient Implementation of MC Cases with Topological Guarantees*** (J. Graphics Tools 8(2)) | Chernyaev MC33를 **실제 구현 가능**하게 완성(모호성 해소·다양체 보장). 오늘날 널리 쓰이는 구현체(예: `MarchingCubes.jl`)의 원형. |
| 2003 | Nielson — *On Marching Cubes* (IEEE TVCG 9(3)) | MC 이론의 종합 정리(케이스·모호성·다양체성). |
| 2019 | Custodio et al. — *An extended triangulation to the MC33 algorithm* (J. Braz. Comp. Soc.) | MC33 구현에 남아있던 잔여 결함을 수정. |

---

## 3. 특징 보존 / Dual 계열 (Sharp features)

문제 (B). 선형 보간 MC는 날카로운 모서리를 뭉갠다. **Hermite 데이터**(교점 + **법선**)를 써서
꼭짓점을 QEF(quadric error function) 최소점에 놓아 코너/에지를 복원하는 흐름.

| 연도 | 논문 | 핵심 |
|---|---|---|
| 2001 | **Kobbelt, Botsch, Schwanecke, Seidel — *Extended Marching Cubes*** (SIGGRAPH ’01) | 법선으로 정의된 접평면들의 교점(QEF 최소)에 **특징 꼭짓점**을 추가 배치 → 날카로운 에지/코너 복원. MC 구조는 유지. |
| 2002 | **Ju, Losasso, Schaefer, Warren — *Dual Contouring of Hermite Data*** (SIGGRAPH ’02, TOG 21(3)) | 큐브 **내부에 꼭짓점 1개**(dual)를 QEF로 두고 부호가 바뀌는 모서리마다 사각형 생성. **octree 적응 격자에서 crack patching 불필요**. 특징 처리 명시적 식별 불필요. (단, 자기교차·비다양체 가능) |
| 2004/05 | **Schaefer & Warren — *Dual Marching Cubes: Primal Contouring of Dual Grids*** (Pacific Graphics ’04 / CGF) | 격자에 **위상적으로 dual인 격자**에서 primal contouring → 얇은 특징을 과세분 없이 재현, **crack-free 적응** 폴리곤화 + 날카로운 특징. |
| 2005 | Ho, Wu, Chen, Chuang, Ouhyoung — *Cubical Marching Squares* (Eurographics ’05) | 문제를 큐브 6면의 **2D marching squares**로 환원 → 적응·특징 보존·위상 해소·**crack 없음**을 동시에. |
| 2007 | Schaefer, Ju, Warren — *Manifold Dual Contouring* (IEEE TVCG 13(3)) | DC의 비다양체 결함을 고쳐 **다양체 보장**. |

> **MC 계열 vs Dual 계열:** MC(+EMC)는 **격자 꼭짓점 부호 → 모서리 교점**(primal)에 삼각형을,
> Dual(DC/DMC)은 **셀 내부 dual 꼭짓점**을 연결한다. Dual은 octree 적응·특징 보존·crack-free에
> 유리하고(→ §4·Q2에서 재등장), MC는 다양체·단순함·병렬화에 유리하다.

---

## 4. 적응 · 다중해상도 & crack 문제 (Q2의 이론적 토대)

문제 (C). 곡률이 큰 곳만 세분해 삼각형 수를 줄이거나(적응), 서로 다른 해상도 블록을 합칠 때
**경계 공유면의 표본/삼각형 불일치 → 균열(crack)**이 생긴다.

| 연도 | 논문 | 핵심 |
|---|---|---|
| 1995 | **Shu, Zhou, Kankanhalli — *Adaptive Marching Cubes*** (The Visual Computer 11(4)) | 곡률 기준으로 큐브를 재귀 세분(2·4·8…). 서로 다른 해상도 큐브가 만나는 면에 **crack 발생** → 공유면 2D 구멍을 메우는 **crack patching** 폴리곤을 명시적으로 생성. |
| 1996 | Shekhar, Fayyad, Yagel, Cornhill — *Octree-based decimation of MC surfaces* (IEEE Vis ’96) | octree로 평탄 영역 병합(decimation), 경계 patching 수반. |
| — | (일반형) **restricted/balanced octree (2:1)** | 인접 셀 해상도 차이를 1단계로 제한 → crack 케이스를 유한한 **transition 패턴**으로 축소. |
| 2009/10 | **Lengyel — Transvoxel Algorithm** (박사논문, UC Davis; transvoxel.org) | 게임의 **LOD 복셀 지형**용 표준 해법. 고해상도 블록의 경계면에 **transition cell**(전이 셀) 전용 테이블을 두어 저해상도 이웃 메시와 **이음매 없이(seamless)** 접합. 512 케이스 룩업. → **“voxel 크기가 다른 두 블록”의 정답 격**. |
| 2025 | **“Resolution Where It Counts: Hash-based GPU-Accelerated 3D Reconstruction via Variance-Adaptive Voxel Grids”** (MrHash, arXiv:2511.21459) — *본 저장소 `AdaptiveVoxelGrid`의 근거 논문* | 분산이 큰(디테일) 영역만 fine, 나머지는 coarse로 저장하는 **variance-adaptive 다중해상도 TSDF** + 다중해상도 MC. |

Dual 계열(§3의 DC/DMC/CMS)은 애초에 **octree 적응에서 crack-free**라, 다중해상도 문제의 또 다른
정공법이다.

---

## 5. 고성능 · 병렬 · GPU (Performance)

| 연도 | 논문 | 핵심 |
|---|---|---|
| 2008 | Dyken, Ziegler, Theobalt, Seidel — *High-speed Marching Cubes using HistoPyramids* (CGF 27(8)) | HistoPyramid로 활성 큐브 압축(stream compaction) → GPU에서 대량 MC. |
| 2015 | **Schroeder, Maynard, Geveci — *Flying Edges*** (IEEE LDAV ’15) | 모서리 단위 **다중 패스**로 완전 독립 처리 → 전처리·검색구조 없이 공유메모리 멀티코어에서 최속. **computational trimming**으로 불필요 계산 제거, 좌표 병합 병목 제거. VTK `vtkFlyingEdges3D`. |

> **본 저장소 연관:** GPU MC(`voxel_tsdf_mc.comp`)와 CPU MC가 **bit-exact**(max-nearest 1.19e-7,
> 정점 수 동일)로 검증됨 — [project 메모리 참조]. GPU는 float atomic이 없는 MoltenVK라 정수
> 고정소수점 리덕션 패턴을 쓴다(엔진 전반의 관례).

---

## 6. 학습 기반 · 미분가능 (Learned / Differentiable — 최신)

신경망 SDF/occupancy에서 메시를 뽑거나, 렌더링 손실로 메시를 **미분가능하게 최적화**하려는 흐름.
고전 MC는 정점 위치가 표본에 국한되고 미분 신호가 약해, 그 두 가지를 학습·완화한다.

| 연도 | 논문 | 핵심 |
|---|---|---|
| 2018 | Liao, Donné, Geiger — *Deep Marching Cubes* (CVPR ’18) | MC를 **end-to-end 미분가능**하게 만들어 메시를 직접 예측(확률적 토폴로지). |
| 2021 | **Chen & Zhang — *Neural Marching Cubes*** (SIGGRAPH Asia, TOG 40(6); arXiv:2106.11272) | 정점 배치·국소 토폴로지를 **학습**해 MC가 뭉개던 특징/얇은 구조 복원. |
| 2021 | **Shen, Gao, Yin, Liu, Fidler — *Deep Marching Tetrahedra (DMTet)*** (NeurIPS ’21) | 변형 가능한 사면체 격자 + SDF의 하이브리드로 **고해상도 3D 생성**(미분가능 MT). |
| 2022 | Chen, Tagliasacchi, Funkhouser, Zhang — *Neural Dual Contouring* (SIGGRAPH, TOG; arXiv:2202.01999) | DC의 장점(특징·octree)을 **학습형**으로, 법선/gradient 입력 없이. |
| 2023 | **Shen et al. — *FlexiCubes: Flexible Isosurface Extraction for Gradient-Based Mesh Optimization*** (SIGGRAPH, TOG) | **Dual Marching Cubes 기반** 추출에 국소 조정 파라미터를 두어 자동미분으로 메시 형상·연결을 최적화. |

---

## 7. Q2 — 서로 다른 voxel 크기(submap)에 Marching Cubes를 적용할 수 있는가?

**요약: 적용 가능하다. 단, “무엇을 추출하느냐”가 관건이다.**

### 7.1 왜 문제가 되는가
표준 MC는 **단일 균일 격자**를 가정한다. 해상도가 다른 두 블록을 각각 **독립적으로** MC하면,
공유 경계면에서 fine 쪽은 여러 개의 짧은 모서리로, coarse 쪽은 하나의 긴 모서리로 표본화되어
**교점 위치와 삼각형 연결이 어긋난다(T-junction) → 균열(crack)/구멍**. 이것이 §4의 고전적
adaptive-MC crack 문제(Shu 1995)다. 핵심은 crack이 **메시 연결성(connectivity)**의 문제라는 점 —
따라서 **연결된 삼각형 메시**를 만들 때만 발생한다.

### 7.2 문헌의 4가지 해법
1. **Crack patching** (Shu 1995): 경계면 2D 구멍을 사후에 메우는 폴리곤 생성. 가장 단순, 다소 임시방편.
2. **Transition cells / Transvoxel** (Lengyel): 고해상도 블록 경계면을 저해상도 이웃과 맞물리는
   **전이 셀 전용 테이블**로 메싱 → 이음매 없음. LOD 복셀 지형의 사실상 표준. **“voxel 크기가
   다른 블록 접합”의 정답**.
3. **Dual 방법** (DC/DMC/CMS, §3): octree/dual 격자에서 **본질적으로 crack-free**(patching 불필요),
   대신 정점 배치 규칙이 다르고 비다양체 주의(→ Manifold DC).
4. **Restricted/balanced octree + 경계 세분화(vertex welding)**: 해상도 차를 2:1로 제한하고 경계
   coarse 셀을 fine 쪽에 맞춰 세분하거나 정점을 공유(weld) → fine/coarse가 정합된 격자에서 동작.

### 7.3 이 저장소의 실제 사례 (핵심)

이 프로젝트에는 “서로 다른 voxel”을 다루는 **두 경로**가 이미 있고, 서로 다른 답을 준다.

**(a) `AdaptiveVoxelGrid::ExtractMesh()` — 진짜 다중해상도 MC(메시).**
2-레벨(fine `h` / coarse `2h`) variance-adaptive TSDF에서 삼각형 메시를 뽑는다. 여기서는 crack이
실제 문제였다: 계획의 단순한 *truncate + collapse* 방식은 경계에서 **1-fine-cell 균열**을 남긴다
(해석적으로 확인됨). 채택한 해법은 위 **(4)번** 계열 —

> **BFS 경계 세분화**: fine 블록에 면-인접(face-adjacent)한 coarse 블록을 8개 fine sub-cell로
> 세분 → fine-pass와 coarse-pass가 **서로소(disjoint) 볼륨**에서 동작하고, 마지막에 `0.25h`
> 격자에서 **한 번만 vertex weld** → **crack-free**.

즉 “다른 voxel 크기에 MC 적용”은 **이미 구현·검증**되어 있다(GPU MC와 bit-exact).

**(b) `SubmapAdvancedTSDF::ExtractPointCloud()` — point cloud(메시 아님).**
base(`baseVoxel`, 전체) + detail(`baseVoxel/2`, dense 블록)을 **precedence dedup**(dense 영역은
detail이 이기고 base는 드롭)으로 합쳐 **oriented point cloud**를 낸다. **삼각형 연결성이 없으므로
위상적 crack 자체가 성립하지 않는다** — 경계에는 국소적인 **점 밀도 불연속(seam)**만 남고, 이는
v1에서 의도적으로 허용됐다(“seamless transitional boundary(MrHash-style)”는 v2로 유예).

### 7.4 결론(실무 지침)
- **메시가 목표**라면: 다른 voxel 크기에 MC는 가능하지만 **경계 처리 필수**. 이 저장소 패턴을
  따르면 **경계 세분화 + 단일 weld**(AdaptiveVoxelGrid 방식)가 간단·견고하고, 더 큰 해상도 격차나
  LOD 스트리밍이면 **Transvoxel 전이 셀**, 특징 보존까지 원하면 **Dual Marching Cubes**가 정공법이다.
- **point cloud가 목표**라면(현재 Submap/AdvancedTSDF 추출): crack은 비적용. precedence dedup으로
  충분하며, 필요 시 경계 seam을 부드럽게 하려면 detail/base 겹침대(overlap band) blending을 v2로.
- 공통 함정(측정됨): detail(half-voxel) 레벨은 타일당 엔트리가 4–8× 많아 **detail 해시가 작으면
  오버플로 → 구멍**(점 수가 single보다 적어짐). 해시 크기를 충분히 크게(기본 `1<<21`) 둘 것.

---

## 8. 타임라인 요약

```
1987  Marching Cubes (Lorensen & Cline) ── 원조 (문제 A/B/C 내재)
        │
   ┌────┼─────────────────────┬───────────────────────┐
 (A)위상               (B)특징/Dual           (C)적응·다중해상도
   │                          │                        │
1991 Asymptotic Decider   2001 Extended MC        1995 Adaptive MC(+crack patch)
1994 Natarajan            2002 Dual Contouring    1996 Octree decimation
1995 MC33 (Chernyaev)     2004 Dual MC            2009 Transvoxel(전이 셀)
2003 Lewiner(구현/다양체) 2005 Cubical M-Squares  2025 Variance-Adaptive(MrHash)
2019 MC33 확장            2007 Manifold DC
        │
   (성능)  2008 GPU HistoPyramid · 2015 Flying Edges
   (학습)  2018 Deep MC · 2021 Neural MC · 2021 DMTet · 2022 Neural DC · 2023 FlexiCubes
```

---

## 9. 참고문헌 (URL)

**근본**
- Lorensen & Cline 1987, *Marching Cubes* — https://dl.acm.org/doi/10.1145/37402.37422
- Lorensen & Johnson 2020, *History of the Marching Cubes Algorithm* — https://ieeexplore.ieee.org/document/9020242/

**위상 정확성**
- Nielson & Hamann 1991, *The Asymptotic Decider* — https://escholarship.org/uc/item/17p025zk ·
  https://en.wikipedia.org/wiki/Asymptotic_decider
- Lewiner et al. 2003, *Efficient Implementation of MC Cases with Topological Guarantees* —
  http://thomas.lewiner.org/pdfs/marching_cubes_jgt.pdf
- Custodio et al. 2019, *An extended triangulation to the MC33 algorithm* —
  https://link.springer.com/article/10.1186/s13173-019-0086-6

**특징 보존 / Dual**
- Kobbelt et al. 2001, *Feature Sensitive Surface Extraction (Extended MC)* —
  https://www.graphics.rwth-aachen.de/media/papers/feature1.pdf
- Ju et al. 2002, *Dual Contouring of Hermite Data* — https://www.cs.rice.edu/~jwarren/papers/dualcontour.pdf ·
  https://dl.acm.org/doi/10.1145/566654.566586
- Schaefer & Warren 2004/05, *Dual Marching Cubes* — https://www.cs.rice.edu/~jwarren/papers/dmc.pdf
- Ho et al. 2005, *Cubical Marching Squares* — https://www.csie.ntu.edu.tw/~cyy/publications/papers/Ho2005CMS.pdf

**적응 · 다중해상도**
- Shu et al. 1995, *Adaptive Marching Cubes* — https://link.springer.com/article/10.1007/BF01901516
- Shekhar et al. 1996, *Octree-based decimation of MC surfaces* — https://dl.acm.org/doi/10.5555/244979.245628
- Lengyel, *Transvoxel Algorithm* — https://transvoxel.org/ · https://transvoxel.org/Lengyel-VoxelTerrain.pdf
- *Variance-Adaptive Voxel Grids (MrHash)* 2025 — https://arxiv.org/pdf/2511.21459

**고성능**
- Schroeder et al. 2015, *Flying Edges* — https://ieeexplore.ieee.org/document/7348069/ ·
  https://vtk.org/doc/nightly/html/classvtkFlyingEdges3D.html

**학습 기반**
- Chen & Zhang 2021, *Neural Marching Cubes* — https://arxiv.org/pdf/2106.11272
- Chen et al. 2022, *Neural Dual Contouring* — https://arxiv.org/pdf/2202.01999
- Shen et al. 2023, *FlexiCubes (Flexible Isosurface Extraction)* — https://dl.acm.org/doi/abs/10.1145/3592430

**참고(개론)**
- Wikipedia, *Marching cubes* — https://en.wikipedia.org/wiki/Marching_cubes

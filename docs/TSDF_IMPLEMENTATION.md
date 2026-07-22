# TSDF: 이 저장소의 구현과 2020년대 이후의 진보

> 복셀에 부호거리를 새겨 표면을 복원하는 방법 — 이 프로젝트에 실제로 구현된 **SimpleTSDF**와 **DirectionalTSDF**를 그림·수식으로 풀고, 2020년 이후 volumetric·neural·Gaussian 계열의 흐름을 지도로 정리한다.
>
> 인터랙티브(그림·수식 스타일링) 버전: <https://claude.ai/code/artifact/c0e7f214-0b01-4ba4-8a9e-faeb5331da7e>
> 학습 경로/전체 계보는 [`TSDF_CURRICULUM.md`](TSDF_CURRICULUM.md) 참고.
>
> 근거: 수식·다이어그램은 실제 셰이더 `src/shader/voxel_tsdf_integrate.comp`, `voxel_tsdf_mc.comp`, `directional_tsdf_integrate.comp`, `directional_tsdf_extract.comp`에서 도출.

---

## Part 0 — 1분 개념: 공간에 부호거리를 새긴다

공간을 작은 정육면체(**voxel**) 격자로 나누고, 각 복셀에 “가장 가까운 표면까지의 **부호 있는 거리**”를 저장한다. 부호 규약:

- 카메라 쪽(표면 앞) = **양수(+)**
- 물체 안쪽(표면 뒤) = **음수(−)**
- 표면 위 = **0**

표면은 곧 “값이 0이 되는 등고면(zero-level isosurface)”이다. **Truncated**(잘린)는 표면에서 `±τ`(truncation, 이 저장소 기본 0.3) 밖은 관심이 없어 잘라낸다는 뜻 — 표면 근처의 얇은 껍질만 저장해 메모리를 아낀다. 여러 관측을 **가중평균**하면 센서 노이즈가 상쇄되고, 이웃 복셀의 부호가 `−`→`+`로 바뀌는 지점을 선형보간해 **복셀보다 정밀한 표면 위치**를 얻는다.

```
관측(depth·점군+법선) ──▶ integrate ──▶ TSDF 볼륨(숨음) ──▶ extract ──▶ 표면(메시·점군)
                          부호거리 누적       복셀 격자          0-등고면
             └────────────── 여러 프레임 반복 통합(incremental) ──────────────┘
```

가운데 “볼륨”은 눈에 안 보이는 중간 상태다. 그래서 이 저장소의 `tsdf_slice_debug`·`tsdf_viewer`는 이 숨은 필드를 슬라이스로 시각화한다.

잘린 부호거리의 1차원 단면 — 표면 근처 `[−τ, +τ]`에서만 선형 램프, 밖은 `−1`/`+1`로 포화:

```
 TSDF
 +1 ┤                    ______   (+ 앞쪽)
    │                   /
  0 ┼──────────────────●──────────▶ 표면까지의 부호거리
    │        _________/    표면(0)
 −1 ┤_______/                       (− 안쪽)
         −τ            +τ
```

DirectionalTSDF는 실제로 `sdf/τ`를 이렇게 [−1, 1]로 정규화해 저장한다.

---

## Part 1 — 이 저장소의 두 구현

두 클래스 모두 GPU 컴퓨트 셰이더로 통합하지만, **필드를 하나로 합치느냐**(Simple) **방향별로 나누느냐**(Directional)가 갈린다. 이 차이가 곧 “각진 모서리 보존”의 핵심이다.

### 1A. SimpleTSDF — KinectFusion 계열 (`voxel_tsdf_integrate.comp`)

고전적인 **투영식(projective) TSDF**. 관측점 `p`와 카메라 `c`가 주어지면 광선 `r = (p−c)/‖p−c‖`를 따라 표면 앞뒤 `±τ` 밴드의 복셀을 훑고, 각 복셀 중심 `v`에 **광선 위 투영 부호거리**를 누적한다.

$$\mathrm{depth} = \lVert p - c \rVert, \qquad \mathrm{sdf}(v) = \mathrm{depth} - (v - c)\cdot r$$

$$|\mathrm{sdf}| > \tau \Rightarrow \text{skip}, \qquad D(v) = \frac{\sum_i d_i}{\sum_i 1} \quad(\textbf{무가중 평균},\ w_i = 1)$$

값은 고정소수점(`×10000`)으로 원자적 누적된다. 핵심 특징 두 가지: **(1) 법선을 쓰지 않고** 카메라 광선만으로 부호를 정하고, **(2) 모든 관측을 무가중 평균**한다.

```
camera c ●──────────── ray r ─────────────▶      │ 표면
                    ┌── −τ ──┬── +τ ──┐          │
                    ○   ○    │   ●   ●            │
                   sdf<0     │  sdf>0             │
                    (뒤)     p   (앞)             │
        sdf(v) = depth − (v−c)·r   → 부호를 '광선 방향'으로만 결정
```

**추출은 Marching Cubes** (`voxel_tsdf_mc.comp`). 이웃한 8개 복셀을 한 큐브로 묶고, 각 코너의 부호(`D<0` = 안)를 8비트 마스크로 만들어 edge/triangle 테이블로 삼각형을 뽑는다. 부호가 바뀌는 에지에서 0-교차 정점을 선형보간한다.

$$\text{cubeIndex bit}_c = (D_c < 0), \qquad t = \frac{D_a}{D_a - D_b}, \qquad \mathrm{vertex} = (1-t)\,a + t\,b$$

(`sumW < ½`인 관측 부족 복셀은 `+τ`로 취급 → 표면 없음.)

**한계 — 왜 코너가 뭉개지나.** 모든 관측이 **한 필드**에 평균된다. 그래서 (a) 0.4mm 간격으로 마주보는 두 벽이 하나로 **합쳐지고**, (b) 90° 코너에서 인접한 두 면의 부호거리가 평균되어 **둥글게 깎인다(chamfer)**. 게다가 무가중·투영식이라 비스듬한 시점에서는 매끈한 면에서도 전역 편향이 생긴다. 이 저장소 실험에서 cube의 **edge max 오차 0.130mm** 스파이크가 바로 이 코너 라운딩의 지문이다.

### 1B. DirectionalTSDF — Splietker & Behnke 2019 (`directional_tsdf_integrate.comp` / `_extract.comp`)

해결책은 단순하면서 강력하다: 복셀마다 필드를 하나가 아니라 **6개 방향 레이어**(±X, ±Y, ±Z)로 나눠 저장하고, 관측점의 **법선**이 향하는 축의 레이어에만 쓴다. 서로 다른 방향을 향한 표면은 애초에 **다른 레이어**에 들어가므로 섞일 수가 없다.

$$a_i = |n \cdot \mathrm{axis}_i|, \quad r_i = a_i^{\,p}\ (p=\texttt{dirExponent}=4), \quad \mathrm{rel}_i = \frac{r_i}{\max_j r_j}$$

$$\text{(rel} \ge 0.05\text{ 인 상위 } K=\texttt{maxDirections}\text{ 개 레이어에 기여)}$$

$$\mathrm{viewFactor} = \max(0,\ n\cdot(-r)), \qquad w = \mathrm{viewFactor} \cdot \mathrm{rel}_i$$

$$\mathrm{value} = \mathrm{clamp}(\mathrm{sdf}/\tau,\ -1,\ 1), \qquad D_{\text{layer}} = \frac{\sum w \cdot \mathrm{value}}{\sum w}$$

즉 SimpleTSDF의 무가중·단일필드와 정반대로, DirectionalTSDF는 **법선 기반 방향 분리 + 시야각 가중 + 정규화**를 쓴다. `K=1`이면 지배축 하나만(고전 동작), `K≥2`면 비스듬한 법선이 여러 레이어에 나눠 기여한다.

```
  SimpleTSDF (단일 필드)              DirectionalTSDF (방향 레이어)
     │                                    │ (+X layer)
     │   __                               │
     │  /    ← 둥글게 깎임                 │
     └─────                        ●───────┘  ← 코너 보존
   두 면 부호거리 평균 → chamfer     (+Y layer)   면A→+Y, 면B→+X : 다른 레이어라 평균 안 됨
```

**추출은 레이어별 지향 점군** (`directional_tsdf_extract.comp`). 각 방향 레이어 안에서만 이웃(+축)과 부호가 바뀌는 0-교차를 찾아 점을 만들고, **같은 레이어의 중앙차분 그래디언트**로 법선을 계산한다. 한 복셀이 +X·+Y 레이어 둘 다에서 점을 내면 코너가 두 방향 점으로 남는다. 이후 후보 점들을 병합해 최종 점군을 만든다.

대가는 **복셀당 최대 6배 저장**과 방향 판정·병합 비용이며, residency(스트리밍) 백엔드로 표면 근처 그룹만 상주시켜 완화한다.

### 1C. 나란히 비교

| 항목 | SimpleTSDF | DirectionalTSDF |
|---|---|---|
| 필드 구조 | 단일 필드 1개 | **6 방향 레이어(±X±Y±Z)** |
| 부호 결정 | 카메라 광선(투영식) | 광선 + 법선 방향 |
| 가중치 | 무가중 (w=1) | **viewFactor × relWeight** |
| 저장 값 | 원거리 평균 `D=ΣdW/ΣW` | 정규화 `clamp(sdf/τ,−1,1)` |
| 추출 | Marching Cubes 메시 | 레이어별 지향 점군 + 병합 |
| 날카로운 코너/능선 | 평균 → 둥글게 뭉갬 | **레이어 분리 → 보존** |
| 마주보는 얇은 벽 | 하나로 합침 | **분리 유지** |
| 메모리 | 1× (기준) | ≤ 6× (residency로 완화) |
| 기반 논문 | Curless&Levoy '96 / KinectFusion '11 | Splietker&Behnke '19 |

**이 저장소 실험 결과** (`tsdf_feature_compare`, cube/cylinder, voxel 0.1mm) — GT 대비 추출점 오차(mm):

| 영역 | Simple mean / max | Directional mean / max |
|---|---|---|
| cube **edge** | 0.050 / **0.130** | 0.027 / 0.071 |
| cube flat | 0.046 / 0.070 | 0.011 / 0.036 |
| cyl **edge** | 0.061 / 0.203 | 0.030 / 0.150 |
| cyl curved | 0.049 / 1.50* | 0.012 / 0.060 |

DirectionalTSDF가 전 영역 우위 — 코너 라운딩은 **edge의 max 스파이크**로 드러난다. Simple이 매끈한 면에서도 밀리는 건 무가중·투영식(법선 미사용)이라는 **알고리즘** 차이 때문이지 코너 효과와는 별개다. (*cyl curved max 1.5 = Simple MC의 축 근처 스트레이 정점 아티팩트.)

---

## Part 2 — 2020년 이후의 진보: TSDF에서 갈라진 여섯 갈래

DirectionalTSDF는 2019년의 “방향성을 더한 고전 TSDF”다. 그 이후 재구성 연구는 크게 여섯 방향으로 뻗었다.

**A. 저장·확장성 (2013–2025).** **Voxel Hashing**(Nießner '13)로 표면 근처 블록만 희소 저장하는 흐름. **VDBFusion**(Vizzo '22)은 OpenVDB로 LiDAR를 CPU 단일코어 20fps 통합. 적응 해상도(**Resolution Where It Counts**, TOG '25)와 **DB-TSDF**('25, directional bitmask)는 DirectionalTSDF의 효율형 후계.

**B. 기하 정확도 (2017–2022).** **Voxblox**('17)는 TSDF→ESDF로 경로계획 지원. **Voxfield**(Pan '22)는 투영식 대신 **비투영(non-projective) + 법선 추정**으로 정확도를 끌어올림 — 이 저장소 SimpleTSDF의 “매끈 영역 편향”을 정면으로 다루는 방향.

**C. 뉴럴 암시적 SDF (2019–2023).** 이산 복셀 대신 신경망이 좌표→SDF를 학습. **DeepSDF**('19) → **NeuS / VolSDF**('21, 볼륨렌더링으로 멀티뷰 이미지에서 표면) → **Instant-NGP**('22, 해시 인코딩으로 실시간급) → **Neuralangelo**('23, 고디테일).

**D. 뉴럴 SLAM 매핑 (2021–2023).** TSDF 매핑의 실시간 후계. **iMAP**('21) → **NICE-SLAM**('22) → **Vox-Fusion**('22) → **Co-SLAM · Point-SLAM · ESLAM**('23). 공통 뼈대는 feature grid를 얕은 디코더로 TSDF/색으로 변환.

**E. Gaussian Splatting 표면화 (2023–2025).** **3DGS**('23)는 렌더는 좋지만 표면이 지저분 → **2DGS**(SIGGRAPH '24), **GOF · SuGaR**로 정확한 메시 추출. **Gaussian-SLAM · SplaTAM**('23/'24)은 3DGS를 맵으로.

**F. 파운데이션·데이터구동 (2024–2025).** **DUSt3R**('24) / **MASt3R** / **MASt3R-SLAM**('25)은 이미지에서 직접 dense 3D를 예측(대응·보정 불필요) — 재구성의 프론트엔드를 재편.

| 대표작 | 연도 | 계열 | 고전 TSDF 대비 핵심 |
|---|---|---|---|
| VDBFusion | 2022 | A 저장 | OpenVDB 희소 · CPU 실시간 LiDAR |
| DB-TSDF | 2025 | A·B 방향 | directional bitmask로 방향 저장 효율화 |
| Voxfield | 2022 | B 정확 | 비투영 + 법선 → 편향 감소 |
| NeuS / VolSDF | 2021 | C 뉴럴 | 이미지만으로 연속 SDF 학습 |
| Neuralangelo | 2023 | C 뉴럴 | 해시 인코딩 + 고디테일 표면 |
| Co-/Point-/ESLAM | 2023 | D SLAM | 학습 디코더가 grid→TSDF/색 |
| 2DGS | 2024 | E 가우시안 | 정확한 표면 메시 + 고품질 렌더 |
| MASt3R-SLAM | 2025 | F 파운데이션 | 이미지→dense 3D 직접 예측 |

### 이 저장소는 어디에 있고, 다음은

- **현재 위치:** DirectionalTSDF = 방향성으로 sharp/thin 구조를 지킨 **고전 계열**(2019). 증분 통합·기하 정확·검증 쉬움이 강점.
- **실용 개선:** SimpleTSDF에 **Voxfield식 비투영 + 법선 가중**을 넣으면 매끈 영역 편향(실험의 confound)이 줄어 “edge에서만 차이나는” 깨끗한 비교가 가능. 방향 저장은 **DB-TSDF식 bitmask**로 6× 메모리를 압축.
- **프론티어 연결:** 같은 GT로 **NeuS·2DGS**와 정량 비교(정확도·완전성·메모리·시간)하는 벤치마크 — [`TSDF_CURRICULUM.md`](TSDF_CURRICULUM.md) L4/L5와 직접 연결.

---

## 참고문헌

- Curless & Levoy, *A Volumetric Method for Building Complex Models from Range Images*, SIGGRAPH 1996. (TSDF 원전)
- Lorensen & Cline, *Marching Cubes*, SIGGRAPH 1987.
- Newcombe et al., *KinectFusion*, ISMAR 2011.
- Splietker & Behnke, *Directional TSDF: Modeling Surface Orientation for Coherent Meshes*, 2019 — [arXiv:1908.05146](https://arxiv.org/pdf/1908.05146). **(이 저장소 DirectionalTSDF의 기반)**
- Splietker & Behnke, *Rendering the Directional TSDF … Point-To-Plane Scale ICP*, 2022/2023 — [arXiv:2301.12796](https://arxiv.org/pdf/2301.12796).
- Vizzo et al., *VDBFusion*, Sensors 2022.
- Pan et al., *Voxfield: Non-Projective Signed Distance Fields*, IROS 2022.
- Wang et al., *NeuS*, NeurIPS 2021 · Yariv et al., *VolSDF*, 2021.
- Müller et al., *Instant-NGP*, SIGGRAPH 2022 · Li et al., *Neuralangelo*, CVPR 2023.
- Zhu et al., *NICE-SLAM*, CVPR 2022 · *Co-SLAM* / *Point-SLAM* / *ESLAM*, 2023.
- Huang et al., *2D Gaussian Splatting*, SIGGRAPH 2024.
- DB-TSDF, 2025 — [arXiv:2509.20081](https://arxiv.org/html/2509.20081v1).

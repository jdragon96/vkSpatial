# DB-TSDF 정리 & 현재 CompactDirectionalTSDF 대비 고정밀 형상 복원 이점

> **한 줄 요약:** DB-TSDF의 진짜 무기는 "더 정밀한 값"이 아니라 **① 해상도를 키워도 프레임 비용이 일정(resolution-invariant)** 해서 *작은 voxel을 실시간으로 쓸 수 있게* 해주는 것, **② 투영거리 대신 공간(준-유클리드)거리** 라 뷰 편향이 적은 것, **③ 명시적 occlusion shadow + 관측 확정 카운터** 로 표면이 깨끗하고 완전한 것 — 이 세 가지다.
> 대신 대가로 **dense 메모리가 해상도³로 폭증** 하고, 현재 구현이 가진 **표면-방향 레이어링(얇은 벽/모서리 보존)** 을 잃는다.

> 논문: **DB-TSDF: Directional Bitmask-based Truncated Signed Distance Fields for Efficient Volumetric Mapping** — J. E. Maese, L. Merino, F. Caballero (Universidad Pablo de Olavide). arXiv:2509.20081v1 (2025-09-24). CPU-only, ROS2, code: `github.com/robotics-upo/DB-TSDF`.
> 수식은 GitHub/마크다운 뷰어에서 렌더됩니다.

---

## Part A — DB-TSDF 논문 정리

### A.1 무엇을 푸는가

**CPU만으로** 고해상도 volumetric mapping을 실시간으로. 대부분의 최신 TSDF/ESDF 파이프라인이 GPU 가속에 의존하거나 해상도가 올라갈수록 비용이 급증하는데, DB-TSDF는 **비트마스크 거리 인코딩 + 방향성 커널** 로 **프레임당 비용을 맵 해상도와 무관하게 일정** 하게 만든다. LiDAR 매핑(D-LIO의 Fast-TDF 백엔드)을 signed 표현으로 확장한 것.

**4대 기여**
1. 방향성 커널 + 비트마스크 거리 인코딩 기반 TSDF 통합 — 전부 CPU 정수 연산
2. **스캔당 계산비용이 전역 grid 크기와 독립** (상수 시간)
3. 고해상도용 8 byte/voxel 저메모리 signed voxel 구조 + ROS2 통합
4. 정확도·속도·메모리 정량 평가

### A.2 Voxel 표현 — 8 byte

각 voxel은 dense, axis-aligned, 고정 해상도 grid에 저장. 필드 3개:

| 필드 | 크기 | 의미 |
|---|---|---|
| **Distance mask** | 32-bit uint | 커널 반경 내 최근접 점유셀까지의 **절단 $L_1$ 거리** 를 voxel 셀 단위로 이산화. 거리는 **켜진 비트 수** 로 암묵 표현. 업데이트는 **bitwise AND** → 상수 시간 |
| **Sign flag** | 1 bit | occupied(0) / free(1). 방향성 커널 업데이트로 결정 (표면 뒤 shadow 반영) |
| **Hit counter** | 8-bit uint | occupied를 지지하는 관측 횟수. 임계값 $T$ 초과 시 occupied 확정 |

$$
S_{\text{voxel}} = 8\,\text{B}, \qquad M_{\text{grid}} = N_x\cdot N_y\cdot N_z \cdot S_{\text{voxel}}
$$

초기화: 모든 distance mask = 최댓값(전 비트 1), sign = free, hit counter = 0. **커널은 거리를 줄이기만(AND는 비트를 지우기만) 하고 새 증거가 있을 때만 점유를 갱신** — 단조(monotonic) 갱신이라 순서 무관·병렬 안전.

### A.3 Directional Kernels — 핵심

빔 방향에 따라 다른 **미리 계산된 커널** 을 적용해, LiDAR 빔의 **비등방 footprint** 와 그 뒤의 **occlusion(shadow)** 를 모델링한다. (원래 D-LIO의 등방 커널을 방향성으로 확장.)

- 센서 주변 방향 공간을 $B_{az}\times B_{el} = 40\times40 = 1600$ 각도 bin으로 이산화. 각 bin = 대표 단위벡터 + **$21\times21\times21$ 큐빅 커널**.
- 점 $\mathbf p=(x,y,z)$(센서 좌표계)의 bin 인덱스:
$$
b_{az}=\Big\lfloor \tfrac{\operatorname{atan2}(y,x)}{2\pi}B_{az}\Big\rfloor,\quad
b_{el}=\Big\lfloor \tfrac{\arcsin(z/\lVert p\rVert)+\pi/2}{\pi}B_{el}\Big\rfloor
$$
- 커널 안 각 voxel이 커널 중심까지의 절단 $L_1$ 거리를 32-bit 마스크로 저장:
$$
m(x,y,z)=\begin{cases}0,& r=0\\ (2^{32}-1)\gg(32-\lceil r\rceil),& r>0\end{cases},\quad r=\sqrt{x^2+y^2+z^2}
$$
- **Occlusion(shadow) 영역** = 접촉점 둘레의 유클리드 구에서 **앞쪽 절반을 버린 truncated $L_2$ 반구(hemisphere)**, 반경 $r_s$ voxel, bin 대표 방향 정렬. ($L_1$ 구는 팔면체가 되어 이산화 아티팩트 → $L_2$ 채택.)
- **반구 모델 선택 이유:** 반구의 평평한 면이 레이에 수직 → 접촉셀의 이웃 voxel도 hit이 쌓여 **적은 프레임으로도 빨리 occupied 확정** + 디테일이 선명. (원뿔 shadow는 중심 voxel에만 hit → 확정이 느림.)

### A.4 Map Update — 전부 정수 연산

각 LiDAR 리턴마다 (Fig.4 파이프라인):
1. 추정 포즈로 전역 좌표계 변환(yaw-only 또는 SE(3) 모션 보정)
2. $21^3$ 이웃이 맵 경계를 벗어나면 스킵
3. azimuth–elevation로 bin 선택 → 해당 커널 로드
4. **distance mask 갱신** (AND는 비트만 지우므로 저장값이 절대 증가 안 함):
$$
m_{\text{grid}}\leftarrow\begin{cases}m_{\text{grid}}\ \&\ m_{\text{kernel}},& \text{바뀌면}\\ m_{\text{grid}},&\text{아니면(쓰기 스킵)}\end{cases}
$$
5. **occupancy** — shadow 영역 voxel만 saturating 카운터 증가:
$$
h^{\text{new}}=\min(h^{\text{old}}+\mathbf 1_{\text{shadow}},\,H_{\max})
$$
6. **sign** — 카운터가 임계값 $T$ 를 넘는 순간에만 occupied로 뒤집음:
$$
s^{\text{new}}=\begin{cases}0(\text{occupied}),& h^{\text{old}}<T \text{ and } h^{\text{new}}\ge T\\ s^{\text{old}},&\text{else}\end{cases}
$$

런타임에 점별 기하 계산이 없다(커널이 미리 계산됨) → 업데이트 = 정수 몇 개 + 메모리 쓰기. OpenMP 병렬. **프레임 비용:**
$$
T_{\text{update}}\approx N_p\cdot K^3\cdot C_{\text{op}}\qquad(K=21\ \text{고정})
$$
$K$ 가 맵 해상도와 무관하게 고정 → **$T_{\text{update}}$ 는 voxel 크기를 바꿔도 일정**. 메시는 Marching Cubes(iso = 0.0)로 추출.

### A.5 평가 결과 (i7-13620H 16스레드, CPU only)

**정확도** (Acc/Comp/C-L1 = cm, 낮을수록 좋음 / Recall·F = %, 높을수록 좋음)

| 데이터셋 | DB-TSDF | 비고 |
|---|---|---|
| **Mai City** (합성 도심, 64-beam) | F-score **96.6 (SOTA 1위)**, C-L1 3.1, Comp 4.6(2위), Recall 93.6 | 모서리·엣지 뚜렷이 보존 |
| **Newer College** (핸드헬드, 실측 노이즈) | Comp 10.7(2위), Recall 92.4(2위), C-L1 9.9(3위) | 실환경 노이즈에도 강건 |

> 참고로 정확도 1위는 종종 **SHINE-Mapping(GPU 필요)**. DB-TSDF는 CPU만으로 그에 필적.

**효율 (핵심 그림)**
- 프레임당 **~150 ms가 voxel 0.3 m → 0.05 m 전 구간에서 거의 일정** (0.05 m no-DS에서 $154.8\pm16.9$ ms). → **해상도 불변**.
- DS=2(20 cm 다운샘플) → 91.1 ms.
- 대조: VDB-GPDF는 0.1 m에서 **500 ms 초과**, 해상도 낮출수록 급증(Fig.7). voxel 크기는 **메모리에만** 영향, 런타임엔 영향 거의 없음.

---

## Part B — 현재 CompactDirectionalTSDF 요약

이 저장소의 구현(`src/TSDF/Backends/CompactDirectionalTSDF.{h,cpp}`, `src/TSDF/Backends/kernel_compact_directional_{integrate,extract}.comp.glsl`)은 **Splietker & Behnke(2019) DirectionalTSDF** 계열의 **GPU(Vulkan) 희소 해시** 버전이다. 자세한 대조는 [`COMPACT_VS_DIRECTIONAL_TSDF.md`](COMPACT_VS_DIRECTIONAL_TSDF.md), 적분 수식은 [`DIRECTIONAL_TSDF_INTEGRATION.md`](DIRECTIONAL_TSDF_INTEGRATION.md).

- **저장:** open-addressing 해시(`wangHash` + linear probing). 엔트리 `DirEntry{ key; weightedDistanceSum(int); weightSum(uint); pad }` = **16 B**. 관측된 (voxel, 방향) 칸만 저장(희소). 키는 이동식 $512^3$ 창의 local 좌표(축 9-bit) + 방향 3-bit.
- **값:** 연속 **가중 평균 투영 부호거리**
$$
\Psi(v,d)=\frac{\sum \psi\,\varphi\,\rho}{\sum \varphi\,\rho},\quad
\psi=\operatorname{clamp}\big((p{-}x_v)\cdot r/\tau,-1,1\big),\ \varphi=\max(0,\,n\cdot(-r))
$$
분자 $\Sigma\psi\varphi\rho$(`weightedDistanceSum`)와 분모 $\Sigma\varphi\rho$(`weightSum`)를 각각 `atomicAdd` 로 누적(fixed-point $\times10000$) → lock-free.
- **"directional"의 의미:** 표면 **법선의 6개 부호축**($\pm X,\pm Y,\pm Z$) 중 정렬 강한 **최대 3개 방향 레이어** 를 한 voxel에 분리 저장(`selectDirections`, 5% 미만 드롭). → 서로 다른 방향의 표면이 한 voxel에서 섞이지 않음.
- **추출:** `kernel_compact_directional_extract.comp.glsl` — 같은 방향 레이어 안에서 +축 이웃과의 zero-crossing을 **서브복셀 보간**, 중앙차분 gradient로 법선 추정 → **oriented point cloud**(메시 아님). 선택적 `MergeCandidates`.

---

## Part C — 먼저 짚을 함정: "directional"이 서로 다른 것을 뜻한다

| | **현재 CompactDirectionalTSDF** | **DB-TSDF** |
|---|---|---|
| "directional"의 뜻 | **표면 법선 방향** 별 레이어(6축) | **빔(관측 광선) 방향** 별 커널(1600 bin) |
| 목적 | 반대/수직 표면의 **상쇄 방지**(anti-aliasing) → 얇은 구조·모서리 보존 | 빔 **비등방 footprint + occlusion** 모델링 → 깨끗·완전한 표면 |
| voxel당 표면 수 | 최대 **3개** (방향 분리) | **1개** (단일 sign/거리) |

→ 둘은 경쟁 기술이라기보다 **서로 다른 축을 최적화**한다. "DB-TSDF가 더 정밀하다"는 단정은 틀리다. 항목별로 봐야 한다.

---

## Part D — 고정밀 형상 복원 관점: 이점 정리

### D.1 DB-TSDF가 유리한 점

**① 해상도 불변 비용 → 작은 voxel을 실시간으로 (가장 큰 레버)**
고정밀의 본질은 결국 **작은 voxel**이다. 그런데 현재 구현의 통합은 레이 마칭이라
$$
\text{steps}=\big\lceil \tau/\text{voxelSize}\big\rceil+1
$$
즉 **voxel을 절반으로 줄이면 포인트당 스텝(=비용)이 2배**. 해상도↑ = 런타임↑.
DB-TSDF는 $T_{\text{update}}\approx N_p K^3 C_{\text{op}}$ 로 $K$ 고정 → **voxel을 줄여도 프레임 비용이 그대로**(메모리만 지불). "실시간을 유지한 채 해상도를 올린다"를 구조적으로 가능케 한다.

**② 투영거리 → 공간(준-유클리드)거리: 뷰 편향 감소**
현재 $\psi$ 는 **레이 방향 투영** 부호거리(KinectFusion식)라, 표면을 비스듬히(grazing) 볼수록 등가면이 실제 표면에서 밀린다. DB-TSDF의 distance mask는 **커널 반경 내 최근접 점유셀까지의 공간 거리**($L_1$, shadow는 $L_2$) → 시야 방향 의존이 작아 **다중뷰 일관성·등가면 위치 정확도** 가 좋다. (논문 Mai City에서 엣지·코너 선명 보존.)

**③ 명시적 occlusion + free-space + 확정 카운터: 깨끗하고 완전한 표면**
DB-TSDF는 표면 뒤 hemispherical shadow에만 hit을 쌓고 $T$ 회 넘으면 occupied 확정, 레이 경로 나머지는 free로 명시. → **단발 오검출 거부(노이즈 강건)**, floating/spurious 표면 억제, 반구 면이 이웃에 증거를 퍼뜨려 **적은 프레임으로 completeness↑**. 현재 구현엔 명시적 free-space carving이 없고(truncation 밴드 밖은 미정), `MIN_WEIGHT` 게이트가 부분적으로만 그 역할을 한다.

**④ 비등방 빔 footprint → 표면 연속성/구멍 메움** (완전성·recall 개선, ③과 같은 메커니즘).

### D.2 현재 CompactDirectionalTSDF가 오히려 유리한 점 (정직하게)

**① 표면-방향 레이어링 = 진짜 얇은구조/모서리 보존**
한 voxel에 방향별 최대 3개 TSDF를 분리 저장 → **voxel보다 얇은 두 표면(얇은 벽)이나 반대 법선 표면이 상쇄되지 않는다.** DB-TSDF는 voxel당 sign·거리가 **하나** 뿐이라, voxel보다 얇은 이중 표면에서 병합/소실 위험이 있다. (DB-TSDF도 커널+bitmask로 엣지를 보존한다고 주장하나, **방향 분리 자체는 없다**.) → **같은 voxel 크기라면 얇은 구조 보존은 현재 구현이 우위.**

**② 연속 값 → 더 미세한 서브복셀 위치 (같은 voxel 크기 기준)**
현재 $\Psi\in[-1,1]$ 는 연속(fixed-point $1/10000$)이라 extract의 zero-crossing 보간이 매끄럽다. DB-TSDF 거리는 **정수 셀 단위 양자화** → MC 보간 해상도가 대략 1셀. 즉 **동일 voxel 크기에서 서브복셀 정밀도는 현재가 더 미세**. (DB-TSDF는 "voxel을 더 줄여서" 이를 상쇄한다.)

**③ 희소 메모리 vs dense 큐빅 증가**
현재는 표면 칸만(≈표면적에 비례). DB-TSDF는 dense $N_x N_y N_z\times8$ B → **해상도를 2배 높이면 메모리 8배**. 고정밀일수록 DB-TSDF 메모리가 급증한다. (반면 현재는 $512^3$ 창 제한 → 큰 장면은 타일링 필요.)

**④ 부드러운 확률적 가중** vs DB-TSDF의 이진 hit-counter occupancy.

### D.3 종합 대조표 (고정밀 관점)

| 항목 | 현재 CompactDirectionalTSDF | DB-TSDF | 고정밀에 누가 유리? |
|---|---|---|---|
| 해상도↑ 시 프레임 비용 | $\propto\lceil\tau/\text{vox}\rceil$ 증가 | **일정(불변)** | **DB-TSDF** |
| 거리 종류 | 투영(레이) 부호거리 | 공간(준-유클리드) 거리 | **DB-TSDF** (뷰 편향↓) |
| Occlusion/free-space | 명시 없음(밴드만) | 명시 shadow + 확정 카운터 | **DB-TSDF** (클린·완전) |
| 노이즈 강건성 | 연속 가중 + `MIN_WEIGHT` | $T$ 회 확정 → 단발 거부 | DB-TSDF 약우위 |
| 얇은 구조/모서리 | **방향 3-레이어 분리** | 단일 sign(방향 분리 없음) | **현재 구현** |
| 서브복셀 정밀(동일 voxel) | 연속 값 | 정수 셀 양자화 | **현재 구현** |
| 메모리 | 희소(표면 비례), 16 B/entry | dense, 8 B/voxel, 해상도³ | 현재(고해상도일수록) |
| 장면 크기 | $512^3$ 이동창(타일링) | dense grid 경계 | 상황별 |
| 하드웨어 | GPU(Vulkan) | CPU(정수/OpenMP) | 이 repo는 GPU |

### D.4 적용 시 고려사항 (이 저장소 맥락)

- DB-TSDF의 방향 bin은 **센서 원점에서 방사되는 LiDAR** 를 가정(azimuth/elevation). 이 repo의 **object-centric point+normal 클라우드**(chair 스캔)에 옮기려면 bin 선택을 per-point 법선 또는 view-ray 기준으로 재정의해야 한다. 참고로 현재 구현도 이미 view-angle 가중($\varphi=\max(0,n\cdot(-r))$)과 법선 기반 방향 선택을 갖고 있어 **비등방성의 일부는 이미 모델링** 되어 있다.
- 여기 하드웨어는 GPU(Vulkan) 기반이므로 DB-TSDF의 "CPU-only"는 이 repo에선 직접 이점이 아니다. **가치는 알고리즘 아이디어**(비트마스크 거리 인코딩, 커널 프리컴퓨트, 해상도 불변 커널 비용, occlusion shadow)에 있으며 이는 GPU로도 이식 가능하다.

---

## Part E — 결론 & 하이브리드 제안

**결론.** "고정밀 형상 복원"에 DB-TSDF가 주는 실질 이점은 **(1) 해상도 불변 비용으로 작은 voxel을 실시간화**, **(2) 투영→공간 거리로 뷰 편향 감소**, **(3) 명시적 occlusion으로 깨끗·완전한 표면** 이다. 그 대가로 **dense 메모리가 해상도³ 로 폭증** 하고, 현재 구현의 강점인 **표면-방향 레이어링(얇은 벽/모서리 보존)과 연속-값 서브복셀 정밀도(동일 voxel 기준)를 잃는다.**

따라서 단순 대체가 아니라 **하이브리드**가 이상적이다:

> 현재의 **방향 레이어 분리(anti-aliasing)** + **연속 가중 값**
> ⊕ DB-TSDF의 **occlusion shadow 커널 + hit-counter 확정** (노이즈·free-space)
> ⊕ **해상도 불변 커널 비용** 아이디어 (작은 voxel 실시간화)

즉 DB-TSDF의 커널/occlusion/상수시간 장점을 이 repo의 GPU 희소-해시·방향 레이어 위에 얹으면, "작은 voxel + 깨끗한 표면 + 얇은 구조 보존"을 동시에 노릴 수 있다.

---

## 참조
- 논문 PDF: [`DB-TSDF.pdf`](DB-TSDF.pdf) (arXiv:2509.20081v1)
- 현재 구현 대조: [`COMPACT_VS_DIRECTIONAL_TSDF.md`](COMPACT_VS_DIRECTIONAL_TSDF.md), 적분 수식: [`DIRECTIONAL_TSDF_INTEGRATION.md`](DIRECTIONAL_TSDF_INTEGRATION.md)
- 구현: `src/TSDF/Backends/CompactDirectionalTSDF.{h,cpp}`, 셰이더: `src/TSDF/Backends/kernel_compact_directional_{integrate,extract}.comp.glsl`

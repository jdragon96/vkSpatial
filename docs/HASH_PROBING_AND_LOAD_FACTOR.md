# 복셀 개수와 선형탐사(linear probing) 길이 — 논문 정리

> "복셀이 몇 개일 때 linear probe를 몇 번 하는가"에 대한 문헌 조사.
>
> 대상 코드: [`AdvancedTSDF.cpp`](../src/TSDF/Backends/AdvancedTSDF.cpp) + [`AdvancedTSDF.integrate.comp.glsl`](../src/TSDF/Backends/AdvancedTSDF.integrate.comp.glsl) + [`LinearProbe.glsl`](../src/TSDF/Memory/Hash/LinearProbe.glsl)
> — open addressing + linear probing(`MAX_PROBE = 128`), load factor 50%에서 리해시.
>
> 관련 문서: [`ADVANCED_TSDF.md`](ADVANCED_TSDF.md) · [`MRHASH_VS_DIRECTIONAL_TSDF.md`](MRHASH_VS_DIRECTIONAL_TSDF.md)

**목차**
1. [요약](#1-요약)
2. [고전 이론 — 평균 탐사 횟수 = f(α)](#2-고전-이론--평균-탐사-횟수--fα)
3. [GPU 해시테이블 — 실측 probe 곡선](#3-gpu-해시테이블--실측-probe-곡선)
4. [복셀/SLAM 시스템 — 실제로 쓰는 파라미터](#4-복셀slam-시스템--실제로-쓰는-파라미터)
5. [우리 구현에 대한 함의](#5-우리-구현에-대한-함의)
6. [참고문헌](#6-참고문헌)

---

## 1. 요약

**"복셀 개수 n의 함수로 탐사 횟수를 준" 논문은 사실상 없다.** 해시 이론은 전부
**load factor α = n/m** (n = 원소 수, m = 테이블 크기)로 기술하며, 복셀 개수는 오직 α를 통해서만
들어온다. n이 직접 등장하는 결과는 딱 하나 — **최장 탐사 길이가 log n에 비례**한다는
Pittel(1987)이고, `MAX_PROBE` 같은 상한을 정할 때 실제로 봐야 하는 식이 바로 이것이다.

| 질문 | 답 | 근거 |
|---|---|---|
| 평균 몇 번 탐사하나 | 삽입 `½(1 + 1/(1−α)²)`, 조회 `½(1 + 1/(1−α))` | Knuth 1963 (§2.1) |
| 최악은 몇 번인가 | `ln m / (α − 1 − ln α)` | Pittel 1987 (§2.2) |
| 복셀이 늘면 나빠지나 | α가 그대로면 **로그로만** 나빠진다 | 같은 식 |
| α는 어디까지 허용되나 | 우리 설정에서 **0.5가 상한** (0.6이면 `MAX_PROBE` 초과) | §2.2 표 |
| 더 높은 α가 필요하면 | 선형탐사를 버리고 bucketed 구조로 | Awad 2021 (§3.1) |
| 넘치면 어떻게 되나 | 복셀이 조용히 드롭 — 실측 7.4%까지 | Mesh-LOAM (§4.3) |

---

## 2. 고전 이론 — 평균 탐사 횟수 = f(α)

### 2.1 Knuth (1963) — 선형탐사의 최초 분석

랜덤 해시 함수 가정 하에서 (Knuth, *TAOCP* Vol.3 §6.4, Ex. 6.4.27):

| | 기대 탐사 횟수 |
|---|---|
| **탐색 성공** (이미 있는 키 조회) | `C(α) ≈ ½ (1 + 1/(1−α))` |
| **탐색 실패 = 삽입 비용** | `C'(α) ≈ ½ (1 + 1/(1−α)²)` |

비교용 uniform probing(클러스터링 없음)은 각각 `(1/α)·ln(1/(1−α))`, `1/(1−α)`. 선형탐사가 더
나쁜 이유가 **primary clustering** — 연속 점유 블록이 서로 잡아먹으며 자라는 현상이고, 이것도
Knuth가 1963년에 발견했다.

| α (점유율) | 성공 탐색 | **삽입** |
|---|---|---|
| 0.10 | 1.06 | 1.12 |
| 0.25 | 1.17 | 1.39 |
| **0.50** | **1.50** | **2.50** |
| 0.60 | 1.75 | 3.63 |
| 0.75 | 2.50 | 8.50 |
| 0.90 | 5.50 | 50.5 |
| 0.95 | 10.5 | 200.5 |
| 0.99 | 50.5 | 5000.5 |

**비대칭이 핵심**: 삽입은 `1/(1−α)²`(2차), 조회는 `1/(1−α)`(1차). TSDF integrate는 매 프레임
수십만 점을 **삽입**하는 워크로드라 나쁜 쪽 지수를 탄다.

### 2.2 Pittel (1987) — 여기서만 n이 직접 등장

> B. Pittel, *Linear probing: the probable largest search time grows logarithmically with the
> number of records*, J. Algorithms 8(2):236–249, 1987.

α를 0과 1 사이에 고정한 채 m, n → ∞ 이면 **최장 탐사열 길이**가 확률적으로

```
maxProbe ≈ ln m / c,      c = α − 1 − ln α
```

로 자란다. 평균이 아니라 **최악(꼬리)** 이며, `MAX_PROBE` 상한을 넘겨 삽입이 실패하는 사건
— 즉 복셀 드롭 — 을 지배하는 게 이 값이다.

우리 설정(tile당 `hashCapacityPerTile = 2²² ≈ 4.19M`, `ln m ≈ 15.25`)으로 계산하면:

| α | c | **최장 탐사 (m=2²²)** | 최장 탐사 (m=2²⁰) |
|---|---|---|---|
| 0.40 | 0.316 | 48 | 44 |
| **0.50** | 0.193 | **79** | **72** |
| 0.60 | 0.111 | 138 ⚠ | 125 |
| 0.75 | 0.038 | 405 | 368 |
| 0.90 | 0.0054 | 2,844 | 2,586 |

두 가지가 바로 읽힌다.

1. **`MAX_PROBE = 128`과 α=0.5 임계값 사이의 여유는 약 1.6배뿐이다.** α를 0.6까지만 올려도 최장
   탐사 기대치가 128을 넘어서(138) 조용한 복셀 드롭이 시작된다 → 리해시를 미루는 튜닝은 금물.
2. **복셀 개수 자체의 영향은 로그라 매우 약하다.** 용량을 2배로 키워도 최장 탐사는 `ln2/c`
   (α=0.5에서 +3.6)만 늘어난다. "복셀이 많아서" 탐사가 길어지는 게 아니라 **"α가 높아서"** 길어진다.

> ⚠️ 위 식은 leading-order 점근이라 상수항을 무시한다. 절대값이 아니라 **자릿수와 α 민감도**를
> 읽는 용도로 쓸 것. 우리 커널은 `wangHash`(랜덤 해시 가정 위반) + 동시 삽입이라 이론값과
> 정확히 일치할 이유는 없다.

### 2.3 그 밖의 이론 결과

| 논문 | 내용 | 우리에게 주는 함의 |
|---|---|---|
| Flajolet, Poblete, Viola (Algorithmica 1998) | 탐사 비용의 **정확한 분포** — 평균뿐 아니라 분산·꼬리 | 평균 2.5 probe여도 꼬리가 두꺼움을 정량화 |
| Pagh, Pagh, Ružić (STOC 2007) | 5-wise independent 해시면 O(1) 기대 시간 보장, 약한 해시는 붕괴 가능 | `wangHash`는 이 보장이 없음 → 이론 곡선보다 나쁠 수 있다 |
| Bender & Kuszmaul, **graveyard hashing** (FOCS 2021) | 인위적 tombstone으로 primary clustering을 완전 제거, 삽입을 Θ(x²) → O(x)로 | 삭제 없는 우리 테이블엔 직접 적용 불가하나, "선형탐사의 2차 항은 불가피하지 않다"는 반례 |
| Bender et al. (2025) 2편 | open addressing 상·하한의 최신 정리 | 이론 상한 확인용 최신 레퍼런스 |

---

## 3. GPU 해시테이블 — 실측 probe 곡선

### 3.1 Awad et al., *Better GPU Hash Tables* (2021)

**"개수 대 탐사 횟수" 실측에 가장 가까운 데이터.** 5천만 키 테이블에서 load factor를 0.6부터
최대치까지 올리며 **키당 평균 probe 수**를 계측한다(Fig. 14). 논문은 "삽입/조회 성능을 지배하는
주 요인은 연산당 probe 수"라고 명시하고 이를 하드웨어 독립 지표로 쓴다.

| 기법 | 삽입 평균 probe | 조회 (전부 hit) | 조회 (전부 miss) |
|---|---|---|---|
| **BCHT** (bucketed cuckoo) | **1.43** @ α=0.99 | 1.38 | 2.80 |
| IHT (iceberg) | 1.46 @ α=0.92 | 1.32 | 3 |
| BP2HT (bucketed power-of-2) | 2.0 (α 무관) | 1.34 | 2 |
| 1CHT (cuckoo, 해시 4개) | 2.75 @ α=0.9 | 2.26 | 3.44 |

α=0.99에서 1.43 probe — **같은 조건의 순수 선형탐사(≈5000 probe)와 세 자릿수 차이**다. 고밀도가
필요하면 bucketed 계열로 갈아타는 게 정답이라는 근거.

### 3.2 그 외

| 논문 | 요지 |
|---|---|
| Lessley, *Data-Parallel Hashing Techniques for GPU Architectures* (2018) | GPU 해싱 서베이 |
| Alcantara et al., *Real-Time Parallel Hashing on the GPU* (SIGGRAPH Asia 2009) | GPU cuckoo의 원조 |
| Lefebvre & Hoppe, *Perfect Spatial Hashing* (SIGGRAPH 2006) | 정적 데이터에 대해 **탐사 1회 보장**(충돌 0). 동적 삽입이 필요한 스캔 중엔 못 쓰지만, 완성된 모델을 굽는 단계엔 유효 |
| Dong, Lao, Kaess, Koltun, *ASH* (TPAMI 2023) | 3D 인식용 GPU 해시맵 프레임워크(Open3D). stdgpu/SlabHash 백엔드 벤치마크 — 우리와 같은 "3D 정수 키" 워크로드의 성능 비교 기준 |

---

## 4. 복셀/SLAM 시스템 — 실제로 쓰는 파라미터

### 4.1 Nießner et al., *Voxel Hashing* (TOG 2013)

이 분야의 기준점. **선형탐사가 아니라 bucket + 링크드 리스트**를 쓴다(버킷당 엔트리 2개, 넘치면
다음 빈 자리에 체인). 해시 함수는 Teschner 2003의 `(x·p₁ ⊕ y·p₂ ⊕ z·p₃) mod n`.

§9의 실측치가 "복셀 개수 vs 탐사 비용"에 대한 가장 직접적인 데이터다.

- 테이블 2²¹ 엔트리(~34MB), 8mm 복셀 스캔 시 **평균 140K 복셀 블록 할당 → 점유율 6.4%**
- 결과: 엔트리 1개 버킷 120K, 2개 버킷 10K, **버킷 오버플로 0.1%**, 최장 체인 **길이 3**, 전체 링크드 리스트 엔트리 ~700개

테이블을 줄여 점유율을 올리면:

| 테이블 크기 | 점유율 | 프레임 시간 |
|---|---|---|
| 2²¹ (기본) | 6.4% | 21 ms |
| 200K | 65% | 24.8 ms (+18%) |
| 160K | 81% | 25.6 ms (+22%) |

저자들의 결론은 "해시 테이블은 복셀 블록 버퍼(1GB)에 비해 메모리가 무시할 만하니 **크게 잡고
점유율을 낮게 유지하라**". 우리가 tile당 hash를 넉넉히 잡는 이유와 같고, `--tile-hash`로 줄일 때
감수하는 게 정확히 이 곡선이다.

### 4.2 InfiniTAM v3 (Kähler et al., 2017)

Nießner 구조의 파생 — 해시 인덱스마다 **ordered bucket**(보통 크기 2) + 넘치면 **unordered excess
list**. 복셀 블록 배열은 2¹⁸ 원소. 충돌 처리가 체이닝이라는 점이 우리(개방주소)와의 비교 포인트.

### 4.3 Mesh-LOAM (2023) — 선형탐사를 명시적으로 고른 사례

LiDAR 매핑용 GPU 복셀 해시. **"robin hood hashing, linear probing, quadratic probing을 시험한 뒤
경험적으로 linear probing을 선택"** 했다고 명시한다. 우리가 겪는 것과 똑같은 실패 모드를
**Information Dropout Ratio(%)** = 충돌로 잃은 복셀 비율로 정량화한다 (Table IV, KITTI 평균):

| 전략 | 복셀 삭제 기법 없음 | 있음 |
|---|---|---|
| Linear probing | 7.4% | 5.3e-05% |
| Quadratic probing | 7.4% | 5.9e-05% |
| Robin Hood hashing | 8.5% | 6.4e-05% |

**공간을 비우지 않으면 평균 7.4%의 복셀이 조용히 사라진다** — 우리 `findOrInsert`가 `~0u`를
리턴하며 점을 버리는 것과 같은 현상이다. 그들의 해법(오래된 복셀을 메시로 굽고 삭제)은 우리의
리해시(용량 확대)와 대비되는 다른 선택지.

### 4.4 De Rebotti et al., *Resolution Where It Counts* (TOG 2025)

MrHash. 충돌은 Nießner식 offset 필드(`o_j ∈ N`)로 처리하며, 탐사 통계는 따로 보고하지 않는다.
저장소 내 PDF: `docs/Variance-Adaptive Voxel Grids.pdf`.

---

## 5. 우리 구현에 대한 함의

현재 상태 — [`AdvancedTSDF.cpp:257-302`](../src/TSDF/Backends/AdvancedTSDF.cpp#L257-L302),
[`LinearProbe.glsl`](../src/TSDF/Memory/Hash/LinearProbe.glsl) (probing now lives in a swappable fragment; see [`HashStrategy.h`](../src/TSDF/Memory/Hash/HashStrategy.h)):

- open addressing + **linear probing**, `MAX_PROBE = 128`, 실패 시 조용한 드롭
- `maybeGrow()`가 **α ≥ 0.5**에서 용량 2배 리해시
- 키에 방향 3비트가 포함 → n = (표면 복셀 수 × 관측 방향 수)

| 항목 | 평가 |
|---|---|
| α = 0.5 임계값 | **적절.** 평균 삽입 2.5 probe(Knuth), 최장 79 probe(Pittel, m=2²²)로 `MAX_PROBE` 아래. 단 여유가 1.6배뿐이라 **더 올리면 안 된다**(α=0.6 → 138 > 128) |
| `MAX_PROBE = 128` | α=0.5 기준 타당. 다만 실패가 **무증상**이라, 드롭 카운터를 추가해 Mesh-LOAM식 dropout ratio를 계측할 가치가 있다 |
| 리해시 = 용량 2배 | Pittel 식에서 최장 탐사는 `+ln2/c`(α=0.5에서 +3.6)만 증가 → 용량을 키우는 방향은 탐사 비용을 거의 악화시키지 않는다. 제약은 메모리뿐 |
| 고밀도가 필요하면 | 선형탐사로 α>0.6을 노리지 말고 **bucketed 구조로 전환**(Nießner식 버킷 2 또는 BCHT). Awad 기준 α=0.99에서도 1.43 probe |
| 해시 함수 | `wangHash`는 Pagh et al.의 독립성 조건을 보장하지 않음. Teschner의 3-소수 XOR 해시(Voxel Hashing이 채택)와 A/B 해볼 만하다 |

---

## 6. 참고문헌

**이론**

- D. E. Knuth, *Notes on "Open" Addressing* (1963) / *TAOCP* Vol. 3 §6.4 — [강의노트 요약 (Zwick, TAU)](https://www.cs.tau.ac.il/~zwick/Adv-Alg-2015/Linear-Probing.pdf)
- B. Pittel, [*Linear probing: the probable largest search time grows logarithmically with the number of records*](https://www.sciencedirect.com/science/article/abs/pii/019667748790040X), J. Algorithms 8(2):236–249, 1987
- P. Flajolet, P. Poblete, A. Viola, [*On the Analysis of Linear Probing Hashing*](https://www.academia.edu/23274710/On_the_Analysis_of_Linear_Probing_Hashing), Algorithmica, 1998
- A. Pagh, R. Pagh, M. Ružić, [*Linear Probing with Constant Independence*](https://arxiv.org/pdf/cs/0612055), STOC 2007
- M. A. Bender, B. C. Kuszmaul, W. Kuszmaul, [*Linear Probing Revisited: Tombstones Mark the Death of Primary Clustering*](https://arxiv.org/pdf/2107.01250), FOCS 2021
- [*Tight Analyses of Ordered and Unordered Linear Probing*](https://arxiv.org/pdf/2501.11582) (2025) · [*Optimal Bounds for Open Addressing Without Reordering*](https://arxiv.org/pdf/2501.02305) (2025)

**GPU 해시테이블**

- M. A. Awad et al., [*Better GPU Hash Tables*](https://arxiv.org/pdf/2108.07232), 2021
- B. Lessley, [*Data-Parallel Hashing Techniques for GPU Architectures*](https://arxiv.org/pdf/1807.04345), 2018
- D. Alcantara et al., *Real-Time Parallel Hashing on the GPU*, SIGGRAPH Asia 2009
- S. Lefebvre, H. Hoppe, *Perfect Spatial Hashing*, SIGGRAPH 2006
- W. Dong, Y. Lao, M. Kaess, V. Koltun, [*ASH: A Modern Framework for Parallel Spatial Hashing in 3D Perception*](https://www.cs.cmu.edu/~kaess/pub/Dong23pami.pdf), TPAMI 2023

**복셀/SLAM 시스템**

- M. Teschner et al., [*Optimized Spatial Hashing for Collision Detection of Deformable Objects*](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf), VMV 2003
- M. Nießner, M. Zollhöfer, S. Izadi, M. Stamminger, [*Real-time 3D Reconstruction at Scale using Voxel Hashing*](https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf), TOG 2013
- O. Kähler et al., [*InfiniTAM v3: A Framework for Large-Scale 3D Reconstruction with Loop Closure*](https://arxiv.org/pdf/1708.00783), 2017
- [*Mesh-LOAM: Real-time Mesh-Based LiDAR Odometry and Mapping*](https://arxiv.org/pdf/2312.15630), 2023
- De Rebotti et al., [*Resolution Where It Counts: Hash-based GPU-Accelerated 3D Reconstruction via Variance-Adaptive Voxel Grids*](https://arxiv.org/html/2511.21459), TOG 2025

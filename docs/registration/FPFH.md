# FPFH 프로세스

`ComputeFpfh` 기준 (`src/Engine/Features/Fpfh.cpp`). 1패스인 SPFH는 [SPFH.md](SPFH.md) 참조.

FPFH(Fast Point Feature Histogram, Rusu 2009)는 SPFH를 이웃끼리 한 번 더 섞어 유효 반경을 넓힌 33-D 기술자다. 점당 하나, 입력 순서 그대로 반환한다.

## 1. 입력 준비

- 입력 `PointCloud`는 **점마다 법선이 있어야 한다** — `normals.size() != points.size()`면 빈 결과를 반환한다.
- 반경 두 개를 받는다: `normalRadius`(SPFH용, 좁음)와 `fpfhRadius`(합산용, 넓음).
- 호출부는 두 반경을 복셀 크기에 비례시킨다 (`RegistrationConfig`: `normalRadiusGain = 3.0`, `fpfhRadiusGain = 5.0`, `voxelSize = 5.0` mm 기준).
- 셀 크기 `fpfhRadius`인 해시 그리드를 **한 번만** 만들어 두 패스가 공유한다.
- 불변식: 그리드 질의는 3×3×3만 훑으므로 질의 반경 ≤ `fpfhRadius`여야 한다. `normalRadius > fpfhRadius`로 부르면 이웃이 조용히 잘린다 — 예외가 아니라 열화로 나타난다.

## 2. 1패스 — SPFH

- 모든 점 $i$에 대해 반경 `normalRadius` 이웃으로 SPFH$(i)$를 계산해 통째로 보관한다.
- 왜: 2패스에서 이웃의 SPFH를 재사용하므로, 미리 다 구해 두면 쌍 특징 계산이 점당 한 번으로 끝난다. 이것이 PFH의 $O(n k^2)$를 $O(n k)$로 낮추는 지점이다.

### 2.1. Darboux Frame

곡면 위를 따라 움직이는 곡선의 방향과 곡면의 법선 방향을 표현하는 좌표계

$$
N = n_{q}
$$

$$
T = normalize(p - q)
$$

$$
L = normalize(N \times T)
$$

|  기호   |         설명         |
| :-----: | :------------------: |
|   $p$   |        이웃점        |
|   $q$   |        기준점        |
| $n_{q}$ | 기준점에서 노말 벡터 |
|   $T$   |    곡선 방향 벡터    |
|   $L$   | 곡면 접선 방향 벡터  |

## 3. 2패스 — 거리 가중 합산

- 점 $i$의 반경 `fpfhRadius` 이웃 $k$개의 SPFH를 거리 역수로 가중해 자기 SPFH에 더한다.

$$
\mathrm{FPFH}(p) = \mathrm{SPFH}(p) + \frac{1}{k} \sum_{q \in \mathcal{N}(p,\ r_{fpfh})} \frac{1}{\|q - p\|} \mathrm{SPFH}(q)
$$

- 자기 자신과 거리 $\le 10^{-12}$인 중복점은 $k$에서도 합에서도 제외한다.
- 이웃이 없으면($k = 0$) 합산 항을 건너뛰고 SPFH$(p)$만 남긴다.
- 왜: 자기 SPFH에 가중치 1을 주고 이웃 평균을 더하므로, 자기 기하가 지배하되 한 링(ring) 바깥 정보가 섞인다 — 반경을 그냥 키우는 것과 달리 국소성이 유지된다.

## 4. 블록 재정규화와 판정

- 합산으로 깨진 스케일을 되돌린다 — 11-bin 블록 3개를 각각 합 1로 다시 나눈다.

$$
h_{[11b,\ 11b+11)} \leftarrow \frac{h_{[11b,\ 11b+11)}}{\sum h_{[11b,\ 11b+11)}}, \qquad b = 0, 1, 2
$$

- 합이 $10^{-12}$ 이하인 블록은 건드리지 않는다 → 고립점은 영벡터로 남는다.
- 결과: 강체 변환 불변인 33-D 벡터. 회전시킨 클라우드에서 같은 값이 나오는지가 `test/test_fpfh.cpp`의 불변성 테스트다.

## 5. 소비처

- **전역 정합** (`GlobalRegistration.cpp`): 두 클라우드를 `voxelSize`로 다운샘플 → FPFH → `MatchFeatures`(33-D L2 최근접 + Lowe 비율 0.95, 최대 5000쌍) → 3점 RANSAC(5000회) → Umeyama 재적합 → Ceres Cauchy 정제.
- **루프 클로저** (`Engine/Backend/LoopClosure.cpp`): 헤더 전용 판본으로 후보 키프레임 쌍의 상대 포즈를 추정한다.
- 왜 초기 추정에만 쓰는가: FPFH는 대응을 _찾는_ 데 강하지 미세 정렬에 쓰기엔 히스토그램 이산화가 거칠다 — 정밀도는 뒤의 ICP가 낸다.

## 판본 차이 (둘 다 살아 있음)

|             | `Fpfh.cpp` (`ComputeFpfh`)          | `FpfhSignature.h` (`ComputeFPFH`)                        |
| ----------- | ----------------------------------- | -------------------------------------------------------- |
| 반경        | `normalRadius` / `fpfhRadius` 두 개 | `FpfhConfig::radius` 하나 (두 패스 공용)                 |
| 쌍의 source | 항상 $p$ (비대칭)                   | $\lvert n \cdot \hat{d} \rvert$가 큰 쪽 (쌍에 대해 대칭) |
| 블록 정규화 | 합 1                                | 합 100 (PCL 호환)                                        |
| 이웃 탐색   | TU 로컬 해시 그리드                 | 주입된 `NeighborQuery` (기본 `CpuGridNeighborhood`)      |
| 소비처      | `GlobalRegistration`                | `Engine::Backend::LoopClosure`                           |

- 두 판본의 기술자는 **수치적으로 다르다** — 스케일(1 vs 100)과 source 선택이 달라서, 한쪽으로 만든 기술자를 다른 쪽과 매칭하면 안 된다.

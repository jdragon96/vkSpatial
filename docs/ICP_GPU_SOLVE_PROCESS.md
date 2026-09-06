# GPU Point-to-Plane ICP 프로세스

`GpuPointToPlaneIcp::Solve` 기준 (`src/Pipeline/Registration/GpuPointToPlaneIcp.cpp`, 커널 `kernel_icp_iterate.comp.glsl`).

## 1. Centroid

- Target Surface에서 Centroid 좌표값을 계산한다 (double 누적 후 float 캐스트).

$$
c_{target} = \frac{1}{N_{target}} \sum_i target_i
$$

- Source와 Target 좌표계를 일치화시킨다 — 두 클라우드 모두 target centroid만큼 이동한다.

$$
target = target - c_{target}
$$

$$
source = source - c_{target}
$$

- Prior 포즈도 같은 centred 프레임으로 켤레변환한다 ($T_c$ = translate($-c_{target}$)).

$$
T = T_c \cdot T_{prior} \cdot T_c^{-1}
$$

- 왜: $|p|$가 원점 근처로 묶여 고정소수점(×10000) GPU 누적이 안전해지고, 회전 선형화의 피벗이 centroid가 되어 6×6 정규방정식의 조건수가 장면의 월드 위치와 무관해진다.
- 불변식: 센서 로컬인 source에서 $c_{target}$을 빼는 것 자체는 무의미하지만, 켤레변환의 $T_c^{-1}$이 정확히 되돌린다 — $T \cdot (source - c) = T_{prior} \cdot source - c$. 뺄셈과 켤레변환은 반드시 쌍으로 유지한다 (한쪽만 지우면 $(R-I)c$ 만큼 틀린다).

## 2. Local Grid (Target 전용)

- Target 포인트로 균일 그리드를 만든다. Source 포인트는 그리드에 넣지 않는다 — 질의(query) 전용이다.
- 셀 크기 = `maxCorrDist` (어닐링 중 가장 넓은 대응 거리). **Solve당 한 번만** 만든다.

1. AABB를 구한다: `origin` = min bound, 축별 셀 개수는

$$
dims_a = \lfloor (max_a - min_a) / cell \rfloor + 1
$$

2. 셀 수 천장(32M)을 넘으면 셀을 키워 다시 잰다 (최대 64회, 1.02는 1-step 수렴 마진):

$$
cell \leftarrow cell \cdot \sqrt[3]{N_{cells} / N_{max}} \cdot 1.02
$$

3. Counting sort로 버킷을 채운다: 히스토그램 → prefix sum(`bucketStart`) → 산개(`bucketIdx`).

- 점의 셀 좌표와 선형 인덱스:

$$
c = \lfloor (p - origin) / cell \rfloor, \qquad idx = (c_z \cdot dims_y + c_y) \cdot dims_x + c_x
$$

- 불변식: 질의 반경 ≤ cell 이므로 3×3×3 이웃 스캔만으로 탐색이 완전하다. 어닐링은 질의 반경(`g_maxCorr`)만 좁히고 그리드는 재구축하지 않는다.

## 3. Iteration

- 매 반복 (`iter < maxIters`):

1. 어닐링 파라미터를 얻는다 (`huberScale`, 이번 반복의 대응 거리).
2. GPU 디스패치 — source 점마다:
   - 현재 포즈를 적용한다: $p = T \cdot source$
   - 그리드 3×3×3 이웃에서 최근접 target $q$를 찾는다 (반경 = 이번 반복의 대응 거리).
   - 법선 게이트: $(R \cdot n_{source}) \cdot n_{target} <$ `normalCompatibilityCosine` 이면 기각한다.
   - point-to-plane 잔차와 Jacobian을 Huber 가중으로 누적한다 (워크그룹당 29슬롯 = H 상삼각 21 + b 6 + inlier 1: 고정소수점 ×10000 원자 누적, $\sum e^2$ 1: float 공유메모리 트리 축약 — 고정소수점이면 ~7 mm 아래 잔차가 0으로 양자화된다).

$$
e = n \cdot (p - q), \qquad J = [\, p \times n,\ n \,]
$$

3. CPU에서 워크그룹 파셜을 합산해 $H$, $b$를 복원한다. inliers < `minInliers`면 중단한다.
4. 6×6을 LDLT로 풀고 포즈를 갱신한다 ($\Delta$ = ZYX 오일러 회전 + 이동):

$$
H x = b, \qquad T \leftarrow \Delta(x) \cdot T
$$

5. $|x| <$ `convEps` 면 수렴 종료한다.

## 4. Un-centre

- 포즈를 원래 프레임으로 되돌린다 (0회 반복이면 정확히 `priorT`로 환원된다):

$$
T_{final} = T_c^{-1} \cdot T \cdot T_c
$$

- 판정: `valid` = inliers ≥ `minInliers` **AND** fitness ≥ `minFitness`, rmse = $\sqrt{\sum e^2 / inliers}$.

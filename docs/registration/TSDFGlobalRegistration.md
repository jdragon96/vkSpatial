# TSDF 글로벌 정합 프로세스

`GlobalRegistrationTracker::Track` → `Registration::Estimate` 기준 (`src/Pipeline/Registration/GlobalRegistrationTracker.cpp`, `GlobalRegistration.cpp`; 기술자·매칭은 `src/Engine/Features/`).

Prior를 쓰지 않는다 — 초기 포즈 없이 프레임을 맵에 붙이는 (재)로컬라이제이션 경로다. 레지스트리 이름은 `global`이고, 합성 트래커 `icp+global`의 폴백으로도 같은 코드가 돈다.

## 1. Target 구성 (TSDF 스냅샷 → 점군)

- `ModelSnapshot::entries`의 TSDF 복셀마다 서브복셀 표면점을 만들어 target 점군으로 쓴다 ($\mu$ = `truncationDistance`).

$$
q_i = center_i - tsdf_i \cdot \mu \cdot n_i
$$

- $center_i$, $n_i$ — 복셀 중심과 그 복셀이 기술하는 표면 법선(`TSDFVoxel::center`, `TSDFVoxel::normal`).
- $tsdf_i$ — 트런케이션으로 정규화된 부호거리(−1~1). $tsdf_i \cdot \mu$가 중심에서 표면까지의 실제 부호거리다.
- $\mu$ — `ModelSnapshot::truncationDistance`(월드 단위). 0이면 서브복셀 보정이 꺼져 $q_i$는 그냥 복셀 중심이다.
- $q_i$ — 그만큼 법선 방향으로 되돌린 target 표면점.

- 왜: 복셀 중심은 자기가 기술하는 표면에서 최대 반 truncation 밴드만큼 떨어져 있다.
- Source는 프레임의 `pts`/`nrm`을 그대로 쓴다.
- `cfg.voxelSize`를 `model->voxel`로 갈아끼운다 — 이후 모든 반경(다운샘플 셀, FPFH, RANSAC 인라이어, Ceres loss)이 이 값의 배수라, mm 스케일 기본값(5.0)을 그대로 두면 미터 스케일 맵에서 파이프라인 전체가 무너진다.

## 2. Downsample + FPFH

- 두 점군을 `voxelSize` 복셀 그리드로 다운샘플한다 (셀당 1점 = centroid, 법선은 합산 후 재정규화).
- 점마다 33-D FPFH를 계산한다 (`normalRadius` = 3×voxel, `fpfhRadius` = 5×voxel).

1. SPFH: 반경 `normalRadius` 이웃과의 Darboux 프레임 쌍 특징을 11빈 히스토그램 3개(= 33-D)에 담는다.

$$
f_1 = v \cdot n_q, \qquad f_2 = u \cdot \hat{d}, \qquad f_3 = \arctan2(w \cdot n_q,\ u \cdot n_q)
$$

- $p$, $q$ — 기준점과 그 이웃점, $n_p$, $n_q$ — 각각의 단위 법선.
- $\hat{d} = (q - p)/\lVert q - p \rVert$ — 두 점을 잇는 단위 방향.
- $(u, v, w)$ — $p$에 세운 Darboux 정규직교 프레임: $u = n_p$, $v = \hat{d} \times u$ (정규화), $w = u \times v$.
- $f_1, f_2, f_3$ — 법선쌍의 상대 자세를 프레임 축에 사영한 세 각도 성분. $f_1, f_2 \in [-1, 1]$, $f_3 \in [-\pi, \pi]$ 범위를 각각 11빈으로 나눈다.
- 예외: $\hat{d}$가 $n_p$와 평행하면 $v$가 정의되지 않아 그 쌍은 버린다.

2. FPFH: 반경 `fpfhRadius` 이웃의 SPFH를 거리 역수로 가중해 더하고, 11빈 블록별로 다시 정규화한다.

$$
FPFH(p) = SPFH(p) + \frac{1}{k} \sum_{q} \frac{1}{\lVert q - p \rVert} SPFH(q)
$$

- $SPFH(\cdot)$ — 1단계에서 점마다 구한 33-D 히스토그램.
- $q$ — $p$의 `fpfhRadius` 이웃(자기 자신과 거리 0인 중복점은 제외).
- $k$ — 실제로 더해진 이웃 수. $k = 0$이면 가중항 없이 $SPFH(p)$만 남는다.
- $1/\lVert q - p \rVert$ — 거리 역수 가중. 가까운 이웃일수록 기술자에 더 실린다.

- 이웃 탐색은 셀 크기 `fpfhRadius`인 CPU 해시 그리드의 3×3×3 스캔이다.
- 불변식: 질의 반경 ≤ 셀 크기여야 탐색이 완전하다 — 그래서 `normalRadius` < `fpfhRadius` = 셀 크기다.

## 3. Feature Matching

- src 기술자마다 tgt에서 1·2위 최근접을 브루트포스로 찾고 Lowe 비율 테스트를 건다 ($d_1/d_2 <$ 0.95).
- 왜: 1위와 2위가 거의 같은 거리면 그 매칭은 변별력이 없다. tgt가 2개 미만이면 비율 자체가 성립하지 않아 빈 결과다.
- 살아남은 대응을 ratio 오름차순으로 정렬해 상위 `numMaxCorr`(5000)개만 남긴다.
- 대응이 3개 미만이면 여기서 실패로 끝난다 (3점 샘플조차 못 만든다).

## 4. RANSAC

- `ransacIters`(5000)회 반복한다. RNG는 고정 시드(12345)라 실행마다 같은 해가 나온다.

1. 서로 다른 대응 3개를 뽑아 Umeyama 닫힌 해로 강체 변환을 푼다 (`with_scaling = false`).
2. 그 $T$로 **전체** 대응의 인라이어를 센다 (임계값 = `ransacInlierGain`(2.0) × voxelSize).

$$
\lVert T p_{src} - p_{tgt} \rVert < 2 \cdot voxelSize
$$

- $T$ — 이번 반복에서 3점 샘플로 푼 후보 강체 변환($4\times4$, 회전+이동).
- $p_{src}$, $p_{tgt}$ — 한 대응이 가리키는 다운샘플된 source·target 점.
- $2 \cdot voxelSize$ — 인라이어 임계 거리(`ransacInlierGain` × `voxelSize`). 맵 해상도의 두 배 안이면 같은 표면으로 본다.

3. 인라이어가 가장 많은 $T$를 유지한다.

- 마지막에 그 인라이어 집합 **전체**로 Umeyama를 다시 풀어 refit한다 (3점 샘플 적합보다 조인다).

## 5. Ceres refine + 판정

- 거친 해가 `valid`가 아니면 정제 없이 그대로 반환한다 (쓰레기를 정제하지 않는다).
- 인라이어 대응마다 3-벡터 잔차를 놓고 쿼터니언 $q$(`QuaternionManifold`) + 이동 $t$를 푼다 (AutoDiff, DENSE_QR, SILENT).

$$
r_i = R(q) p_{src,i} + t - p_{tgt,i}
$$

- $q$ — 최적화 대상 단위 쿼터니언 $[w,x,y,z]$(`ceres::QuaternionManifold`), $R(q)$ — 그것이 만드는 회전.
- $t$ — 최적화 대상 이동 벡터. 둘 다 RANSAC의 거친 $T$에서 초기화한다.
- $p_{src,i}$, $p_{tgt,i}$ — 거친 해의 **인라이어** 대응 $i$번 점쌍(전체 대응이 아니다).
- $r_i$ — 그 대응의 point-to-point 3-벡터 잔차. Cauchy 손실을 씌워 합을 최소화한다.

- 로버스트 손실은 `CauchyLoss(ceresLossGain(1.0) × voxelSize)`다 — RANSAC 임계값이 걸러내지 못한 잔여 아웃라이어용 M-추정.
- 정제된 $T$로 **같은 대응 집합**에서 인라이어/fitness를 다시 센다 (다운샘플·FPFH·매칭·RANSAC은 재실행하지 않는다).
- 판정: `valid` = 인라이어 ≥ 3 **AND** fitness > 0.1(내부 상수 `kMinFitness`), fitness = 인라이어 / 전체 대응 수.
- 실패는 인라이어 < 3이면 `TooFewInliers`, 아니면 `LowOverlap`으로 분류한다. **`NoModel`로 두면 안 된다** — `ShouldFuse()`가 `NoModel`을 융합하므로 틀린 포즈가 맵에 들어간다.
- `rmse`는 이 경로에서 채워지지 않는다(항상 0) — 품질 판단은 fitness로 한다.

# ICP 분석 — Local vs Global 두 파트

[ICP_METHODS.md](ICP_METHODS.md)가 방법 카탈로그라면, 이 문서는 registration을 관통하는
가장 중요한 축인 **local(정밀화) vs global(초기 정렬)** 을 분석한다. 핵심 명제 두 개:

1. **ICP 본체는 local optimizer다** — 이미 근사적으로 맞춰진 상태를 정밀화하며, 목적함수가
   비볼록이라 *가까운 극소값*으로만 수렴한다. → 좋은 초기값이 없으면 실패한다.
2. **Global registration은 그 초기값(basin)을 만들어 준다** — 초기 자세 없이 임의 배치에서
   대략 정렬한다. 정밀도는 낮지만 ICP의 seed가 된다.

전체 파이프라인은 거의 항상 **Global(coarse) → Local ICP(fine)** 이다.

---

## Part 1. Local Registration (= ICP 본체)

### 1.1 정의 — 무엇을 푸는가
근사 정렬 `T₀`가 주어졌을 때, correspondence를 반복 갱신하며 `T`를 정밀화한다. 대상은
"거의 맞은" 두 점군. 출력은 **높은 정밀도**의 rigid 변환.

### 1.2 왜 "local"인가 — 비볼록성
목적함수
```
E(T) = Σ_i  dist( T·p_i , Q )²
```
는 `T`(그리고 숨은 correspondence)에 대해 **비볼록**이다. 대응관계가 자세에 따라 이산적으로
바뀌므로(누가 누구의 최근접인지) 표면에 수많은 local minimum이 생긴다. ICP는 이 위에서
좌표하강(coordinate descent)을 할 뿐이다.

### 1.3 수렴 이론 — ICP ≈ EM (단조감소, 그러나 극소)
point-to-point ICP는 EM으로 볼 수 있다:
- **E-step (Matching):** 현재 `T`를 고정하고 각 `p_i`의 대응 `q_{c(i)}`를 결정.
- **M-step (Minimization):** 대응을 고정하고 `E`를 최소화하는 `T`를 closed-form(SVD)으로 계산.

두 단계 모두 `E`를 증가시키지 않으므로 **단조 감소 + 유한 correspondence 집합 → 수렴 보장.**
단, 보장되는 것은 *극소값으로의 수렴*일 뿐 전역 최적이 아니다. 이것이 ICP의 본질적 한계다.

### 1.4 무엇이 "수렴 basin"을 결정하나
초기 `T₀`가 정답에서 얼마나 벗어나도 되는지(basin of convergence)는:
- **Error metric** — point-to-plane은 접선 방향으로 미끄러질 자유를 줘 basin이 넓고 수렴이
  빠르다(point-to-point는 좁고 느림). symmetric/GICP는 더 넓다. (자세한 metric은 ICP_METHODS §2)
- **Overlap** — 두 점군의 겹침이 적으면 잘못된 대응이 많아 basin이 급격히 좁아진다.
- **Geometry 축퇴(degeneracy)** — 평면 1개(법선 1축만 구속 → 나머지 5-DoF 미구속),
  구/원통(회전 미구속) 같은 형상은 **underconstrained** → local에서도 특정 DoF가 표류.
  (point-to-plane의 6×6 정규방정식 `Hessian`의 조건수/최소 고유값으로 진단 가능.)
- **초기 오차** — 경험적으로 point-to-point는 수십 도/객체 크기의 수십 % 이내에서 안정.

### 1.5 비용과 실패 모드
- **비용:** `O(iterations × N × 대응탐색비용)`. 병목은 nearest-neighbor 탐색 → kd-tree / **BVH**.
- **실패 모드:** (a) 잘못된 basin(초기값 나쁨), (b) 낮은 overlap, (c) 형상 축퇴/대칭,
  (d) outlier·부분겹침 → robust rejection/kernel 필요(ICP_METHODS §3).

### 1.6 Local을 강하게 만드는 지렛대
좋은 초기값(→ Part 2), coarse-to-fine(voxel downsample로 basin 넓히고 점점 정밀),
robust kernel(Huber/Cauchy) + distance/normal rejection, 축퇴 DoF 정규화.

---

## Part 2. Global Registration (= 초기값 없는 정렬)

### 2.1 정의 — 목표
**초기 자세 없이** 임의 배치의 두 점군을 대략 정렬. 정밀도보다 **올바른 basin 획득**이 목적.
출력은 ICP에 넘길 coarse `T₀`.

### 2.2 세 가지 접근

**(A) Feature + robust estimation — 빠르고 근사적, 실전 주류**
1. 각 점에 local descriptor 계산: **FPFH**(Rusu), SHOT 등(법선·기하 히스토그램, 회전불변).
2. descriptor 매칭 → putative correspondence(다수가 오답).
3. robust하게 변환 추정:
   - **RANSAC** — 3점 샘플 → 변환 → inlier 세기, 반복.
   - **FGR** (Fast Global Registration, Zhou 2016) — line process로 outlier를 연속최적화에 흡수, RANSAC보다 빠름.
   - **TEASER++** (Yang 2020) — graph-theoretic maximal clique로 outlier 제거 + **certifiable**(최적성 인증) 회전 추정. 극단적 outlier 비율에도 강함.
- 장점: 빠름, partial overlap OK, 초기값 불필요. 단점: **feature 반복성(repeatability)에 의존** — 매끈하거나 반복 패턴 형상에서 약함.

**(B) Globally-optimal ICP — 느리지만 전역 최적 보장**
- **Go-ICP** (Yang 2013) — `SE(3)` 위에서 **branch-and-bound**로 L2 목적함수의 전역 최적을 탐색,
  각 분기 안에서 지역 ICP로 하한을 조인다. 전역 최적 **증명 가능**, 그러나 느려 대규모/실시간엔 부적합.
- 파생: GOGMA(GMM+BnB) 등.

**(C) Correspondence-free / 학습 기반**
- 분포 정합: 두 점군을 **GMM**으로 보고 분포 간 정합(대응 명시 안 함), 전역 NDT.
- 딥러닝: DGR, PointDSC(outlier 분류), **PREDATOR**(낮은 overlap 특화), **GeoTransformer**(강인한 대응).
  학습 데이터 도메인에 의존하지만 낮은 overlap/텍스처리스에서 강함.

### 2.3 트레이드오프

| 접근 | 초기값 | 보장 | 속도 | 낮은 overlap | 비고 |
|---|---|---|---|---|---|
| (A) Feature+RANSAC/FGR/TEASER++ | 불필요 | 확률적/인증(TEASER) | 빠름 | 중~강 | feature 반복성 의존 |
| (B) Go-ICP (BnB) | 불필요 | **전역 최적** | 느림 | 약 | 소규모/오프라인 |
| (C) GMM/학습 | 불필요 | 없음(경험적) | 중 | 강(PREDATOR) | 도메인 의존 |

---

## Part 3. 둘은 어떻게 합쳐지나 — 파이프라인

```
   [두 점군, 초기값 없음]
        │
        ▼  Global registration (coarse)   ← basin 확보, 정밀도 낮음
     coarse T₀
        │
        ▼  Local ICP (fine)               ← 정밀화, T₀ 근처 극소로 수렴
     정밀 T*
```

- **왜 둘 다 필요한가:** global은 *올바른 골짜기*를 찾지만 바닥까지 못 간다(정밀도↓);
  local은 정밀하지만 *가까운 골짜기*만 본다(전역성↓). 상보적이다.
- 실무 팁: global 후 곧장 fine ICP보다, **coarse→fine multiscale ICP**(큰 voxel→작은 voxel)로
  basin을 이어붙이면 안정적.

---

## Part 4. Local vs Global 한눈 비교

| 관점 | **Local (ICP 본체)** | **Global (초기 정렬)** |
|---|---|---|
| 초기값 | **필요** | 불필요 |
| 목적 | 정밀화 | basin 획득 |
| 최적성 | 극소값(단조수렴) | 전역(BnB) 또는 근사(feature/학습) |
| 정밀도 | **높음** | 낮음(coarse) |
| 비용 | 반복당 저렴, 여러 번 | 1회, 상대적 고가 |
| overlap 민감도 | 높음 | 방법별(TEASER/PREDATOR는 강) |
| 실패 원인 | 나쁜 init, 축퇴, outlier | feature 미반복, 대칭 형상 |
| 대표 | point-to-point/plane, GICP | FPFH+RANSAC, FGR, TEASER++, Go-ICP |

---

## Part 5. SLAM에서의 Local/Global — 이 축이 어디에 나타나나

SLAM은 이 두 파트를 **역할별로 다른 곳에** 배치한다.

- **Local ICP = odometry / scan-matching front-end.**
  연속 프레임에서는 직전 pose(또는 IMU/constant-velocity)가 **훌륭한 초기값**이므로
  *local ICP만으로 충분*하다. frame-to-frame 또는 frame-to-model(누적 맵) 정합으로 6-DoF pose를 낸다.
  → SLAM이 매 프레임 비싼 global registration을 피할 수 있는 이유가 **시간적 연속성(motion prior)** 이다.

- **Global이 등장하는 두 지점:**
  1. **Global *registration*** — loop closure / relocalization. 오래전 방문한 곳으로 돌아왔을 때는
     상대 pose 초기값이 없다(궤적이 끊김) → 여기서만 §2의 global 정합(feature/place recognition)을 쓴다.
  2. **Global *optimization*** — back-end. odometry의 **drift**(local 정합의 누적 오차)를 교정하려면
     전 궤적을 전역 일관되게 재추정해야 한다 → **pose-graph optimization**(GTSAM/g2o/Ceres),
     필요시 global bundle adjustment. loop closure가 만든 constraint가 그래프를 "닫아" drift를 분산·흡수한다.

**요약:** *local ICP는 순간의 상대 pose(측정)*, *global은 (a) 끊긴 관계를 잇는 재정합 + (b) 전 궤적의 일관성*.
SLAM = **local front-end(ICP) + global back-end(pose graph/loop closure).**

---

## Part 6. 이 저장소(VkLBVH)로의 매핑 & 구현 순서

**먼저 Local part부터 (지금 가능):**
- Correspondence: `BVH`(`src/BVH/BVH.h`)의 GPU `KNN`/`RadiusSearch`.
- Error metric의 normal: `DirectionalTSDF`(point cloud + normal, frame-to-model model).
- 검증: `Engine::Eval`(`SyntheticSurface`/`ScanSampler`로 알려진 변환 생성, `RmseMetrics`로 정확도 회귀).
- 순서: point-to-point(SVD) → point-to-plane+robust → frame-to-model odometry.
  초기값은 **직전 pose(motion prior)** → 이 단계에선 global registration이 아예 불필요.

**Global part는 나중에 (SLAM 단계로 넘어갈 때):**
- 먼저 필요한 것은 **global *optimization*(pose graph back-end)** — drift 교정. (GTSAM/Ceres 도입)
- 그 다음 **loop closure detection + global *registration***(재정합). 초기에는 생략하고 순수 odometry로 시작해도 됨.

**선행 과제:** 대규모 correspondence를 GPU `KNN`으로 돌리기 전
[Engine::Core large-N 비결정 버그](KNOWN_ISSUES_engine_core_large_n.md)를 먼저 해결.
그전까지는 작은 N / CPU 참조로 local ICP를 프로토타이핑.

---

### 참고
방법별 수식·논문은 [ICP_METHODS.md](ICP_METHODS.md) §2–§4, SLAM 시스템 사례는 §5 참조.
추가 핵심: Yang et al., *Go-ICP* (PAMI 2016) — 전역 최적 ICP(BnB). Zhou et al., *FGR* (ECCV 2016).
Yang et al., *TEASER++* (T-RO 2020). Huang et al., *PREDATOR* (CVPR 2021). Qin et al., *GeoTransformer* (CVPR 2022).

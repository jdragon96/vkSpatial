# Direct TSDF-Gradient Odometry (실시간 프레임→모델 트래킹) — 설계

> **한 줄 요약:** DirectionalTSDF의 **저장된 gradient**를 이용해, 새 프레임 점 $p_i$를 변환한 $x_i=Tp_i$의 부호거리 $\text{sdf}(x_i)$를 직접 0으로 만드는 SE(3) 포즈를 Gauss–Newton으로 추정한다. 대응점 탐색·레이캐스트·프레임별 BVH가 전혀 없어 실시간이며, 잔차가 곧 point-to-plane이고 Jacobian이 저장 gradient에서 해석적으로 나온다.

> 상태: 설계(브레인스토밍 승인). 다음 단계: writing-plans. 이것은 "실시간 track + 고품질 맵" 비전의 **sub-project A**(4개 중 첫째)이며, B(submap)·C(pose-graph/loop closure)·D(live loop)는 각각 별도 spec.
> 수식은 GitHub/마크다운 뷰어에서 렌더됩니다.

---

## 0. 확정사항 (브레인스토밍 결과)
- 운용 모드: **실시간 track + 고품질 맵**(dense-SLAM 계열).
- 트래킹 방식: **① Direct TSDF-gradient**(대응점/레이캐스트/BVH 불필요, 저장 gradient 활용 극대화).
- 표현: 이미 머지된 **stored-gradient DirectionalTSDF**(block/streaming residency, `GpuTsdfVoxel{sumDW,sumW,sumNx,sumNy,sumNz}`).
- 입력: 비정렬 point cloud + 법선(chair PLY 및 `ScanDataset` 합성 프레임 모두).
- coarse-to-fine: **단일 해상도로 시작(measure-first)**, ATE에서 수렴 실패 관측 시에만 피라미드 추가.

## 1. 배경 · 목표 · 범위

### 1.1 배경
기존 저장소에는 **FPFH 전역 정합**(`Engine/Registration`)과 **PoseGraph/LoopClosure**(`Engine/Backend`)는 있으나 **실시간 국소 ICP odometry는 없다**. `registration_chair_demo`는 연속 프레임을 FPFH로 프레임-투-프레임 정합 후 `DirectionalTSDF::Integrate`한다 — 무겁고(실시간 아님) drift가 크다(프레임-투-프레임). 본 설계는 그 자리에 **프레임-투-모델 실시간 트래커**를 넣는다.

### 1.2 목표
- 새 프레임과 현재 TSDF 모델 사이의 SE(3) 포즈를 실시간(프레임당 GN 3~8회)으로 추정.
- 머지된 저장 gradient를 트래킹의 잔차·Jacobian에 직접 사용(정밀 축과 트래킹 축의 통일).
- 합성 GT 궤적에서 **ATE < ~voxelSize** 달성, chair에서 FPFH 대비 drift 개선.

### 1.3 참고 문헌 근거
Direct SDF 카메라 트래킹은 확립된 기법이다: Bylow et al., *"Real-Time Camera Tracking and 3D Reconstruction Using Signed Distance Functions"*, RSS 2013; Canelhas et al., *"SDF Tracker"*, IROS 2013. 본 설계는 이를 **directional·stored-gradient TSDF** 위에서 재정식화(방향 레이어 선택 + 저장 gradient로 Jacobian) 한 것.

### 1.4 범위 밖 (YAGNI)
- submap 생명주기(sub-project B), pose-graph/loop closure(C), live 스레딩/뷰어(D).
- relocalization(트래킹 실패 복구) — 본 설계는 `valid=false` 신호만 낸다.
- 해상도 피라미드 — 측정 후 필요 시.
- RGB/광도(photometric) 항 — 기하만.

---

## 2. 수학

### 2.1 잔차 (residual)
모델의 제로 레벨 표면이 목표. 프레임 점 $p_i$, 현재 포즈 $T$, 변환점 $x_i=Tp_i$. 이상적으로 $\text{sdf}(x_i)=0$. 포즈 증분 $\xi\in\mathfrak{se}(3)$에 대해
$$
E(\xi)=\sum_i w_i\,\rho\!\big(r_i(\xi)\big)^2,\qquad r_i(\xi)=\text{sdf}\big(\exp(\hat\xi)\,x_i\big),\quad \xi=[\nu;\ \omega]\in\mathbb R^6
$$
$\nu$=병진, $\omega$=회전. 좌증분 규약 $T\leftarrow\exp(\hat\xi)T$.

### 2.2 Jacobian (저장 gradient에서 해석적으로)
$\nabla\text{sdf}(x)=\hat g(x)$ = 저장 gradient(단위). 1차: $\exp(\hat\xi)x\approx x+\nu+\omega\times x$, $\ \mathrm d\,\text{sdf}=\hat g\cdot \mathrm d x$. 항등식 $\hat g\cdot(\omega\times x)=\omega\cdot(x\times\hat g)$ 로
$$
J_i=\frac{\partial r_i}{\partial\xi}\Big|_{\xi=0}=\big[\ \hat g_i^{\top}\ \ (x_i\times\hat g_i)^{\top}\ \big]\in\mathbb R^{1\times6}
$$

### 2.3 Gauss–Newton
$$
\Big(\sum_i w_i J_i^\top J_i\Big)\,\xi=-\sum_i w_i J_i^\top r_i,\qquad T\leftarrow\exp(\hat\xi)\,T
$$
6×6 정규방정식을 LDLT로 풀고 반복(수렴 또는 최대 반복까지).

### 2.4 단위(units) — 중요
저장값 $c=\text{sumDW}/\text{sumW}\in[-1,1]$는 정규화 TSDF(`clamp(sdf/τ,-1,1)`), $\tau$=truncation. **metric 부호거리** $=c\,\tau$. voxel 중심에서 1차 전개한 점 $x$의 metric sdf:
$$
r_i \approx c\,\tau + \hat g\cdot(x_i-\text{voxelCenter})
$$
($\hat g$는 단위이고 proper SDF의 $\lVert\nabla\text{sdf}\rVert\approx1$ 가정.) $J_i$의 gradient도 동일한 metric 스케일.

---

## 3. TSDF 샘플러 (`TsdfSampler`)

월드 점 $x$ + 법선 $n$ → $(c,\ \hat g,\ \text{valid})$.

1. **방향 레이어 선택:** $d=\text{signedAxisIndex}(\text{dominantAxis}(R\,n_i))$ — 프레임 점의 (현재 R로 회전된) 법선의 지배 부호축. `directional_tsdf_integrate.comp`의 `selectDirections` top-1과 동일 규약. (GN 반복마다 $R$이 바뀌므로 $d$는 매 반복 재계산 — 저렴.)
2. **주소 계산(block 경로):** `directional_tsdf_extract.comp`의 조회 로직 재사용 — 월드 voxel + $d$ → group key → `indexGrid` 오프셋 → pool slot → 그룹 내 voxel 오프셋 → `GpuTsdfVoxel`. (index grid가 `kInvalidPoolIndex`면 미상주/미관측 → invalid.)
3. **값·gradient:** $c=\text{sumDW}/\text{sumW}$, $\hat g=\text{normalize}(\text{sumN})$.
4. **유효성 게이트:** `sumW < MIN_WEIGHT` → invalid; $|c|\ge1$(밴드 밖/포화) → invalid; $\lVert\text{sumN}\rVert$ 퇴화 → invalid.
5. **1차 샘플:** $r=c\,\tau+\hat g\cdot(x-\text{voxelCenter})$. 해시/보간 없이 **점당 조회 1회**.

> 설계 노트: 트래킹은 현재 streaming 윈도우에 상주한 그룹만 샘플할 수 있다 — 프레임의 관측 영역은 active 윈도우가 덮으므로 정합한다(bounded-VRAM 스트리밍과의 시너지). 미상주 점은 invalid로 스킵.

---

## 4. GPU 리덕션 셰이더 (`tsdf_track_reduce.comp`)

프레임 점마다 $r_i, J_i, w_i$ 를 계산해 **정규방정식 성분을 누적**. 부동소수 atomic(확장 의존)을 피하려고 **워크그룹 단위 shared-memory 리덕션 → 워크그룹당 부분합 1개 기록 → CPU 최종합**.

- 워크그룹당 출력 = 다음 성분의 부분합:
  - $J^\top J$ 상삼각 **21개** float,
  - $J^\top r$ **6개** float,
  - 잔차 제곱합 **1개**, 인라이어 수 **1개** → 총 **29개**.
- 프레임 점 수/256 ≈ 수백 워크그룹 → CPU가 수백 개의 29-벡터를 합산(무시할 비용, 결정적).
- 입력 바인딩: `frame points`, `frame normals`, 모델의 `indexGrid`/`pool`/`meta`, push-const(현재 $T$, $\tau$, voxelSize, localBase, Huber 파라미터, MIN_WEIGHT, numPoints).

**가중치:** $w_i=\text{huber}(r_i/\sigma)$; invalid 샘플은 $w_i=0$(누적 제외). 인라이어 = $w_i>0$.

---

## 5. 호스트 트래커 (`DirectTsdfTracker`)

```
Result Track(model, points, normals, Tinit, cfg):   # 시그니처는 §5.1이 authoritative
    T = Tinit
    for iter in 0..maxIters:
        dispatch tsdf_track_reduce with (T, frame, model buffers)   # GPU
        download per-workgroup partials; sum -> A(6x6), b(6x1), ssr, inliers   # CPU
        if inliers < minInliers or A ill-conditioned: return {valid=false, ...}
        xi = solve A xi = -b   (LDLT)
        T = exp(hat(xi)) * T
        if ||xi|| < eps: break
    return {valid=true, T, fitness = inliers/numPoints, meanResidual}
```

- **모션 prior:** 호출자가 `Tinit`을 등속 예측 $T_{prev}\,(T_{prev-1}^{-1}T_{prev})$로 제공(초기 몇 프레임은 $T_{prev}$ 또는 identity).
- **수렴:** $\lVert\xi\rVert<\varepsilon$ 또는 `maxIters`(기본 8). 반복마다 GPU 재디스패치(갱신된 $T$).
- **유효성:** 인라이어 비율·$A$ 조건수(rcond) 게이트 → 실패 시 `valid=false`(D의 relocalization 트리거용).

### 5.1 인터페이스 (writing-plans가 정확히 쓸 시그니처)
```cpp
namespace Engine::Odometry {
    struct TrackResult {
        bool valid = false;
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        float fitness = 0.0f;       // inliers / numPoints
        float meanResidual = 0.0f;  // metric, mm
        int   iterations = 0;
    };
    struct TrackConfig {
        int   maxIters = 8;
        float convEps = 1e-5f;      // ||xi|| stop
        float huberDelta = 1.0f;    // * voxelSize (metric)
        int   minInliers = 200;
        float rcondMin = 1e-6f;
    };
    class DirectTsdfTracker {
    public:
        void Build(Engine::Core::Context& ctx);
        TrackResult Track(const Engine::Spatial::DirectionalTSDF& model,
                          const std::vector<Eigen::Vector3f>& points,
                          const std::vector<Eigen::Vector3f>& normals,
                          const Eigen::Matrix4f& Tinit,
                          const TrackConfig& cfg = {});
    };
}
```

---

## 6. 모델 버퍼 접근

트래커는 모델의 `indexGrid`/`pool`/`meta` GPU 버퍼가 필요. `IResidencyBackend`가 이미 `IndexGridBuffer()/PoolVoxelBuffer()/MetaBuffer()/LocalBase()/PoolCapacity()`를 노출하고, `DirectionalTSDF`는 세션에서 추가된 `DebugBackend()`로 backend 참조를 준다. **프로덕션 접근자**로 `DirectionalTSDF`에 `const IResidencyBackend& Backend() const`(또는 필요한 핸들만 노출하는 접근자)를 추가하고, 샘플러 셰이더가 그 버퍼들을 바인딩. 주소 계산 상수(`kLocalGroupGrid`, `kGroupDim`, `kNumDirections`, `IndexGridOffset`)는 `DirectionalTSDFTypes.h` 공유.

---

## 7. 강건성

- 밴드 밖/미상주/저가중치/퇴화-gradient 점 제거(§3.4).
- Huber M-estimator(잔차 $\sigma$=`huberDelta·voxelSize`).
- 등속 모션 prior로 수렴 반경(≈truncation 밴드) 확보.
- 조건수·인라이어 게이트로 degenerate 형상(평면만 보이는 등)에서 실패를 정직하게 보고.

---

## 8. 검증

### 8.1 합성 ATE 오라클 (헤드라인, 결정적)
`Engine::Eval::ScanDataset`/`ObjectScanner`는 analytic 표면을 trackball 궤적으로 sphere-trace해 **GT 카메라 포즈가 알려진** `{points,normals,cameraPos}` 프레임 시퀀스를 만든다. 절차:
1. 합성 시퀀스 생성(cube/sphere 등, GT 궤적 $\{T^{gt}_k\}$).
2. 첫 프레임은 GT로 seed·integrate. 이후 각 프레임: `Track(model, frame, motionPrior)` → integrate.
3. 추정 궤적 $\{T_k\}$ 대 $\{T^{gt}_k\}$의 **ATE(RMSE), RPE** 계산.
- **게이트:** ATE < ~voxelSize(합성, 노이즈 무). GT-포즈 integrate만 한 참조 재구성과 트래킹 재구성이 시각적으로 동일.
- **메타모픽:** 첫 프레임 seed 후 항등-모션(정지) 시퀀스에서 트래커가 $T\approx$identity 유지(drift ≈ 0).

### 8.2 실제 chair 시퀀스 (정성)
`scanData/frame_*.ply`로 track→integrate, 재구성 품질 + drift를 기존 FPFH 프레임-투-프레임 데모와 비교.

### 8.3 단위 테스트
- `TsdfSampler`: 알려진 평면 TSDF에서 샘플 $r$·$\hat g$가 해석값과 일치(밴드 내), 밴드 밖/미상주에서 invalid.
- Jacobian: 수치미분(central difference) 대비 해석 $J_i$ 일치(임의 점/포즈).
- 리덕션: GPU 부분합 CPU 합산 = 직접 CPU 계산과 일치(소규모 점군).

---

## 9. Open questions
- (Q1) 방향 레이어 선택을 top-1(단일 $d$)로 할지, 여러 후보 레이어를 결합할지(모서리/얇은 구조에서 안정성 vs 편향).
- (Q2) `huberDelta`·`minInliers`·`maxIters` 기본값 튜닝(합성 ATE 스윕).
- (Q3) 수렴 실패 시 단일→피라미드 승격 기준(어떤 ATE/모션 크기에서).
- (Q4) 트래킹 대상 모델 = active streaming 윈도우 전체인지, 최근 관측 서브영역인지(B와의 접점).
- (Q5) 프로덕션 backend 접근자 형태(`Backend()` 전체 vs 핸들만).

---

## 10. 파일 구조 (writing-plans 입력)
- 신규: `src/Engine/Odometry/DirectTsdfTracker.{h,cpp}`, `src/shader/tsdf_track_reduce.comp`(+ 공유 샘플러 헬퍼, 필요 시 `tsdf_sampler.glsl`).
- 수정: `src/Engine/Spatial/DirectionalTSDF.h`(프로덕션 backend 접근자), `src/Engine/Spatial/DirectionalTSDFTypes.h`(샘플러가 쓰는 주소 상수는 이미 존재 — 재사용).
- 테스트: `test/test_directTsdfTracker.cpp`(샘플러/Jacobian/리덕션/합성 ATE), 데모: `example2/odometry_demo.cpp`(선택).
- 참조 재사용: `directional_tsdf_extract.comp`(조회 주소 계산), `Engine::Eval::ScanDataset`(ATE 오라클).

## 11. 참조
- Bylow, Sturm, Kerl, Cremers, "Real-Time Camera Tracking and 3D Reconstruction Using Signed Distance Functions", RSS 2013.
- Canelhas, Stoyanov, Lilienthal, "SDF Tracker", IROS 2013.
- Newcombe et al., "KinectFusion", ISMAR 2011 (GPU ICP reduction 패턴).
- 내부: [`2026-07-26-highprecision-submap-tsdf-design.md`](2026-07-26-highprecision-submap-tsdf-design.md)(stored-gradient/streaming 근거), `2026-07-24-directional-tsdf-stored-gradient-design.md`, 구현: `src/Engine/Spatial/DirectionalTSDF.*`, `src/shader/directional_tsdf_{integrate,extract}.comp`.

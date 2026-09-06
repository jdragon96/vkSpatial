# ICP 정합 품질 개선 (Local ICP Registration-Quality)

로컬 ICP 트래커(`GpuIcpTracker` = `"icp"`, `PointToPlaneIcpTracker` = `"icp-cpu"`)의
정합 품질을 **3단계(Tier)로 누적 개선**한 작업 정리.

- **브랜치:** `feature/icp-registration-quality`
- **커밋 범위:** `fc19da3..0a56fa5` (baseline + Tier 1~3 + 측정체계 + 최종 보강)
- **적용 대상:** GPU 경로(`GpuPointToPlaneIcp` + `kernel_icp_iterate.comp.glsl`)와 CPU 경로
  (`AlignPointToPlaneIcp`) **양쪽 모두**, 그리고 두 경로가 공유하는 `RegistrationThread`
- **플랫폼:** macOS / MoltenVK / Apple M4 Max (UMA). GPU float atomic이 없어 정수 고정소수점 리덕션 사용
- **관련 문서:** [ICP_LOCAL_VS_GLOBAL.md](ICP_LOCAL_VS_GLOBAL.md), [ICP_METHODS.md](ICP_METHODS.md)

---

## 1. 문제 정의

기존 로컬 ICP는 정합 정확도가 낮았고, 그 **원인을 영향도 순으로** 정리하면:

1. **타깃이 양자화된 복셀 중심(voxel center)이었다.** 두 트래커 모두 타깃 점군을
   `AdvancedEntry.center`(맵 격자에 스냅된 복셀 중심, 예: 0.5 m)로 만들었다. `AdvancedEntry`는
   `tsdf`(정규화 부호거리)와 `normal`(저장된 gradient)도 갖고 있어 실제 표면은
   `center − tsdf·truncationDistance·normal`에 있는데, `tsdf`를 버리면서 정합 정밀도가
   **복셀 절반 수준**에 고정됐다.
2. **대응(correspondence) 게이팅이 느슨한 hard-cutoff였다.** `maxCorrespondenceDistance = 2 × voxel`
   의 이진(accept/reject) 판정에 robust weighting이 없어, 잘못된 대응을 그대로 동등하게 신뢰했다.
3. **타깃 노멀이 거칠었다.** 복셀화된 모델의 노멀은 point-to-plane 방향성을 부정확하게 만들었다.
4. **prior가 직전 pose뿐이었다.** `RegistrationThread`가 `previousPose`만 seed로 줘서, 실제
   카메라 움직임에서는 ICP가 정답에서 먼 지점에서 출발했다.

또한 **정량 품질 지표가 없었다** (`RegistrationResult`에는 inlier 비율 `fitness`만 존재) —
개선을 측정할 수단부터 필요했다.

---

## 2. Tier 0 — 측정 체계 (가장 먼저 구축)

각 Tier의 효과를 증명하려면 계측기가 먼저 있어야 한다. **서로 다른 것을 재는 2종의 RMSE**를 둔다.

### 2.1 재구성-vs-정답 RMSE (1차 품질 게이트)
기존 `Engine::Eval::RmseMetrics`(`NearestNeighbourRMSE`, `AccuracyRMSE`)를 **재사용**.
정합된 소스가 실제 표면에 얼마나 정확히 안착하는지를 측정한다. CPU float 연산이라 정밀도 상한이 없다.

### 2.2 ICP residual RMSE (2차 / 실시간 프레임 신호)
`RegistrationResult`에 `float rmse` 추가. `sqrt(sumOfSquaredResiduals / numInliers)`이며 각 잔차는
point-to-plane 거리 `(변환된_소스점 − 타깃점) · 타깃노멀`.

- **GPU** (`kernel_icp_iterate.comp.glsl`): 워크그룹당 고정소수점 리덕션이 기존 28 슬롯
  (21 상삼각 정규행렬 + 6 우변 + 1 inlier)이었는데, **29번째 슬롯**에 `Σ residual²`를 누적
  (동일 SCALE=10000; centred 프레임에서 크기가 작아 int32 오버플로 없음).
- **CPU** (`AlignPointToPlaneIcp`): inlier 카운트 옆에 `sumOfSquaredResiduals`를 함께 누적.

> **고정소수점 바닥(floor):** SCALE=10000이라 `|잔차| < ~7 mm`는 0으로 반올림된다. residual RMSE는
> **2차 지표**이므로 허용 — 1차 게이트는 CPU float의 재구성-vs-정답 NN RMSE다.

### 2.3 Perturbation-recovery 하네스 (전체 회귀 게이트)
결정론적(seeded) 벤치마크 테스트(`test/test_gpuIcp.cpp`):
- **현실적 모델 구축:** 다평면 코너 표면을 `AdvancedEntry`로 만들되 `center`는 거친 복셀 격자에 스냅,
  `tsdf`/`normal`은 실제 sub-voxel 표면을 인코딩(+미세 노이즈) → 양자화 문제와 Tier 1이 쓰는
  sub-voxel 데이터를 **동시에** 실험. 코너의 apex는 격자 배수에서 일부러 벗어나(`0.37,-0.22,0.29`)
  `tsdf`가 0이 되지 않게 해 진짜 양자화 오차를 만든다.
- **알려진 SE(3) 섭동**을 가한 소스 프레임 → 트래커의 `Track()`을 직접 구동(단순 `Solve`가 아님)해
  sub-voxel 타깃 구성(Tier 1) + robust/rejection/annealing(Tier 2~3)을 **end-to-end**로 실험.
- 보고: **복원 pose 오차**(translation norm, rotation angle) + `NearestNeighbourRMSE` + residual RMSE.
- 각 Tier는 이 수치를 **줄이거나 유지**해야 한다.

### 2.4 라이브 RMSE
뷰어 stats 패널에 최신 트래킹 RMSE 표시(`PipelineStats.trackerRmseAvg`, `RegistrationThread`의 러닝 평균).

---

## 3. Tier 1 — Sub-voxel 타깃 (CPU-side, 셰이더 변경 없음)

두 트래커의 타깃 구성에서 raw center를 저장된 sub-voxel 표면점으로 교체:

```
surfacePoint = entry.center − entry.tsdf * truncationDistance * entry.normal
```

- `ModelSnapshot`에 `float truncationDistance` 추가, `IntegrationThread::buildSnapshot`에서
  맵 config의 truncation(integrate 셰이더와 동일 값)으로 채움. `float voxel`도 함께 전달해
  트래커가 대응 거리를 맵 해상도에 맞춰 스케일할 수 있게 함.
- `tsdf`는 [−1, 1]로 정규화(`voxel2point / truncateDistance`)되어 있으므로 월드 offset은
  `tsdf * truncationDistance`. **부호**는 integrate/extract 컨벤션에 맞춰 검증(하네스의 복원오차가
  부호 오류를 즉시 드러냄).
- GPU 경로는 이미 투영된 표면점을 업로드하므로 **셰이더 변경 불필요**.

---

## 4. Tier 2 — Robust 대응

- **Huber robust weighting:** 대응마다 가중치 `w(residual)` — `|residual| ≤ huberScale`이면 `1`,
  넘으면 `huberScale / |residual|` — 를 정규행렬·우변 기여 양쪽에 적용(기존 hard 이진 accept 대체).
  `huberScale ≈ voxel`(튜너블).
- **노멀 호환성 rejection:** `sourceNormal · targetNormal < cos(compatibilityAngle)`(기본
  `normalCompatibilityCosine = 0.5`, ~60°)이면 대응 제거. 이를 위해 GPU에 **소스 노멀 스토리지 버퍼
  (binding 6)** 추가 — `GpuPointToPlaneIcp`가 `frame.nrm`을 업로드하고 셰이더에서 호환성 검사.
  CPU 경로는 소스 노멀을 직접 사용. (소스 노멀이 비어 있으면 rejection은 건너뜀 = 하위호환.)
- **기본 게이트를 조임:** Tier 3의 스케줄이 고정 게이트를 대체하기 전, 기본 대응 거리를 축소.

---

## 5. Tier 3 — Coarse-to-fine annealing + 모션 모델

### 5.1 Coarse-to-fine annealing (per-solve grid hoist 유지)
per-iteration grid 재생성을 다시 만들지 않도록 hoist와 조화:
- 대응 grid를 **가장 거친(가장 넓은 annealed 거리) 셀 크기로 한 번만** 생성(기존 hoist 그대로).
- per-iteration `currentMaxCorrespondenceDistance`를 **push-constant**로 전달 — 넓게(~2–3×voxel,
  큰 수렴 basin) 시작해 반복마다 기하급수적으로 ~0.5–1×voxel까지 축소. grid의 이웃 스캔은 고정,
  **거리 필터(와 Huber scale)만** 매 반복 조인다 → solve당 업로드/grid-build 1회 유지.
- CPU `AlignPointToPlaneIcp`도 동일: grid는 가장 거친 셀, per-iteration 거리 필터.
- GPU/CPU가 **동일 스케줄**을 쓰도록 `RegistrationTypes.h`의 `inline AnnealIcpIteration(...)`를 공유.
  `minCorrespondenceDistance = 0`이면 고정 값으로 단락(short-circuit) = 기존 동작.

### 5.2 상수속도(constant-velocity) 모션 모델 (`RegistrationThread::Run`)
- `previousPose`와 `previousPreviousPose`가 모두 유효할 때 prior를
  `previousPose * (previousPreviousPose.inverse() * previousPose)`(직전 상대운동 재적용)로 seed.
  처음 두 프레임 또는 트랙 실패 후에는 `previousPose`로 fallback.
- `RegistrationThread`(트래커 무관)에 있어 **모든 로컬 트래커가 동시에** 이득.

---

## 6. GPU/CPU 수치 일관성

두 경로가 모든 Tier에서 수치적으로 동일해야 한다(바인딩 요구사항):
- **centring이 translation-only**이므로 centred `mat3(g_T)` ≡ 월드 `R` → 노멀 rejection 검사
  `dot(R·srcN, n)`이 양쪽 동일.
- 잔차 `e=(p−q)·n`, Huber 가중치(`|e|`의 함수), rejection 모두 translation-invariant.
- residual 누적은 양쪽 **unweighted**.
- annealing 스케줄은 공유 `AnnealIcpIteration`로 분기 불가.

**테스트 보증:**
- `GpuIcp.SolveMatchesCpuOnCorner` — clean 코너에서 `gpu.T ≈ cpu.T`(tol 5e-3).
- `GpuIcp.ResidualRmseMatchesCpu` — residual RMSE 일치.
- `GpuIcp.RobustPathMatchesCpuOnNoisyFixture` — **노이즈/아웃라이어** 픽스처에서 Huber 감쇠와
  노멀 rejection이 **실제로 발동**하는 상태로 `gpu.T ≈ cpu.T` 보증. 측정된 GPU-vs-CPU 최대 절대차
  **0.000055**(tol 5e-3, ~90× 여유) → robust/rejection 경로에서도 일관성 확인.

---

## 7. 측정 결과

결정론적 perturbation 하네스 기준.

| 단계 | 픽스처 / 섭동 | translation 오차 | 재구성 NN RMSE |
|---|---|---|---|
| **Baseline** (Tier 0, raw center) | 코너 apex off-grid | `0.03001` | `0.02237` |
| **+ Tier 1** (sub-voxel 타깃) | 동일 | `0.00006` | `0.00005` |
| **+ Tier 2** (robust, non-robust 대비) | 노이즈+아웃라이어 | `0.00160 → 0.00075` | `0.00084 → 0.00054` |
| **+ Tier 3** (annealing, 고정게이트 대비) | 대섭동 0.22 m + 0.25 rad | `0.20686 → 0.00007` (GPU) / `0.0` (CPU) | — |

- **모션 모델:** 직선 운동 시퀀스에서 예측 prior 오차 `0.10 → 0.0` (직전-pose prior 대비)
  — `RegistrationThread.ConstantVelocityPriorBeatsPreviousPoseOnStraightLine`.
- **전체 스위트:** **241 passed / 1 skipped / 0 failed** (`vkspatial_tests`).

---

## 8. 파일별 변경

| 파일 | 변경 |
|---|---|
| `src/Pipeline/Registration/RegistrationTypes.h` | `RegistrationResult.rmse`; `RegistrationParam`에 `huberScale`, `normalCompatibilityCosine`, `minCorrespondenceDistance`; 공유 `AnnealIcpIteration` |
| `src/Pipeline/Registration/kernel_icp_iterate.comp.glsl` | 29-slot 리덕션(Σe² slot 28); 소스 노멀 binding 6; `g_huberScale`/`g_normalCompatibilityCosine`/per-iter `g_maxCorr` push constant |
| `src/Pipeline/Registration/GpuPointToPlaneIcp.{h,cpp}` | annealed 거리로 `dispatchCentred`; `IterOut.sumOfSquaredResiduals`; 소스 노멀 버퍼 |
| `src/Pipeline/Registration/PointToPlaneIcp.h` | `AlignPointToPlaneIcp(src, sourceNormals, tgt, priorT, params)` — Huber + rejection + annealing; grid 1회 생성 |
| `src/Pipeline/Registration/GpuIcpTracker.cpp`, `PointToPlaneIcpTracker.cpp` | sub-voxel 타깃 구성; `frame.nrm` 전달; `huberScale = model->voxel`; annealing opt-in 주석 |
| `src/Pipeline/Types.h` | `ModelSnapshot.voxel`, `.truncationDistance`; `PipelineStats.trackerRmseAvg` |
| `src/Pipeline/Integration/IntegrationThread.cpp` | `buildSnapshot`에서 `voxel`/`truncationDistance` 채움 |
| `src/Pipeline/Registration/RegistrationThread.cpp` | 상수속도 모션 모델; `m_trackerRmse` 러닝 평균 |
| `test/test_gpuIcp.cpp`, `test/test_pipeline.cpp` | perturbation 하네스(+노이즈/annealing 변형), GPU≡CPU 가드, 모션 모델 테스트 |

---

## 9. 사용 / 설정

- 트래커 선택은 독립 전략(`identity` / `icp` / `icp-cpu` / `global`), `TrackerRegistry::Default()` 허브.
- **프로덕션 트래커는 Tier 1~2 + 모션 모델이 활성**. Tier 3 annealing은 **의도적으로 dormant**:
  트래커가 `minCorrespondenceDistance`를 설정하지 않기 때문(고정 게이트).
  - **활성화하려면:** `minCorrespondenceDistance > 0`으로 설정 **AND** `maxCorrDist`를 넓혀 큰 수렴
    basin 확보.
  - **비용:** `maxCorrDist`를 넓히면 GPU `LocalGrid` 셀 수/메모리가 커진다(라이브 게이트를 2×voxel로
    유지하는 이유). scan_out 같은 대형 씬에서 게이트가 과도하면 셀 폭발로 수 GB 할당 → `LocalGrid`에
    방어적 셀 상한 존재.
- 확인 예: `./build-rel/example2/voxel_fill_debugger --dir scan_out --voxel 0.5` (헤드리스; 트래커를
  거는 뷰어 경로는 취득 축에서 PLY가 빠지면서 없어졌다 — 트래커 A/B는 `icp_quality_diag --replay`로)

---

## 9.5. 재현성 — 파이프라인을 통과하는 측정의 전제 (2026-08)

**§7의 수치는 전부 결정론적 하네스(테스트 픽스처)에서 나온 것이고, 파이프라인을 통과하는 측정에는
그대로 적용되지 않았다.** `capture/`(477프레임 D435 녹화)에서 동일 명령
`icp_quality_diag --replay capture --voxel 0.05 --trackers icp`를 네 번 돌린 결과:

| 실행 | path length | step max | turn max | rejected | noModel/noLocal/fewInliers/lowOverlap |
|---|---|---|---|---|---|
| A | 1.47 m | 0.1235 | 4.42° | 233 | 6 / 0 / 227 / 0 |
| B | 6.45 m | 0.2308 | 22.19° | 233 | 6 / 28 / 93 / 106 |
| C | 7.84 m | 0.7683 | 46.59° | 232 | 6 / 2 / 156 / 68 |
| D | **136.76 m** | **40.16** | **179.88°** | 267 | 6 / 47 / 44 / 170 |

**원인:** 블로킹 채널(`3e8a6e7`)은 프레임 *개수*만 맞춘다. 맵은 latest-wins `Mailbox`로 트래커에
전달되고 정합은 `trackedFrames` 용량(4)만큼 융합보다 앞서 달릴 수 있어서, 프레임 N이 *어느 버전의
맵*에 정합하는지가 쓰레드 스케줄링에 달렸다. 그 맵이 정합 타깃이므로 포즈가 달라지고 → 다음 맵이
달라진다 → 발산한다.

**따라서 `96402c4`의 `minFitness` 스윕(0.2→166k, 0.4→50k, 0.6→67k)은 신호가 아니라 노이즈다** —
비단조성 자체가 그 증거였다.

**수정 2건:**
1. `FrameHandshake`(`Pipeline/CommunicationModule.h`) — 녹화 모드에서 정합이 매 프레임 융합 완료를
   기다린다(lock-step). 프레임 N은 항상 0..N-1을 담은 맵에 정합한다.
   테스트 `Pipeline.ALosslessReplayAlignsEachFrameAgainstEveryEarlierFrame`가 관측된 맵 인덱스
   수열이 정확히 `-1, 0, 1, 2, …`임을 단언한다("두 실행이 일치"보다 강한 조건 — 타이밍 버그는
   운으로 두 실행을 일치시킬 수 있지만 이 수열은 매 프레임 실제로 기다렸을 때만 성립한다).
2. **타깃 정규 순서**(`SortTargetIntoCanonicalOrder`, `RegistrationTypes.h`) — TSDF compaction 커널이
   `atomicAdd(g_count, 1u)`로 append하므로 `ModelSnapshot::entries` 순서가 매 실행 다르다. 그 순서가
   두 경로로 결과에 샌다: `Solve`가 타깃 centroid를 float으로 합산하므로(비결합적) 10만 점 centroid가
   마지막 비트에서 흔들리고, 모든 잔차가 `int(round(x*SCALE))`로 양자화되므로 마지막 비트 차이가
   고정소수점 기여의 일부를 뒤집는다. 그리고 `LocalGrid::Nearest`의 정확한 거리 동점은 먼저 방문한
   후보가 이긴다. centroid 합산은 double로도 바꿨다.

**수정 후 확인** (같은 바이너리로 3회):

| 실행 | frames | entries | step avg | step max | turn max | path | rejected | skipped |
|---|---|---|---|---|---|---|---|---|
| 1 | 476 | 1,088,202 | 0.0475 | 0.2592 | 19.23° | 11.731 | 230 | 229 |
| 3 | 476 | 1,088,202 | 0.0475 | 0.2592 | 19.23° | 11.731 | 230 | 229 |
| 2 | *348* | 933,170 | — | — | — | — | 166 | 165 |

완주한 두 실행은 **모든 보고 수치가 동일**하다. 실행 2가 다른 이유는 발산이 아니라
**도구의 180초 타임아웃**에 프레임 348에서 걸린 것 — lock-step이 두 단계의 중첩을 없애므로
replay가 두 단계 비용의 *합*으로 느려진다(align 85→333 ms). 타임아웃을 1800초로 올렸다.
**이것이 lock-step의 실제 비용이다: 재현성을 얻는 대신 replay wall-clock이 대략 2배가 된다.**
라이브 센서는 handshake를 켜지 않으므로 영향 없다.

또한 거부 원인 분포가 정리됐다: 수정 전 `6/28/93/106`처럼 네 원인에 흩어져 있던 것이
수정 후 `1 / 0 / 0 / 229`(noModel 1, lowOverlap 229)로 모인다. 스케줄링이 만들던
`NoLocalTarget`·`TooFewInliers`가 사라지고 **남은 실패는 전부 하나의 실제 원인** — 프레임 대비
대응점 비율이 `minFitness` 0.4를 못 넘기는 것 — 이라는 뜻이다. 이제 이 하나를 공략할 수 있다.

## 9.6. 진동의 근본 원인 — 상수속도 prior (2026-08-18)

§9.5의 재현성 수정 후에도 남던 "477 중 229 거부"의 원인을 계측으로 특정하고 고쳤다.

**진단 경로** (모두 결정론 replay라서 유효):
1. 거부 프레임 분포를 찍자 **정확히 홀수 프레임마다 교대** — 주기-2 진동 (229개 전부 길이-1 run).
2. fitness 분포가 완전 이봉: 채택 median 0.87 / 거부 median 0.16, 겹침 없음 — "게이트 경계선" 기각.
3. `--min-fitness` 0.4→0.2→0.1 스윕: 게이트를 낮추면 궤적이 **악화**(path 6→17→43 m) — "새 영역 프레임을 게이트가 억울하게 자른다" 기각.
4. 트래커 계측: 거부 프레임은 **좋은 prior에서 solve가 0.1~0.26 m 튐**(제2 어트랙터), 같은 prior의 이웃 프레임은 정상 수렴 — 교대하는 것은 prior 모드(직전포즈 vs 상수속도)뿐.
5. 상수속도 prior를 끄자 **476/476 추적** — 확정.

**메커니즘 2겹:**
- **재무장 버그**: 거부 후 첫 채택에서 상수속도가 재무장되는데, 그때의 delta는 2프레임 간격이라 1프레임처럼 재적용하면 과외삽 → 다음 solve 발산 → 거부 → 반복 (주기-2).
- 재무장을 "연속 2회 채택 후"로 고쳐도(1-간격 delta 보장) **주기-3으로 재발**(153 거부): 이 데이터(~5 mm/frame)에서는 delta가 포즈 추정 노이즈에 지배되어, 외삽이 노이즈를 재적용하는 것과 같다. §5.2가 적어둔 전제 그대로 — 모션 모델은 "per-frame motion이 대응 게이트 대비 클 때" 가치가 있다.

**수정 3건** (테스트 각각: `RegistrationThread.VelocityPriorNeedsTwoConsecutiveAdoptionsAfterARejection`, `RegistrationThread.SlowMotionUsesThePreviousPosePrior`, `Pipeline.GpuIcpTrackerRejectsAPhysicallyImplausibleStep`):
1. 상수속도 prior는 **연속 2회 채택** 후에만 재무장 (1-간격 delta 보장).
2. 상수속도 prior는 **직전 step > 2 cm**일 때만 사용 (`kVelocityPriorMinimumStepMeters`; 그 아래에선 직전 포즈가 이미 basin 안이고 외삽은 노이즈만 보탬). straight-line 픽스처(0.1 m/frame)는 임계 위라 모델 유지.
3. **물리 스텝 게이트** `RegistrationParam::maxStepMeters` (기본 0=off, 트래커 기본 `kDefaultTrackerMaxStepMeters`=0.08): prior에서 그보다 멀리 간 solve는 fitness와 무관하게 `ETrackFailure::ImplausibleMotion`으로 거부. 실측된 오수렴이 fitness 0.571로 게이트를 통과해 맵을 오염시킨 사건의 2차 방어선.

**capture/ 결과** (voxel 0.05, before → after):

| | 거부 | step avg | step max | turn max | path | entries |
|---|---|---|---|---|---|---|
| before (p2p+prefilter5) | **229** | 0.0246 | 0.2253 | 5.06° | 6.081 | 387,950 |
| **after (p2p+prefilter5)** | **1** (NoModel 부트스트랩) | 0.0046 | **0.0404** | 5.77° | **2.204** | 211,431 |
| after (p2p, 원본 depth) | 1 | 0.0049 | 0.0336 | 4.32° | 2.337 | 245,924 |

step max가 30 fps 핸드헬드 물리 한계(0.05 m) 안으로 들어왔고, 원본 depth의 p2p도 이제 전 프레임 추적된다
(좁은 basin 문제의 대부분이 prior 유발이었다는 뜻; 프리필터는 맵 크기 211k vs 246k로 여전히 유리).
`scan_out` identity 무회귀(거부 0), 전체 스위트 241 passed.

**Local→Global fallback에 대한 판정**: 위 수정 후 이 데이터에는 구할 실패가 남지 않는다(거부 1 = 맵 없음).
global 재정위가 의미 있는 것은 *건강한 맵에서 연속 N프레임 실패*(진짜 tracking lost)가 관측될 때이고,
그때의 올바른 자리는 `TrackerRegistry`에 새 이름으로 등록하는 composite 트래커(local 시도 → 연속 실패 시
global)다 — 파이프라인 구조 변경이 아니라 트래커 하나 추가.

### 9.7 composite 트래커 `icp+global` (구현됨)

위 판정대로 `RelocalizingIcpTracker`(레지스트리 이름 `icp+global`)로 구현했다. 동작:

- 평상시엔 내부 `GpuIcpTracker`에 위임. **건강한 맵 대비 실패**(`TooFewInliers`/`LowOverlap`/
  `ImplausibleMotion`)만 연속 실패로 카운트한다 — `NoModel`/`NoLocalTarget`은 재정위할 맵이 없는
  상태이므로 카운트하지 않는다(부트스트랩에서 global이 발동하면 안 됨,
  `Pipeline.RelocalizingTrackerDoesNotFireGlobalWhileBootstrapping`).
- 연속 `kDefaultFailuresBeforeGlobal`(=3)회 실패 시 prior-free global 정합
  (`GlobalRegistrationTracker` = FPFH+RANSAC+Ceres, full-model 타깃)을 1회 시도하고, 그 포즈를
  시드로 local ICP를 다시 돌려 **일반 local 게이트를 통과할 때만 채택**한다. step 게이트는 특별
  취급이 필요 없다 — refine의 prior가 global 포즈 자체라서 큰 재정위 점프는 게이트에 보이지 않는다
  (설계 초안의 "게이트 해제" 가정은 뮤테이션 테스트로 죽은 코드임이 확인되어 제거).
- refine이 실패하면 **local의 실패 원인을 그대로 반환**한다(절대 fusible한 원인으로 위장하지 않음).
  카운터는 시도 직전에 리셋되어 실패한 시도는 N회 실패를 새로 채워야 재시도된다(CPU 비용 스로틀,
  `Pipeline.RelocalizingTrackerThrottlesFailedGlobalAttempts`).
- 관측: `TrackerStats.relocalizationAttempts`/`.relocalizationSuccesses`.

같은 작업에서 단독 `global` 트래커의 버그 3건도 수정: (1) 실패 시 `failure`를 안 채워 기본값
`NoModel`로 보고 → `ShouldFuse`가 융합해버림(이제 `TooFewInliers`/`LowOverlap` 분류,
`Pipeline.GlobalTrackerFailureIsNotFusible`), (2) mm-스케일 기본 `voxelSize`(5.0)를 `model->voxel`로
스케일하지 않음(`Pipeline.GlobalTrackerScalesToModelVoxelAndRecovers`), (3) 타깃을 raw
`entry.center`로 구성 → sub-voxel surface point로 교정.

### 9.8 step 게이트의 복셀 스케일링 (실측 검증 포함)

`kDefaultTrackerMaxStepMeters = 0.08`은 **절대 단위 상수**(미터, 30 fps 핸드헬드)라서 맵 단위가
미터가 아니거나 복셀이 굵으면 solve 지터만으로도 게이트를 넘는다. 실측: scan_out(extent 247,
voxel 0.5)에서 90프레임 중 76개, scanData(mm 단위, voxel 5.7)에서 60프레임 중 59개가
`ImplausibleMotion`으로 거부되어 재구성이 부트스트랩 프레임 근처에서 멈췄다.

수정: **기본 게이트를 `max(0.08, 1.6 × model->voxel)`로 스케일**(`kDefaultTrackerMaxStepVoxels
= 1.6` = capture/ 튜닝비 0.08/0.05). 명시적 `SetMaxStepMeters`는 절대값 그대로 존중
(`Pipeline.GpuIcpTrackerScalesDefaultStepGateToMapVoxel`).

세 데이터셋 실측 검증 (icp 트래커, before → after):

| 데이터셋 | 거부(implausible) | 판정 |
|---|---|---|
| capture/ (실캡처, m, voxel≤0.05) | 1(NoModel)/0 → 1/0, step max 0.0095 | **비트 동일** — 게이트 max(0.08, 1.6×0.05)=0.08, 튜닝 보존 |
| scan_out (합성, 사전 정합, voxel 0.5) | 76 → 41, 융합 13→48프레임 | 게이트는 열렸지만 채택 포즈가 드리프트(path 10.9, 참값 0) — **애초에 identity가 정답인 데이터** |
| scanData (실스캔, 사전 정합, mm, voxel 5.7) | 59 → 59 | solve가 9.2mm 게이트도 넘는 오수렴 — **게이트가 의도대로 가비지를 차단** |

부수 발견 2건: (1) **scanData도 scan_out처럼 사전 정합돼 있다** — frame 간 NN 거리 중앙값
0.00mm(p90 0.81), centroid 이동 10.9mm는 모션이 아니라 커버리지 차이다. 트래커 기준선은
identity. (2) 사전 정합 데이터에 icp를 걸면 게이트가 "정지된 깨끗한 모델"과 "움직이는 드리프트
모델" 중 하나를 고르게 될 뿐이다 — 게이트 문제가 아니라 도구 선택 문제.

## 10. 남은 후속 과제 (parked, 비차단)

최종 리뷰에서 병합 비차단으로 분류된 항목:

- **sub-voxel 공식이 두 트래커에 중복** → 공유 헬퍼로 추출.
- `GpuIcpTracker`가 `entry.center`로 crop하는데 sub-voxel 보정이 `2×voxel` 마진을 초과할 수 있음
  (narrow-band 경계 엔트리 누락 가능; `|tsdf|≈1`인 가장 신뢰도 낮은 점이라 영향 미미) →
  보정된 surface point로 crop하거나 마진을 `maxCorrDist + truncation`으로 확대.
- `GlobalRegistrationTracker.rmse`가 inert(상위 `Estimate`가 `rmse`를 안 채움) — 표시 전용, 무해.
- GPU residual RMSE 고정소수점 바닥(~7 mm) — 2차 지표 한계, 문서화됨.
- ~~상수속도 모델이 프레임 드롭 직후 1 구간 과외삽~~ — §9.6에서 수정: 실데이터에서는 단일 프레임이 아니라 자기유지 진동이었다.

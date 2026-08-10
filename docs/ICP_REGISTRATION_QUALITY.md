# ICP 정합 품질 개선 (Local ICP Registration-Quality)

로컬 ICP 트래커(`GpuIcpTracker` = `"icp"`, `PointToPlaneIcpTracker` = `"icp-cpu"`)의
정합 품질을 **3단계(Tier)로 누적 개선**한 작업 정리.

- **브랜치:** `feature/icp-registration-quality`
- **커밋 범위:** `fc19da3..0a56fa5` (baseline + Tier 1~3 + 측정체계 + 최종 보강)
- **적용 대상:** GPU 경로(`GpuPointToPlaneIcp` + `icp_iterate.comp.glsl`)와 CPU 경로
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

- **GPU** (`icp_iterate.comp.glsl`): 워크그룹당 고정소수점 리덕션이 기존 28 슬롯
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
| `src/Engine/Pipeline/Registration/RegistrationTypes.h` | `RegistrationResult.rmse`; `RegistrationParam`에 `huberScale`, `normalCompatibilityCosine`, `minCorrespondenceDistance`; 공유 `AnnealIcpIteration` |
| `src/shader/icp_iterate.comp.glsl` | 29-slot 리덕션(Σe² slot 28); 소스 노멀 binding 6; `g_huberScale`/`g_normalCompatibilityCosine`/per-iter `g_maxCorr` push constant |
| `src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.{h,cpp}` | annealed 거리로 `dispatchCentred`; `IterOut.sumOfSquaredResiduals`; 소스 노멀 버퍼 |
| `src/Engine/Pipeline/Registration/PointToPlaneIcp.h` | `AlignPointToPlaneIcp(src, sourceNormals, tgt, priorT, params)` — Huber + rejection + annealing; grid 1회 생성 |
| `src/Engine/Pipeline/Registration/GpuIcpTracker.cpp`, `PointToPlaneIcpTracker.cpp` | sub-voxel 타깃 구성; `frame.nrm` 전달; `huberScale = model->voxel`; annealing opt-in 주석 |
| `src/Engine/Pipeline/Types.h` | `ModelSnapshot.voxel`, `.truncationDistance`; `PipelineStats.trackerRmseAvg` |
| `src/Engine/Pipeline/Integration/IntegrationThread.cpp` | `buildSnapshot`에서 `voxel`/`truncationDistance` 채움 |
| `src/Engine/Pipeline/Registration/RegistrationThread.cpp` | 상수속도 모션 모델; `m_trackerRmse` 러닝 평균 |
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
- 뷰어 예: `./build-rel/example2/voxel_fill_debugger --dir scan_out --voxel 0.5 --tracker icp`

---

## 10. 남은 후속 과제 (parked, 비차단)

최종 리뷰에서 병합 비차단으로 분류된 항목:

- **sub-voxel 공식이 두 트래커에 중복** → 공유 헬퍼로 추출.
- `GpuIcpTracker`가 `entry.center`로 crop하는데 sub-voxel 보정이 `2×voxel` 마진을 초과할 수 있음
  (narrow-band 경계 엔트리 누락 가능; `|tsdf|≈1`인 가장 신뢰도 낮은 점이라 영향 미미) →
  보정된 surface point로 crop하거나 마진을 `maxCorrDist + truncation`으로 확대.
- `GlobalRegistrationTracker.rmse`가 inert(상위 `Estimate`가 `rmse`를 안 채움) — 표시 전용, 무해.
- GPU residual RMSE 고정소수점 바닥(~7 mm) — 2차 지표 한계, 문서화됨.
- 상수속도 모델이 프레임 드롭 직후 1 구간 과외삽 — 복구 후 단일 프레임에만 영향, ICP가 prior를 정제.

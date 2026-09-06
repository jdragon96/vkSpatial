# 실데이터 capture/ 에서 Point-to-Plane과 정합 살리기

> **한 줄 요약:** 실제 D435 핸드헬드 녹화(`capture/`, 477프레임)에서 point-to-plane TSDF와
> ICP 정합이 둘 다 동작하지 않던 것을, 결함 6건을 계측으로 특정해 고쳤다.
> **477 중 229프레임 거부·비물리적 궤적 → 476/477 추적·물리적 궤적.**

- **커밋:** `d4a1f8d..68b16e1` + `3a467c2` (2026-08-17~18)
- **데이터:** `capture/` — 640×480 float32 depth(미터), fx=fy=383.18, 30 fps, depth 0.25–6.8 m,
  카메라 이동 ~5 mm/frame, **입사각 median 44° / p90 67°**, 이웃 법선 흔들림 median 24°
- **상세 문서:** [`ADVANCED_TSDF.md`](ADVANCED_TSDF.md) §3(밴드 버그),
  [`ADVANCED_TSDF_A1_A2.md`](ADVANCED_TSDF_A1_A2.md)(capture 실측),
  [`ICP_REGISTRATION_QUALITY.md`](ICP_REGISTRATION_QUALITY.md) §9.5–9.6(재현성·진동)

---

## 최종 결과

`icp_quality_diag --replay capture --voxel 0.05 --trackers icp` (결정론 replay):

| | 거부 | step avg | step max | turn max | path | entries |
|---|---:|---:|---:|---:|---:|---:|
| **수정 전** | **229** | 0.0246 | **0.225 m** | 19.2° | 6.08 m | 388k~1,088k |
| **수정 후 (p2p + `--prefilter 5`)** | **1**¹ | 0.0046 | **0.040 m** | 5.8° | 2.20 m | 211k |
| 수정 후 (p2p, 원본 depth) | 1¹ | 0.0049 | 0.034 m | 4.3° | 2.34 m | 246k |

¹ 프레임 0의 `NoModel`(맵이 아직 없음) — 구조적으로 불가피.
step max가 30 fps 핸드헬드 물리 한계(~0.05 m/frame) **안**으로 들어왔고, 전 프레임 fitness 0.79–0.93.

TSDF 품질(GT 있는 `scan_out`, voxel 0.5, `tsdf_folder_eval`):

| | accuracy mean | precision@1vox | recall@1vox |
|---|---:|---:|---:|
| 수정 전 p2p | 0.325 | 0.849 | 0.126 |
| projective | 0.305 | 0.886 | 0.125 |
| **수정 후 p2p** | **0.179 (−45%)** | **0.998** | 0.104 |

**판정 역전**: 수정 전에는 p2p가 projective보다 *나빴다*. 수정 후 41% 낫다.
(recall −17%는 제거된 접선 방향 smear의 몫 — 정확도 우선 채택.)

---

## 결함 6건 — 증상 → 진단 → 수정

### 1. p2p 밴드를 ray로 행진하며 normal로 측정 (`d4a1f8d`)

- **증상:** grazing 각도에서 truncation 밴드가 잘림 — 60°에서 coverage 83%, 75°에서 50%(실측).
  같은 평면을 0°와 75°에서 본 맵이 1045 vs 573 엔트리 — **시점 의존 맵**.
- **진단:** `kernel_AdvancedTSDF.integrate.comp.glsl`이 밴드를 **ray로 행진**하면서 소속 판정은
  **normal로 측정**. `dot(ray, n) = −cosθ` 이므로 도달 범위가 `steps·voxel·cosθ`로 축소.
  손익분기 θ=41.4° < capture/ median 44°.
- **수정:** p2p 모드에서 `unitNormal`로 행진(projective는 ray 유지). 샘플 수 불변 → 비용 0.
- **테스트:** `AdvancedTSDF.PointToPlaneMapDoesNotDependOnTheViewpoint`(byte-identical 단언 +
  projective 음성 대조군), `…BandFillsTheFullTruncationDepthAtEveryIncidence`(각도별 표).
  뮤테이션: `steps`만 늘리는 대안은 시점 의존이 남아 **여전히 빨강**.
- 자매 커널 2개(compact_directional / directional)는 살아있는 호출자가 없고 문서 수치의 기준이라
  **frozen comparator** 주석만 남김.

### 2. GT 채점이 죽어 있었음 (`e00877d`)

- `tsdf_folder_eval`의 RMSE 블록이 주석 처리 — `data/chair.ply`가 **binary PLY**인데 로더는
  ascii 전용이라 항상 스킵됐던 것. `scan_out/ground_truth.ply`(ascii, 자동 발견)로 살림.
- `GridNN` 링 탐색 32셀 상한 → 먼 GT 점이 `inf`를 반환해 **평균 전체를 오염**시키던 버그 수정.
- accuracy/completeness 평균만으로는 점수 조작 가능(점을 버리면 accuracy↑, 뿌리면 completeness↑)
  → **precision/recall/F1@1voxel** 추가.

### 3. 소스 법선이 노이즈 지배 (`b092ab7`)

- **증상:** 1픽셀 forward-difference 법선의 이웃 간 흔들림 median **24°**(매끈한 면 기준 1–3°).
  p2p는 SDF 값과 (수정 1 이후) 밴드 방향이 **둘 다** 법선에 걸린다.
- **수정:** `DepthFilterOptions::prefilterWindow` — discontinuity-aware mean(같은 면 이웃만 평균,
  flying-pixel 가드와 같은 규칙 재사용). 5×5로 24°→**2.9°**. 필터된 depth가 점·법선 **모두**에
  들어간다(법선만 필터하면 둘이 다른 표면을 기술).
- **기본값 0(off)**: capture/에서 p2p는 크게 좋아지지만 **projective는 오히려 나빠져서**(고주파
  depth에 의존) 전역 기본 전환은 기각. p2p와 짝으로 `--prefilter 5`.
- **테스트:** `DepthFrontend.ThePrefilter*` 4건. 뮤테이션 교훈: 단순 box mean은 step을 문질러도
  하류 jump 가드가 증거를 지워 위치 단언을 통과 — **점 개수 보존** 단언이 잡는다.

### 4. Replay가 무손실이지만 비결정적 (`b894f76`)

- **증상:** 동일 명령 4회 → path **1.47 / 6.45 / 7.84 / 136.76 m**. 어떤 A/B도 무의미.
- **진단:** 맵이 latest-wins `Mailbox`로 전달되고 정합이 융합보다 최대 4프레임 앞서 달림 →
  프레임 N이 *어느 버전의 맵*에 정합하는지가 쓰레드 스케줄링 소관. 그 맵이 정합 타깃이라
  카오스적으로 발산. (기존 `minFitness` 스윕 `96402c4`는 이것 때문에 노이즈였다.)
- **수정:** `FrameHandshake`(녹화 모드 lock-step: 프레임 N은 항상 0..N−1을 담은 맵에 정합) +
  타깃 정규 정렬 `SortTargetIntoCanonicalOrder`(compaction의 atomicAdd 순서가 float centroid 합산과
  NN 동점 처리로 새던 것) + centroid 합산 double화. **완주 실행 bit-identical** 확인.
- **비용:** replay wall-clock ~2배(단계 중첩 소멸). 라이브 센서는 handshake 비활성 — 영향 없음.
- **테스트:** `Pipeline.ALosslessReplayAlignsEachFrameAgainstEveryEarlierFrame` — "두 실행 일치"가
  아니라 **관측된 맵 인덱스 수열이 정확히 −1,0,1,2,…** 임을 단언(운으로 통과 불가).

### 5. 거부 프레임이 직전 포즈로 융합됨 (`b894f76`)

- **증상:** 거부된 프레임(=자인한 틀린 포즈)이 그대로 맵에 융합 → 오염된 맵이 다음 프레임의
  정합 타깃 → 자기강화.
- **수정:** `ShouldFuse()` — **원인별** 정책. `TooFewInliers`/`LowOverlap`/`ImplausibleMotion`은
  융합 안 함; `NoModel`/`NoLocalTarget`은 융합(오염시킬 로컬 맵이 없고, 거부하면 맵이 부트스트랩
  못 하거나 새 영역으로 못 자람 — 전부 거부하는 정책은 프레임 0에서 **데드락**).
  건너뛴 프레임도 frame index는 전진(완료 판정 유지), `PipelineStats::skippedFusions`로 관측.
- **테스트:** `Pipeline.AFrameWhoseOverlapGateFailedIsNotFusedIntoTheMap`(엔트리 수가 아니라
  **맵 extent** 단언 — 개수 단언은 직전-포즈 융합 버그를 통과시킴), `…AFrameWithNoMapYetIsStillFused`.

### 6. 상수속도 prior가 자기유지 진동을 만듦 (`b894f76`) — "이상한 정합"의 직접 원인

- **증상:** 재현성 수정 후 거부 229건이 전부 `LowOverlap` 하나로 수렴, 그런데 분포가
  **정확히 홀수 프레임마다 교대**(주기-2). fitness 완전 이봉(채택 0.87 / 거부 0.16).
- **진단 사슬:**
  - `--min-fitness` 0.4→0.2→0.1 스윕: 게이트를 낮추면 궤적 **악화**(path 6→17→43 m) → 게이트 무죄.
  - 트래커 계측: 거부 프레임은 **좋은 prior에서 solve가 0.1–0.26 m 튐**(제2 어트랙터), 같은
    prior의 이웃 프레임은 정상 수렴. 교대하는 것은 **prior 모드**(직전포즈 vs 상수속도)뿐.
  - 상수속도를 끄자 **476/476 추적** — 확정.
- **메커니즘 2겹:**
  1. **재무장 버그**: 거부 후 첫 채택에서 2프레임 간격 delta를 1프레임처럼 재적용 → 과외삽 →
     발산 → 거부 → 반복 (주기-2).
  2. 재무장을 고쳐도(1-간격 delta 보장) **주기-3 재발**(153 거부): ~5 mm/frame에서는 delta가
     포즈 추정 노이즈에 지배 — 외삽 = 노이즈 재적용.
- **수정:** 상수속도 prior는 **연속 2회 채택** AND **직전 step > 2 cm**
  (`kVelocityPriorMinimumStepMeters`)일 때만. 저속에선 직전 포즈가 이미 수렴 basin 안이라 외삽은
  노이즈만 보탠다. straight-line 픽스처(0.1 m/frame)는 임계 위라 모션 모델의 측정된 가치 유지.
- **+ 물리 스텝 게이트:** `RegistrationParam::maxStepMeters`(트래커 기본 0.08 m) — prior에서
  그보다 멀리 간 solve는 fitness와 무관하게 `ETrackFailure::ImplausibleMotion`으로 거부.
  fitness 게이트가 못 잡는 오수렴(맵을 처음 오염시킨 fit 0.571 / 0.225 m 점프)의 2차 방어선.
- **테스트:** `RegistrationThread.VelocityPriorNeedsTwoConsecutiveAdoptionsAfterARejection`,
  `…SlowMotionUsesThePreviousPosePrior`, `Pipeline.GpuIcpTrackerRejectsAPhysicallyImplausibleStep`.

---

## 사용법

```bash
# capture/ 정합 + TSDF 진단 (결정론, ~5분)
./build-rel/example2/icp_quality_diag --replay capture --voxel 0.05 --trackers icp --prefilter 5
#   p2p on/off A/B: --no-p2p 추가

# GT 채점 (scan_out/ground_truth.ply 자동 발견)
./build-rel/example2/tsdf_folder_eval --dir scan_out --voxel 0.5 [--no-p2p] [--out mesh.ply]
```

- **p2p를 쓸 때는 `--prefilter 5`를 같이** — 맵이 더 조밀(211k vs 246k)하고 궤적도 최선.
- 파이프라인 A/B 측정은 **재현성 수정 이후의 것만** 신뢰할 것. 이전 수치(예: `minFitness` 스윕,
  `downsampleVoxel` 주석의 복셀 수)는 스케줄러를 잰 것일 수 있다.

## Local → Global fallback 판정

수정 후 이 데이터에는 global 재정위가 구할 실패가 없다(거부 1 = 맵 없음). 건강한 맵에서
**연속 N프레임 실패**(진짜 tracking lost)가 관측되는 데이터가 생기면, 그때의 올바른 자리는
파이프라인 구조 변경이 아니라 `TrackerRegistry`에 **새 이름으로 등록하는 composite 트래커**
(local 시도 → 연속 실패 시 global)다. 그 전까지는 만들지 않는다.

## 남은 것 (비차단)

- ~~GPU residual RMSE의 고정소수점 바닥(~7 mm, SCALE=10000)~~ — **해결(2026-08-26)**: 잔차 슬롯만
  float 공유메모리 트리 축약으로 교체(H/b는 고정소수점 유지, 덧셈 순서 고정이라 재생 결정론 유지).
  테스트 `GpuIcp.AccumulateResidualSumHasNoFixedPointFloor`. 이 문서 위 표의 trackerRmse 수치는
  바닥이 있던 시절 측정이므로 여전히 참고용.
- ~~`PointToPlaneIcpTracker`(icp-cpu)가 `ETrackFailure`를 부분적으로만 채움~~ — **해결(2026-08-26)**:
  실패 분류·minFitness 기본 0.4·스텝 게이트 복셀 스케일링 모두 GpuIcpTracker와 동일 판정으로 정렬.
  테스트 `Pipeline.CpuIcpTracker*` 3종.
- 좁은 수렴 basin 자체(평면 지배 장면에서 30 mm prior 오차에 0.1 m+ sliding)는 남아 있다 —
  prior 수정으로 **노출이 사라졌을 뿐**. Tier-3 annealing(dormant)이 후보 완화책.
- 자매 커널 2개의 밴드 버그(frozen comparator) — 고치려면 관련 비교 문서 재측정과 함께.

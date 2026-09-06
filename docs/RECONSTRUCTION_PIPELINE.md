# 재구성 파이프라인

depth 이미지 한 장이 들어와 TSDF 맵의 복셀이 되기까지. 스레드 셋, 그 사이를 잇는 채널 셋, 그리고
"이 프레임을 맵에 넣어도 되는가"를 세 번 나누어 묻는 정책이 전부다.

코드는 `src/Pipeline/`, 조립은 `Pipeline::Pipeline`(`src/Pipeline/Pipeline.h`).

---

## 1. 전체 그림

```
 ┌─────────────────────┐                    ┌──────────────────────┐                  ┌────────────────────┐
 │  AcquisitionThread  │                    │  RegistrationThread  │                  │ IntegrationThread  │
 │                     │                    │                      │                  │                    │
 │  IDepthProvider     │  Channel<Frame>    │  Tracker::Track      │ Channel<Tracked  │  TSDF::Integrate   │
 │       ↓             │  capacity 8        │       ↓              │        Frame>    │       ↓            │
 │  DepthFrontEnd(GPU) │ ─────────────────▶ │  ShouldFuse          │  capacity 4      │  TSDF::Download    │
 │       ↓             │                    │       ↓              │ ───────────────▶ │       ↓            │
 │  Frame(점+법선)      │                    │  FusionGate::Admit   │                  │  ModelSnapshot     │
 └─────────────────────┘                    └──────────────────────┘                  └─────────┬──────────┘
                                                       ▲                                        │
                                                       │   Mailbox<ModelSnapshot> (latest-wins)  │
                                                       └────────────────────────────────────────┘
                                                                      ↓
                                                            호출자 / 렌더 스레드
```

**프레임은 한 방향으로 흐르고, 맵은 비동기로 되돌아온다.** 정합 스레드는 매 루프 첫머리에
`m_comm.model.Latest()`를 읽는데, 그건 큐가 아니라 **latest-wins Mailbox**다. 실시간 모드에서는 둘
사이에 아무 동기화가 없으므로 프레임 N이 **어느 버전의 맵**에 정합하는지가 스케줄링에 달렸다. 이게
왜 중요한지는 §5에서 다룬다.

세 스테이지는 모두 `PipelineStage`를 상속하고, 워커에서 던져진 예외는 삼켜지지 않고
`Pipeline::CheckErrors()`로 호출자에게 다시 던져진다.

---

## 2. Acquisition — depth 이미지에서 점군까지

`src/Pipeline/Acquisition/AcquisitionThread.cpp`

### 2.1 소스는 하나의 축, 하나의 인터페이스

취득 축에 있는 것은 **depth 이미지뿐**이고, 추상화는 `Realsense::IDepthProvider` 하나다.

| `EAcquisitionSource` | provider | 비고 |
|---|---|---|
| `Realsense` | `Realsense::RealSenseD435` | 라이브 D400. `MakeDepthProvider`가 `Open(config.stream)`까지 한다 |
| `RealsenseFile` | `Realsense::RealSenseD435Recorder` | 녹화 재생. `recordingDirectory` 없으면 던진다 |

`AcquisitionConfig::makeProvider`가 설정되어 있으면 `source`를 **무조건 이긴다.** 테스트가 합성
장치를 주입하는 통로이고, `realsense_scan --record`가 장치를 recorder로 감싸는 통로다 — enum은
장치를 서술하지 체인을 서술하지 않는다.

provider는 **생성자에서** 만들어진다. 그래서 없는 카메라는 창이 뜨기 전에 CLI 한 줄로 보고된다.

**깊이는 언제나 Z16이다.** `DepthFrame`은 `rawZ16` 포인터만 싣고(provider 소유, 다음 `Grab`까지 유효),
metres로 푸는 단계가 파이프라인 어디에도 없다. 풀었다 되돌리면 드라이버가 시작한 자리로 돌아오는 데
프레임당 호스트 전체 패스를 두 번 쓴다.

### 2.2 DepthFrontEnd — GPU 한 번에 도는 여섯 패스

`AcquisitionThread.cpp` 파일 안에만 사는 클래스다. 자기 `Engine::Core::Context`를 소유하고,
`Realsense::RealSensePipeline::Execute` 하나로 아래를 **한 커맨드 배치에** 기록한다:

```
upload + score → threshold + back-project → normals → downsample → count → scan → scatter
```

| # | 패스 | 하는 일 | 산출 |
|---|---|---|---|
| 1 | `ValidationMask::Execute` | Z16(+IR) 업로드, 픽셀마다 신뢰도 점수 | `{valid, emitted, score}` |
| 2 | `kernel_RemainValidDepth` | `scoreThreshold` 비교 + 역투영 | 조밀 `m_vertices` |
| 3 | `NormalEstimation` | 평면 피팅 법선 (`normal.enabled`일 때만) | `m_normals`, 거부 카운터 |
| 4 | `DownSample` | 디바이스에서 복셀 솎기 (`downSample.enabled`일 때만) | `emitted` 취소 |
| 5 | `kernel_CountEmittedPerRow` → `kernel_ScanRows` | 행별 개수 → prefix sum | `m_rowOffset` |
| 6 | `kernel_ScatterValidPoints` | 압축 | `m_points`, `m_compactNormals`, `m_compactScores` |

**순서가 임의가 아니다.** 법선 패스와 다운샘플 패스는 둘 다 `emitted`를 **취소**한다. 그래서 둘 다
count보다 **앞**에 와야 한다 — 뒤집으면 나중에 버려진 픽셀에 압축 슬롯이 하나 배정된다.

세 압축 배열은 인덱스를 공유한다: 점 `i`, 그 신뢰도, 그 법선이 같은 픽셀을 가리킨다.

`out.cam`은 항상 원점이다. depth 프레임은 센서 좌표계이고, 월드 카메라 위치는 정합 후
`pose * f.cam`으로 만들어진다.

### 2.3 다운샘플링은 두 군데에 있다

- **GPU**(`downSample`) — readback **전**, 전송량까지 줄인다. `realsense_scan`은 이걸 켜고 셀 크기를
  `baseVoxel / 2`로 **파생**시킨다(맵의 detail 레벨).
- **호스트**(`downsampleVoxel`) — readback **후** `Features::DownsampleVoxel`. `realsense_scan`은
  `0.0f`로 꺼둔다. 같은 축소를 두 번 낼 이유가 없다.

`MapConfig::downsample`(융합 직전 세 번째 자리)은 `SubmapAdvancedTSDF`만 읽는데,
`IntegrationThread::ConfigureMap`이 그 필드를 전달하지 않는다 — **이 경로에서는 죽은 노브다**(§7).

---

## 3. Registration — 프레임을 맵에 맞춘다

`src/Pipeline/Registration/RegistrationThread.cpp`

### 3.1 prior 포즈 — 상수속도는 두 조건이 다 차야 켜진다

```cpp
const bool velocityWorthUsing =
        consecutiveAdoptions >= 2 &&
        lastDelta.translation().norm() > 0.02f;   // kVelocityPriorMinimumStepMeters
const Eigen::Isometry3f prior = velocityWorthUsing ? previousPose * lastDelta : previousPose;
```

1. **직전 2회 연속 채택** — 1-프레임 간격 delta임을 보장한다.
2. **직전 step > 2 cm** — 저속에서 상수속도 외삽은 포즈 노이즈를 재적용하는 것이라, 자기유지 진동을
   만든다.

거부가 한 번 나면 `consecutiveAdoptions`가 0으로 리셋되므로 다음 두 프레임은 직전-포즈 prior로 돈다.
실측 근거는 `ICP_REGISTRATION_QUALITY.md` §9.6.

### 3.2 트래커

| 이름 | 클래스 |
|---|---|
| `identity` | `IdentityTracker` |
| `icp` | `GpuIcpTracker` |
| `icp-cpu` | `PointToPlaneIcpTracker` |
| `global` | `GlobalRegistrationTracker` |
| `icp+global` | `RelocalizingIcpTracker` |

`TrackerRegistry::Create`는 모르는 이름에 **`nullptr`을 돌려준다**(던지지 않는다) — 호출자가 확인해야
한다.

### 3.3 실패는 `valid` 하나로 뭉뚱그리지 않는다

`ETrackFailure`가 원인을 나눈다. "맵이 아직 없다"와 "엉뚱한 점에 걸렸다"는 정반대 대응을 요구하는데
`valid == false`만 보면 구분되지 않기 때문이다.

| 값 | 뜻 |
|---|---|
| `NoModel` | 맵이 아직 없거나 프레임이 비었다 — 초기 프레임에서 정상 |
| `NoLocalTarget` | 그 위치에 맵이 거의 없다 — 새 영역이다 |
| `TooFewInliers` | 대응점이 `minInliers`보다 적다 |
| `LowOverlap` | 개수는 찼지만 프레임 대비 비율이 낮다 |
| `ImplausibleMotion` | prior에서 물리적으로 불가능한 거리를 움직였다 — fitness가 좋아도 오수렴 |

포즈 delta 통계(`poseDeltaMetersMax`, `trajectoryLengthMeters`)가 따로 있는 이유: 30 fps 핸드헬드는
프레임당 0.05 m를 훨씬 밑돈다. 잔차 RMSE는 **트래커가 스스로 고른 대응점**만 채점하므로 발산을
못 잡지만, 이건 잡는다.

### 3.4 융합 여부는 두 층으로 결정된다

```cpp
tf.fuse = ShouldFuse(a.valid, a.failure);        // 1층: 맵을 오염시키는가
if (tf.fuse) tf.fuse = m_fusionGate.Admit(...);  // 2층: 넣을 만큼 좋은가
else         ++m_skippedFusions;
```

**1층 `ShouldFuse`** — 고정 규칙이다.
- `TooFewInliers` / `LowOverlap` / `ImplausibleMotion` → **융합 안 함.** 맵이 있는데도 solve가 게이트를
  못 넘긴 경우이고, 그 틀린 포즈로 오염된 맵이 다음 프레임의 정합 타깃이 된다.
- `NoModel` / `NoLocalTarget` → **융합함.** 오염시킬 맵이 애초에 없다. 거부하면 맵이 부트스트랩되지
  않거나(프레임 0) 새 영역으로 자라지 못한다.

**2층 `FusionGate`** — 센서마다 튜닝하는 판단이고 **전부 기본 OFF**다. 기본이 켜져 있으면
`identity` 트래커(fitness를 계산하지 않아 항상 0.0)가 `scan_out`/`scanData` 전체를 거부해 아무것도
재구성하지 못한다.

`Admit`의 순서:
1. 첫 프레임은 무조건 통과 — 맵의 씨앗이다.
2. 부트스트랩 중이면 `fitness >= bootstrapMinFitness`인 프레임을 세고, **게이트를 무장시킨 그 프레임까지
   포함해 전부 거부**한다.
3. `NoModel`/`NoLocalTarget`은 정상 게이트를 우회한다.
4. `minimumFusionFitness` / `maximumFusionRmse` 미달이면 각각 카운터를 올리고 거부.

2층은 1층이 **이미 예라고 답했을 때만** 물어본다 — 아니오를 예로 뒤집는 경로는 없다.

`skippedFusions`는 **1층 거부만** 센다. 2층 거부는 게이트 안의 별도 카운터
(`fusionRejectedByFitness` / `fusionRejectedByRmse`)로 관측한다.

---

## 4. Integration — 융합과 스냅샷

`src/Pipeline/Integration/IntegrationThread.cpp`

자기 `Engine::Core::Context`와 `TSDF`를 소유한다. 매 프레임:

1. `++processed` — **`fuse`와 무관하게** 항상 증가한다. 프레임 인덱스는 호출자의 완료 신호이고
   (`icp_quality_diag`가 `ProcessedFrame()`을 기다린다), 조용히 건너뛰면 모든 소비자가 멈춘다.
2. `if (tf.fuse)`일 때만 `IntegrateWorld` → `BuildSnapshot` → `model.Publish`. 건너뛴 프레임은
   download와 publish도 건너뛴다 — 맵이 변하지 않았고, Mailbox가 직전 스냅샷을 그대로 들고 있다.
3. `handshake.NoteCompleted()` — **건너뛴 프레임에서도 반드시 호출한다.** 아니면 융합을 거절하는 첫
   프레임에서 파이프라인이 교착한다.

스냅샷은 풀에서 재사용된다(`use_count() == 1`이면 자유). `entryBuckets`는 재사용 전에 명시적으로
비운다 — 남은 인덱스가 살아남으면 안 된다.

`ModelSnapshot`이 결과 복셀만 싣지 않는 이유:

- `voxel` — 트래커가 ICP 대응 거리를 **맵 해상도에 맞추기 위해**. 고정값을 쓰면 대응점이 너무 적게
  잡히는 동시에 GPU `LocalGrid` 셀 수가 폭발한다.
- `truncationDistance` — 서브복셀 타깃 점 복원용.
- `entryBuckets` — 트래커는 매 프레임 스냅샷을 프레임 AABB로 크롭한다. 인덱스가 없으면 그 크롭이
  **정합 스레드에서** O(entries) 전수 스캔이 되고, 프레임 이웃은 그대로인데 맵 전체와 함께 자란다.
  `entryBucketSize == 0`이면 선형 스캔으로 폴백하므로 정확성에 필수는 아니다.

---

## 5. 채널 정책 — 드롭이냐 블로킹이냐

`CommunicationModule(dropWhenBehind)` 하나가 두 링크의 오버플로 정책을 함께 정하고, 그 값은
`AcquisitionConfig::realTime`에서 온다.

| `realTime` | 채널 | handshake | 쓰는 곳 |
|---|---|---|---|
| `true` (기본) | 가득 차면 **가장 오래된 것을 버린다** | 꺼짐 | 라이브 센서 |
| `false` | 가득 차면 **생산자가 블록** | 켜짐 | 녹화 재생 |

라이브 센서는 맵이 따라오든 말든 계속 생산하므로 드롭해서 지연을 묶어야 한다. 녹화를 드롭 모드로
돌리면 **느린 설정이 조용히 더 적은 프레임을 처리해서 설정 간 비교 측정이 전부 오염된다.**

### 무손실은 재현성이 아니다

블로킹 채널은 프레임 *개수*만 맞춘다. 맵은 latest-wins Mailbox로 전달되고 정합은 `trackedFrames`
용량(4)만큼 융합보다 앞서 달릴 수 있으므로, 프레임 N이 **어느 버전의 맵**에 정합하는지가 여전히
스케줄링에 달렸다. 그 맵이 정합 타깃이므로 포즈가 달라지고, 다음 맵이 달라진다 — 실행마다 발산한다.

그래서 녹화 모드는 `FrameHandshake`도 켠다. 정합이 매 프레임 융합 완료를 기다리는 lock-step이고,
동시에 흐르는 프레임은 정확히 하나가 된다. **파이프라인을 통과하는 A/B 측정은 이것 없이 무의미하다.**

실측: 핸드셰이크 이전에 한 명령을 네 번 돌려 궤적 길이가 1.47 / 6.45 / 7.84 / 136.76 m로 나왔다.

---

## 6. 재구성(Reconfigure)과 실패

`Pipeline::Reconfigure(cfg, tracker)`는 `Stop()` → `buildStages()` → `Start()`다. 같은 `Pipeline`
객체를 유지하므로 렌더 스레드가 들고 있는 참조가 살아남는다. **누적된 맵은 도중에 바꿀 수 없으므로
프레임 0부터 다시 재생한다** — 그리고 그게 A/B 측정이 성립하는 조건이기도 하다. 반쯤 이 설정이고
반쯤 저 설정인 맵은 아무것도 측정하지 않는다.

`buildStages`는 **새 스테이지를 만들기 전에 옛 스테이지를 파괴한다.** 그래야만 한다 — 옛 취득
스테이지가 장치를 붙잡고 있어서, 풀리기 전엔 새 것이 못 연다. 그 대가로 생성자가 던지면 세 포인터가
모두 null로 남는다. 그래서 `Pipeline`의 모든 접근자가 그 상태를 견딘다: `GetStats()`는 0을 돌려주고,
`CheckErrors()`는 없는 스테이지를 건너뛰고, `Stop`/`Start`/`SetPaused`는 no-op이다. 호출자가 뷰어이고,
반쯤 무너진 파이프라인이야말로 **실패를 표시하기 위해 계속 그려야 하는** 상황이다.

---

## 7. 설정에서 실제로 연결된 것과 아닌 것

`MapConfig`는 백엔드 **이름을 고르지 않는다** — `IntegrationThread::ConfigureMap`이
`config.backend = "advanced"`를 하드코딩한다. 등록된 백엔드도 `"advanced"` 하나뿐이다.

`MapConfig`가 실제로 고르는 이름 축은 splitter다: `submap ? "dense" : "none"`.

해시 주소법(`TSDFBackendConfig::hash`, `"linear"` / `"bucketed"`)은 `MapConfig`에 필드가 없어
**항상 `"linear"`로 돈다.** 모르는 이름은 조용히 linear로 폴백하므로, 비교를 라벨링해야 하는 호출자는
`Resolve`에 물어봐야 한다.

**이 경로에서 읽히지 않는 필드**(선언은 있으나 `ConfigureMap`이 전달하지 않는다):

- `MapConfig::detailK`
- `MapConfig::detailTruncVoxels`
- `MapConfig::downsample`

**채워지지 않는 스냅샷 필드**: `ModelSnapshot::isNew`, `ModelSnapshot::firstFrame`. 최초 관측 프레임은
복셀마다 `TSDFVoxel::firstFrame`에 GPU가 찍는다. 스냅샷은 풀에서 재사용되므로 이 두 벡터에는 이전
사용의 잔재가 남을 수 있다 — **읽지 말 것.**

---

## 8. 관측 지점

`Pipeline::GetStats()`가 네 곳에서 모은다: 채널 깊이/드롭은 `CommunicationModule`, 나머지는 각
스테이지의 원자 카운터.

증상별로 볼 곳:

| 증상 | 먼저 볼 것 |
|---|---|
| 점이 안 나온다 | `acquiredFrames`(프레임이 오는가) → `GpuFrontEndStats::lastFramePoints`(프론트엔드가 다 버리는가) |
| 법선이 적다 | `normalOutOfDomain`(스텐실 경계 — 추정기 도메인) vs `normalNoSupport`(장면이 거부) — **절대 합치지 말 것** |
| 맵이 안 자란다 | `fusionArmed`(부트스트랩이 잡고 있는가) → `skippedFusions` → `rejected*` 분류 |
| 트래커가 발산한다 | `poseDeltaMetersMax`, `trajectoryLengthMeters` — 잔차 RMSE는 이걸 못 잡는다 |
| 프레임이 사라진다 | `trackDropped`(라이브 드롭 모드에서만 0이 아니다) |
| 맵에 구멍이 있다 | `map.insertFailureCount`, `windowLimitRefusals`, `downSampleInsertFailures` |

용량 천장은 전부 실패 카운터를 노출한다. 조용히 잘리면 증상 없이 결과만 망가지기 때문이다.
다운샘플의 두 천장은 **fail OPEN**이다 — 테이블에 못 넣은 픽셀은 emitted를 유지한다. 덜 솎이는 건
처리량 손해지만, 떨어뜨리면 표면에 구멍이 뚫리고 그건 메시가 틀릴 때까지 증상이 없다.

---

## 관련 문서

- `ICP_REGISTRATION_QUALITY.md` — 정합 품질, 상수속도 prior 실측 근거
- `ADVANCED_TSDF.md`, `TSDF_IMPLEMENTATION.md` — 융합 단계 내부
- `DENSE_REGION_SEPARATION.md` — splitter `"dense"`가 하는 일
- `src/Realsense/RealSensePipeline.md` — 프론트엔드 GPU 패스 상세
- `REALDATA_CAPTURE_FIXES.md` — 실데이터에서 나온 수정들

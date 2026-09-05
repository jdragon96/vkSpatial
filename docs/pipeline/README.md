# 재구성 파이프라인 프로세스

`Pipeline::Pipeline` 기준 (`src/Pipeline/Pipeline.cpp`). 세 워커 스레드와 그 사이를 잇는 `CommunicationModule` 하나를 소유·기동·정지하는 파사드다.

## 1. 구성 (buildStages)

- 기존 스테이지를 **하류부터** 파괴한다 (`m_integration` → `m_registration` → `m_reconstruction`).
  - 왜: 상류가 살아 있는 채로 하류를 지우면 남은 push가 죽은 큐를 친다.
- `CommunicationModule(cfg.acquisition.realTime)`을 만들어 두 링크의 오버플로 정책을 한 번에 정한다.
  - `realTime == true`(라이브 센서) → 큐가 차면 오래된 프레임을 버려 지연을 묶는다.
  - `realTime == false`(녹화 재생) → 블로킹 = 무손실이고, `FrameHandshake`가 함께 켜져 lock-step이 된다.
- 새 스테이지를 **상류부터** 만든다: `AcquisitionThread`(취득) → `RegistrationThread`(정합 + `Tracker`) → `IntegrationThread`(융합 + `MapConfig`).
- 세 축(취득 전략 / 트래커 / TSDF 백엔드)이 각각 독립적으로 교체된다.

## 2. 기동 (Start)

- **하류부터** 시작한다: 융합 → 정합 → 취득.
- 왜: 소비자가 먼저 서 있어야 첫 프레임이 곧바로 흘러간다.

## 3. 프레임 흐름

```
AcquisitionThread --Channel<Frame>(8)--> RegistrationThread
                     --Channel<TrackedFrame>(4)--> IntegrationThread
                     --Mailbox<ModelSnapshot>--> 호출자 / 렌더 스레드
```

1. 취득: `IDepthProvider::Grab`한 depth 이미지를 GPU 프론트엔드로 `Frame`으로 만들거나(PLY 소스면 파일에서 읽어), `downsampleVoxel > 0`이면 복셀 다운샘플까지 마친 뒤 push한다.
2. 정합: `model.Latest()`를 타깃으로 `Tracker::Track(frame, model, prior)`를 돌린다.
   - prior는 직전 포즈이며, 연속 2회 채택 AND 직전 step > 0.02 m일 때만 상수속도 외삽을 얹는다.
   - 실패는 `ETrackFailure`별로 카운트되고, 포즈는 직전 포즈로 되돌린다.
3. 융합 판정: `ShouldFuse(valid, failure)`가 먼저, 통과한 프레임만 `FusionGate::Admit`이 다시 본다. 둘 다 yes여야 `tf.fuse`가 선다.
4. 융합: `tf.fuse`인 프레임만 월드 좌표로 변환해 `TSDF::Integrate` → `Download` → `ModelSnapshot` 발행(`Mailbox`는 latest-wins).
   - 스킵된 프레임도 `processed`는 올리고 `handshake.NoteCompleted()`를 부른다.
   - 불변식: 스킵된 프레임이 handshake를 놓으면 첫 거부에서 파이프라인이 교착한다.

## 4. 관측

- `LatestModel()` — `Mailbox`의 최신 스냅샷. 트래커는 여기 실린 `voxel`로 대응 거리를 맵 해상도에 맞춘다.
- `GetStats()` — 큐 깊이·드롭 수는 `CommunicationModule`에서, 단계별 평균 시간과 실패 원인별 카운터는 각 스테이지에서 그 자리에서 모아 `PipelineStats`로 만든다(내부 상태 보관 없음).
- `CheckErrors()` — 세 스테이지의 `Error()`를 순서대로 훑어 첫 워커 예외를 호출자 스레드에서 다시 던진다.

## 5. 정지와 재구성

- `Stop()`: 세 채널(`capturedFrames`, `trackedFrames`, `handshake`)을 **먼저 닫아** 블로킹된 Pop/Wait을 깨운 뒤, **상류부터** join한다.
  - 왜: 닫기 전에 join하면 상류가 채널에서 대기 중일 때 join이 돌아오지 않는다.
- `Reconfigure()`: `Stop()` → `buildStages()` → `Start()`를 같은 객체 위에서 한다. 보유된 참조(렌더 스레드 등)는 유효한 채로 남지만, 누적 중이던 맵은 버려지고 프레임 0부터 다시 재생된다.
- 소멸자는 `Stop()`을 부른다. 각 `PipelineStage` 파생 클래스도 자기 소멸자에서 `Stop()`을 부른다.

## 6. 스레드별 의사코드

- 세 스테이지 모두 `PipelineStage`의 `Run()` / `Interrupt()` 두 훅만 구현한다.

- `Run()`은 `StopRequested()`가 서면 돌아오고, `Interrupt()`는 `Run()`이 막혀 있을 채널을 닫아 join을 풀어준다.

```python
# 아래와 같은 순서로 스레드 알고리즘이 동작한다.
AcquisitionThread()
|
|- CommuniationModule.capturedFrames.Push()
|
RegistrationThread()
|
|- CommuniationModule.trackedFrames.Push()
|
IntegrationThread()
|
|- CommuniationModule.trackedFrames.Push()
|
```

### 6.1. AcquisitionThread (`Acquisition/AcquisitionThread.cpp`)

```python
def Run():
    source.Open()
    while not StopRequested() and source:
        # 1. 일시정지 대기, 정지 요청이면 false
        if not waitWhilePaused(): break

        # 2. 취득 + 다운샘플을 한 덩어리로 계측
        with ScopedMean(acquireMs):
            ok = source.Next(frame)
            # 3. Downsampling
            if ok and downsampleVoxel > 0: reduceFrame(frame)
        if not ok: break
        # 4. 큐에 현재 프레임을 저장
        if not capturedFrames.Push(frame): break
    source.Close()
    capturedFrames.Close()

def Interrupt():
    source.Close()
    pauseCv.notify_all()
```

### 6.2. RegistrationThread (`Registration/RegistrationThread.cpp`)

```python
def Run():
    previousPose = previousPreviousPose = Identity
    consecutiveAdoptions = 0
    while not StopRequested() and capturedFrames.Pop(frame):
        # 1. 가장 마지막 integration 시, 해당 위치에서 Target을 가져옴
        model = GetLastModel()
        # 2. Pose 변화량 계산
        lastDelta = previousPreviousPose⁻¹ * previousPose
        # 3. Adaptive Threshold를 활용한 움직임 예측값 반영
        prior = (consecutiveAdoptions >= 2 and |lastDelta.t| > 0.02 m)
                    ? previousPose · lastDelta
                    : previousPose
        # 4. Registration 수행
        result = tracker.Track(frame, model, prior)
        # 5. 정합 결과 반영하기
        if result.valid:
            step = previousPose⁻¹ · result.pose
            누적: poseDelta 평균/최대, trajectoryLength, trackerRmse
            previousPreviousPose = previousPose; previousPose = result.pose
            consecutiveAdoptions = min(consecutiveAdoptions + 1, 2)
        else:
            consecutiveAdoptions = 0
            rejected++
            rejected[result.failure]++
            pose = previousPose

        tracked.pose = pose
        tracked.cameraWorld = pose · frame.cam
        tracked.fuse = ShouldFuse(result.valid, result.failure)
        if tracked.fuse: tracked.fuse = fusionGate.Admit(valid, failure, fitness, rmse)
        if not tracked.fuse: skippedFusions++

        handshake.NotePushed()
        trackedFrames.Push(tracked)

        # lock-step(녹화)일 때만 실제로 대기
        if not handshake.WaitForDrain(): break
    trackedFrames.Close()

def Interrupt():
    capturedFrames.Close()
    handshake.Close()
```

- 불변식: 거부된 프레임도 `previousPose`를 실어 반드시 push한다 — 하류의 프레임 인덱스가 호출자의 완료 신호다.

### 6.3. IntegrationThread (`Integration/IntegrationThread.cpp`)

```
Run():
    ctx = Engine::Core::Context; tsdf.Build(ctx, ConfigureMap(cfg))   # 이 스레드가 GPU 컨텍스트를 소유
    pool = []                                                        # ModelSnapshot 재사용 풀
    processed = -1
    while not StopRequested() and trackedFrames.Pop(tracked):
        with ScopedMean(integrateMs):        # integrate + download + snapshot
            processed++
            if tracked.fuse:
                IntegrateWorld(tsdf, tracked)          # pose가 Identity면 변환 생략
                snap = pool에서 use_count()==1인 것 재사용, 없으면 새로 만듦
                BuildSnapshot(snap, tsdf, processed, baseVoxel, truncation)
                comm.model.Publish(snap)
        m_processed = processed                        # 스킵 프레임도 전진한다
        handshake.NoteCompleted()                      # 스킵이어도 반드시 놓는다

Interrupt(): trackedFrames.Close(); handshake.Close()
```

- `BuildSnapshot`은 `Download` → 엔트리 버킷 인덱스 → 타일/해시 통계 → 월드 AABB 순으로 채운다.
  - 왜 인덱스를 여기서 만드나: 정합 스레드의 프레임별 crop이 O(entries) 스캔 대신 버킷 조회로 끝나기 때문이다.
- 스킵된 프레임은 download/publish도 건너뛴다 — 맵이 안 변했고 `Mailbox`가 직전 스냅샷을 그대로 들고 있다.

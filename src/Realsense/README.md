# Realsense — D400 깊이 프론트엔드

raw Z16 한 장을 받아 **좌표·법선·신뢰도가 붙은 압축된 점 배열**을 내놓는다. 전 과정이 GPU에서 돌고,
살아남은 점만 호스트로 내려온다.

이 모듈은 이 저장소의 **새 알고리즘 모듈 레퍼런스**다 — 파이프라인이 조합을 갖고, `Algorithm/` 밑에
세부 알고리즘이 각자 한 벌씩 앉는다. 새 모듈을 만든다면 이 폴더를 열어서 따라 쓴다
(`.claude/skills/algorithm-module`).

## 무엇을 하나

```
raw Z16
  → ValidationMask        점수: c = [z>0] · c_range · c_ir · c_nb
  → RemainValidDepth      문턱값 + 역투영
  → NormalEstimation      법선 (forward / central / planefit)
  → DownSample            detail voxel당 점 하나
  → Count → Scan → Scatter  결정적 압축
= { points, normals, scores }
```

D4 VPU는 자기 신뢰도를 40여 개 파라미터로 계산해놓고 **결과만 depth == 0으로 공개하고 버린다**. 가중
융합이 원하는 연속 신뢰도는 그래서 합성해야 하고, 그게 이 모듈이 존재하는 이유다.

각 스테이지는 `emitted`를 **끄기만** 한다. 채택은 점수 문턱값이 정하고, 뒤 스테이지는 거부권만 갖는다.

## 파일

| | |
| --- | --- |
| `RealSensePipeline.h` | 스테이지 사이의 버퍼와 **순서**를 소유한다. 알고리즘은 없다 |
| `RealSensePipeline.md` | 그 순서의 단계별 흐름 + Pipeline 통합 A/B |
| `RealSenseTypes.h` | 경계 전용 POD — 옵션, 카운터, 푸시상수 미러. GLSL 미러와 `static_assert`로 묶여 있다 |
| `RealSenseD435.h/.cpp` | 장치. Z16을 **그대로** 넘긴다(언팩 없음). librealsense2 없으면 던지는 스텁 |
| `Algorithm/Common.glsl` | 커널들이 공유하는 struct와 같은표면 허용치. `main()`이 없으므로 접두사 없음 |
| `Algorithm/ValidationMask.*` | 점수 + 문턱값 + 압축 커널들 |
| `Algorithm/NormalEstimation.*` | 법선 추정 3전략 (`.md`에 정확도 검토) |
| `Algorithm/DownSample.*` | 복셀당 하나로 솎기 (`.md`에 감소량 실측) |

세부 알고리즘의 **왜**는 각 `.md`에 있다. 기본값이 왜 그 값인지도 거기 있고, 전부 측정 근거가 붙어 있다.

## 쓰는 법

```cpp
Realsense::RealSensePipeline pipeline(context, width, height);

Realsense::ValidationScoreOptions score;   // 장치에서: camera.MakeScoreOptions()
Realsense::NormalEstimationOptions normal; // 기본 planefit, radius 4
Realsense::DownSampleOptions downSample;   // 기본 OFF
downSample.enabled = true;
downSample.detailVoxelMeters = 0.010f;

{
    Engine::Compute::CommandBatch batch(context);
    pipeline.Execute(batch, depthZ16, score, intrinsics, 0.9f, normal, downSample);
    batch.Submit();
}
auto points  = pipeline.DownloadValidPoints();
auto normals = pipeline.DownloadValidNormals();
auto scores  = pipeline.DownloadValidScores();   // 세 배열의 인덱스가 1:1
```

점수와 압축은 따로 부를 수 있다(`RecordScore` / `RecordExtract`). 문턱값만 바꿔 여러 번 압축하는 것이
점수를 다시 계산하는 것과 같은 답을 내야 하고, 문턱값이 자기 커널에 사는 이유가 그것이다.

**재구성 파이프라인에서 쓰려면** `Pipeline::GpuDepthFrameSource`가 이 모듈을 `IFrameSource`로 감싼다.
opt-in이고, `AcquisitionConfig::downsampleVoxel`은 0으로 둔다(여기 다운샘플이 readback **전**에 돈다).

## 튜닝된 기본값 — 셋을 동시에

`capture/` 실측. 세 목표가 서로 당기므로 노브는 짝으로 움직인다.

| 노브 | 값 | 무엇을 위해 |
| --- | --- | --- |
| 점수 문턱값 | **0.9** | 깊이 절벽 위 픽셀(flying pixel)을 0%로 |
| `planeFitRadius` | **4** | 인접 법선 불일치 3.22° → 1.89° |
| `detailVoxelMeters` | 호출자 | 5 mm면 28.6%, 10 mm면 8.5%가 남는다 |

**반경만 올리면 노이즈가 오히려 늘어난다**(cliff 1.06% → 1.31%) — 넓은 적합은 실 위에서도 살아남기
때문이다. 근거 표는 `RealSensePipeline.md`의 튜닝 절.

## 도구

```bash
validation_score_lab --replay capture --loop              # 뷰어: 노브를 돌리며 본다
validation_score_lab --replay capture --sweep k,subpixel  # 헤드리스 표
icp_quality_diag --replay capture --frontend gpu          # 파이프라인 통과 A/B
```

## 주의

- **공용 include에 `#version`을 쓰지 않는다.** 셰이더 오류는 런타임에만 터지므로, 이 폴더의 커널이
  하나도 컴파일된 적 없는 상태로 테스트가 통과할 수 있다 — 실제로 그랬다.
- **워크그룹 패딩 레인의 `y*width + x`는 범위 밖이 아니라 다음 행이다.** 경계 밖 쓰기는 반드시
  `if (inside)`로 막는다. 848은 32의 배수가 아니다.
- 천장은 전부 카운터로 관측 가능하고 **fail open**이다 — 못 넣으면 점을 살려둔다. 덜 솎이는 건 성능
  손해지만 떨어뜨리면 표면에 구멍이 뚫린다.

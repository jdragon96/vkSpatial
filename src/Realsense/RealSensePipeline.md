# 깊이 프론트엔드 프로세스

`Realsense::RealSensePipeline::RecordExtract` 기준 (`src/Realsense/RealSensePipeline.h`).

`ValidationMask`는 이제 **점수 스테이지 하나**이고, 아래 다섯 패스의 버퍼와 순서는 파이프라인이
소유한다 — 어느 한 패스의 이름을 단 클래스에 순서가 숨어 있지 않게 하려는 분리다.
점수는 `RecordScore`(→ `ValidationMask`)가 `properties[].score`에 채워 둔 상태에서 시작한다.
커널 7개를 한 배치에 순서대로 기록하며, 사이마다 `batch.Barrier()`를 넣는다.

## 1. Threshold & Back-projection

`ValidationMask.RemainValidDepth.glsl`, 16×16 워크그룹, 픽셀당 1 스레드.

- `emitted`와 `g_vertices[pixel]`을 먼저 0으로 지운다.
- `valid == 0`인 픽셀은 곧바로 반환한다.
- Z16을 다시 읽어 미터로 바꾸고 카메라 좌표로 역투영해 `g_vertices`에 적재한다.

$$ p = \left( \frac{u - c_x}{f_x} z,\; \frac{v - c_y}{f_y} z,\; z \right), \quad z = Z \cdot \text{depthScale} $$

- `score >= scoreThreshold`인 픽셀만 `emitted = 1`로 표시한다.

**왜:** 문턱값은 프레임에 대해 새로 알려주는 게 없는 유일한 노브다. 점수 커널과 분리해 두면 문턱값을
움직여도 4항 곱(픽셀당 3×3 gather)을 다시 치르지 않는다.

## 2. Normal Estimation

`NormalEstimation.EstimateNormal.glsl`, 16×16 워크그룹. 상세는 [NormalEstimation.md](NormalEstimation.md).

- `emitted == 1`인 픽셀에만 법선을 추정해 `m_normals`에 적재한다.
- 실패한 픽셀은 `emitted = 0`으로 **취소**한다 — 세우지는 않는다.
- `NormalEstimationOptions::enabled = false`면 패스 전체를 건너뛴다.

**왜 여기냐:** 역투영이 끝나야 `g_vertices`가 있고, 행 카운트가 최종 판정을 봐야 하므로 3단계보다
**앞**이다. 순서를 뒤집으면 취소된 픽셀이 슬롯을 배정받는다.

## 3. DownSample

`DownSample.ToDetailVoxel.glsl`, 16×16 워크그룹, 두 패스(CLAIM / CANCEL). 상세는
[Algorithm/DownSample.md](Algorithm/DownSample.md).

- detail voxel 하나당 점 하나만 남기고 나머지의 `emitted`를 **취소**한다.
- `DownSampleOptions::enabled`가 기본 OFF다.

**왜 여기냐:** 법선 뒤여야 모든 게이트를 통과한 픽셀만 솎고, 카운트 앞이어야 압축이 최종 판정을 본다 —
법선 패스와 같은 양쪽 제약이다. 실측(capture/, detail 5 mm)에서 점이 **28.6%로** 줄어 readback과 그
뒤가 그만큼 가벼워진다.

## 4. Row Count

`ValidationMask.CountEmittedPerRow.glsl`, `local_size_x = 64`, **행당 1 스레드**.

- 각 행을 직렬로 훑어 `emitted`의 합을 `g_rowOffset[row]`에 쓴다.

## 5. Row Scan

`ValidationMask.ScanRows.glsl`, `local_size_x = 1`, **단일 invocation**.

- 행 개수를 exclusive prefix sum으로 제자리 변환한다.
- 총합을 `g_rowOffset[height]`에 남긴다.

**왜:** 병렬 스캔도 atomicAdd append도 아니다. 이 체인의 출력은 ICP source cloud이고 그 centroid는
float 합이라 **순서가 solve까지 도달한다.** 480행 직렬 주사는 픽셀당 패스 옆에서 무시할 수준이다.

## 6. Scatter

`ValidationMask.ScatterValidPoints.glsl`, `CHUNK = 256`, **행당 워크그룹 1개**(`Dispatch(height,1,1)`).

- `gl_WorkGroupID.x`가 곧 행이므로 `DispatchElements`로 띄우면 안 된다.
- 행을 256열 청크로 끊어, 청크마다 `emitted` 플래그에 Hillis-Steele inclusive scan을 돌린다.
- 청크 총합을 `s_rowBase`에 누적해 다음 청크로 넘긴다.

$$ slot = rowOffset[row] + \left| \{\, c < column : emitted(row, c) \,\} \right| $$

- 살아남은 픽셀의 좌표를 `g_points[slot]`에, 점수를 `g_compactScores[slot]`에, 법선을
  `g_compactNormals[slot]`에 **함께** 싣는다.

**왜:** 점수는 하류의 fusion weight이고 법선은 point-to-plane이 푸는 대상이다. 이미지 공간에 두고
오면 압축된 클라우드와 다시 짝지을 수 없다 — 그때는 이미지 좌표가 사라진 뒤다.

**불변식:** `slot`은 위치만의 순수 함수다 — 어떤 레인이 경쟁에서 이겼는지에 의존하지 않으므로 같은
프레임은 실행마다 바이트 단위로 같은 배열을 낸다.

## 7. Readback

- `ValidPointCount()`가 `g_rowOffset[height]`를 읽어 생존 개수를 돌려준다.
- `DownloadValidPoints()`는 그 개수만큼 vec4를 읽고 w를 버려 `Eigen::Vector3f`로 준다.
- `DownloadValidScores()`와 `DownloadValidNormals()`는 같은 개수를 그대로 복사한다 — 세 배열의
  인덱스가 서로 1:1로 대응한다.

---

# 튜닝 (2026-09-05)

목표 셋을 동시에 만족시킨다: **점 수를 줄이고**(50 mm 다운샘플), **길게 늘어지는 노이즈를 죽이고**,
**법선을 깨뜨리지 않는다.**

## 지표를 먼저 고쳐야 했다

"늘어지는 노이즈"를 재는 대리 지표를 두 번 잘못 골랐고, 둘 다 정상 기하와 노이즈를 구분하지 못했다.

- **고립도**(최근접 이웃 > 1.5 voxel) — 실은 사슬이라 서로가 서로의 이웃이 된다. 어느 설정에서나
  0.09%로 붙박이였다.
- **국소 선형성**(PCA로 $(\lambda_2-\lambda_1)/\lambda_2 > 0.8$) — 50 mm로 솎인 구름에서는
  **실루엣 경계도 선형**이다. 7~8%에서 안 움직였고, 아래 표에서 cliff가 0%가 되어도 7.60%로 남아
  정상 기하를 재고 있었음이 확인된다.

쓸 수 있었던 것은 원인에 붙인 물리적 정의였다: **깊이 절벽 위에 앉은 픽셀.** depth 이미지에서 3×3 깊이
범위가 10 cm를 넘으면 그 픽셀은 전경과 배경 사이에 보간된 것이다. 그 라벨을 최종 클라우드에 투영해 센다.

## 실측 (`capture/` 한 프레임, 640×480, f 383, 다운샘플 50 mm)

| 설정 | 최종 점 | **cliff%** | linear% | adj p50 | adj p90 | 다운샘플 전 |
| --- | --- | --- | --- | --- | --- | --- |
| thr 0.0 · r2 (이전 기본) | 1,129 | 1.06% | 7.82% | 3.22° | 7.84° | 262,291 |
| thr 0.5 · r2 | 1,104 | 0.91% | 7.45% | 3.18° | 7.67° | 246,358 |
| thr 0.7 · r2 | 1,074 | 0.09% | 8.04% | 3.16° | 7.59° | 235,412 |
| thr 0.9 · r2 | 1,019 | 0.00% | 7.62% | 3.14° | 7.50° | 220,108 |
| **thr 0.9 · r4 (현재)** | **1,009** | **0.00%** | 7.60% | **1.89°** | **4.97°** | 218,276 |
| thr 0.0 · r4 | 1,143 | **1.31%** | 6.92% | 1.97° | 5.54° | 268,729 |

## 고른 값과 근거

**점수 문턱값 0.9** — 목표 2. `c_nb`(같은표면 이웃/8)는 절벽 위 픽셀을 정확히 겨냥한다. 그 픽셀은 자기
표면 위에 이웃이 없어 점수가 0으로 떨어진다. 0.7이면 0.09%, 0.9면 **하나도 남지 않는다.**

**`planeFitRadius` 4** — 목표 3. 인접 법선 불일치가 3.14° → 1.89°. 넓은 창은 정확도와 수율을 **같이**
올린다(구멍이 있어도 남은 표본으로 적합하므로).

**둘은 짝이어야 한다.** 마지막 줄이 그 증거다: 반경만 올리고 문턱값을 0에 두면 cliff가 1.06% →
**1.31%로 악화**된다. 넓은 적합은 실 위에서도 살아남기 때문이다.

**50 mm 다운샘플** — 목표 1. 218,276 → **1,009점**. 다운샘플이 목표 2를 어렵게 만드는 원인이기도 하다:
표면은 99.5% 솎이지만 실은 각 점이 자기 복셀을 차지해 거의 다 남아, 최종 클라우드에서 쓰레기 비율이
스무 배 가까이 증폭된다. **그래서 노이즈 제거는 반드시 다운샘플 앞이어야 하고, 체인이 이미 그 순서다.**

## 시도했다가 되돌린 것

측정이 이득을 보이지 않아 **넣지 않았다.** 둘 다 구현해서 재본 뒤 되돌렸다.

- **입사각 게이트** — 문턱값 0.9에서 cliff가 이미 0%라 더 지울 것이 없었다(1,003점 대 1,009점, 나머지
  지표 동일).
- **복셀당 최소 표본 수** — 표면과 노이즈를 같은 비율로 지웠다. minVox 32에서 점이 1,009 → 778로
  줄지만 cliff는 이미 0이고 다른 지표도 안 움직였다.

---

# Pipeline 통합 (2026-09-05)

`Pipeline::AcquisitionThread`(`src/Pipeline/Acquisition/`)가 이 프론트엔드를 직접 돌린다. 감싸는
클래스는 없다 — depth 이미지에서 `Frame`을 만드는 방법이 이것 하나뿐이라 인터페이스로 뺄 축이 아니다.
`AcquisitionConfig::source`가 `Realsense`(라이브 D435) / `RealsenseFile`(녹화) / `PlyFolder`를 고른다.

## 무엇을 대체했나

이전 `DepthCameraFrameSource`는 전부 CPU였다 — `PrefilterDepth` 후 `BackprojectDepth`의 픽셀 루프로
역투영과 법선 추정을 하고, 그다음 호스트에서 voxel 솎기를 했다. 지금은 프레임이 `RealSensePipeline`으로
가서 점수·문턱값·역투영·법선·다운샘플·압축을 전부 디바이스에서 하고 **살아남은 점만 읽어온다.**
CPU 경로는 아래 A/B로 근거를 남기고 삭제됐다.

그래서 `AcquisitionConfig::downsampleVoxel`은 0으로 둔다. 그건 readback **후**에 도는 CPU 축소이고,
여기 `DownSampleOptions`는 readback **전**에 솎으므로 전송량까지 준다.

입력은 두 갈래이고 손실이 붙는 것은 한쪽뿐이다.

- **라이브(`Realsense`)** — `D435DepthProvider`가 드라이버의 Z16 버퍼를 `DepthFrame::rawZ16`에 **그대로**
  실어 넘긴다. float을 거치지 않으므로 왕복 손실이 없고, `focalLengthPixels`/`baselineMeters`/`depthScale`도
  장치의 실제 캘리브레이션에서 온다(`MakeScoreOptions`) — 호출자가 그 숫자를 알 필요도, 조용히 틀릴
  방법도 없다.
- **녹화·테스트(`RealsenseFile`)** — `IDepthProvider`가 float 미터를 주므로 소스가 Z16으로
  재양자화한다. RealSense 녹화에서는 그 float이 애초에 같은 스케일의 Z16에서 나왔으므로 왕복이
  정확하고, 테스트가 한 양자 이내임을 고정한다.

## A/B — 같은 녹화, 같은 하류 (`capture/` 476프레임, `--trackers icp`)

```
icp_quality_diag --replay capture --trackers icp --frontend cpu
icp_quality_diag --replay capture --trackers icp --frontend gpu --gpu-downsample 0.01
```

| | CPU (`BackprojectDepth`) | GPU (`RealSensePipeline`) |
| --- | --- | --- |
| ICP align (프레임당) | 240.61 ms | **30.43 ms** (7.9배) |
| 맵 엔트리 | 3,208,407 | **616,294** (5.2배 적음) |
| tracker RMSE | 0.001214 | 0.001902 |
| 경로 길이 | 1.282 m | 1.238 m |
| 최대 step | 0.0112 m | **0.0085 m** |
| 추적 거부 | 1 (프레임 0, no-model) | 1 (동일) |

**ICP가 8배 빨라진다** — 점이 5배 적기 때문이고, 이게 통합의 근거다. 추적 품질은 유지된다: 거부가 늘지
않고(`TooFewInliers` 0), 최대 step은 오히려 작아져 움직임이 더 매끈하다.

잔차 RMSE는 0.0012 → 0.0019로 올라간다. 5배 성긴 맵에 대한 잔차이므로 오르는 것이 예상되는 방향이고,
827 mm 규모 장면에서 둘 다 1~2 mm다. **이 숫자로 "정확도가 나빠졌다"고 말할 수는 없다** — 같은 밀도에서
비교한 것이 아니기 때문이다. 그 비교가 필요하면 점 수를 맞춰 다시 재야 한다.

`--gpu-downsample 0.01`에 CPU reduce를 껐을 때(`--downsample 0`) 결과가 616,294로 **동일**하다. GPU
출력이 이미 10 mm 이상 떨어져 있어 CPU 축소가 지울 것이 없다는 뜻이고, 두 축소가 겹쳐 이중으로 솎이지
않는다는 확인이다.

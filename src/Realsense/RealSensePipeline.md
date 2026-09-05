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

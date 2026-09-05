# NormalEstimation 프로세스

`Realsense::NormalEstimation::RecordEstimate` 기준 (`src/Realsense/Algorithm/NormalEstimation.h`).
`ValidationMask.RemainValidDepth`가 역투영을 끝내고 `emitted`를 세운 뒤, 행 카운트 **전에** 한 번 돈다.
커널 하나(`NormalEstimation.EstimateNormal.glsl`), 16×16 워크그룹, 픽셀당 1 스레드.

이웃 탐색이 없다 — organized point cloud라 $(u\pm1, v\pm1)$이 곧 이웃이고, KD-tree 없이 O(1)로 닿는다.

## 1. 대상 선별과 허용치

- `emitted == 0`인 픽셀은 곧바로 반환한다.
- 같은표면 허용치를 센서 상수에서 유도한다. `Common.glsl`의 한 정의를 score 커널과 공유한다.

$$ \tau = k \cdot \sigma_z(z), \qquad \sigma_z = \frac{s \cdot z^2}{f \cdot B} $$

**왜:** `valid`가 아니라 `emitted`를 읽는다. 채택은 점수 문턱값이 이미 정했고, 이 패스는 아무도
요구하지 않은 픽셀에 비용을 쓰지 않는다.

## 2. 전략 디스패치

- `-D` 매크로가 고른 프래그먼트 하나가 `EstimateSurfaceNormal`을 구현한다.
- 세 이름이 등록돼 있다: `forward`(전방차분 삼각형), `central`(중앙차분), `planefit`(창 안 같은표면
  표본의 최소자승 평면, 기본값).
- 모든 프래그먼트는 `SameSurfaceSample` 하나를 통과한다 — 이웃은 존재하고, 측정을 갖고,
  $|z_p - z_c| \le \tau$여야 쓸 수 있다.
- 반환은 **방향이 정해지지 않은** 단위 법선과 결과 코드 셋 중 하나다: `OK` / `OUT_OF_DOMAIN`
  (스텐실이 이미지 밖) / `NO_SUPPORT`(스텐실은 맞는데 표본 부족).

**왜:** 전략 선택은 이미지 전체에 균일하므로 픽셀당 분기가 아니라 컴파일 변형이어야 한다. 디스패처
`.glsl`을 쓰는 이유는 glslang이 `#include MACRO`를 지원하지 않기 때문이다.

**불변식:** 방향 정렬·거부권·카운터는 전부 커널에 남는다 — 어느 원인에 거부가 청구되는지가 어느
프래그먼트가 컴파일됐는지에 좌우되면 안 된다.

## 3. 방향 정렬

$$ \hat n \leftarrow -\hat n \quad\text{if}\quad \hat n \cdot P > 0 $$

- 카메라 좌표계에서 점을 향하는 시선 벡터가 곧 점 $P$이므로, 내적의 부호가 판정의 전부다.

**왜:** 융합 가중치와 point-to-plane 잔차가 둘 다 법선 부호와 함께 뒤집힌다. 뒤집힌 법선은 작은
오차가 아니다.

## 4. 거부권과 카운터

- 성공하면 `g_normals[centre]`에 법선을 쓴다. **`emitted`는 건드리지 않는다.**
- 실패하면 `emitted = 0`으로 취소하고 원인별 카운터를 원자적으로 올린다
  (`outOfDomain` / `noSupport`).

**왜:** 이 패스는 거부권만 갖는다. 여기서 다시 채택하면 confidence가 거부한 픽셀이 두 번째 문으로
들어온다. 반대로 취소는 필요하다 — 법선 없는 점은 point-to-plane ICP도 가중 TSDF 융합도 못 쓰므로,
`emitted == 1`이 "좌표와 법선이 둘 다 있다"를 함의하게 만든다.

**불변식:** 두 카운터는 `ValidationScoreCounters`와 **합쳐지지 않는다.** 그쪽은 이미지 전체를
분할한다는 불변식이 있고, 이 둘은 나중 패스가 되돌린 픽셀을 센다.

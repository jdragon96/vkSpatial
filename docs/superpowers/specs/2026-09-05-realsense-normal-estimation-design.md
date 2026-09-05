# Realsense 법선 추정 — 설계

`src/Realsense/`의 D400 프론트엔드에 법선 추정을 붙인다. 정확도가 목표이고, 어느 추정기가 정확한지는
이 저장소가 이미 측정으로 답해 두었으므로 이 설계는 **새 알고리즘을 고르는 문서가 아니라, 이미 이긴
알고리즘을 새 모듈의 규약에 맞게 옮기는 문서**다.

날짜: 2026-09-05. 대상: `src/Realsense/Algorithm/`.

## 1. 지금 무엇이 문제인가

`Realsense::ValidationMask`는 점수 → 문턱값 → 역투영 → 압축까지 하고 **좌표만** 내보낸다. 하류의
point-to-plane ICP와 TSDF 융합은 둘 다 점당 법선을 요구하므로, 지금 상태로는 이 프론트엔드의 출력을
그대로 쓸 수 없다. `ValidationMask.ScatterValidPoints.glsl`의 주석이 그 자리를 미리 비워 두었다 —
"NormalEstimation이 들어오면 바인딩 한 쌍이 늘어난다".

## 2. 이미 있는 것 — 새로 만들 알고리즘은 없다

`src/Pipeline/Reconstruction/Algorithm/`에 세 추정기가 레지스트리와 함께 구현돼 있고, 정확도는
`docs/DEPTH_NOISE_FILTERING.md`에 측정돼 있다.

**합성 평면, 참값 대비 평균 각오차** ($\sigma_z$ = 2 mm, 횡방향 간격 3.9 mm):

| | `forward` | `central` | `planefit` |
| --- | --- | --- | --- |
| 각오차 | 50.6° | 32.8° | 12.5° |
| `forward` 대비 | 1× | 1.9× | 5.6× |

**`capture/` 477프레임 실측** (D435 640×480, 인접 방출 픽셀 간 법선 불일치 중앙값): prefilter **없는**
`planefit`(3.28°)이 5×5 prefilter를 건 `forward`(3.22°)와 동급인데 점 좌표를 뭉개지 않고, 점을 1.6%
**더** 방출하며, 스텐실 거부가 12.7배 줄어든다(306만 → 24만). 비용은 전 구성 0.3–0.7 ms/frame이고
**추정기 차이가 제출 오버헤드에 묻혀 측정되지 않는다.**

`NormalPlaneFit.glsl`은 닫힌 형식 대칭 3×3 고유분해, 트레이스 정규화, 중심점 원점 이동, 공선 표본
거부를 이미 갖추고 있다. 이식 대상은 이 코드다.

## 3. 결정된 사항

### 3-1. 세 전략을 레지스트리째 옮긴다 (planefit 기본값)

하나만 옮기지 않는 이유는 `normal_estimator_eval`류 비교의 가치가 **추정기 교차**에 있고, 이 저장소
규약이 "새 알고리즘은 새 클래스가 아니라 새 등록 이름으로 들어간다"이기 때문이다. 전략 축은 C++
가상함수가 아니라 **디스패처 `.glsl` + `-D` 매크로**로 가른다 — 선택이 이미지 전체에 균일하므로 픽셀당
분기가 아니라 컴파일 변형이어야 하고, `ValidationMask`의 score 변형 캐시와 같은 패턴이다.

### 3-2. 법선 실패는 `emitted`를 취소한다 (그 이상은 하지 않는다)

`emitted == 1`이 "좌표와 법선이 둘 다 있다"를 함의하게 한다. 법선 없는 점은 point-to-plane ICP도 TSDF
융합도 쓸 수 없으므로, 하류가 0 법선을 특례 처리하는 것보다 여기서 빼는 편이 싸다.

입사각 게이트(`[H6]`)는 **이번에 넣지 않는다.** 넣는다면 `cos(incidence)`를 score에 곱하는 연속 항이
맞지만, 그러면 score의 의미가 "법선 패스를 돌았느냐"에 의존하게 되고 이득을 따로 측정해야 한다.

### 3-3. "같은 표면"의 정의는 모듈 안에 하나만 둔다

이식에서 가장 조심할 지점이다. Pipeline의 법선 커널은 허용치를
$\tau = \max(\text{minimumDepthJump},\ \text{relative}\cdot z)$로 잡지만, Realsense 모듈은 이미 센서에서
유도된 정의를 score 커널에 갖고 있다:

$$ \tau = k \cdot \sigma_z(z), \qquad \sigma_z = \frac{s \cdot z^2}{f \cdot B} $$

한 모듈에 두 정의가 공존하면 `c_nb`가 세는 이웃과 `planefit`이 적합하는 표본이 달라진다. 따라서
`AxialNoiseSigma`를 `Algorithm/Common.glsl`로 올려 **두 커널이 같은 함수를 쓴다.** 법선 커널의
푸시상수는 `relativeDepthJump`/`minimumDepthJump` 대신
`subpixelRms`/`focalLengthPixels`/`baselineMeters`/`sameSurfaceSigmaMultiplier`를 받는다.

### 3-4. `[H4]`/`[H5]`는 이식하지 않는다

`c_nb = |같은표면 이웃|/8`이 `[H5]`의 연속판이고, 단차 인접(`[H4]`)은 `c_nb < 1`로 이미 반영된다.
점수 문턱값이 그것을 소비하므로 법선 커널이 다시 판정하면 같은 검사를 두 번 하는 것이다.

### 3-5. 거부는 두 종류로 나누어 관측한다

`NormalEstimationCounters { outOfDomain, noSupport }`를 별도 버퍼로 둔다. 스텐실이 이미지 밖으로
나가는 것(모든 추정기가 잃는 액자, 넓을수록 더 잃음)과 스텐실이 들어맞는데도 같은표면 표본이 부족한
것(장면이 거부한 픽셀)은 성격이 다르다. `ValidationScoreCounters`에는 넣지 않는다 — 그쪽은 이미지
전체를 분할한다는 불변식이 있다.

score 카운터와 달리 **매크로로 끄지 않고 항상 쓴다.** score 커널은 모든 픽셀이 다섯 버킷 중 하나에
들어가므로 픽셀당 atomic이 필요하지만, 여기서는 거부된 픽셀만 원자적으로 더하면 된다 — 정상 프레임에서
드문 사건이라 상시 켜 둘 비용이 아니다.

### 3-6. 버퍼와 순서는 `ValidationMask`가 계속 소유한다

`NormalEstimation`은 전략 레지스트리와 커널 변형 캐시만 소유하고, 실행은
`RecordEstimate(batch, vertices, properties, normals, counters, width, height, options)`로 받는다.
파사드가 수명·라우팅을 갖고 구현이 알고리즘을 갖는 이 저장소의 형태를 따른다.

`NormalEstimationOptions`의 기본값은 `estimator = "planefit"`, `planeFitRadius = 2`(k = 5),
`minimumPlaneFitSamples`는 감싸는 구현체의 멤버 기본값과 정확히 같게 둔다 — 노브를 노출하는 변경이
동작을 바꾸면 안 된다.

## 4. 만들 것

```
src/Realsense/Algorithm/
  NormalEstimation.h                     클래스 + 이름→매크로 레지스트리 + Options
  NormalEstimation.EstimateNormal.glsl   커널
  NormalEstimation.Strategy.glsl         디스패처 + SameSurfaceSample 공용 정의
  NormalEstimation.ForwardDifference.glsl
  NormalEstimation.CentralDifference.glsl
  NormalEstimation.PlaneFit.glsl
```

`NormalEstimation.Depth2Normal.glsl`은 삭제한다 — 바인딩 1번이 두 번 선언된 미완성 스텁이고, 위 커널이
그 자리를 대신한다.

`Common.glsl`에 `AxialNoiseSigma`와 `SameSurfaceTolerance`를 올린다.

`ValidationMask`의 변경:

| # | 패스 | 변경 |
| --- | --- | --- |
| 1 | RemainValidDepth | 그대로 |
| **2** | **EstimateNormal** | **신규. `m_normals` 작성, 법선 실패 시 `emitted = 0`** |
| 3 | CountEmittedPerRow | 그대로 |
| 4 | ScanRows | 그대로 |
| 5 | ScatterValidPoints | 바인딩 2개 추가 (normals in, compactNormals out) |

순서가 강제된다: 법선이 `emitted`를 지우는 일은 세는 것보다 **먼저**여야 한다.

버퍼 둘이 는다 — `m_normals`(device-local, 디바이스 밖으로 안 나감), `m_compactNormals`(readback,
`m_points`와 평행). 읽기는 `DownloadValidNormals()`가 맡고 인덱스가 점과 1:1로 대응한다.

## 5. 테스트 전략

`test/test_realsenseNormalEstimation.cpp`. 해석적 도형을 Z16 depth로 합성해 **GT 법선 대비** 중앙
각오차를 잰다. 카메라도 녹화도 필요 없고 결정적이다.

1. **무잡음 기울어진 평면** — 세 전략 모두 참값과 거의 일치한다. 역투영·부호·정규화를 핀한다.
2. **$\sigma_z$ = 2 mm 잡음 평면** — 단언은 **순서**(`planefit < central < forward`)와 **비**(planefit이
   forward보다 최소 3배 낮다)로 건다. §2의 12.5°/32.8°/50.6°는 방향을 잡아주는 참고값이지 단언값이
   아니다 — 합성 장면의 횡방향 간격과 난수열이 다르면 절대 각도는 움직이지만 비는 이론
   비($1.41/0.71/0.32$)를 따라간다. 난수는 고정 시드로 결정적으로 만든다. planefit의 두 함정(원점
   이동·트레이스 정규화)은 이 테스트에서만 드러나므로, **원점 이동을 빼는 뮤테이션으로 이 단언이 실제로
   빨개지는지 확인한다.**
3. **깊이 단차 두 평면** — 경계에서 법선이 늘어지지 않는다.
4. **방향 일관성** — 모든 방출 픽셀에서 $\hat n \cdot P \le 0$.
5. **게이트** — `emitted == 1`이면 반드시 단위 법선이 있다 (§3-2를 핀).
6. **창에 구멍** — planefit은 방출하고 차분 스텐실은 잃는다 (실측의 +1.6%가 나오는 성질).
7. **레지스트리 합의** — 등록된 이름이 전부 빌드되고, 모르는 이름은 던진다.

회귀 확인은 기존 `RealsenseValidationScore` 10개와 `validation_score_lab --replay capture --sweep`이
그대로 통과하는 것으로 한다.

## 6. 비목표

- **SAT(적분영상)** — 창 크기 무관 O(1)은 이 저장소에 없는 문제를 푼다(비용이 제출 오버헤드에 묻힘).
  더 중요하게 prefix sum은 깊이 불연속 이웃을 창에서 뺄 수 없어 같은표면 게이트와 충돌한다.
- **bilateral/guided prefilter** — 실측이 `planefit`이 그것을 대체한다는 것이고, prefilter는 TSDF에
  들어갈 점 좌표를 뭉갠다.
- **적응형 창 반경**($r \approx f R / z$), **곡률 신뢰도**($\lambda_0 / \sum \lambda$) — 둘 다 정확도에
  실효가 있을 후보지만 이번 범위 밖이다. 새 등록 이름으로 나중에 들어온다.
- **입사각 게이트** — §3-2 참조.
- **`normal_estimator_eval`의 Realsense 이식** — dense grid 다운로드용 진단 접근자를 열어야 해서
  범위가 커진다. §5의 합성 GT가 정확도 판정을 맡는다.

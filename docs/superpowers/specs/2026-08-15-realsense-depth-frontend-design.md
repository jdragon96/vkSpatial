# RealSense depth 프론트엔드 — 설계

> 목적: 재구성 파이프라인에 **실제 센서 입력**을 붙인다. 지금 파이프라인이 먹는 것은 포즈도
> 시점도 없는 전체장면 점군이고, 그래서 트래커가 `identity` 말고는 동작하지 않는다.
>
> 대상: `src/Pipeline/Reconstruction/`
> 작성일: 2026-08-15 · 상태: 리뷰 대기
> 관련: [`TSDF_ROUTING.md`](../../TSDF_ROUTING.md)

---

## 1. 지금 무엇이 문제인가

`scan_out/`의 프레임은 **각각이 이미 장면 전체를 덮는다** — 세 프레임을 재보니 모두
x ∈ [−95, 95]로 190 m 전 범위였다. 한 시점에서 본 부분집합이 아니다.

그 결과 세 가지가 무너져 있다.

- `EstimateCamera`(중심 + 평균법선 × 3×대각)가 의미 없는 값을 낸다. 장면을 감싸는 점들의
  평균 법선은 상쇄된다.
- 시점이 없으니 point-to-plane ICP가 풀 문제가 없다 → `identity`가 유일하게 동작하는 트래커.
- 점 간격(~0.7 m)이 어떤 복셀 크기보다도 넓다 → MC 셀의 코너가 안 차서 메시에 구멍,
  dense 판정도 항상 0.

D435는 이 셋을 한 번에 뒤집는다. 0.3–5 m 장면에서 점 간격이 ~2 mm라 **복셀보다 촘촘**하고,
매 프레임이 한 시점에서 본 부분집합이며, 손으로 들고 움직이면 진짜 드리프트가 쌓인다.

## 2. 이미 있는 것 — 새로 만들 것은 provider 하나

`DepthCameraFrameSource.h`에 자리가 이미 파여 있다.

```cpp
struct CameraIntrinsics { float fx, fy, cx, cy; int width, height; };
struct DepthFrame       { std::vector<float> depth; };  // metres, <=0 = invalid
class  IDepthProvider   { virtual const CameraIntrinsics &Intrinsics() const = 0;
                          virtual bool Grab(DepthFrame &out) = 0; };
Frame  BackprojectDepth(const DepthFrame &, const CameraIntrinsics &);
```

`BackprojectDepth`는 역투영과 **법선 추정까지 이미 한다** — 정렬된 격자의 이웃 차분 외적,
부호는 카메라 쪽으로.

$$n = \widehat{(P_{u+1,v} - P_{u,v}) \times (P_{u,v+1} - P_{u,v})}, \qquad n_z > 0 \Rightarrow n \leftarrow -n$$

즉 이 스펙의 범위는 **provider 두 개(장치/재생) + 법선 보강**이다. 파이프라인 구조는 손대지 않는다.

## 3. 결정된 사항

### 3-1. 깊이 불연속 검사를 넣는다 (없으면 실측에서 바로 깨진다)

현재 `BackprojectDepth`는 이웃이 유효하기만 하면 외적한다. 물체 경계에서는 **앞면과 뒷면의 점**으로
외적하게 되어 법선이 표면과 무관한 방향을 가리킨다. D435는 스테레오라 경계에 "날아다니는 픽셀"이
특히 많다.

이웃과의 깊이 차가 임계값을 넘으면 그 픽셀에서 법선을 만들지 않는다.

$$|z_n - z_c| > \max(\epsilon_{\min},\; \kappa z_c) \;\Rightarrow\; \text{버림}$$

$\kappa$는 상대 임계값(기본 0.02 = 2%), $\epsilon_{\min}$은 절대 하한(기본 5 mm). **깊이에 비례**해야
하는 이유는 스테레오 깊이 오차가 $z^2/(fb)$로 커지기 때문 — 고정 임계값을 쓰면 가까이서는
과하게 버리고 멀리서는 못 거른다.

법선은 **버리되 점은 남긴다**. 법선 없는 점도 TSDF의 projective 경로에는 쓸 수 있고, 점까지 버리면
경계가 통째로 사라진다. → `Frame`에 법선 유효 플래그가 없으므로, 이번에는 **점도 함께 버린다**.
`Frame`의 계약이 `pts.size() == nrm.size()`이고 파이프라인 전체가 그것을 가정한다.

### 3-2. 녹화는 `DepthFrame` 수준에서 한다

역투영 **이전**의 원시 depth를 저장한다. 그래야 재생이 역투영과 법선 추정을 **똑같이 다시 통과**하고,
그 단계의 변경을 재생 데이터로 검증할 수 있다. `Frame`을 저장하면 프론트엔드가 고정돼버린다.

형식은 의존성 없는 최소 구성으로 한다.

```
<dir>/intrinsics.txt      fx fy cx cy width height     (공백 구분 한 줄)
<dir>/depth_0000.bin      float32 × width*height, row-major, metres
```

PNG를 쓰지 않는 이유: 16비트 정수로 낮추면 스케일을 다시 정해야 하고, 그 스케일 실수는 조용히
전체 재구성을 망친다. float 원본이 디버깅 가능하다.

### 3-3. librealsense는 선택적 의존으로 링크한다

이 머신에는 Homebrew로 2.58.3이 있지만, 없는 머신에서도 빌드가 되어야 한다. CMake가 찾으면
`RealSenseDepthProvider`를 컴파일하고 `VKBVH_HAS_REALSENSE`를 정의한다. 못 찾으면 그 파일만
빠지고 나머지는 그대로 빌드된다.

### 3-4. 컬러는 이번에 다루지 않는다

`VoxelAttribute`가 빈 구조체이고 파이프라인이 색을 나르지 않는다. depth만으로 재구성이 도는 것을
먼저 확인한다.

## 4. 만들 것

| 파일 | 내용 |
|---|---|
| `Reconstruction/RealSenseDepthProvider.{h,cpp}` | `IDepthProvider` 구현. `rs2::pipeline`, `depth_frame` → metres, `rs2_intrinsics` → `CameraIntrinsics` |
| `Reconstruction/DepthRecorder.{h,cpp}` | `IDepthProvider`를 감싸 `Grab` 결과를 디스크에 떨구는 데코레이터 |
| `Reconstruction/RecordedDepthProvider.{h,cpp}` | 그 디렉터리를 다시 읽는 `IDepthProvider` |
| `DepthCameraFrameSource.h` (수정) | `BackprojectDepth`에 불연속 임계값 파라미터 추가 |
| `example2/depth_capture.cpp` | 장치 → 녹화, 또는 녹화 → 파이프라인 재생. 하드웨어 없이도 재생은 동작 |

D435 기본 설정: 848×480 @ 30 fps, depth만. `rs2::depth_sensor::get_depth_scale()`로 uint16을
미터로 바꾼다 — 하드코딩하지 않는다(장치마다 다르다).

## 5. 테스트 전략

**하드웨어 없이 검증 가능한 것부터.** CI에도 이 머신에도 장치가 항상 붙어 있지 않다.

| 무엇을 | 어떻게 |
|---|---|
| 불연속 검사가 경계를 거른다 | 합성 depth: 좌우로 $z=1$과 $z=2$인 계단. 계단을 가로지르는 법선이 하나도 안 나와야 한다 |
| 평면에서 법선이 정확하다 | 합성 depth: 카메라를 향한 평면. 모든 법선이 $(0,0,-1)$에 수렴 |
| 기울어진 평면 | 알려진 각도의 평면 → 법선이 그 각도와 일치 (불연속 검사가 정상 표면을 죽이지 않는지) |
| 녹화 왕복 | `DepthRecorder`로 쓰고 `RecordedDepthProvider`로 읽어 intrinsics와 depth가 비트 단위 일치 |
| 재생이 프레임을 만든다 | 녹화된 디렉터리 → `DepthCameraFrameSource` → `Frame`의 점·법선 수가 같고 0이 아님 |
| 장치 | 하드웨어가 있을 때만 도는 수동 스모크. 자동 테스트로 만들지 않는다 |

계단 테스트가 핵심이다 — **불연속 검사를 지우면 반드시 빨개져야 한다.**

## 6. 비목표

- 컬러/텍스처 — `VoxelAttribute`가 빈 구조체다
- IMU — D435에는 없다(D435i가 그것)
- 포즈 그래프/루프 클로저 연결 — 다음 스펙
- 정답 궤적 — D435는 주지 않는다. ATE는 TUM RGB-D 같은 벤치마크에서 재고, 여기서는 루프 폐합
  오차와 표면 일관성으로 대신한다

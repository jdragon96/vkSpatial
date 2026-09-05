# LocalRegistration — point-to-plane ICP

이미 대략 맞춰진 두 점군을 마지막까지 붙인다. 초기 추정이 있다는 전제이고, 전역 탐색은
`src/GlobalRegistration`이 한다.

## 구성

CPU와 GPU 두 solver가 있고 **둘은 단계가 아니라 대안**이다. 그래서 이 모듈에는 조합할 것이 없고
파이프라인 클래스도 없다 — 어느 쪽이 도는지는 `Pipeline`의 `TrackerRegistry`가 이름으로 고른다
(`icp` → GPU, `icp-cpu` → CPU). 선택만 넘기는 래퍼를 두면 아무것도 하지 않는 층이 하나 더 생긴다.

```
Algorithm/
  PointToPlaneIcp.h                    CPU. AlignPointToPlaneIcp() 하나, 헤더 온리
  GpuPointToPlaneIcp.h / .cpp          GPU. 대응 탐색과 정규방정식 누적이 디바이스에서
  GpuPointToPlaneIcp.Iterate.glsl      그 커널
```

## 왜 point-to-plane인가

point-to-point는 평면 위에서 미끄러진다 — 두 평면 조각이 어긋나 있어도 잔차가 0에 가깝게 나온다.
법선 방향 거리만 재면 그 자유도가 사라져서 평면이 많은 실내 장면에서 수렴이 훨씬 빠르다. 대신 **법선이
필요하고**, 법선이 나쁘면 그 대가를 그대로 치른다.

## 쓸 때 알아야 할 것

- **대응 거리는 맵 해상도에 맞춘다.** 고정값을 쓰면 대응점이 너무 적게 잡히는 동시에 GPU LocalGrid
  셀 수가 폭발한다. `ModelSnapshot::voxel`이 그 값을 싣고 온다.
- **`maxStepMeters`는 물리적 게이트다.** solve가 prior에서 한 프레임에 불가능한 거리를 이동하면
  fitness가 좋아도 오수렴이다(`ETrackFailure::ImplausibleMotion`). 기본값은
  `max(0.08, 1.6×voxel)`.
- 실패 원인은 `valid` 하나로 뭉뚱그리지 않는다. `ETrackFailure`가 "맵이 아직 없다"와 "엉뚱한 점에
  걸렸다"를 가른다 — 정반대 대응을 요구하는데 `valid == false`만 보면 구분되지 않는다.

측정과 곡선은 `docs/ICP_REGISTRATION_QUALITY.md`, 방법 비교는 `docs/ICP_METHODS.md`.

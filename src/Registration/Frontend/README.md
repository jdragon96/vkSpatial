# Registration::Frontend

프레임 하나를 맵(또는 다른 프레임)에 맞추는 정합. 여러 프레임을 한꺼번에 푸는 것은 `../Backend`다.

두 갈래는 **단계가 아니라 대안**이다 — 어느 쪽이 도는지는 Pipeline의 `TrackerRegistry`가 고르고,
그 선택을 전달만 하는 래퍼는 아무 일도 하지 않으므로 여기에 조합 클래스가 없다.

| | 초기 추정 | 파일 |
|---|---|---|
| **Global** | 필요 없음 | `GlobalRegistrationPipeline.*` |
| **Local** | 필요함 | `PointToPlaneIcp.h` (CPU), `GpuPointToPlaneIcp.*` (GPU) |


---

## Global

두 점군을 **초기 추정 없이** 붙인다. 겹침이 작거나 위치를 잃었을 때 쓰고, 결과는 그대로 쓰기보다
`src/LocalRegistration`의 ICP에 넘겨 마무리한다.

### 구성

```
GlobalRegistrationPipeline.h / .cpp
  Estimate()          FPFH → RANSAC → refine 를 조합한다. 이 모듈의 결과물
  EstimateRansac()    대응 후보에서 RANSAC으로 강체 변환을 찾는다
  SolveRigidUmeyama() 대응 세 쌍에서 닫힌 형식 강체 해
```

특징 계산(FPFH)과 매칭은 `Features`에 있다 — 정합만의 것이 아니라 기술자 일반이라서다.
이 모듈은 그것을 **조합**한다.

### 왜 RANSAC 뒤에 refine이 붙나

RANSAC은 인라이어 집합을 찾지 대응을 최적화하지 않는다. 찾은 집합 위에서 Ceres로 다시 풀어야 그
집합이 지지하는 최선의 변환이 나온다. 그 두 단계를 하나로 합치면 이상치 하나가 해 전체를 끌고 간다.

### 스케일 의존

`RegistrationConfig`는 물리값이 아니라 **gain**을 담는다 — `normalRadiusGain`, `fpfhRadiusGain`,
`ransacInlierGain`. 실제 반경은 `voxelSize`를 곱해 파생된다. C++ 기본 멤버 초기화가 다른 멤버를
참조할 수 없어서이기도 하고, 데이터 스케일이 바뀌어도 한 값만 고치면 되기 때문이기도 하다.

**`voxelSize`를 데이터에 맞추지 않으면 나머지 전부가 조용히 틀어진다.** 이 모듈에서 가장 흔한 실패다.

### 쓰는 곳

`Pipeline`의 `global` 트래커(`GlobalRegistrationTracker`)가 이걸 부르고,
`RelocalizingIcpTracker`가 로컬 ICP가 놓쳤을 때 여기로 넘어간다.

비교 실험은 `docs/ICP_LOCAL_VS_GLOBAL.md`.

---

## Local

이미 대략 맞춰진 두 점군을 마지막까지 붙인다. 초기 추정이 있다는 전제이고, 전역 탐색은
`src/GlobalRegistration`이 한다.

### 구성

CPU와 GPU 두 solver가 있고 **둘은 단계가 아니라 대안**이다. 그래서 이 모듈에는 조합할 것이 없고
파이프라인 클래스도 없다 — 어느 쪽이 도는지는 `Pipeline`의 `TrackerRegistry`가 이름으로 고른다
(`icp` → GPU, `icp-cpu` → CPU). 선택만 넘기는 래퍼를 두면 아무것도 하지 않는 층이 하나 더 생긴다.

```
Algorithm/
  PointToPlaneIcp.h                    CPU. AlignPointToPlaneIcp() 하나, 헤더 온리
  GpuPointToPlaneIcp.h / .cpp          GPU. 대응 탐색과 정규방정식 누적이 디바이스에서
  GpuPointToPlaneIcp.Iterate.glsl      그 커널
```

### 왜 point-to-plane인가

point-to-point는 평면 위에서 미끄러진다 — 두 평면 조각이 어긋나 있어도 잔차가 0에 가깝게 나온다.
법선 방향 거리만 재면 그 자유도가 사라져서 평면이 많은 실내 장면에서 수렴이 훨씬 빠르다. 대신 **법선이
필요하고**, 법선이 나쁘면 그 대가를 그대로 치른다.

### 쓸 때 알아야 할 것

- **대응 거리는 맵 해상도에 맞춘다.** 고정값을 쓰면 대응점이 너무 적게 잡히는 동시에 GPU LocalGrid
  셀 수가 폭발한다. `ModelSnapshot::voxel`이 그 값을 싣고 온다.
- **`maxStepMeters`는 물리적 게이트다.** solve가 prior에서 한 프레임에 불가능한 거리를 이동하면
  fitness가 좋아도 오수렴이다(`ETrackFailure::ImplausibleMotion`). 기본값은
  `max(0.08, 1.6×voxel)`.
- 실패 원인은 `valid` 하나로 뭉뚱그리지 않는다. `ETrackFailure`가 "맵이 아직 없다"와 "엉뚱한 점에
  걸렸다"를 가른다 — 정반대 대응을 요구하는데 `valid == false`만 보면 구분되지 않는다.

측정과 곡선은 `docs/ICP_REGISTRATION_QUALITY.md`, 방법 비교는 `docs/ICP_METHODS.md`.

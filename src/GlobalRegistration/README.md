# GlobalRegistration — 초기 추정 없는 정합

두 점군을 **초기 추정 없이** 붙인다. 겹침이 작거나 위치를 잃었을 때 쓰고, 결과는 그대로 쓰기보다
`src/LocalRegistration`의 ICP에 넘겨 마무리한다.

## 구성

```
GlobalRegistrationPipeline.h / .cpp
  Estimate()          FPFH → RANSAC → refine 를 조합한다. 이 모듈의 결과물
  EstimateRansac()    대응 후보에서 RANSAC으로 강체 변환을 찾는다
  SolveRigidUmeyama() 대응 세 쌍에서 닫힌 형식 강체 해
```

특징 계산(FPFH)과 매칭은 `Features`에 있다 — 정합만의 것이 아니라 기술자 일반이라서다.
이 모듈은 그것을 **조합**한다.

## 왜 RANSAC 뒤에 refine이 붙나

RANSAC은 인라이어 집합을 찾지 대응을 최적화하지 않는다. 찾은 집합 위에서 Ceres로 다시 풀어야 그
집합이 지지하는 최선의 변환이 나온다. 그 두 단계를 하나로 합치면 이상치 하나가 해 전체를 끌고 간다.

## 스케일 의존

`RegistrationConfig`는 물리값이 아니라 **gain**을 담는다 — `normalRadiusGain`, `fpfhRadiusGain`,
`ransacInlierGain`. 실제 반경은 `voxelSize`를 곱해 파생된다. C++ 기본 멤버 초기화가 다른 멤버를
참조할 수 없어서이기도 하고, 데이터 스케일이 바뀌어도 한 값만 고치면 되기 때문이기도 하다.

**`voxelSize`를 데이터에 맞추지 않으면 나머지 전부가 조용히 틀어진다.** 이 모듈에서 가장 흔한 실패다.

## 쓰는 곳

`Pipeline`의 `global` 트래커(`GlobalRegistrationTracker`)가 이걸 부르고,
`RelocalizingIcpTracker`가 로컬 ICP가 놓쳤을 때 여기로 넘어간다.

비교 실험은 `docs/ICP_LOCAL_VS_GLOBAL.md`.

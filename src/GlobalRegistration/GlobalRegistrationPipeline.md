# GlobalRegistration 프로세스

`Engine::Registration::Estimate` 기준 (`src/GlobalRegistration/GlobalRegistrationPipeline.cpp`).
초기 추정 없이 `src`를 `tgt`에 맞추는 변환을 낸다.

물리 반경은 전부 `cfg.voxelSize`에서 파생된다 — `RegistrationConfig`는 gain만 담는다. **`voxelSize`를
데이터 스케일에 맞추지 않으면 아래 모든 단계가 조용히 틀어진다.**

## 1. Downsample

- 두 점군을 `cfg.voxelSize`로 voxel 축소한다 (`Engine::Features::DownsampleVoxel`).

**왜:** 뒤 단계의 비용이 점 수에 제곱으로 붙는다. 그리고 FPFH는 밀도가 고르지 않으면 같은 표면에서도
다른 기술자를 낸다 — 축소가 그 밀도를 균일하게 만든다.

## 2. Normal + FPFH

- `normalRadiusGain × voxelSize` 반경으로 법선을, `fpfhRadiusGain × voxelSize`로 FPFH를 만든다.
- 둘 다 `Engine::Features`에 있다. 정합만의 것이 아니라 기술자 일반이라서다.

**불변식:** FPFH 반경은 법선 반경보다 커야 한다. 기술자가 법선이 서술하는 것보다 넓은 이웃을 봐야
구분력이 생긴다.

## 3. 대응 후보

- 특징 공간 최근접으로 `src`→`tgt` 대응을 만든다 (`Engine::Features::FeatureMatching`).
- 이 단계의 출력은 **후보**다. 상당수가 틀렸다는 전제로 다음 단계가 설계돼 있다.

## 4. RANSAC

- `cfg.ransacIters`만큼: 대응 3쌍을 뽑아 `SolveRigidUmeyama`로 가설 변환을 만들고,
  `ransacInlierGain × voxelSize` 안에 들어오는 대응 수로 점수를 매긴다.
- 최고 득점 가설의 **인라이어 집합 전체**로 다시 한 번 `SolveRigidUmeyama`.

$$ T^* = \arg\max_{T} \left| \{\, (p,q) : \lVert T p - q \rVert < \tau \,\} \right| $$

**왜 닫힌 형식인가:** 가설 하나가 한 번의 호출이라 수천 개를 뽑을 수 있다. 가설마다 반복 해를 풀면
그 예산이 나오지 않는다. `Algorithm/RigidTransform.h` 참조.

**왜 3쌍인가:** 강체 변환이 잘 정의되는 최소 개수다. 더 뽑으면 이상치가 섞일 확률이 올라가고, 적게
뽑으면 해가 퇴화한다.

## 5. Refine

- 인라이어 위에서 Ceres로 다시 푼다. 손실 스케일은 `ceresLossGain × voxelSize`.

**왜 RANSAC만으로 끝내지 않나:** RANSAC은 인라이어 **집합**을 찾지 대응을 최적화하지 않는다. 그 집합이
지지하는 최선의 변환은 다시 풀어야 나온다. 반대로 두 단계를 합치면 이상치 하나가 해 전체를 끌고 간다.

## 6. 결과

- `RegistrationResult{ T, fitness, inlierCount, ... }`.
- 이 변환은 **초기 추정**으로 쓰라는 것이다. 마무리는 `src/LocalRegistration`의 point-to-plane ICP가
  한다 — `RelocalizingIcpTracker`가 그 두 단계를 잇는다.

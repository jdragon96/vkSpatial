# TSDF 로컬 정합 프로세스

`RegistrationThread::Run` → `GpuIcpTracker::Track` 기준 (`src/Pipeline/Registration/RegistrationThread.cpp`, `GpuIcpTracker.cpp`). 내부 solve는 [ICP_GPU_SOLVE_PROCESS.md](../ICP_GPU_SOLVE_PROCESS.md)에 따로 있다.

Prior 포즈에서 출발해 프레임을 TSDF 맵에 붙이는 프레임-투-모델 추적 경로다. 레지스트리 이름은 `icp`(GPU)이고 `icp-cpu`는 타깃 크롭만 없을 뿐 같은 파라미터·같은 판정을 쓴다.

## 1. Prior 선택

- 직전 채택 포즈 $P_{-1}$을 기본 prior로 쓴다.
- 두 조건이 **다 차면** 상수속도 외삽으로 바꾼다.

$$
prior = P_{-1} \cdot (P_{-2}^{-1} P_{-1}), \qquad \text{if } adoptions \ge 2 \ \wedge\ \lVert \Delta t_{-1} \rVert > 0.02\,\text{m}
$$

- 왜: 연속 2회 채택이라야 delta가 1-프레임 간격이 되고, 2 cm 문턱 아래에서는 외삽이 신호가 아니라 포즈 노이즈를 재적용해 주기적 거부 진동을 만든다(측정 근거는 `ICP_REGISTRATION_QUALITY.md` §9.6).
- 정합이 거부되면 채택 카운터를 0으로 되돌리고 포즈는 $P_{-1}$을 유지한다.
- 맵은 latest-wins `Mailbox`에서 최신 스냅샷 하나를 꺼내 쓴다.

## 2. 로컬 타깃 크롭 (TSDF 스냅샷 → 점군)

1. prior로 프레임 점을 월드로 보내 AABB를 잡고, 각 축을 `maxCorrDist`만큼 넓힌다.
2. 그 박스 안의 TSDF 엔트리만 모은다 — 스냅샷의 버킷 인덱스(`ForEachEntryInBox`, 버킷 한 변 = 16 voxel)를 타고, 없으면 선형 스캔으로 폴백한다.
3. 엔트리마다 서브복셀 표면점을 만든다 ($\mu$ = `truncationDistance`).

$$
q_i = center_i - tsdf_i \cdot \mu \cdot n_i
$$

- 왜 크롭하는가: 타깃 비용이 맵 전체가 아니라 프레임 이웃에 비례하고, GPU NN 그리드 셀 수도 함께 묶인다.
- 타깃이 3점 미만이면 `NoLocalTarget`으로 즉시 반환한다 (맵이 아직 거기까지 자라지 않았다).
- 마지막에 `SortTargetIntoCanonicalOrder`로 좌표 사전순 정렬한다 — 해시 순회 순서가 실행마다 달라지면 동점 최근접이 갈려 재현성이 깨진다.

## 3. 해상도 앵커링

- 맵 해상도(`model->voxel`)에 solve 파라미터를 맞춘다.

$$
maxCorrDist = 2 \cdot voxel, \qquad huberScale = voxel
$$

- 왜: 고정 대응 거리는 거친 맵에서 대응을 너무 적게 잡는 **동시에** GPU LocalGrid 셀 수를 폭발시킨다.
- 게이트 기본값: `minFitness` = 0.4, `maxStepMeters` = $\max(0.08,\ 1.6 \cdot voxel)$. 호출자가 명시로 준 값은 절대값으로 그대로 쓴다.

## 4. Solve

- `GpuPointToPlaneIcp::Solve(frame.pts, frame.nrm, target, prior, params)`를 부른다.
- Centroid 정렬 → 타깃 전용 균일 그리드(Solve당 1회) → point-to-plane 반복 → un-centre 순이다. 상세는 [ICP_GPU_SOLVE_PROCESS.md](../ICP_GPU_SOLVE_PROCESS.md).
- `icp-cpu`는 같은 자리에서 `AlignPointToPlaneIcp`를 부른다 — 두 트래커는 solve를 똑같이 판정해야 하므로 기본값이 서로의 거울이다.

## 5. 게이트 · 분류 · 융합

- 물리 스텝 게이트를 먼저 건다 — prior 대비 이동이 `maxStepMeters`를 넘으면 fitness가 좋아도 기각하고 포즈를 prior로 되돌린다.

$$
\lVert t_{icp} - t_{prior} \rVert > maxStepMeters \ \Rightarrow\ \texttt{ImplausibleMotion}
$$

- 왜: fitness 게이트가 오수렴을 다 못 잡는다 — 실측에서 fitness 0.571로 0.225 m를 주장한 solve가 통과해 이후 맵 전체를 오염시켰다.
- 그 외 실패는 인라이어 < `minInliers`면 `TooFewInliers`, 아니면 `LowOverlap`이다.
- `valid` = 인라이어 ≥ `minInliers` **AND** fitness ≥ `minFitness`.
- 융합 여부는 원인별로 갈린다(`ShouldFuse`): `TooFewInliers`/`LowOverlap`/`ImplausibleMotion`은 융합하지 않고, `NoModel`/`NoLocalTarget`은 융합한다 — 오염시킬 맵이 없고, 거부하면 맵이 부트스트랩되거나 자라지 못한다.
- 거부는 원인별 카운터로, 건너뛴 융합은 `SkippedFusions()`로 관측한다.

## 부록 — `icp+global` 폴백

- 위 로컬 추적이 **맵이 멀쩡한데도**(`TooFewInliers`/`LowOverlap`/`ImplausibleMotion`) 연속 3회 실패하면 글로벌 정합([TSDFGlobalRegistration.md](TSDFGlobalRegistration.md))을 1회 시도한다.
- 성공한 글로벌 포즈는 그대로 채택하지 않고 **그것을 prior로 로컬 ICP를 다시 돌려** 같은 게이트를 통과할 때만 채택한다.
- 실패하면 글로벌이 아니라 **로컬의 실패 결과**를 반환한다 — 글로벌 실패를 그대로 돌려주면 `NoLocalTarget`류로 분류돼 나쁜 포즈가 융합된다.

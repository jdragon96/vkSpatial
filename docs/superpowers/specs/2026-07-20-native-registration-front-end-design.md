# 네이티브 점군 정합 front-end (`Engine::Spatial::Registration`) 설계

> 알고리즘 레퍼런스: KISS-Matcher (MIT-SPARK, arXiv 2409.15615, MIT license) — **링크하지 않고** 알고리즘 구조를 재구성한다.
> 정합 맥락: [`docs/ICP_LOCAL_VS_GLOBAL.md`](../../ICP_LOCAL_VS_GLOBAL.md), [`docs/ICP_METHODS.md`](../../ICP_METHODS.md)

## 배경 / 목적

`Engine::Spatial::DirectionalTSDF::Integrate(points, normals, cameraPos, aabbCenterHint)`는 **프레임이 이미 정합(registered)돼 있다고 가정**한다 (설계 문서가 "ICP/Tracking은 상류"로 명시). `directional_tsdf_chair_benchmark`도 이미 등록된 `scanData/frame_*.ply`를 그대로 통합했다 — 즉 "raw 센서 프레임을 어떻게 정렬해 넣는가"라는 정합 front-end가 비어 있다.

이 스펙은 그 갭을 **VkLBVH 네이티브로 재작성**해 채운다. KISS-Matcher가 제공하는 global(feature+robust) 정합을, 외부 라이브러리 링크(TBB/ROBIN/TEASER++, macOS/Linux 전용, CPU-only) 대신 **순수 Eigen + Vulkan compute로 재구성**한다. 이유: VkLBVH의 크로스플랫폼(Windows+macOS) 타깃과 GPU-first 아키텍처에 맞추고, 외부 의존성 없이 코드를 소유하기 위함이다 (링크 vs 재작성 비교와 그 결론은 브레인스토밍에서 확정).

핵심 caveat: **어려운 robust-estimation 수학은 처음부터 새로 짜지 않는다.** M1은 재구현이 안전한 estimator(FGR/RANSAC)로 시작하고, certifiable robustness가 필요하면 TEASER의 GNC 코어만 벤더링한다(M3).

## 범위 / 마일스톤

- **M1 (이번 스펙의 핵심)**: CPU end-to-end 정합 — voxel downsample → FPFH descriptor → feature matching → robust SE(3)(FGR, RANSAC baseline) → `{valid, T, inliers}`. **알려진-변환 회복 검증** + raw chair 프레임 정합 데모. 순수 Eigen(+표준 병렬), 크로스플랫폼, 외부 deps 0.
- **M2 (후속)**: FPFH·matching을 Engine::Core compute(Vulkan)로 GPU 가속. **선행: large-N BVH 버그**([`KNOWN_ISSUES_engine_core_large_n.md`](../../KNOWN_ISSUES_engine_core_large_n.md)) 해결 또는 downsample로 N↓.
- **M3 (선택)**: 극단 outlier 강인성이 필요하면 TEASER++ GNC solver 코어만 벤더링(Eigen-only, 크로스플랫폼). from-scratch 재작성 금지.

### 범위 밖
- 외부 라이브러리(KISS-Matcher/TBB/ROBIN/TEASER++) **링크** — 안 한다(알고리즘 레퍼런스로만 소스 참조).
- Local ICP(정밀화) — 별도 작업. 이 스펙은 global(coarse) 정합. (파이프라인상 global→local ICP→TSDF; local ICP는 후속.)
- Loop closure / pose-graph back-end — SLAM 단계, 후속.
- GPU 가속(M2)·certifiable solver(M3) — 이번 스펙은 M1(CPU 정확성)에 집중.
- Windows CI — M1은 크로스플랫폼으로 **설계**하되 검증은 macOS(현재 개발기). Windows 빌드 검증은 후속.

## 아키텍처

새 모듈 `src/Engine/Spatial/Registration.{h,cpp}` (+ 스테이지별 헬퍼). `SpatialIndex`/`DirectionalTSDF`와 나란히 `Engine::Spatial`에 둔다.

### 파이프라인
```
GlobalRegistration::Estimate(src, tgt):
  src, tgt : std::vector<Eigen::Vector3f> (+ 선택적 노멀)
   → 1. voxel downsample        (keypoint set 축소)
   → 2. FPFH descriptor         (각 점의 33-bin 히스토그램; 노멀 필요)
   → 3. feature matching        (descriptor 공간 KNN + mutual/ratio test → putative 대응)
   → 4. robust SE(3) 추정        (FGR 기본 / RANSAC baseline)
   → RegistrationResult { bool valid; Eigen::Matrix4f T; size_t numInliers; float fitness; }
```

### 1. Voxel downsample (`downsampleVoxel`)
- 셀 크기 `voxelSize`로 voxel-grid 다운샘플: 각 점유 셀당 대표점 1개(centroid). 노멀도 셀 평균 후 normalize.
- KISS-Matcher처럼 `voxelSize`가 이후 반경 파라미터의 기준: `normalRadius = 3·voxelSize`, `fpfhRadius = 5·voxelSize`.
- M1은 CPU(`unordered_map<Vector3i>` 해시 그리드). 데이터 스케일(chair=mm)에 맞춰 `voxelSize`(예: 수 mm) 설정.

### 2. FPFH descriptor (`computeFPFH`)
- 입력: downsampled 점 + 노멀. scan 프레임은 `nx,ny,nz`를 제공하므로 **제공 노멀 사용**(재추정 불필요). 노멀 없으면 `normalRadius` 이웃의 공분산 최소고유벡터로 추정(후속).
- FPFH(Rusu 2009): 각 점 p에 대해
  - `SPFH(p)` = `fpfhRadius` 이웃 각각과의 3개 각도 특징 `(α, φ, θ)`를 11-bin씩 히스토그램 → 33-bin.
  - `FPFH(p) = SPFH(p) + (1/k) Σ_i (1/w_i) SPFH(n_i)` (w_i = 거리 가중).
- 반경 이웃 탐색: **M1은 CPU kdtree/hash-grid**(downsampled N에서 저렴, large-N BVH 버그 우회). 출력: `std::vector<Eigen::Matrix<float,33,1>>`.

### 3. Feature matching (`matchFeatures`)
- descriptor 공간(33-D)에서 src→tgt 최근접(+ tgt→src) → **mutual nearest** 또는 **ratio test**(1st/2nd < thr)로 putative 대응 필터.
- 대응 수 상한 `numMaxCorr`. M1은 CPU kdtree(33-D). 출력: `std::vector<std::pair<int,int>>` (src idx, tgt idx).

### 4. Robust SE(3) 추정
- **M1 기본: FGR** (Fast Global Registration, Zhou 2016) — 모든 대응에 Geman-McClure robust kernel + line-process, scale μ를 graduated non-convexity로 낮추며 매 iteration 가중 최소자승(SVD로 R,t)을 푼다. threshold 샘플링 없이 고outlier에 강함. ~150–250 LOC Eigen.
- **M1 baseline: RANSAC** — 3-점 대응 샘플 → SVD로 T → inlier(대응 잔차 < `inlierThr`) 카운트, 최다 inlier 선택. 자명하게 정확 → **FGR 검증의 대조군**으로도 사용.
- 출력 유효성: `numInliers`/`fitness`(inlier 비율)로 `valid` 판정. `solveRigidSVD`(가중 Kabsch/Umeyama)는 두 estimator가 공유.

### TSDF 연동 (데모/파이프라인)
```
T = Estimate(newFrame, model_or_prevFrame).T    // src=new, tgt=model → T: new를 model 좌표로
if (result.valid && result.numInliers >= minInliers):
    transform points by T, normals by R(=T.block<3,3>)   // 노멀은 회전만
    cam' = R·cam + t
    DirectionalTSDF.Integrate(points', normals', cam', centroid')
else: skip/fallback  // 오정합 프레임은 TSDF 오염 방지
```
- frame-to-model: tgt = 누적 모델의 추출 포인트클라우드(다운샘플). frame-to-frame: tgt = 직전 프레임(드리프트 있음, 데모엔 단순).

### 검증 (자기완결 oracle — KISS-Matcher 링크 불필요)
- **알려진-변환 회복**(핵심 CI anchor): chair 한 프레임을 알려진 `T_gt`(무작위 회전 + 병진)로 교란한 사본을 만들어 `Estimate(perturbed, original)` → 복원 `T`가 `T_gt⁻¹`를 각도·병진 tolerance 이내로 회복하는지. `Engine::Eval` RMSE 관례.
- **연속 프레임 near-identity**: scanData가 이미 등록돼 있으므로 인접 프레임 정합 결과가 ≈identity(작은 상대 pose)인지.
- **데모**: raw chair 프레임들을 정합해 누적 통합 → 재구성 PLY. pre-registered 가정 없이 동작 증명.

## 주요 결정 (확정)
1. **재작성(네이티브) — 링크 아님.** 순수 Eigen+Vulkan, 크로스플랫폼, deps 0.
2. **M1 solver = FGR(기본) + RANSAC(baseline/대조)**, TEASER GNC는 M3 벤더링(from-scratch 금지).
3. **M1 = CPU 정확성 우선**, GPU(M2)는 large-N BVH 버그 해결 후.
4. **검증 = 알려진-변환 회복**(외부 참조 불필요).
5. **제공 노멀 사용**(scan 프레임의 nx,ny,nz) — FPFH 노멀 재추정 회피.

## 리스크
| 리스크 | 대응 |
|---|---|
| FPFH 재구현 미묘함(각도 특징·bin·정규화) | 알려진-변환 회복 테스트로 회귀 고정; RANSAC 대조군; KISS-Matcher/PCL 소스와 수식 대조 |
| FGR 수렴/튜닝 finicky | RANSAC baseline을 먼저 통과시켜 파이프라인(FPFH+매칭)을 분리 검증한 뒤 FGR 도입 |
| large-N BVH 버그 | M1은 CPU 이웃탐색으로 우회; downsampled N은 작음 |
| 스케일/파라미터(mm) | voxelSize를 데이터 스케일로; 반경은 voxelSize 배수 |
| 오정합이 TSDF 오염 | valid+inliers 게이팅, 실패 시 skip |
| Windows 미검증 | 순수 Eigen+표준 라이브러리로 설계; Windows 빌드 검증은 후속 |
```

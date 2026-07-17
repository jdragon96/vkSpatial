# DirectionalTSDF 검증 하니스 설계 (합성 데이터셋 + GT + RMSE)

## 배경 / 목적

`Engine::Spatial::DirectionalTSDF`의 integrate→extract 파이프라인(Phase 3에서 구현)이 표면을 얼마나 정확히 복원하는지 **정량 지표**가 없다. 지금까지의 검증은 정성적(interproximal 데모의 육안 PLY, 두 면 분리 여부)이거나 구조적(submit 수, 재사용 카운트)이었다. 이 스펙은 참값(ground truth)을 정확히 아는 합성 해석 형상을 이용해 재구성 정확도를 **RMSE 두 종**으로 측정하는 평가 하니스를 정의한다.

핵심 아이디어: 형상을 해석적으로(구·평면) 정의하면 (1) 임의의 점에서 참표면까지의 거리를 closed-form으로 계산할 수 있고, (2) 참표면을 원하는 밀도로 조밀 샘플링해 GT 포인트 클라우드를 만들 수 있다. 이 둘로 각각 **정확도**(재구성 점이 표면 위에 있는가)와 **완전성**(표면 전체가 재구성 점으로 커버됐는가)을 측정한다.

## 범위

- `src/Engine/Eval/`에 header-only 평가 유틸 3종 추가: `SyntheticSurface.h`, `ScanSampler.h`, `RmseMetrics.h`. 순수 CPU 수학이라 별도 라이브러리 타깃 없이 헤더온리로 두고 example2·test 양쪽에서 include한다(둘 다 `${CMAKE_SOURCE_DIR}/src`가 이미 include 경로에 있어 CMake 변경 불필요).
- `example2/directional_tsdf_eval.cpp` 신규: 구·평면 각각 합성 스캔 → DirectionalTSDF 재구성 → 정확도/완전성 RMSE 출력 + recon/GT PLY export.
- (수치 확인 후) `test/test_directionalTSDFEval.cpp` 신규: 실측 RMSE 기반 threshold로 회귀 가드 assert.

### 범위 밖 (YAGNI)

- **노이즈 모델**: v1은 노이즈 없는 이상적 스캔. 센서 노이즈/아웃라이어는 후속.
- **치아/자유곡면 형상**: 해석 거리 함수가 없어 point-to-mesh 거리 하니스가 필요 → 별도 스펙.
- **SimpleTSDF와의 비교**: SimpleTSDF의 추출은 marching-cubes 삼각형 메시(`ExportMC`)라 DirectionalTSDF의 point extraction과 산출물 종류가 달라 RMSE 직접 비교가 사과-오렌지. 필요하면 별도로 다룬다.
- **normal 정확도**: 추출 normal은 SDF gradient 기반이고 방향 관례(orientation) 이슈가 남아 있어(스펙 §15) RMSE에서 제외한다. position만 사용한다.
- **vkBVH GPU 최근접**: N,M ~ 수천 규모라 CPU brute-force로 충분. GPU KNN은 성능 문제가 실측되면 후속.

## 아키텍처

### 1. `SyntheticSurface.h` — 해석 형상 인터페이스 + 구/평면

```cpp
namespace Engine::Eval {

    // Analytic surface with a closed-form unsigned distance, a surface normal, and a
    // dense sampler for ground-truth point clouds.
    class Surface {
    public:
        virtual ~Surface() = default;
        // Unsigned distance from an arbitrary point to the surface.
        virtual float Distance(const Eigen::Vector3f &p) const = 0;
        // Outward unit normal of the closest surface point to p (used to synthesise scans).
        virtual Eigen::Vector3f NormalAt(const Eigen::Vector3f &p) const = 0;
        // A roughly uniform dense sampling of the surface (ground-truth point cloud).
        virtual std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const = 0;
    };

    class SphereSurface : public Surface {
    public:
        SphereSurface(const Eigen::Vector3f &center, float radius);
        float Distance(const Eigen::Vector3f &p) const override;      // |‖p−c‖ − r|
        Eigen::Vector3f NormalAt(const Eigen::Vector3f &p) const override; // (p−c)/‖p−c‖
        std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const override; // Fibonacci sphere
    private:
        Eigen::Vector3f m_center;
        float m_radius;
    };

    class PlaneSurface : public Surface {
    public:
        // Finite plane patch: point q, unit normal n, two in-plane half-extents (uExtent, vExtent).
        PlaneSurface(const Eigen::Vector3f &point, const Eigen::Vector3f &normal,
                     float uExtent, float vExtent);
        float Distance(const Eigen::Vector3f &p) const override;       // |(p−q)·n|
        Eigen::Vector3f NormalAt(const Eigen::Vector3f &) const override; // n (constant)
        std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const override; // grid over the patch
    private:
        Eigen::Vector3f m_point, m_normal, m_u, m_v; // m_u/m_v: orthonormal in-plane axes
        float m_uExtent, m_vExtent;
    };

} // namespace Engine::Eval
```

- `SphereSurface::SampleDense`: Fibonacci-sphere 분포로 대략 균일한 `approxCount`개 표면 점 생성.
- `PlaneSurface`: `m_u`/`m_v`는 `m_normal`에 직교하는 정규직교 축을 생성자에서 구성; `SampleDense`는 `[-uExtent,uExtent]×[-vExtent,vExtent]` 격자.
- **Distance는 부호없는 거리**(정확도 RMSE는 표면과의 근접도만 보므로 부호 불필요). 평면의 유한 패치라도 Distance는 무한 평면 기준 수직거리를 쓴다(재구성 점은 패치 내부 근처에만 생기므로 실용상 동일; 경계 밖 재구성 점은 스캔이 패치 밖을 안 만들므로 발생하지 않음).

### 2. `ScanSampler.h` — 합성 스캐너 (궤도 카메라)

```cpp
namespace Engine::Eval {

    struct ScanFrame {
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> normals;
        Eigen::Vector3f cameraPos;
        Eigen::Vector3f aabbCenterHint; // centroid of this frame's visible points
    };

    struct OrbitParams {
        std::vector<Eigen::Vector3f> surfaceSamples; // dense candidate surface points to scan
        float cameraRadius = 3.0f;   // orbit distance from the scene origin
        Eigen::Vector3f orbitCenter = Eigen::Vector3f::Zero();
        int numElevation = 9;        // latitude rings, −80°..+80°
        int numAzimuth = 36;         // longitudes per ring
        float cosVisibility = 0.15f; // reject grazing samples (normal·viewDir <= this)
    };

    // Generates one ScanFrame per camera pose. A surface sample is included in a frame if
    // its outward normal faces the camera (normal · dir(sample→cam) > cosVisibility).
    // Mirrors the visibility logic already used in example2/voxel_tsdf_mc.cpp, so the
    // multi-view running average and the directional layering are exercised realistically.
    std::vector<ScanFrame> GenerateOrbitScan(const Surface &surface, const OrbitParams &params);

} // namespace Engine::Eval
```

- 입력 `surfaceSamples`는 스캔 대상이 되는 표면 후보 점(형상별로 example에서 `SampleDense`로 생성하거나 별도 생성). 각 점의 normal은 `surface.NormalAt(p)`로 구한다.
- 각 카메라 포즈에서 가시 점만 모아 프레임 구성; 빈 프레임은 스킵. `aabbCenterHint`는 그 프레임 가시 점들의 centroid.

### 3. `RmseMetrics.h` — 정확도/완전성 RMSE

```cpp
namespace Engine::Eval {

    // Accuracy: how close reconstructed points sit to the true surface (exact, closed-form).
    //   sqrt( mean_i Distance(recon_i)^2 )
    float AccuracyRMSE(const std::vector<Eigen::Vector3f> &reconPoints, const Surface &surface);

    // Completeness: how well the true surface is covered by reconstructed points.
    //   for each GT point, nearest reconstructed point distance; RMSE of those.
    //   sqrt( mean_j min_i ‖gt_j − recon_i‖^2 )   (CPU brute-force nearest neighbour)
    float CompletenessRMSE(const std::vector<Eigen::Vector3f> &gtDense,
                           const std::vector<Eigen::Vector3f> &reconPoints);

    // Cross-check (accuracy via nearest GT, bounded by GT sampling density):
    //   for each recon point, nearest GT point distance; RMSE of those.
    float ReconToGtNnRMSE(const std::vector<Eigen::Vector3f> &reconPoints,
                          const std::vector<Eigen::Vector3f> &gtDense);

} // namespace Engine::Eval
```

- 세 함수 모두 순수 CPU. `CompletenessRMSE`/`ReconToGtNnRMSE`는 이중 루프 brute-force 최근접(각 ~수천 × 수천 = 수천만 연산, sub-second).
- `reconPoints`가 비면 `AccuracyRMSE`는 0이 아니라 큰 값 또는 정의 불가 — 빈 입력은 `std::numeric_limits<float>::infinity()`를 반환해 "재구성 실패"가 threshold를 반드시 넘게 한다. `CompletenessRMSE`도 recon이 비면 infinity.

### 4. 데이터 흐름

```
Surface(sphere/plane)
   │  surfaceSamples = surface.SampleDense(Nscan)
   │  frames = GenerateOrbitScan(surface, {surfaceSamples, ...})
   ▼
DirectionalTSDF tsdf; tsdf.Build(ctx, voxelSize=0.1, ...)
   for each frame: tsdf.Integrate(frame.points, frame.normals, frame.cameraPos, frame.aabbCenterHint)
   ▼
recon = tsdf.PointCloud()                       // vector<ExtractedPoint>; positions만 추출
gtDense = surface.SampleDense(Ngt)              // 별도 조밀 샘플
   ▼
accuracy      = AccuracyRMSE(reconPos, surface)
completeness  = CompletenessRMSE(gtDense, reconPos)
reconToGt     = ReconToGtNnRMSE(reconPos, gtDense)   // 교차확인
```

## 예제/테스트 계획

### `example2/directional_tsdf_eval.cpp`

두 시나리오를 순차 실행:
1. **Sphere**: 반지름 1.0, 중심 원점. `SampleDense`로 스캔 후보 생성 → 궤도 스캔 → 재구성. 구는 모든 방향의 표면을 포함해 6개 direction layer를 골고루 자극한다.
2. **Plane**: 원점 통과, normal +Z, 패치 반경 1.0×1.0. 단일 방향 레이어의 평탄 정확도.

각 시나리오에서 출력:
- 재구성 점 수, GT 점 수
- `accuracyRMSE`, `completenessRMSE`, `reconToGtNnRMSE` (world unit, voxelSize=0.1 기준으로 해석)
- `recon_sphere.ply`/`gt_sphere.ply`, `recon_plane.ply`/`gt_plane.ply` (position만; 육안 대조용)

`example2/CMakeLists.txt`에 `voxel_tsdf_mc`와 동일 패턴으로 실행 파일 등록(`Engine::Spatial` 링크, `VKBVH_SHADER_DIR` define). 헤더온리 eval 유틸은 링크 불필요.

### `test/test_directionalTSDFEval.cpp` (수치 확인 후 작성)

example 실행으로 실제 RMSE를 확인한 뒤, 그 값에 여유를 둔 threshold로 회귀 가드 작성. 예상 스케일: voxelSize=0.1에서 정확도 RMSE는 대략 반복셀~복셀(≈0.03–0.1), 완전성은 스캔/샘플 밀도에 의존. **실측 전에는 threshold를 확정하지 않는다**(플랜에서 실측 단계 후 값을 채운다).

케이스(초안, 실측 후 상수 확정):
1. `SphereReconstructionAccuracyWithinTolerance` — `AccuracyRMSE(sphere) < TH_acc`
2. `SphereReconstructionCompletenessWithinTolerance` — `CompletenessRMSE(sphere) < TH_comp`
3. `PlaneReconstructionAccuracyWithinTolerance` — `AccuracyRMSE(plane) < TH_acc_plane`
4. `EmptyReconstructionYieldsInfiniteRMSE` — 빈 recon → infinity (GPU 불필요 순수 CPU 유닛 테스트)
5. `RmseMetricsSanity` — 알려진 소규모 점 집합에 대해 세 metric의 손계산 값과 일치 (GPU 불필요)

## 파일 변경 목록

| 파일 | 변경 |
|---|---|
| `src/Engine/Eval/SyntheticSurface.h` (신규) | 해석 형상 인터페이스 + Sphere/Plane (header-only) |
| `src/Engine/Eval/ScanSampler.h` (신규) | 궤도 카메라 합성 스캐너 (header-only) |
| `src/Engine/Eval/RmseMetrics.h` (신규) | 정확도/완전성/교차 RMSE (header-only) |
| `example2/directional_tsdf_eval.cpp` (신규) | 구/평면 평가 실행 파일 |
| `example2/CMakeLists.txt` | eval 실행 파일 등록 |
| `test/test_directionalTSDFEval.cpp` (신규, 수치 확인 후) | RMSE threshold 회귀 가드 + 순수 CPU 유닛 테스트 |

`DirectionalTSDF`/`Engine::Spatial`/`Engine::Compute`/`Engine::Core`는 이 스펙에서 변경하지 않는다.

## 검증 기준 (성공 조건)

- 구 재구성 점이 참구면 근처에 분포(정확도 RMSE가 복셀 스케일 이내), 표면 전체 커버(완전성 RMSE가 스캔 밀도 스케일 이내) — 실측 수치로 확인.
- 평면 재구성 정확도가 구보다 같거나 더 좋음(단일 레이어, 곡률 없음).
- 빈 재구성/열화 시 RMSE가 무한대 또는 큰 값이 되어 회귀 가드가 반드시 실패.
- 순수 CPU metric 유닛 테스트(손계산 대조)가 GPU 없이 통과.

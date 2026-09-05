# Realsense 법선 추정 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Realsense::ValidationMask`의 압축 출력이 좌표와 함께 법선을 싣도록, 이미 측정으로 검증된 세 법선 추정기(`forward`/`central`/`planefit`)를 `src/Realsense/Algorithm/`로 레지스트리째 이식한다.

**Architecture:** `NormalEstimation`이 전략 레지스트리와 커널 변형 캐시만 소유하고, 버퍼와 실행 순서는 `ValidationMask`가 계속 소유한다. 전략은 C++ 가상함수가 아니라 디스패처 `.glsl` + `-D` 매크로로 가른다. 법선 패스는 `RemainValidDepth`와 `CountEmittedPerRow` 사이에 들어가 `emitted`를 **취소만** 한다.

**Tech Stack:** C++17 헤더 온리, GLSL 450 컴퓨트(런타임 shaderc 컴파일), Vulkan, GTest.

**Spec:** [docs/superpowers/specs/2026-09-05-realsense-normal-estimation-design.md](../specs/2026-09-05-realsense-normal-estimation-design.md)

## Global Constraints

- 코드 주석은 영어, 문서는 한국어. 주석은 **왜**를 적는다.
- **축약어 금지**: `minimumPlaneFitSamples`이지 `minPlaneSamples`가 아니다.
- `assert` 금지 — 릴리스 전용 구성이라 no-op이 된다. 실패는 `throw std::runtime_error("<ClassName>::<Method>: ...")`.
- 커널 파일명은 `<클래스명>.<역할>.glsl`. `main()` 없는 include는 접두사·접미사 없음.
- GLSL 서식: `///` 섹션 배너, Allman 중괄호(`for`는 K&R), 번호 붙인 단계 주석, 탭.
- `src/`에 로깅 없음 — 카운터를 노출하고 판단은 호출자에게 맡긴다.
- **`src/`에 파일을 추가하면 `cmake -S . -B build-rel`를 반드시 다시 돌린다** (GLOB_RECURSE).
- **셰이더 경로 오류는 런타임에만 터진다.** 매 태스크는 빌드가 아니라 실행으로 끝난다.
- 기본값은 감싸는 구현체의 멤버 기본값과 정확히 같아야 한다: `planeFitRadius = 2`(창 5), `minimumPlaneFitSamples = 8` — Pipeline의 `planeFitWindow = 5` / `minimumPlaneFitSamples = 8`과 일치. `estimator` 기본값만 의도적으로 다르다(`"planefit"`, 스펙 §3-1의 측정 근거).
- **커밋은 하지 않는다.** Realsense 작업 전체가 아직 미커밋이라 사용자가 요청할 때 한 번에 한다. 각 태스크는 "테스트 통과 + 작업 트리 정상"으로 끝난다.

**빌드·실행 명령 (모든 태스크 공통):**

```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-rel --target vkspatial_tests -j8
./build-rel/test/vkspatial_tests --gtest_filter='<필터>'
```

셰이더만 빠르게 확인:

```bash
glslc -fshader-stage=compute -I src <파일> -o /dev/null
```

---

## File Structure

| 파일 | 책임 |
| --- | --- |
| `src/Realsense/Algorithm/Common.glsl` | 두 커널이 공유하는 struct 미러 + **같은표면 허용치 한 정의** |
| `src/Realsense/RealSenseTypes.h` | 경계 전용 POD: property, 카운터, 푸시상수 미러 |
| `src/Realsense/Algorithm/NormalEstimation.h` | 전략 레지스트리 + 커널 변형 캐시 + `RecordEstimate` |
| `src/Realsense/Algorithm/NormalEstimation.EstimateNormal.glsl` | 커널: 허용치 계산 → 추정 → 방향 정렬 → `emitted` 취소 |
| `src/Realsense/Algorithm/NormalEstimation.Strategy.glsl` | 디스패처 + `SameSurfaceSample` 공용 정의 + 결과 코드 |
| `src/Realsense/Algorithm/NormalEstimation.ForwardDifference.glsl` | 전략 1 |
| `src/Realsense/Algorithm/NormalEstimation.CentralDifference.glsl` | 전략 2 |
| `src/Realsense/Algorithm/NormalEstimation.PlaneFit.glsl` | 전략 3 (닫힌 형식 고유분해) |
| `src/Realsense/Algorithm/ValidationMask.h` | 체인 순서 + 버퍼 소유 (수정) |
| `src/Realsense/Algorithm/ValidationMask.ScatterValidPoints.glsl` | 바인딩 2개 추가 (수정) |
| `test/test_realsenseNormalEstimation.cpp` | 합성 GT 정확도 + 게이트 + 레지스트리 |

---

## Task 1: 같은표면 허용치를 한 정의로 모은다

법선 커널이 score 커널과 **다른** 허용치를 쓰면 `c_nb`가 세는 이웃과 `planefit`이 적합하는 표본이 달라진다. 뒤 태스크가 이 함수를 쓰므로 먼저 옮긴다.

**Files:**
- Modify: `src/Realsense/Algorithm/Common.glsl`
- Modify: `src/Realsense/Algorithm/ValidationMask.CalculateScore.glsl` (지역 `AxialNoiseSigma` 제거)

**Interfaces:**
- Consumes: 없음
- Produces: `float AxialNoiseSigma(float depth, float subpixelRms, float focalLengthPixels, float baselineMeters)`, `float SameSurfaceTolerance(float depth, float subpixelRms, float focalLengthPixels, float baselineMeters, float sigmaMultiplier)` — Task 2·4가 쓴다.

- [ ] **Step 1: `Common.glsl` 끝에 두 함수를 추가한다**

```glsl
/// sigma_z = s * z^2 / (f * B)
///
/// sigma_z : axial depth noise of one sample, one standard deviation [m]
/// s       : subpixel disparity matching error, RMS [px] -- 0.08 with texture, 0.25 without
/// z       : depth [m]
/// f       : focal length [px]
/// B       : stereo baseline [m] -- D435 0.05, D455 0.095
/// z^2     : exact for active stereo, not a fit -- z = f*B/d, so a disparity error e propagates
///           as z^2*e/(f*B)
/// 0       : returned when f*B <= 0. That makes every tolerance 0, which scores the WHOLE frame
///           zero, so the C++ side must reject such a configuration rather than lean on this
///
/// The thresholds are parameters rather than push-constant reads: a shared include cannot name a
/// PC member that every including kernel is guaranteed to declare.
float AxialNoiseSigma(float depth, float subpixelRms, float focalLengthPixels, float baselineMeters)
{
	float denominator = focalLengthPixels * baselineMeters;
	if (denominator <= 0.0) return 0.0;
	return subpixelRms * depth * depth / denominator;
}

/// tau = k * sigma_z(z)
///
/// The ONE definition of "same surface" in this module. The score kernel's c_nb term and every
/// normal estimator go through it, so a neighbour counted as support and a sample admitted to a
/// plane fit cannot come to mean two different things.
float SameSurfaceTolerance(float depth, float subpixelRms, float focalLengthPixels,
                           float baselineMeters, float sigmaMultiplier)
{
	return sigmaMultiplier * AxialNoiseSigma(depth, subpixelRms, focalLengthPixels, baselineMeters);
}
```

- [ ] **Step 2: `ValidationMask.CalculateScore.glsl`에서 지역 정의를 지우고 공용 함수를 부른다**

지역 `float AxialNoiseSigma(float depth) { ... }` 블록(배너 주석 포함)을 통째로 삭제하고, `main()`의 허용치 계산을 바꾼다:

```glsl
		// The tolerance is derived from the sensor, not tuned. Shared with every normal estimator
		// through Common.glsl so "same surface" has one meaning in this module.
		float tolerance     = SameSurfaceTolerance(depth, g_subpixelRms, g_focalLengthPixels,
		                                           g_baselineMeters, g_sameSurfaceSigmaMultiplier);
		neighbourConfidence = NeighbourConfidence(local, depth, tolerance);
```

- [ ] **Step 3: 세 변형이 컴파일되는지 확인한다**

```bash
for m in "" "-DVALIDATION_SCORE_WITH_COUNTERS" "-DVALIDATION_SCORE_WITH_INFRARED -DVALIDATION_SCORE_WITH_COUNTERS"; do
  glslc -fshader-stage=compute -I src $m src/Realsense/Algorithm/ValidationMask.CalculateScore.glsl -o /dev/null || echo "FAIL: $m"
done
```
Expected: 출력 없음(전부 통과).

- [ ] **Step 4: 점수가 바뀌지 않았는지 확인한다 — 순수 리팩터링이다**

```bash
cmake --build build-rel --target vkspatial_tests -j8
./build-rel/test/vkspatial_tests --gtest_filter='RealsenseValidationScore*'
```
Expected: 10 PASSED. 특히 `TheSameSurfaceToleranceScalesWithTheSquareOfDepth`(1 m에서 5/8, 3 m에서 1.0)가 통과해야 한다 — 이 테스트가 곧 허용치 공식의 회귀 가드다.

---

## Task 2: NormalEstimation 스캐폴딩 + `forward` + 체인 결선

커널·레지스트리·체인 결선이 함께 도착해야 처음으로 테스트 가능한 산출물이 된다 — 리뷰어가 결선을 거부하며 커널만 승인할 수 없다.

**Files:**
- Create: `src/Realsense/Algorithm/NormalEstimation.h`
- Create: `src/Realsense/Algorithm/NormalEstimation.Strategy.glsl`
- Create: `src/Realsense/Algorithm/NormalEstimation.ForwardDifference.glsl`
- Create: `src/Realsense/Algorithm/NormalEstimation.EstimateNormal.glsl`
- Delete: `src/Realsense/Algorithm/NormalEstimation.Depth2Normal.glsl`
- Modify: `src/Realsense/RealSenseTypes.h` (카운터 + 푸시상수 미러)
- Modify: `src/Realsense/Algorithm/Common.glsl` (카운터 struct 미러)
- Modify: `src/Realsense/Algorithm/ValidationMask.h` (버퍼 2개, 패스 1개, 다운로드 1개)
- Modify: `src/Realsense/Algorithm/ValidationMask.ScatterValidPoints.glsl` (바인딩 2개)
- Test: `test/test_realsenseNormalEstimation.cpp`

**Interfaces:**
- Consumes: Task 1의 `SameSurfaceTolerance`
- Produces:
  - `Realsense::NormalEstimationOptions { std::string estimator; int planeFitRadius; int minimumPlaneFitSamples; }`
  - `Realsense::NormalEstimationCounters { std::uint32_t outOfDomain; std::uint32_t noSupport; }`
  - `Realsense::NormalEstimation::RecordEstimate(Engine::Compute::CommandBatch&, Engine::Core::Buffer &vertices, Engine::Core::Buffer &properties, Engine::Core::Buffer &normals, Engine::Core::Buffer &counters, int width, int height, const ValidationScoreOptions&, const NormalEstimationOptions&)`
  - `Realsense::ValidationMask::RecordRemainValidDepth(..., const NormalEstimationOptions &normalOptions)` — 인자 하나 추가
  - `std::vector<Eigen::Vector3f> Realsense::ValidationMask::DownloadValidNormals() const`
  - `Realsense::NormalEstimationCounters Realsense::ValidationMask::DownloadNormalCounters() const`
  - GLSL: `int EstimateSurfaceNormal(int column, int row, float depth, float tolerance, out vec3 normal)` — Task 3·4가 이 시그니처로 구현한다. 반환값은 `NORMAL_ESTIMATE_OK` / `NORMAL_ESTIMATE_OUT_OF_DOMAIN` / `NORMAL_ESTIMATE_NO_SUPPORT`.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`test/test_realsenseNormalEstimation.cpp`:

```cpp
#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Realsense/Algorithm/ValidationMask.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

using Realsense::NormalEstimationOptions;
using Realsense::ValidationMask;
using Realsense::ValidationScoreOptions;

namespace {

    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    constexpr float kFocal = 425.0f;

    ValidationScoreOptions TestOptions() {
        ValidationScoreOptions options;
        options.focalLengthPixels = kFocal;
        options.baselineMeters = 0.05f;
        options.subpixelRms = 0.08f;
        options.depthScale = 0.001f;
        options.nearFadeStart = 0.1f;
        options.nearFadeEnd = 0.2f;
        options.farFadeStart = 5.0f;
        options.farFadeEnd = 6.0f;
        return options;
    }

    // A plane through (0,0,distance) with unit normal `normal`, sampled on the pixel grid and
    // quantised to Z16. The ray through (u,v) is d = ((u-cx)/fx, (v-cy)/fy, 1); it meets the plane
    // at t = (n . p0) / (n . d), and the sample's depth is that t (because d.z == 1).
    std::vector<std::uint16_t> RenderPlane(const Eigen::Vector3f &normal, float distance,
                                           float depthScale) {
        const float cx = float(kWidth) * 0.5f, cy = float(kHeight) * 0.5f;
        const Eigen::Vector3f pointOnPlane = Eigen::Vector3f(0.0f, 0.0f, distance);
        const float numerator = normal.dot(pointOnPlane);

        std::vector<std::uint16_t> depth(std::size_t(kWidth) * kHeight, 0);
        for (int row = 0; row < kHeight; ++row)
            for (int column = 0; column < kWidth; ++column) {
                const Eigen::Vector3f ray((float(column) - cx) / kFocal,
                                          (float(row) - cy) / kFocal, 1.0f);
                const float denominator = normal.dot(ray);
                if (std::abs(denominator) < 1e-6f) continue;
                const float z = numerator / denominator;
                if (!(z > 0.0f)) continue;
                const float units = std::round(z / depthScale);
                if (units > 65535.0f) continue;
                depth[std::size_t(row) * kWidth + column] = std::uint16_t(units);
            }
        return depth;
    }

    struct NormalRun {
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> normals;
        Realsense::NormalEstimationCounters counters;
    };

    NormalRun RunFrontEnd(Engine::Core::Context &context,
                          const std::vector<std::uint16_t> &depth,
                          const std::string &estimator) {
        ValidationScoreOptions options = TestOptions();
        NormalEstimationOptions normalOptions;
        normalOptions.estimator = estimator;

        ValidationMask mask(context, kWidth, kHeight);
        const float cx = float(kWidth) * 0.5f, cy = float(kHeight) * 0.5f;
        {
            Engine::Compute::CommandBatch batch(context);
            mask.Execute(batch, depth.data(), options);
            batch.Barrier();
            mask.RecordRemainValidDepth(batch, options, kFocal, kFocal, cx, cy, 0.0f, normalOptions);
            batch.Submit();
        }
        NormalRun run;
        run.points = mask.DownloadValidPoints();
        run.normals = mask.DownloadValidNormals();
        run.counters = mask.DownloadNormalCounters();
        return run;
    }

    // Median rather than mean: at a depth step a large disagreement is CORRECT, and a mean would
    // mostly track how much boundary the fixture has.
    float MedianAngleDegrees(const std::vector<Eigen::Vector3f> &normals,
                             const Eigen::Vector3f &truth) {
        std::vector<float> angles;
        angles.reserve(normals.size());
        for (const Eigen::Vector3f &normal: normals) {
            const float cosine = std::clamp(normal.dot(truth), -1.0f, 1.0f);
            angles.push_back(std::acos(cosine) * 180.0f / float(M_PI));
        }
        if (angles.empty()) return 180.0f;
        std::nth_element(angles.begin(), angles.begin() + angles.size() / 2, angles.end());
        return angles[angles.size() / 2];
    }

} // namespace

// A noise-free plane has one right answer, so this pins back-projection, the cross-product order,
// normalisation and the orientation flip all at once. Any estimator that gets this wrong is wrong
// for a reason no noise test would isolate.
TEST(RealsenseNormalEstimation, ANoiselessTiltedPlaneMatchesItsTrueNormal) {
    Engine::Core::Context context;
    // Tilted, not fronto-parallel: a plane facing straight down -z would pass even if the two
    // tangents were swapped or a sign were dropped.
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
    const std::vector<std::uint16_t> depth = RenderPlane(truth, 1.5f, 0.001f);

    const NormalRun run = RunFrontEnd(context, depth, "forward");
    ASSERT_GT(run.normals.size(), 0u);
    ASSERT_EQ(run.normals.size(), run.points.size());
    EXPECT_LT(MedianAngleDegrees(run.normals, truth), 1.0f);
}

// Every emitted normal must face the camera: the fusion weight and the point-to-plane residual
// both change sign with it, so a flipped normal is not a small error.
TEST(RealsenseNormalEstimation, EveryEmittedNormalFacesTheCamera) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
    const std::vector<std::uint16_t> depth = RenderPlane(truth, 1.5f, 0.001f);

    const NormalRun run = RunFrontEnd(context, depth, "forward");
    ASSERT_GT(run.points.size(), 0u);
    for (std::size_t i = 0; i < run.points.size(); ++i)
        EXPECT_LE(run.normals[i].dot(run.points[i]), 0.0f) << "point " << i;
}

// The gate this module chose: `emitted == 1` has to mean "coordinate AND normal", so no downstream
// consumer has to special-case a zero normal.
TEST(RealsenseNormalEstimation, EveryCompactedPointCarriesAUnitNormal) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
    const std::vector<std::uint16_t> depth = RenderPlane(truth, 1.5f, 0.001f);

    const NormalRun run = RunFrontEnd(context, depth, "forward");
    ASSERT_GT(run.normals.size(), 0u);
    for (const Eigen::Vector3f &normal: run.normals)
        EXPECT_NEAR(normal.norm(), 1.0f, 1e-4f);
}
```

`<algorithm>`과 `<string>`도 include한다 (`std::clamp`, `std::nth_element`, `std::string`).

- [ ] **Step 2: 실패를 확인한다**

```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-rel --target vkspatial_tests -j8
```
Expected: 컴파일 실패 — `NormalEstimationOptions`, `DownloadValidNormals`, 7-인자 `RecordRemainValidDepth`가 없다.

- [ ] **Step 3: 경계 타입을 추가한다 — `RealSenseTypes.h`**

`ValidationScoreCounters` 뒤에 넣는다:

```cpp
    // Kept apart from ValidationScoreCounters on purpose: that one partitions the image and must
    // keep summing to width*height, while these two count pixels a LATER pass refused.
    struct NormalEstimationCounters {
        // The stencil hung off the edge of the image. Every estimator loses a frame this way and a
        // wider one loses more, so this tracks the estimator's domain, not the scene.
        std::uint32_t outOfDomain = 0;
        // The stencil fitted and still found too few same-surface samples. This one is the scene
        // refusing the pixel, which is the number worth watching.
        std::uint32_t noSupport = 0;
    };

    static_assert(sizeof(NormalEstimationCounters) == 8, "the GLSL mirror is two 4-byte scalars");
    static_assert(offsetof(NormalEstimationCounters, noSupport) == 4, "field order drifted");

    // Must match the push_constant block in NormalEstimation.EstimateNormal.glsl.
    struct NormalPushConstants {
        std::int32_t width;
        std::int32_t height;
        float subpixelRms;
        float focalLengthPixels;
        float baselineMeters;
        float sameSurfaceSigmaMultiplier;
        std::int32_t planeFitRadius;
        std::int32_t minimumPlaneFitSamples;
    };
```

- [ ] **Step 4: GLSL 미러를 `Common.glsl`에 추가한다**

`ValidationMaskProperty` 뒤에 넣는다:

```glsl
struct NormalEstimationCounters {
	uint outOfDomain;  // the stencil hung off the edge of the image
	uint noSupport;    // the stencil fitted but found too few same-surface samples
};
```

- [ ] **Step 5: 디스패처 `NormalEstimation.Strategy.glsl`을 만든다**

```glsl
/// *********************************************
/// Normal estimator dispatcher
///
/// NormalEstimation.EstimateNormal.glsl includes only this file. Which fragment it pulls in is
/// decided by a preprocessor definition the C++ side passes through ComputePipeline::Define,
/// listed in Realsense/Algorithm/NormalEstimation.h.
///
/// Why a dispatcher instead of `#include NORMAL_STRATEGY_INCLUDE`: glslang does NOT macro-expand
/// #include ("must be followed by a header name"), so the include name has to be a literal. This
/// keeps each fragment a real file that editors and grep can follow.
///
/// The fragment includes below use the full src-relative path (not a bare filename):
/// ComputePipeline resolves an #include against a fixed root list computed once for the top-level
/// compiled file -- [that file's own folder, VKBVH_SHADER_DIR, VKBVH_SRC_DIR] -- not against THIS
/// file's folder, even though this file is itself reached via #include.
///
/// Every fragment defines exactly one entry point:
///
///     int EstimateSurfaceNormal(int column, int row, float depth, float tolerance, out vec3 normal)
///
/// It returns an UNORIENTED unit normal. Orientation against the pixel's own view ray, the
/// `emitted` gate and both counters stay in the kernel: which cause a rejection is charged to must
/// not depend on which fragment happened to be compiled in.
///
/// The fragments read g_vertices, g_properties, g_width and g_height, which the including kernel
/// declares above this include.
/// *********************************************

/// Outcome of one estimator call.
///
/// The two failures are kept apart because only one of them is a discarded measurement. A stencil
/// hanging off the edge of the image is the estimator's domain ending -- every estimator loses a
/// border, and a wider one loses more -- while a stencil that fits and still cannot find
/// same-surface samples is a pixel the scene refused, which is what the counter is for.
#define NORMAL_ESTIMATE_OK            0
#define NORMAL_ESTIMATE_OUT_OF_DOMAIN 1
#define NORMAL_ESTIMATE_NO_SUPPORT    2

/// A neighbour is usable only if it exists, carries a measurement, and sits on the centre pixel's
/// surface. Every fragment goes through this one definition so "same surface" cannot come to mean
/// three different things across the three estimators -- the same reason SameSurfaceTolerance is
/// shared by the score kernel and every estimator.
bool SameSurfaceSample(int column, int row, float depth, float tolerance, out vec3 point)
{
	point = vec3(0.0);
	if (column < 0 || column >= g_width || row < 0 || row >= g_height) return false;

	int index = row * g_width + column;
	if (g_properties[index].valid == 0u) return false;
	if (abs(g_vertices[index].z - depth) > tolerance) return false;

	point = g_vertices[index].xyz;
	return true;
}

#if defined(NORMAL_PLANE_FIT)
#include "Realsense/Algorithm/NormalEstimation.PlaneFit.glsl"
#elif defined(NORMAL_CENTRAL_DIFFERENCE)
#include "Realsense/Algorithm/NormalEstimation.CentralDifference.glsl"
#else
#include "Realsense/Algorithm/NormalEstimation.ForwardDifference.glsl"
#endif
```

Task 3·4가 도착하기 전까지 `#if` 가지의 두 파일이 없지만, 매크로가 정의되지 않으면 전처리기가 그 `#include`에 닿지 않으므로 컴파일된다. Task 3·4는 파일만 추가하면 된다.

- [ ] **Step 6: `NormalEstimation.ForwardDifference.glsl`을 만든다**

```glsl
/// *********************************************
/// "forward" -- one forward-difference triangle. The historical estimator, kept as the noisy
/// baseline every other strategy is measured against.
///
///     n = (P(u+1,v) - P(u,v)) x (P(u,v+1) - P(u,v))
///
/// Two samples per tangent, so each tangent carries sqrt(2) * sigma of the per-pixel depth noise.
/// It is also anchored at a corner: the normal of that triangle is the surface normal at
/// (u+0.5, v+0.5), stored at (u,v), which shifts the whole normal field half a pixel on a curved
/// surface. "central" fixes both at the same cost.
/// *********************************************

int EstimateSurfaceNormal(int column, int row, float depth, float tolerance, out vec3 normal)
{
	normal = vec3(0.0);

	// 1. The stencil reaches u+1 and v+1, so the last row and column have no domain.
	if (column + 1 >= g_width || row + 1 >= g_height) return NORMAL_ESTIMATE_OUT_OF_DOMAIN;

	// 2. Both differencing partners must be on this pixel's surface. A step in depth is two
	//    surfaces, not one, and a normal differenced across it belongs to neither.
	vec3 right, below;
	if (!SameSurfaceSample(column + 1, row, depth, tolerance, right)) return NORMAL_ESTIMATE_NO_SUPPORT;
	if (!SameSurfaceSample(column, row + 1, depth, tolerance, below)) return NORMAL_ESTIMATE_NO_SUPPORT;

	// 3. Cross the two edges of the triangle.
	vec3 point = g_vertices[row * g_width + column].xyz;
	normal = cross(right - point, below - point);
	if (length(normal) < 1e-9) return NORMAL_ESTIMATE_NO_SUPPORT;

	normal = normalize(normal);
	return NORMAL_ESTIMATE_OK;
}
```

- [ ] **Step 7: 커널 `NormalEstimation.EstimateNormal.glsl`을 만든다**

Pipeline판과의 결정적 차이: **`emitted`를 세우지 않고 취소만 한다.** 채택은 이미 점수 문턱값이 정했고, 이 패스는 거부권만 갖는다.

```glsl
#version 450

#include "Realsense/Algorithm/Common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PC
{
	int   g_width;
	int   g_height;

	float g_subpixelRms;
	float g_focalLengthPixels;
	float g_baselineMeters;
	float g_sameSurfaceSigmaMultiplier;

	int   g_planeFitRadius;
	int   g_minimumPlaneFitSamples;
};

layout(std430, set = 0, binding = 0) readonly buffer VertexGrid  { vec4 g_vertices[]; };
layout(std430, set = 0, binding = 1) buffer          ValidMask   { ValidationMaskProperty g_properties[]; };
layout(std430, set = 0, binding = 2) writeonly buffer NormalGrid { vec4 g_normals[]; };
layout(std430, set = 0, binding = 3) buffer          Counters    { NormalEstimationCounters g_counters; };

#include "Realsense/Algorithm/NormalEstimation.Strategy.glsl"

/// The normal pass may only CANCEL a point, never elect one.
///
/// Which pixels are wanted was decided by the score threshold in ValidationMask.RemainValidDepth;
/// re-electing here would let a pixel the confidence rejected back in through a second door. What
/// this pass adds is a veto: a point without a normal is useless to point-to-plane ICP and to a
/// weighted TSDF fusion alike, so `emitted == 1` is made to mean "coordinate AND normal" and no
/// consumer downstream has to special-case a zero normal.
void main()
{
	int column = int(gl_GlobalInvocationID.x);
	int row    = int(gl_GlobalInvocationID.y);
	if (column >= g_width || row >= g_height) return;

	int centre = row * g_width + column;
	g_normals[centre] = vec4(0.0);

	// 1. Nothing to do where the threshold pass already dropped the pixel. Reading `emitted`
	//    rather than `valid` keeps this pass off the pixels nobody asked for.
	if (g_properties[centre].emitted == 0u) return;

	float depth     = g_vertices[centre].z;
	float tolerance = SameSurfaceTolerance(depth, g_subpixelRms, g_focalLengthPixels,
	                                       g_baselineMeters, g_sameSurfaceSigmaMultiplier);

	// 2. Estimate. The fragment compiled in decides HOW; the two failure causes are charged here
	//    so the attribution does not depend on which one it was.
	vec3 normal;
	int outcome = EstimateSurfaceNormal(column, row, depth, tolerance, normal);
	if (outcome == NORMAL_ESTIMATE_OUT_OF_DOMAIN)
	{
		g_properties[centre].emitted = 0u;
		atomicAdd(g_counters.outOfDomain, 1u);
		return;
	}
	if (outcome == NORMAL_ESTIMATE_NO_SUPPORT)
	{
		g_properties[centre].emitted = 0u;
		atomicAdd(g_counters.noSupport, 1u);
		return;
	}

	// 3. Orient against this pixel's own view ray. In camera space the ray to the point IS the
	//    point, so the sign of the dot product is the whole test.
	vec3 point = g_vertices[centre].xyz;
	if (dot(normal, point) > 0.0) normal = -normal;

	g_normals[centre] = vec4(normal, 0.0);
}
```

- [ ] **Step 8: 미완성 스텁을 지운다**

```bash
rm src/Realsense/Algorithm/NormalEstimation.Depth2Normal.glsl
```
바인딩 1번이 두 줄 중복 선언된 미완성 파일이고, Step 7의 커널이 그 자리를 대신한다.

- [ ] **Step 9: `NormalEstimation.h`를 만든다**

```cpp
#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include "Realsense/Algorithm/ValidationMask.h" // ValidationScoreOptions
#include "Realsense/RealSenseTypes.h"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Realsense {

    // Name -> preprocessor definition for NormalEstimation.EstimateNormal.
    //
    // The strategy axis lives inside the shader: the three estimators differ only in which
    // neighbours they read and how they combine them, so they are split by a dispatcher .glsl plus
    // a -D definition rather than by a C++ virtual call. One kernel, one dispatch, and no per-pixel
    // branch on a value that is uniform across the whole image.
    //
    // "forward" carries no definition on purpose: NormalStrategy falls through to it when nothing
    // is defined, so a build that forgets to pass a definition degrades to the simplest estimator
    // rather than failing to compile.
    struct NormalEstimatorStrategy {
        const char *name;
        const char *macroName; // nullptr = compile the dispatcher's default fragment
    };

    inline const std::vector<NormalEstimatorStrategy> &NormalEstimatorStrategies() {
        // One forward-difference triangle. Two samples per tangent, so the gradient carries
        // sqrt(2)*sigma of the per-pixel depth noise, and the normal it produces actually belongs
        // to (u+0.5, v+0.5) while it is stored at (u,v).
        static const std::vector<NormalEstimatorStrategy> strategies = {
                {"forward", nullptr},
                // Symmetric differences. Same cost and same 3x3 footprint, but the half-pixel
                // offset cancels and the gradient noise drops to sigma/sqrt(2) -- half the angular
                // noise of "forward" for free.
                {"central", "NORMAL_CENTRAL_DIFFERENCE"},
                // Least-squares plane through every same-surface sample in a window. The slope of
                // a least-squares fit over k evenly spaced samples carries sigma*sqrt(12/(k(k^2-1)))
                // -- 0.32*sigma at k=5 against 1.41*sigma for "forward", so 4.5x less gradient
                // noise, at the price of a k*k gather and a 3x3 eigen solve per pixel.
                {"planefit", "NORMAL_PLANE_FIT"},
        };
        return strategies;
    }

    inline std::vector<std::string> NormalEstimatorNames() {
        std::vector<std::string> names;
        names.reserve(NormalEstimatorStrategies().size());
        for (const NormalEstimatorStrategy &strategy: NormalEstimatorStrategies())
            names.emplace_back(strategy.name);
        return names;
    }

    // Throws rather than falling back to the default: a misspelled name that silently ran
    // "forward" would report the timings and the point counts of an estimator nobody selected.
    inline const NormalEstimatorStrategy &FindNormalEstimator(const std::string &name) {
        for (const NormalEstimatorStrategy &strategy: NormalEstimatorStrategies())
            if (name == strategy.name) return strategy;

        std::string known;
        for (const NormalEstimatorStrategy &strategy: NormalEstimatorStrategies()) {
            if (!known.empty()) known += ", ";
            known += strategy.name;
        }
        throw std::runtime_error("Realsense::FindNormalEstimator: unknown normal estimator '" +
                                 name + "'; registered names are " + known);
    }

    struct NormalEstimationOptions {
        // "planefit" rather than Pipeline's "forward": measured at 12.5 degrees against the ground
        // truth where "forward" reads 50.6, and on real frames it needs no depth prefilter, so the
        // point that reaches the TSDF is not smoothed to fix the normal.
        std::string estimator = "planefit";
        // The fit spans planeFitRadius pixels either side of the centre; 2 is a 5x5 window, which
        // matches Pipeline's planeFitWindow = 5. Ignored by the two difference estimators.
        int planeFitRadius = 2;
        int minimumPlaneFitSamples = 8;
    };

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Surface normals over an organised point cloud.
    //
    // Owns the strategy axis and the kernel variants and nothing else -- the buffers and the order
    // of the passes stay with ValidationMask, which is what sequences them.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class NormalEstimation {
    public:
        explicit NormalEstimation(Engine::Core::Context &context) : m_context(context) {
            // Build the default variant now so a broken shader surfaces here rather than at the
            // first frame; the others are compiled the first time they are asked for.
            estimatorKernel(NormalEstimationOptions{});
        }

        void RecordEstimate(Engine::Compute::CommandBatch &batch,
                            Engine::Core::Buffer &vertices,
                            Engine::Core::Buffer &properties,
                            Engine::Core::Buffer &normals,
                            Engine::Core::Buffer &counters,
                            int width,
                            int height,
                            const ValidationScoreOptions &scoreOptions,
                            const NormalEstimationOptions &normalOptions) {
            ValidateOptions(normalOptions);

            const NormalPushConstants pushConstants{width,
                                                    height,
                                                    scoreOptions.subpixelRms,
                                                    scoreOptions.focalLengthPixels,
                                                    scoreOptions.baselineMeters,
                                                    scoreOptions.sameSurfaceSigmaMultiplier,
                                                    normalOptions.planeFitRadius,
                                                    normalOptions.minimumPlaneFitSamples};

            Engine::Core::ComputePipeline &kernel = estimatorKernel(normalOptions);
            kernel.Bind(0, vertices).Bind(1, properties).Bind(2, normals).Bind(3, counters);
            kernel.Args(pushConstants);

            const VkExtent3D localSize = kernel.GetLocalSize();
            if (localSize.width == 0 || localSize.height == 0)
                throw std::runtime_error("Realsense::NormalEstimation::RecordEstimate: the kernel "
                                         "reported a zero local size");
            batch.Dispatch(kernel,
                           (std::uint32_t(width) + localSize.width - 1) / localSize.width,
                           (std::uint32_t(height) + localSize.height - 1) / localSize.height,
                           1);
        }

        static void ValidateOptions(const NormalEstimationOptions &options) {
            FindNormalEstimator(options.estimator); // throws on an unknown name
            if (options.planeFitRadius < 1)
                throw std::runtime_error("Realsense::NormalEstimation: planeFitRadius is " +
                                         std::to_string(options.planeFitRadius) +
                                         "; the fit needs at least one ring around the centre");
            // Three non-collinear samples are the algebraic minimum for a plane; anything less
            // cannot be a fit, and the caller's threshold decides how over-determined it must be.
            if (options.minimumPlaneFitSamples < 3)
                throw std::runtime_error("Realsense::NormalEstimation: minimumPlaneFitSamples is " +
                                         std::to_string(options.minimumPlaneFitSamples) +
                                         "; a plane needs at least 3");
        }

    private:
        // One pipeline per strategy, compiled on first use and cached. Rebuilding rather than
        // caching would re-allocate descriptors on every frame of an A/B run that alternates
        // estimators.
        Engine::Core::ComputePipeline &estimatorKernel(const NormalEstimationOptions &options) {
            auto found = m_kernels.find(options.estimator);
            if (found != m_kernels.end()) return *found->second;

            const NormalEstimatorStrategy &strategy = FindNormalEstimator(options.estimator);
            auto pipeline = std::make_unique<Engine::Core::ComputePipeline>(m_context);
            if (strategy.macroName) pipeline->Define(strategy.macroName);
            pipeline->Build("Realsense/Algorithm/NormalEstimation.EstimateNormal.glsl");
            return *m_kernels.emplace(options.estimator, std::move(pipeline)).first->second;
        }

        Engine::Core::Context &m_context;
        std::map<std::string, std::unique_ptr<Engine::Core::ComputePipeline>> m_kernels;
    };

} // namespace Realsense
```

- [ ] **Step 10: `ValidationMask.h`에 버퍼 2개를 추가한다**

`MakeBuffers`의 `m_compactScores` 뒤:

```cpp
            // Parallel to m_vertices: dense, device-local, never leaves the device. Only the
            // compacted run below is read back.
            m_normals = std::make_unique<Engine::Core::Buffer>(m_context);
            m_normals->Allocate(std::uint32_t(pixels * 4u * sizeof(float)));

            m_compactNormals = std::make_unique<Engine::Core::Buffer>(m_context);
            m_compactNormals->AllocateHostVisibleReadback(std::uint32_t(pixels * 4u * sizeof(float)));

            m_normalCounters = std::make_unique<Engine::Core::Buffer>(m_context);
            m_normalCounters->AllocateHostVisibleReadback(std::uint32_t(sizeof(NormalEstimationCounters)));
```

멤버 선언에 넷을 추가한다 (`m_normals`, `m_compactNormals`, `m_normalCounters`, 그리고 `std::unique_ptr<NormalEstimation> m_normalEstimation`).

`NormalEstimation`은 `ValidationMask.h`를 include하므로 **여기서 헤더를 include하면 순환**이 된다. `ValidationMask.h` 위쪽에 전방 선언을 두고, 생성자·`RecordRemainValidDepth`의 정의는 `NormalEstimation.h`가 include된 뒤에 오도록 `ValidationMask.h` **맨 아래**에 `#include "Realsense/Algorithm/NormalEstimation.h"`를 두는 대신 — 더 단순하게, **`NormalEstimationOptions`와 레지스트리를 `RealSenseTypes.h`로 옮기고** `NormalEstimation.h`가 `ValidationMask.h`를 include하지 않게 한다. `RecordEstimate`의 `ValidationScoreOptions` 인자도 `RealSenseTypes.h`에 있으면 순환이 사라진다.

→ **Step 10a:** `ValidationScoreOptions`와 `NormalEstimationOptions`, 레지스트리 세 함수를 `RealSenseTypes.h`로 옮긴다. `ValidationMask.h`는 `RealSenseTypes.h`만 include하면 되고, `NormalEstimation.h`도 마찬가지다. `ValidationMask.h`가 `NormalEstimation.h`를 include한다(단방향).

- [ ] **Step 11: 체인에 패스를 끼운다 — `RecordRemainValidDepth`**

시그니처에 `const NormalEstimationOptions &normalOptions`를 마지막 인자로 추가하고, RemainValidDepth 디스패치 **뒤**, CountEmittedPerRow **앞**에 넣는다:

```cpp
            dispatchOverImage(batch, *kernel_RemainValidDepth);
            batch.Barrier();

            // Before the count, not after: this pass CANCELS emitted pixels, and the row counts
            // have to see the final verdict.
            batch.FillBuffer(m_normalCounters->Handle(), 0, sizeof(NormalEstimationCounters), 0u);
            batch.Barrier();
            m_normalEstimation->RecordEstimate(batch, *m_vertices, *m_properties, *m_normals,
                                               *m_normalCounters, m_width, m_height, options,
                                               normalOptions);
            batch.Barrier();
```

- [ ] **Step 12: scatter에 바인딩 2개를 추가한다**

`ValidationMask.ScatterValidPoints.glsl`:

```glsl
layout(std430, set = 0, binding = 0) readonly  buffer ValidMask      { ValidationMaskProperty g_properties[]; };
layout(std430, set = 0, binding = 1) readonly  buffer VertexGrid     { vec4 g_vertices[]; };
layout(std430, set = 0, binding = 2) readonly  buffer RowOffset      { uint g_rowOffset[]; };
layout(std430, set = 0, binding = 3) writeonly buffer CompactPoints  { vec4 g_points[]; };
layout(std430, set = 0, binding = 4) writeonly buffer CompactScores  { float g_compactScores[]; };
layout(std430, set = 0, binding = 5) readonly  buffer NormalGrid     { vec4 g_normals[]; };
layout(std430, set = 0, binding = 6) writeonly buffer CompactNormals { vec4 g_compactNormals[]; };
```

쓰는 곳:

```glsl
			g_points[slot]         = g_vertices[pixel];
			g_compactScores[slot]  = g_properties[pixel].score;
			// The normal travels WITH the point for the same reason the score does: a compacted
			// cloud whose normals were left behind in image space cannot be re-paired with them.
			g_compactNormals[slot] = g_normals[pixel];
```

파일 상단의 "Points only. ... this module has no normals yet" 문단을 지운다 — 더 이상 사실이 아니다.

C++ 쪽 바인딩도 맞춘다:

```cpp
            kernel_ScatterValidPoints->Bind(0, *m_properties)
                    .Bind(1, *m_vertices)
                    .Bind(2, *m_rowOffset)
                    .Bind(3, *m_points)
                    .Bind(4, *m_compactScores)
                    .Bind(5, *m_normals)
                    .Bind(6, *m_compactNormals);
```

- [ ] **Step 13: 다운로드 두 개를 추가한다**

```cpp
        // Parallel to DownloadValidPoints: the normal at each surviving point, same index.
        std::vector<Eigen::Vector3f> DownloadValidNormals() const {
            const std::uint32_t count = ValidPointCount();
            std::vector<Eigen::Vector3f> out(count);
            if (count == 0) return out;
            m_compactNormals->MakeVisibleToCPU(count * 4u * std::uint32_t(sizeof(float)));
            const float *words = static_cast<const float *>(m_compactNormals->MappedPtr());
            for (std::uint32_t i = 0; i < count; ++i)
                out[i] = Eigen::Vector3f(words[4 * i + 0], words[4 * i + 1], words[4 * i + 2]);
            return out;
        }

        NormalEstimationCounters DownloadNormalCounters() const {
            m_normalCounters->MakeVisibleToCPU(std::uint32_t(sizeof(NormalEstimationCounters)));
            NormalEstimationCounters out;
            std::memcpy(&out, m_normalCounters->MappedPtr(), sizeof out);
            return out;
        }
```

- [ ] **Step 14: 셰이더가 컴파일되는지 먼저 확인한다**

```bash
for f in NormalEstimation.EstimateNormal ValidationMask.ScatterValidPoints; do
  glslc -fshader-stage=compute -I src src/Realsense/Algorithm/$f.glsl -o /dev/null || echo "FAIL: $f"
done
```
Expected: 출력 없음.

- [ ] **Step 15: 테스트를 통과시킨다**

```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-rel --target vkspatial_tests -j8
./build-rel/test/vkspatial_tests --gtest_filter='RealsenseNormalEstimation*'
```
Expected: 3 PASSED.

- [ ] **Step 16: 기존 테스트가 안 깨졌는지 확인한다**

`RecordRemainValidDepth`에 인자가 하나 늘었으므로 `test_realsenseValidationScore.cpp`의 호출부 3곳(`RunCompaction`, `ReThresholdingDoesNotNeedAReScore`, `CompactionCarriesAcrossChunksOnAWideFrame`)에 `NormalEstimationOptions{}`를 넘긴다. 단, 그 테스트들은 **법선 게이트가 점을 줄인다**는 사실을 모른다 — `CompactionKeepsExactlyThePixelsAboveTheThreshold`는 평평한 프레임의 테두리를 세므로 `forward`의 액자가 겹친다. 기대값을 고치지 말고 그 테스트들에는 **`estimator = "forward"`가 아니라 법선 패스를 건너뛰는 경로**가 필요하다 → `NormalEstimationOptions`에 `bool enabled = true;`를 두고, 그 테스트들은 `enabled = false`로 넘겨 기존 기대값을 그대로 유지한다. 커널 디스패치를 통째로 건너뛴다.

```bash
./build-rel/test/vkspatial_tests --gtest_filter='Realsense*'
```
Expected: 13 PASSED (기존 10 + 신규 3).

---

## Task 3: `central` 전략

**Files:**
- Create: `src/Realsense/Algorithm/NormalEstimation.CentralDifference.glsl`
- Modify: `test/test_realsenseNormalEstimation.cpp`

**Interfaces:**
- Consumes: Task 2의 `EstimateSurfaceNormal` 시그니처와 `SameSurfaceSample`
- Produces: 등록 이름 `"central"`이 실제로 빌드된다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```cpp
// Every registered name must actually build and run, and an unknown one must throw rather than
// silently running the default -- a misspelled name that quietly ran "forward" would report the
// accuracy of an estimator nobody selected.
TEST(RealsenseNormalEstimation, EveryRegisteredEstimatorBuildsAndAnUnknownOneThrows) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
    const std::vector<std::uint16_t> depth = RenderPlane(truth, 1.5f, 0.001f);

    for (const std::string &name: Realsense::NormalEstimatorNames()) {
        const NormalRun run = RunFrontEnd(context, depth, name);
        EXPECT_GT(run.normals.size(), 0u) << "estimator '" << name << "' emitted nothing";
        EXPECT_LT(MedianAngleDegrees(run.normals, truth), 1.0f) << "estimator '" << name << "'";
    }

    NormalEstimationOptions unknown;
    unknown.estimator = "centarl"; // transposed on purpose
    EXPECT_THROW(Realsense::NormalEstimation::ValidateOptions(unknown), std::runtime_error);
}
```

- [ ] **Step 2: 실패를 확인한다**

```bash
cmake --build build-rel --target vkspatial_tests -j8
./build-rel/test/vkspatial_tests --gtest_filter='*EveryRegisteredEstimatorBuilds*'
```
Expected: FAIL — `"central"`을 고르면 디스패처가 없는 파일을 include하려 해서 `ComputePipeline::Build`가 런타임에 던진다.

- [ ] **Step 3: `NormalEstimation.CentralDifference.glsl`을 만든다**

```glsl
/// *********************************************
/// "central" -- symmetric differences over the same 3x3 footprint.
///
///     n = ((P(u+1,v) - P(u-1,v)) / 2) x ((P(u,v+1) - P(u,v-1)) / 2)
///
/// Two changes against "forward", both free:
///
///  - The estimate is anchored ON the pixel instead of at the corner of a triangle, so the
///    half-pixel shift of the normal field disappears.
///  - The tangent spans two pixels rather than one, so a difference of two samples with noise
///    sigma is divided by 2 instead of 1: the gradient carries sigma/sqrt(2) where "forward"
///    carries sqrt(2)*sigma. Half the angular noise, for the same four loads.
///
/// It does NOT fall back to a one-sided difference when a partner is missing. A frame whose normals
/// came from two different estimators cannot answer "did the symmetric stencil help", which is the
/// only reason to have this axis; the pixel is refused instead and charged to the stencil counter.
/// *********************************************

int EstimateSurfaceNormal(int column, int row, float depth, float tolerance, out vec3 normal)
{
	normal = vec3(0.0);

	// 1. The stencil reaches one pixel either side, so a one-pixel border has no domain.
	if (column - 1 < 0 || column + 1 >= g_width || row - 1 < 0 || row + 1 >= g_height)
		return NORMAL_ESTIMATE_OUT_OF_DOMAIN;

	// 2. All four partners must be on this pixel's surface.
	vec3 left, right, above, below;
	if (!SameSurfaceSample(column - 1, row, depth, tolerance, left)) return NORMAL_ESTIMATE_NO_SUPPORT;
	if (!SameSurfaceSample(column + 1, row, depth, tolerance, right)) return NORMAL_ESTIMATE_NO_SUPPORT;
	if (!SameSurfaceSample(column, row - 1, depth, tolerance, above)) return NORMAL_ESTIMATE_NO_SUPPORT;
	if (!SameSurfaceSample(column, row + 1, depth, tolerance, below)) return NORMAL_ESTIMATE_NO_SUPPORT;

	// 3. Cross the two central differences. The 0.5 factors scale the normal, not its direction,
	//    and normalize() drops them -- they are written out because the tangents are what the
	//    noise argument above is about, not the cross product.
	vec3 tangentAlongColumns = (right - left) * 0.5;
	vec3 tangentAlongRows    = (below - above) * 0.5;

	normal = cross(tangentAlongColumns, tangentAlongRows);
	if (length(normal) < 1e-9) return NORMAL_ESTIMATE_NO_SUPPORT;

	normal = normalize(normal);
	return NORMAL_ESTIMATE_OK;
}
```

- [ ] **Step 4: 통과를 확인한다**

```bash
glslc -fshader-stage=compute -I src -DNORMAL_CENTRAL_DIFFERENCE src/Realsense/Algorithm/NormalEstimation.EstimateNormal.glsl -o /dev/null
cmake --build build-rel --target vkspatial_tests -j8
./build-rel/test/vkspatial_tests --gtest_filter='Realsense*'
```
Expected: `"planefit"`만 아직 실패한다(파일 없음). `"forward"`/`"central"`은 통과. Task 4에서 마저 통과시킨다.

---

## Task 4: `planefit` 전략 + 잡음 정확도 순서

정확도가 이 작업의 목적이고, 이 태스크가 그것을 증명한다.

**Files:**
- Create: `src/Realsense/Algorithm/NormalEstimation.PlaneFit.glsl`
- Modify: `test/test_realsenseNormalEstimation.cpp`

**Interfaces:**
- Consumes: Task 2의 `EstimateSurfaceNormal` 시그니처, 푸시상수 `g_planeFitRadius` / `g_minimumPlaneFitSamples`
- Produces: 등록 이름 `"planefit"` (기본값)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```cpp
namespace {

    // A deterministic normal deviate. std::mt19937 with a fixed seed rather than rand(): the
    // assertion below is about the RATIO between estimators, and two estimators must see the same
    // frame or the comparison measures the noise draw instead.
    std::vector<std::uint16_t> RenderNoisyPlane(const Eigen::Vector3f &normal, float distance,
                                                float depthScale, float sigmaMetres) {
        std::vector<std::uint16_t> depth = RenderPlane(normal, distance, depthScale);
        std::mt19937 generator(20260905u);
        std::normal_distribution<float> deviate(0.0f, sigmaMetres);
        for (std::uint16_t &sample: depth) {
            if (sample == 0) continue;
            const float metres = float(sample) * depthScale + deviate(generator);
            const float units = std::round(metres / depthScale);
            sample = (units > 0.0f && units <= 65535.0f) ? std::uint16_t(units) : std::uint16_t(0);
        }
        return depth;
    }

} // namespace

// The reason this module exists at its measured accuracy. sigma_z = 2 mm is the noisy regime the
// repository characterised: docs/DEPTH_NOISE_FILTERING.md reports 50.6 / 32.8 / 12.5 degrees
// against ground truth for forward / central / planefit.
//
// The assertion is on the ORDER and the RATIO, not on those absolute degrees: the synthetic
// scene's lateral spacing and the noise draw move the absolute angle, while the ratio follows the
// theoretical gradient-noise factors 1.41 / 0.71 / 0.32.
TEST(RealsenseNormalEstimation, PlaneFitIsTheMostAccurateUnderDepthNoise) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
    const std::vector<std::uint16_t> depth = RenderNoisyPlane(truth, 1.5f, 0.001f, 0.002f);

    const float forward = MedianAngleDegrees(RunFrontEnd(context, depth, "forward").normals, truth);
    const float central = MedianAngleDegrees(RunFrontEnd(context, depth, "central").normals, truth);
    const float planefit = MedianAngleDegrees(RunFrontEnd(context, depth, "planefit").normals, truth);

    EXPECT_LT(central, forward) << "the symmetric stencil must beat the corner triangle";
    EXPECT_LT(planefit, central) << "the least-squares fit must beat the symmetric stencil";
    EXPECT_LT(planefit * 3.0f, forward)
            << "planefit " << planefit << " deg vs forward " << forward
            << " deg -- the gain collapsed, which is what a lost origin shift or a missing trace "
               "normalisation looks like";
}
```

`<random>`을 include한다.

- [ ] **Step 2: 실패를 확인한다**

```bash
cmake --build build-rel --target vkspatial_tests -j8
./build-rel/test/vkspatial_tests --gtest_filter='*PlaneFitIsTheMostAccurate*'
```
Expected: FAIL — `"planefit"` 변형이 없는 파일을 include하려 해서 던진다.

- [ ] **Step 3: `NormalEstimation.PlaneFit.glsl`을 만든다**

`src/Pipeline/Reconstruction/Algorithm/NormalPlaneFit.glsl`을 **내용 변경 없이** 그대로 옮긴다. `SmallestEigenvalueOfSymmetric`, `NullDirectionOfSymmetric`, `EstimateSurfaceNormal` 세 함수와 배너 주석 전부를 포함한다. 그 파일은 `g_planeFitRadius`와 `g_minimumPlaneFitSamples`를 푸시상수에서 읽는데, Task 2의 커널이 같은 이름으로 선언했으므로 그대로 맞는다.

바꾸지 말 것 (전부 뮤테이션으로 검증된 것들):
- 모멘트는 **중심점을 원점으로** 누적한다 — 1.5 m에서 좌표는 O(1 m), 창 안 퍼짐은 O(1 mm)라 `E[p^2]-E[p]^2`가 float32에서 7자리 상쇄된다.
- 고유분해 전에 **트레이스 정규화**한다 — 공분산 성분이 O(1e-6)이라 그 세제곱이 float32 하한 근처다.
- 영공간 외적은 **세 쌍 중 가장 긴 것**을 쓴다 — 어느 한 쌍은 거의 평행할 수 있다.
- `dot(normal, normal) < 1e-12`면 거부한다 — 공선 표본(1 px 폭 표면)은 개수 하한으로 못 막는다.

- [ ] **Step 4: 통과를 확인한다**

```bash
glslc -fshader-stage=compute -I src -DNORMAL_PLANE_FIT src/Realsense/Algorithm/NormalEstimation.EstimateNormal.glsl -o /dev/null
cmake --build build-rel --target vkspatial_tests -j8
./build-rel/test/vkspatial_tests --gtest_filter='Realsense*'
```
Expected: 전부 PASSED. 실패하면 각도를 출력해 어느 쪽이 무너졌는지 본다.

- [ ] **Step 5: 뮤테이션으로 단언이 실제로 무는지 확인한다**

`NormalEstimation.PlaneFit.glsl`에서 원점 이동을 없앤다 — `vec3 offset = neighbourPoint - centrePoint;`를 `vec3 offset = neighbourPoint;`로 바꾼다.

```bash
cmake --build build-rel --target vkspatial_tests -j8
./build-rel/test/vkspatial_tests --gtest_filter='*PlaneFitIsTheMostAccurate*'
```
Expected: **FAIL.** float32 상쇄로 planefit의 이득이 무너져야 한다. 통과해 버리면 테스트가 이 함정을 못 잡는 것이므로, 잡음 σ를 키우거나 비 임계값을 조이고 다시 확인한다.

- [ ] **Step 6: 뮤테이션을 되돌리고 다시 통과시킨다**

```bash
cmake --build build-rel --target vkspatial_tests -j8
./build-rel/test/vkspatial_tests --gtest_filter='Realsense*'
```
Expected: 전부 PASSED.

---

## Task 5: 경계 거동 + 문서

**Files:**
- Modify: `test/test_realsenseNormalEstimation.cpp`
- Create: `src/Realsense/Algorithm/NormalEstimation.md`
- Modify: `src/Realsense/Algorithm/ValidationMask.md` (체인에 패스 하나 추가)

**Interfaces:**
- Consumes: Task 2–4 전부
- Produces: 없음 (최종 태스크)

- [ ] **Step 1: 단차와 구멍 테스트를 쓴다**

```cpp
// Two planes at different depths meeting mid-image. A normal differenced ACROSS the step belongs
// to neither surface, so the estimators must refuse those pixels rather than emit a normal that
// smears from one plane to the other.
TEST(RealsenseNormalEstimation, NormalsDoNotSmearAcrossADepthStep) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth(0.0f, 0.0f, -1.0f);
    std::vector<std::uint16_t> depth = RenderPlane(truth, 1.0f, 0.001f);
    // The right half jumps 30 cm further away -- far beyond tau = 3*sigma_z (11.3 mm at 1 m).
    for (int row = 0; row < kHeight; ++row)
        for (int column = kWidth / 2; column < kWidth; ++column)
            depth[std::size_t(row) * kWidth + column] += 300;

    const NormalRun run = RunFrontEnd(context, depth, "planefit");
    ASSERT_GT(run.normals.size(), 0u);
    // Both halves are fronto-parallel, so every SURVIVING normal must still be the true one. A
    // normal that straddled the step would tilt hard and show up here.
    EXPECT_LT(MedianAngleDegrees(run.normals, truth), 1.0f);
    for (const Eigen::Vector3f &normal: run.normals)
        EXPECT_LT(std::acos(std::clamp(normal.dot(truth), -1.0f, 1.0f)) * 180.0f / float(M_PI), 15.0f);
}

// A hole in the window costs planefit accuracy but not the pixel: it fits whatever same-surface
// samples the window holds. A difference stencil loses the pixel outright when one partner is
// missing, which is where the measured +1.6% emitted points comes from.
TEST(RealsenseNormalEstimation, PlaneFitSurvivesHolesThatDefeatADifferenceStencil) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth(0.0f, 0.0f, -1.0f);
    std::vector<std::uint16_t> depth = RenderPlane(truth, 1.5f, 0.001f);
    // Drop every fourth column: a forward difference loses its right-hand partner on the column
    // before each hole, while a 5x5 fit still sees ~19 of 25 samples.
    for (int row = 0; row < kHeight; ++row)
        for (int column = 0; column < kWidth; column += 4)
            depth[std::size_t(row) * kWidth + column] = 0;

    const NormalRun forward = RunFrontEnd(context, depth, "forward");
    const NormalRun planefit = RunFrontEnd(context, depth, "planefit");

    EXPECT_GT(planefit.normals.size(), forward.normals.size())
            << "planefit " << planefit.normals.size() << " vs forward " << forward.normals.size();
    EXPECT_GT(forward.counters.noSupport, planefit.counters.noSupport)
            << "the difference stencil must be the one charged for the holes";
}
```

- [ ] **Step 2: 통과를 확인한다**

```bash
cmake --build build-rel --target vkspatial_tests -j8
./build-rel/test/vkspatial_tests --gtest_filter='Realsense*'
```
Expected: 전부 PASSED (기존 10 + 신규 7).

- [ ] **Step 3: 회귀 전체를 돌린다**

```bash
./build-rel/test/vkspatial_tests
cmake --build build-rel --target validation_score_lab -j8
./build-rel/example2/validation_score_lab --replay capture --sweep k --frames 5
```
Expected: 스위트 전체 통과(기존 339 + 신규 7), sweep 표의 네 열이 100%로 분할된다.

- [ ] **Step 4: `NormalEstimation.md`를 쓴다**

`process-summary` 형식(`ValidationMask.md`가 레퍼런스)을 따라 4단계로: 허용치 → 전략 디스패치 → 방향 정렬 → `emitted` 취소와 카운터. 30~60줄.

- [ ] **Step 5: `ValidationMask.md`의 체인 표에 패스를 추가한다**

`## 1. Threshold & Back-projection`과 `## 2. Row Count` 사이에 법선 패스를 넣고, 뒤 섹션 번호를 하나씩 민다. `## 4. Scatter`의 불릿에 법선도 함께 실린다는 것을 적는다.

---

## Self-Review

**스펙 커버리지:**

| 스펙 절 | 태스크 |
| --- | --- |
| §3-1 세 전략 레지스트리, planefit 기본값 | Task 2 (Step 9), 3, 4 |
| §3-2 법선 실패 → `emitted` 취소, 입사각 없음 | Task 2 (Step 7 커널, Step 1 게이트 테스트) |
| §3-3 같은표면 허용치 한 정의 | Task 1 |
| §3-4 `[H4]`/`[H5]` 미이식 | Task 2 Step 7 (커널이 둘을 부르지 않음) |
| §3-5 카운터 두 종류, 상시 켬 | Task 2 (Step 3·4·7), Task 5 (Step 1이 소비) |
| §3-6 버퍼·순서는 ValidationMask 소유 | Task 2 (Step 10·11) |
| §4 파일 목록, Depth2Normal 삭제 | Task 2 (Step 8) |
| §5 테스트 1–7 | 1·2·4·5 → Task 2, 7 → Task 3, 2 → Task 4, 3·6 → Task 5 |
| §6 비목표 | 계획에 없음 (의도됨) |

**플레이스홀더 스캔:** 없음. Task 4 Step 3만 "그대로 옮긴다"인데, 원본 경로와 바꾸지 말아야 할 네 가지를 명시했으므로 실행자가 판단할 여지가 없다.

**타입 일관성:** `NormalEstimationOptions`(estimator/planeFitRadius/minimumPlaneFitSamples/enabled), `NormalEstimationCounters`(outOfDomain/noSupport), `NormalPushConstants` 8필드가 GLSL 푸시상수 블록과 순서·타입이 일치한다. `EstimateSurfaceNormal`의 시그니처가 세 프래그먼트에서 동일하다. `RecordRemainValidDepth`의 새 인자가 Task 2 Step 11과 테스트 헬퍼에서 같은 위치다.

**발견해 고친 것:** Task 2 Step 10에서 `NormalEstimation.h` ↔ `ValidationMask.h` **순환 include**를 발견해 Step 10a(옵션·레지스트리를 `RealSenseTypes.h`로 이동)를 추가했다. Task 2 Step 16에서 기존 압축 테스트가 법선 액자 때문에 깨진다는 것을 발견해 `NormalEstimationOptions::enabled`를 추가했다.

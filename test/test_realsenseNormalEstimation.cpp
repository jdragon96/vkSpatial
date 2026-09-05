#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Realsense/RealSensePipeline.h"
#include "Realsense/RealSenseD435.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using Realsense::NormalEstimationOptions;
using Realsense::ValidationScoreOptions;

namespace {

    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    constexpr float kFocal = 425.0f;

    // Z16 quantisation is itself a noise source, and a coarse one: at 1.5 m the lateral spacing
    // between pixels is z/f = 3.5 mm, so the sensor's own 1 mm quantum tilts a one-pixel forward
    // difference by about 4.8 degrees all by itself. A fixture meant to have ONE right answer has
    // to quantise finely enough that the geometry, not the quantum, is what is being measured.
    // depthScale is a stream property, so asking for a finer one is legitimate rather than a fudge.
    constexpr float kFineDepthScale = 0.00005f;  // 0.05 mm -- 1.5 m lands at 30000 Z16 units
    constexpr float kSensorDepthScale = 0.001f;  // what a D435 actually reports

    ValidationScoreOptions TestOptions(float depthScale) {
        ValidationScoreOptions options;
        options.focalLengthPixels = kFocal;
        options.baselineMeters = 0.05f;
        options.subpixelRms = 0.08f;
        options.depthScale = depthScale;
        options.nearFadeStart = 0.1f;
        options.nearFadeEnd = 0.2f;
        options.farFadeStart = 5.0f;
        options.farFadeEnd = 6.0f;
        return options;
    }

    // A plane through (0,0,distance) with unit normal `normal`, sampled on the pixel grid and
    // quantised to Z16. The ray through (u,v) is d = ((u-cx)/fx, (v-cy)/fy, 1); it meets the plane
    // at t = (n . p0) / (n . d), and that t IS the sample's depth because d.z == 1.
    std::vector<std::uint16_t> RenderPlane(const Eigen::Vector3f &normal, float distance,
                                           float depthScale, float focal = kFocal) {
        const float cx = float(kWidth) * 0.5f, cy = float(kHeight) * 0.5f;
        const Eigen::Vector3f pointOnPlane(0.0f, 0.0f, distance);
        const float numerator = normal.dot(pointOnPlane);

        std::vector<std::uint16_t> depth(std::size_t(kWidth) * kHeight, 0);
        for (int row = 0; row < kHeight; ++row)
            for (int column = 0; column < kWidth; ++column) {
                const Eigen::Vector3f ray((float(column) - cx) / focal,
                                          (float(row) - cy) / focal, 1.0f);
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
                          const std::string &estimator,
                          float depthScale, float focal = kFocal) {
        ValidationScoreOptions options = TestOptions(depthScale);
        options.focalLengthPixels = focal;
        NormalEstimationOptions normalOptions;
        normalOptions.estimator = estimator;

        Realsense::RealSensePipeline pipeline(context, kWidth, kHeight);
        const Realsense::PinholeIntrinsics intrinsics{focal, focal, float(kWidth) * 0.5f,
                                                      float(kHeight) * 0.5f};
        {
            Engine::Compute::CommandBatch batch(context);
            pipeline.Execute(batch, depth.data(), options, intrinsics, 0.0f, normalOptions);
            batch.Submit();
        }
        NormalRun run;
        run.points = pipeline.DownloadValidPoints();
        run.normals = pipeline.DownloadValidNormals();
        run.counters = pipeline.DownloadNormalCounters();
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
// normalisation and the orientation flip all at once. An estimator that gets this wrong is wrong
// for a reason no noise test would isolate.
TEST(RealsenseNormalEstimation, ANoiselessTiltedPlaneMatchesItsTrueNormal) {
    Engine::Core::Context context;
    // Tilted, not fronto-parallel: a plane facing straight down -z would pass even if the two
    // tangents were swapped or a sign were dropped.
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
    const std::vector<std::uint16_t> depth = RenderPlane(truth, 1.5f, kFineDepthScale);

    const NormalRun run = RunFrontEnd(context, depth, "forward", kFineDepthScale);
    ASSERT_GT(run.normals.size(), 0u);
    ASSERT_EQ(run.normals.size(), run.points.size());
    EXPECT_LT(MedianAngleDegrees(run.normals, truth), 1.0f);
}

// Every emitted normal must face the camera: the fusion weight and the point-to-plane residual
// both change sign with it, so a flipped normal is not a small error.
TEST(RealsenseNormalEstimation, EveryEmittedNormalFacesTheCamera) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
    const std::vector<std::uint16_t> depth = RenderPlane(truth, 1.5f, kFineDepthScale);

    const NormalRun run = RunFrontEnd(context, depth, "forward", kFineDepthScale);
    ASSERT_GT(run.points.size(), 0u);
    for (std::size_t i = 0; i < run.points.size(); ++i)
        EXPECT_LE(run.normals[i].dot(run.points[i]), 0.0f) << "point " << i;
}

// The gate this module chose: `emitted == 1` has to mean "coordinate AND normal", so no downstream
// consumer has to special-case a zero normal.
TEST(RealsenseNormalEstimation, EveryCompactedPointCarriesAUnitNormal) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
    const std::vector<std::uint16_t> depth = RenderPlane(truth, 1.5f, kFineDepthScale);

    const NormalRun run = RunFrontEnd(context, depth, "forward", kFineDepthScale);
    ASSERT_GT(run.normals.size(), 0u);
    for (const Eigen::Vector3f &normal: run.normals)
        EXPECT_NEAR(normal.norm(), 1.0f, 1e-4f);
}

// Every registered name must actually build and run, and an unknown one must throw rather than
// silently running the default -- a misspelled name that quietly ran "forward" would report the
// accuracy of an estimator nobody selected.
TEST(RealsenseNormalEstimation, EveryRegisteredEstimatorBuildsAndAnUnknownOneThrows) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
    const std::vector<std::uint16_t> depth = RenderPlane(truth, 1.5f, kFineDepthScale);

    for (const std::string &name: Realsense::NormalEstimatorNames()) {
        const NormalRun run = RunFrontEnd(context, depth, name, kFineDepthScale);
        EXPECT_GT(run.normals.size(), 0u) << "estimator '" << name << "' emitted nothing";
        EXPECT_LT(MedianAngleDegrees(run.normals, truth), 1.0f) << "estimator '" << name << "'";
    }

    NormalEstimationOptions unknown;
    unknown.estimator = "centarl"; // transposed on purpose
    EXPECT_THROW(Realsense::NormalEstimation::ValidateOptions(unknown), std::runtime_error);
}

namespace {

    // A deterministic normal deviate. Fixed seed rather than rand(): the assertion below is about
    // the RATIO between estimators, and two estimators must see the SAME frame or the comparison
    // measures the noise draw instead of the estimator.
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

// The reason this module runs planefit by default. sigma_z = 2 mm is the noisy regime the
// repository characterised: docs/DEPTH_NOISE_FILTERING.md reports 50.6 / 32.8 / 12.5 degrees
// against ground truth for forward / central / planefit.
//
// The assertion is on the ORDER and the RATIO, not on those absolute degrees: the synthetic scene's
// lateral spacing and the noise draw move the absolute angle, while the ratio follows the
// theoretical gradient-noise factors 1.41 / 0.71 / 0.32.
TEST(RealsenseNormalEstimation, PlaneFitIsTheMostAccurateUnderDepthNoise) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
    const std::vector<std::uint16_t> depth =
            RenderNoisyPlane(truth, 1.5f, kSensorDepthScale, 0.002f);

    const float forward =
            MedianAngleDegrees(RunFrontEnd(context, depth, "forward", kSensorDepthScale).normals, truth);
    const float central =
            MedianAngleDegrees(RunFrontEnd(context, depth, "central", kSensorDepthScale).normals, truth);
    const float planefit =
            MedianAngleDegrees(RunFrontEnd(context, depth, "planefit", kSensorDepthScale).normals, truth);

    EXPECT_LT(central, forward) << "the symmetric stencil must beat the corner triangle: central "
                                << central << " deg vs forward " << forward << " deg";
    EXPECT_LT(planefit, central) << "the least-squares fit must beat the symmetric stencil: planefit "
                                 << planefit << " deg vs central " << central << " deg";
    EXPECT_LT(planefit * 3.0f, forward)
            << "planefit " << planefit << " deg vs forward " << forward
            << " deg -- the gain collapsed, which is what a lost origin shift or a missing trace "
               "normalisation looks like";
}

// The plane fit accumulates its moments FROM THE CENTRE POINT, and that shift is load-bearing
// rather than cosmetic: a covariance is the difference of two nearly equal quantities, so when the
// coordinates dwarf the spread across the window, E[p^2] - E[p]^2 cancels most of float32's seven
// digits and the eigenvector becomes rounding noise.
//
// How badly depends on focal / windowExtent, not on range -- the lateral spacing z/f scales with z
// exactly as the coordinates do. A D435 (f = 425, 5x5 window) sits near 100:1, which costs about
// four digits: degraded, but not enough to change which estimator wins, so the accuracy test above
// does NOT pin this. A narrow-FOV sensor reaches 1000:1, and there the shift is the difference
// between an exact normal and a useless one. Mutation-verified: deleting the shift takes this
// fixture from ~0.1 to 10.7 degrees.
TEST(RealsenseNormalEstimation, PlaneFitStaysConditionedWhenCoordinatesDwarfTheWindow) {
    Engine::Core::Context context;
    const float narrowFieldFocal = 4250.0f;
    // 25 um: the finest quantum that still fits 1.5 m into Z16, so what this measures is the
    // conditioning of the fit and not the quantisation of the fixture.
    const float fineQuantum = 0.000025f;
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
    const std::vector<std::uint16_t> depth =
            RenderPlane(truth, 1.5f, fineQuantum, narrowFieldFocal);

    const NormalRun run = RunFrontEnd(context, depth, "planefit", fineQuantum, narrowFieldFocal);
    ASSERT_GT(run.normals.size(), 0u);
    EXPECT_LT(MedianAngleDegrees(run.normals, truth), 1.0f);
}

// Two fronto-parallel planes meeting mid-image. A normal differenced ACROSS the step belongs to
// neither surface, so every estimator must refuse those pixels rather than emit one that smears
// from the near plane to the far one.
TEST(RealsenseNormalEstimation, NormalsDoNotSmearAcrossADepthStep) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth(0.0f, 0.0f, -1.0f);
    std::vector<std::uint16_t> depth = RenderPlane(truth, 1.5f, kFineDepthScale);
    // The right half jumps 0.3 m further away -- far beyond tau = 3*sigma_z, which is 25 mm here.
    const std::uint16_t step = std::uint16_t(0.3f / kFineDepthScale);
    for (int row = 0; row < kHeight; ++row)
        for (int column = kWidth / 2; column < kWidth; ++column)
            depth[std::size_t(row) * kWidth + column] += step;

    const NormalRun run = RunFrontEnd(context, depth, "planefit", kFineDepthScale);
    ASSERT_GT(run.normals.size(), 0u);
    // Both halves face the camera squarely, so every SURVIVING normal must still be the true one.
    // One that straddled the step would tilt hard and show up here.
    EXPECT_LT(MedianAngleDegrees(run.normals, truth), 1.0f);
    for (const Eigen::Vector3f &normal: run.normals) {
        const float degrees =
                std::acos(std::clamp(normal.dot(truth), -1.0f, 1.0f)) * 180.0f / float(M_PI);
        EXPECT_LT(degrees, 15.0f) << "a normal was fitted across the step";
    }
}

// A hole in the window costs planefit accuracy but not the pixel: it fits whatever same-surface
// samples the window holds and refuses only below minimumPlaneFitSamples. A difference stencil
// loses the pixel outright when one partner is missing, which is where the measured +1.6% emitted
// points on real frames comes from.
TEST(RealsenseNormalEstimation, PlaneFitSurvivesHolesThatDefeatADifferenceStencil) {
    Engine::Core::Context context;
    const Eigen::Vector3f truth(0.0f, 0.0f, -1.0f);
    std::vector<std::uint16_t> depth = RenderPlane(truth, 1.5f, kFineDepthScale);
    // Drop every fourth column: a forward difference loses its right-hand partner on the column
    // before each hole, while a 5x5 fit still sees about 19 of 25 samples.
    for (int row = 0; row < kHeight; ++row)
        for (int column = 0; column < kWidth; column += 4)
            depth[std::size_t(row) * kWidth + column] = 0;

    const NormalRun forward = RunFrontEnd(context, depth, "forward", kFineDepthScale);
    const NormalRun planefit = RunFrontEnd(context, depth, "planefit", kFineDepthScale);

    EXPECT_GT(planefit.normals.size(), forward.normals.size())
            << "planefit " << planefit.normals.size() << " vs forward " << forward.normals.size();
    EXPECT_GT(forward.counters.noSupport, planefit.counters.noSupport)
            << "the difference stencil must be the one charged for the holes";
}

// A camera that was never opened reports no focal length, and the scaling below would turn that
// into a radius of 0 -- which NormalEstimation::ValidateOptions refuses. Falling back to the
// default is the right failure: a window sized for the wrong camera still produces normals, while
// a throwing option struct takes down a caller that only asked for defaults.
TEST(RealsenseNormalEstimation, AnUnopenedCameraStillYieldsValidOptions) {
    Realsense::RealSenseD435 camera; // not Open()ed -- no device required for this test
    const Realsense::NormalEstimationOptions options = camera.MakeNormalOptions();

    EXPECT_GE(options.planeFitRadius, 1);
    EXPECT_NO_THROW(Realsense::NormalEstimation::ValidateOptions(options));
    EXPECT_EQ(options.planeFitRadius, Realsense::NormalEstimationOptions{}.planeFitRadius);
}

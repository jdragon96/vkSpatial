#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Realsense/Algorithm/ValidationMask.h"
#include "Realsense/RealSensePipeline.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using Realsense::ValidationMask;
using Realsense::ValidationScoreCounters;
using Realsense::ValidationScoreOptions;

namespace {

    // A D435 at 848x480 with the projector on. The two fades are pushed wide of the fixture depths
    // so that c_range is 1 and any score below 1 comes from the term under test.
    ValidationScoreOptions TestOptions() {
        ValidationScoreOptions options;
        options.focalLengthPixels = 425.0f;
        options.baselineMeters = 0.05f;
        options.subpixelRms = 0.08f;
        options.depthScale = 0.001f;
        options.nearFadeStart = 0.1f;
        options.nearFadeEnd = 0.2f;
        options.farFadeStart = 5.0f;
        options.farFadeEnd = 6.0f;
        return options;
    }

    struct ScoreRun {
        std::vector<float> scores;
        ValidationScoreCounters counters;
    };

    ScoreRun RunScore(Engine::Core::Context &context, int width, int height,
                      const std::vector<std::uint16_t> &depth,
                      const ValidationScoreOptions &options,
                      const std::vector<std::uint8_t> &infrared = {}) {
        ValidationMask mask(context, width, height);
        {
            Engine::Compute::CommandBatch batch(context);
            mask.Execute(batch, depth.data(), options, infrared.empty() ? nullptr : infrared.data());
            batch.Submit();
        }
        return ScoreRun{mask.DownloadScores(), mask.DownloadCounters()};
    }

} // namespace

// A frame of uniform depth well inside the fade band: every interior pixel has all eight
// neighbours on its own surface, so the product is 1. The image border cannot -- the halo reads 0
// outside the image -- which is the behaviour the kernel relies on instead of a border special
// case, so it is pinned here rather than left as an accident.
TEST(RealsenseValidationScore, AFlatFrameScoresOneInsideAndFallsOffAtTheBorder) {
    Engine::Core::Context context;
    const int width = 40, height = 24;
    const std::vector<std::uint16_t> depth(std::size_t(width) * height, 1500); // 1.5 m

    const ScoreRun run = RunScore(context, width, height, depth, TestOptions());
    ASSERT_EQ(run.scores.size(), std::size_t(width) * height);

    for (int row = 1; row + 1 < height; ++row)
        for (int column = 1; column + 1 < width; ++column)
            EXPECT_NEAR(run.scores[std::size_t(row) * width + column], 1.0f, 1e-5f)
                    << "pixel (" << column << "," << row << ")";

    // A corner keeps three of its eight neighbours; an edge pixel keeps five.
    EXPECT_NEAR(run.scores[0], 3.0f / 8.0f, 1e-5f);
    EXPECT_NEAR(run.scores[std::size_t(width) / 2], 5.0f / 8.0f, 1e-5f);
}

// depth == 0 is the VPU's own verdict, and it has to zero the pixel outright rather than merely
// lower it -- the point of the leading [z > 0] factor.
//
// The score alone does NOT pin that factor, which mutation showed: with nearFadeStart above zero,
// c_range already drives a zero-depth pixel to 0, so deleting [z > 0] leaves the score identical.
// What separates them is the ATTRIBUTION -- which bucket the pixel is charged to -- so the counter
// assertion below is the one doing the work here, not the score.
TEST(RealsenseValidationScore, APixelTheSensorRejectedScoresZeroAndIsChargedToTheSensor) {
    Engine::Core::Context context;
    const int width = 16, height = 16;
    std::vector<std::uint16_t> depth(std::size_t(width) * height, 1500);
    depth[std::size_t(8) * width + 8] = 0;

    ValidationScoreOptions options = TestOptions();
    options.countRejections = true;
    const ScoreRun run = RunScore(context, width, height, depth, options);

    EXPECT_EQ(run.scores[std::size_t(8) * width + 8], 0.0f);
    EXPECT_EQ(run.counters.zeroedByNoMeasurement, 1u)
            << "the pixel was zeroed by some other term before [z > 0] saw it";
    EXPECT_EQ(run.counters.zeroedByRange, 0u);
    // ... and it costs each of its eight neighbours one unit of support.
    EXPECT_NEAR(run.scores[std::size_t(8) * width + 7], 7.0f / 8.0f, 1e-5f);
}

// tau = 3 * sigma_z, and sigma_z is quadratic in z, so the SAME step in millimetres is a
// discontinuity up close and ordinary noise far away. A fixed tolerance cannot do this, which is
// the reason the analytic sigma_z is in the kernel at all.
TEST(RealsenseValidationScore, TheSameSurfaceToleranceScalesWithTheSquareOfDepth) {
    Engine::Core::Context context;
    const int width = 16, height = 16;
    const ValidationScoreOptions options = TestOptions();

    // sigma_z = s*z^2/(f*B); at f=425, B=0.05, s=0.08 that is 3.76 mm at 1 m and 33.9 mm at 3 m.
    // tau = 3*sigma is then 11.3 mm and 101.6 mm, so a 40 mm step straddles the two.
    const std::uint16_t stepMillimetres = 40;

    const auto scoreOfSteppedFrame = [&](std::uint16_t base) {
        std::vector<std::uint16_t> depth(std::size_t(width) * height, base);
        for (int row = 0; row < height; ++row)
            for (int column = 8; column < width; ++column)
                depth[std::size_t(row) * width + column] = std::uint16_t(base + stepMillimetres);
        // Column 7 sits against the step, with three of its neighbours across it.
        return RunScore(context, width, height, depth, options).scores[std::size_t(8) * width + 7];
    };

    EXPECT_NEAR(scoreOfSteppedFrame(1000), 5.0f / 8.0f, 1e-5f)
            << "at 1 m a 40 mm step is 3.5 sigma and must break the surface";
    EXPECT_NEAR(scoreOfSteppedFrame(3000), 1.0f, 1e-5f)
            << "at 3 m the same step is 1.2 sigma and must not";
}

// The counters are compiled out by default, so a caller reading them without asking for them gets
// zeros rather than stale or partial numbers.
TEST(RealsenseValidationScore, CountersAreZeroUnlessTheyWereAskedFor) {
    Engine::Core::Context context;
    const int width = 32, height = 16;
    std::vector<std::uint16_t> depth(std::size_t(width) * height, 1500);
    depth[0] = 0;

    ValidationScoreOptions options = TestOptions();
    const ScoreRun without = RunScore(context, width, height, depth, options);
    EXPECT_EQ(without.counters.scoredPixels, 0u);
    EXPECT_EQ(without.counters.zeroedByNoMeasurement, 0u);

    options.countRejections = true;
    const ScoreRun with = RunScore(context, width, height, depth, options);

    // Every pixel lands in exactly one bucket, so the five partition the image.
    const std::uint32_t total = with.counters.scoredPixels + with.counters.zeroedByNoMeasurement +
                                with.counters.zeroedByRange + with.counters.zeroedByInfrared +
                                with.counters.zeroedByNeighbourSupport;
    EXPECT_EQ(total, std::uint32_t(width * height));
    EXPECT_EQ(with.counters.zeroedByNoMeasurement, 1u);
}

// Each of these silently rescales sigma_z or collapses a smoothstep, and the frame still comes out
// looking plausible -- a confidence map that is uniformly wrong reads exactly like a right one.
TEST(RealsenseValidationScore, AConfigurationThatWouldSilentlyMisscoreIsRefused) {
    Engine::Core::Context context;
    const int width = 8, height = 8;
    const std::vector<std::uint16_t> depth(std::size_t(width) * height, 1500);

    const auto expectRefused = [&](ValidationScoreOptions options, const char *why) {
        EXPECT_THROW(RunScore(context, width, height, depth, options), std::runtime_error) << why;
    };

    ValidationScoreOptions unset = TestOptions();
    unset.focalLengthPixels = 0.0f;
    expectRefused(unset, "f*B == 0 makes every tolerance 0 and scores the whole frame black");

    ValidationScoreOptions equalEdges = TestOptions();
    equalEdges.nearFadeEnd = equalEdges.nearFadeStart;
    expectRefused(equalEdges, "smoothstep is undefined when its edges are equal");

    ValidationScoreOptions crossed = TestOptions();
    crossed.farFadeStart = 0.05f;
    expectRefused(crossed, "the fades overlap, so no depth scores 1");

    ValidationScoreOptions missingInfrared = TestOptions();
    missingInfrared.useInfrared = true;
    expectRefused(missingInfrared, "useInfrared without an infrared frame");
}

// The infrared term compiles in as a fourth binding and a second sampled image. Bright, unclipped
// pixels must pass it unchanged; a clipped one is unmeasured, not bright.
TEST(RealsenseValidationScore, TheInfraredTermGatesOnSaturationWhenCompiledIn) {
    Engine::Core::Context context;
    const int width = 16, height = 16;
    const std::vector<std::uint16_t> depth(std::size_t(width) * height, 1000); // 1 m -> I*z^2 == I

    std::vector<std::uint8_t> infrared(std::size_t(width) * height, 200); // healthy, unclipped
    infrared[std::size_t(8) * width + 8] = 255;                           // retro-reflector

    ValidationScoreOptions options = TestOptions();
    options.useInfrared = true;
    options.infraredFloor = 20.0f;
    options.infraredReference = 120.0f;
    options.infraredSaturation = 250.0f;

    const ScoreRun run = RunScore(context, width, height, depth, options, infrared);

    EXPECT_EQ(run.scores[std::size_t(8) * width + 8], 0.0f) << "a clipped pixel must be refused";
    EXPECT_NEAR(run.scores[std::size_t(4) * width + 4], 1.0f, 1e-5f)
            << "200 counts is above the reference, so the infrared term is 1";
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Thresholding and compaction, which now belong to RealSensePipeline. It back-projects, marks
// the pixels whose score
// clears the bar, and compacts them -- count, serial row scan, then a per-row shared-memory scatter.
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    // The compaction tests below pin EXACT point counts against a hand-computed border, so they
    // ask for the chain without the normal pass. That pass has a stencil of its own and eats a
    // further ring of pixels; folding it in would make these tests measure the estimator's domain
    // instead of the compaction they exist to pin. Normal accuracy has its own suite.
    Realsense::NormalEstimationOptions WithoutNormals() {
        Realsense::NormalEstimationOptions options;
        options.enabled = false;
        return options;
    }

    struct CompactRun {
        std::vector<Eigen::Vector3f> points;
        std::vector<float> scores;
        std::uint32_t count = 0;
    };

    CompactRun RunCompaction(Engine::Core::Context &context, int width, int height,
                             const std::vector<std::uint16_t> &depth,
                             const ValidationScoreOptions &options, float threshold) {
        Realsense::RealSensePipeline pipeline(context, width, height);
        const Realsense::PinholeIntrinsics intrinsics{options.focalLengthPixels,
                                                      options.focalLengthPixels,
                                                      float(width) * 0.5f, float(height) * 0.5f};
        {
            Engine::Compute::CommandBatch batch(context);
            pipeline.Execute(batch, depth.data(), options, intrinsics, threshold, WithoutNormals());
            batch.Submit();
        }
        CompactRun run;
        run.count = pipeline.ValidPointCount();
        run.points = pipeline.DownloadValidPoints();
        run.scores = pipeline.DownloadValidScores();
        return run;
    }

} // namespace

// The compacted array must hold exactly the pixels at or above the threshold, and every score it
// carries must clear the bar -- the point and its confidence travel together or the weight cannot
// be re-paired with the sample downstream.
TEST(RealsenseValidationScore, CompactionKeepsExactlyThePixelsAboveTheThreshold) {
    Engine::Core::Context context;
    const int width = 32, height = 24;
    const std::vector<std::uint16_t> depth(std::size_t(width) * height, 1500);
    const ValidationScoreOptions options = TestOptions();

    // On a flat frame only the border loses neighbour support, so a threshold just under 1
    // separates the interior from the frame around it.
    const CompactRun run = RunCompaction(context, width, height, depth, options, 0.99f);
    EXPECT_EQ(run.count, std::uint32_t((width - 2) * (height - 2)));
    ASSERT_EQ(run.points.size(), run.count);
    ASSERT_EQ(run.scores.size(), run.count);
    for (const float score: run.scores) EXPECT_GE(score, 0.99f);

    // Lowering the bar can only ever admit more points, never fewer.
    const CompactRun permissive = RunCompaction(context, width, height, depth, options, 0.3f);
    EXPECT_GT(permissive.count, run.count);
    EXPECT_EQ(permissive.count, std::uint32_t(width * height)) << "every pixel scores at least 3/8";
}

// The compacted order is row-major, and it is load-bearing: the cloud's centroid is a float sum,
// so the ORDER reaches the solve. An atomicAdd append would be shorter and would make a replay
// diverge -- this repository measured trajectories of 1.47 / 6.45 / 7.84 / 136.76 m from four runs
// of one command before the order leaks were closed.
TEST(RealsenseValidationScore, CompactionIsRowMajorAndByteIdenticalAcrossRuns) {
    Engine::Core::Context context;
    const int width = 24, height = 16;

    // A depth ramp down the columns, so a point's x tells you which column it came from and a
    // transposed or reordered walk cannot produce the same array.
    std::vector<std::uint16_t> depth(std::size_t(width) * height);
    for (int row = 0; row < height; ++row)
        for (int column = 0; column < width; ++column)
            depth[std::size_t(row) * width + column] = std::uint16_t(1400 + column);

    const ValidationScoreOptions options = TestOptions();
    const CompactRun first = RunCompaction(context, width, height, depth, options, 0.99f);
    ASSERT_GT(first.count, 0u);

    // Row-major: x is non-decreasing within a row, and each row restarts at the left.
    const float fx = options.focalLengthPixels;
    const float cx = float(width) * 0.5f;
    std::size_t index = 0;
    for (int row = 1; row + 1 < height; ++row) {
        float previousX = -1e9f;
        for (int column = 1; column + 1 < width; ++column) {
            ASSERT_LT(index, first.points.size());
            const float z = float(1400 + column) * options.depthScale;
            const float expectedX = (float(column) - cx) / fx * z;
            EXPECT_NEAR(first.points[index].x(), expectedX, 1e-5f)
                    << "point " << index << " is not the pixel row-major order expects";
            EXPECT_GT(first.points[index].x(), previousX);
            previousX = first.points[index].x();
            ++index;
        }
    }
    EXPECT_EQ(index, first.points.size());

    for (int repeat = 0; repeat < 3; ++repeat) {
        const CompactRun again = RunCompaction(context, width, height, depth, options, 0.99f);
        ASSERT_EQ(again.count, first.count) << "run " << repeat;
        EXPECT_EQ(std::memcmp(again.points.data(), first.points.data(),
                              first.points.size() * sizeof(Eigen::Vector3f)),
                  0)
                << "run " << repeat << " compacted the same frame into a different order";
    }
}

// Re-thresholding must not need a re-score: that is the whole reason the threshold lives in its
// own kernel rather than inside the score kernel. Scoring once and compacting twice at different
// bars has to give the same answer as scoring again each time.
TEST(RealsenseValidationScore, ReThresholdingDoesNotNeedAReScore) {
    Engine::Core::Context context;
    const int width = 32, height = 24;
    const std::vector<std::uint16_t> depth(std::size_t(width) * height, 1500);
    const ValidationScoreOptions options = TestOptions();
    const float fx = options.focalLengthPixels, cx = float(width) * 0.5f, cy = float(height) * 0.5f;

    Realsense::RealSensePipeline pipeline(context, width, height);
    const Realsense::PinholeIntrinsics intrinsics{fx, fx, cx, cy};
    {
        Engine::Compute::CommandBatch batch(context);
        pipeline.RecordScore(batch, depth.data(), options); // scored ONCE
        batch.Submit();
    }

    std::vector<std::uint32_t> counts;
    for (const float threshold: {0.99f, 0.6f, 0.3f}) {
        Engine::Compute::CommandBatch batch(context);
        pipeline.RecordExtract(batch, options, intrinsics, threshold, WithoutNormals());
        batch.Submit();
        counts.push_back(pipeline.ValidPointCount());
    }

    EXPECT_EQ(counts[0], std::uint32_t((width - 2) * (height - 2)));
    EXPECT_LT(counts[0], counts[1]);
    EXPECT_LE(counts[1], counts[2]);
    EXPECT_EQ(counts[2], std::uint32_t(width * height));
}

// A row wider than the scatter's CHUNK (256), which is the one path the smaller fixtures above
// never reach: the kernel scans 256 columns at a time and carries the running count into the next
// chunk, so a bug in that carry writes the whole right-hand side of every row over the left.
//
// 700 is deliberately not a multiple of 256 -- the last chunk is partial, and lanes past the end
// must contribute nothing to the prefix rather than a stale flag. It is also not a multiple of the
// score kernel's 32-wide workgroup, so a padding lane that writes outside `if (inside)` lands on
// the NEXT ROW's pixels and blanks scores another workgroup already computed.
TEST(RealsenseValidationScore, CompactionCarriesAcrossChunksOnAWideFrame) {
    Engine::Core::Context context;
    const int width = 700, height = 6;

    // Every third column is dropped, so the survivors are not contiguous and a chunk boundary
    // lands in the middle of a run.
    std::vector<std::uint16_t> depth(std::size_t(width) * height, 1500);
    for (int row = 0; row < height; ++row)
        for (int column = 0; column < width; column += 3)
            depth[std::size_t(row) * width + column] = 0;

    ValidationScoreOptions options = TestOptions();
    options.countRejections = true;
    const float fx = options.focalLengthPixels;
    const float cx = float(width) * 0.5f, cy = float(height) * 0.5f;

    Realsense::RealSensePipeline pipeline(context, width, height);
    const Realsense::PinholeIntrinsics intrinsics{fx, fx, cx, cy};
    {
        Engine::Compute::CommandBatch batch(context);
        // 0.0f keeps every measured pixel.
        pipeline.Execute(batch, depth.data(), options, intrinsics, 0.0f, WithoutNormals());
        batch.Submit();
    }

    const std::uint32_t count = pipeline.ValidPointCount();
    const std::vector<Eigen::Vector3f> points = pipeline.DownloadValidPoints();
    ASSERT_EQ(points.size(), count);

    // Exactly the measured pixels, in row-major order, with no column repeated or skipped.
    std::vector<float> expectedX;
    for (int row = 0; row < height; ++row)
        for (int column = 0; column < width; ++column) {
            if (depth[std::size_t(row) * width + column] == 0) continue;
            expectedX.push_back((float(column) - cx) / fx * 1.5f);
        }
    ASSERT_EQ(points.size(), expectedX.size());
    for (std::size_t i = 0; i < points.size(); ++i)
        EXPECT_NEAR(points[i].x(), expectedX[i], 1e-5f)
                << "point " << i << " -- the chunk carry lost or duplicated a slot";
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// The confidence bar as a flying-pixel gate. This is the property the tuned default rests on.
///////////////////////////////////////////////////////////////////////////////////////////////////

// A depth cliff with interpolated pixels across it -- what a stereo matcher produces at a silhouette
// and what shows up in a viewer as a trail strung between the foreground and the wall behind it.
//
// The pixels ON the cliff have no neighbours on their own surface, so c_nb drives their score to
// zero, and the bar is what removes them. Measured on capture/ with cliff pixels labelled from the
// depth image: they are 1.06% of the delivered cloud at a bar of 0.0, 0.09% at 0.7 and none at 0.9.
//
// Downsampling is what makes this matter so much rather than a little: thinning to a 50 mm voxel
// removes over 99% of a surface but keeps a trail nearly intact, because each trail point owns its
// own voxel. The share of junk in the delivered cloud is amplified about twentyfold.
TEST(RealsenseValidationScore, TheConfidenceBarRemovesPixelsStrungAcrossADepthCliff) {
    Engine::Core::Context context;
    const int width = 48, height = 32;
    const ValidationScoreOptions options = TestOptions();

    // Left half at 1 m, right half at 2 m, and one column between them holding the midpoint -- a
    // pixel the sensor reports where there is no surface at all.
    std::vector<std::uint16_t> depth(std::size_t(width) * height);
    const int cliffColumn = width / 2;
    for (int row = 0; row < height; ++row)
        for (int column = 0; column < width; ++column) {
            std::uint16_t value = column < cliffColumn ? 1000 : 2000;
            if (column == cliffColumn) value = 1500;
            depth[std::size_t(row) * width + column] = value;
        }

    const auto countOnCliff = [&](float bar) {
        Realsense::RealSensePipeline pipeline(context, width, height);
        const Realsense::PinholeIntrinsics intrinsics{options.focalLengthPixels,
                                                      options.focalLengthPixels,
                                                      float(width) * 0.5f, float(height) * 0.5f};
        {
            Engine::Compute::CommandBatch batch(context);
            pipeline.Execute(batch, depth.data(), options, intrinsics, bar, WithoutNormals());
            batch.Submit();
        }
        std::size_t onCliff = 0;
        for (const Eigen::Vector3f &p: pipeline.DownloadValidPoints()) {
            if (!(p.z() > 0.0f)) continue;
            // The interpolated column is the only place a point sits at 1.5 m.
            if (std::abs(p.z() - 1.5f) < 0.01f) ++onCliff;
        }
        return onCliff;
    };

    EXPECT_GT(countOnCliff(0.0f), 0u) << "the fixture has no cliff pixels to remove";
    EXPECT_EQ(countOnCliff(0.9f), 0u)
            << "the confidence bar let pixels through that sit on a half-metre depth step";
}

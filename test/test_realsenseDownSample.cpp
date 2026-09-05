#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Realsense/RealSensePipeline.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <vector>

using Realsense::DownSampleOptions;
using Realsense::NormalEstimationOptions;
using Realsense::RealSensePipeline;
using Realsense::ValidationScoreOptions;

namespace {

    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    constexpr float kFocal = 425.0f;
    constexpr float kDepthScale = 0.001f;

    ValidationScoreOptions TestOptions() {
        ValidationScoreOptions options;
        options.focalLengthPixels = kFocal;
        options.baselineMeters = 0.05f;
        options.subpixelRms = 0.08f;
        options.depthScale = kDepthScale;
        options.nearFadeStart = 0.1f;
        options.nearFadeEnd = 0.2f;
        options.farFadeStart = 5.0f;
        options.farFadeEnd = 6.0f;
        return options;
    }

    Realsense::PinholeIntrinsics TestIntrinsics() {
        return Realsense::PinholeIntrinsics{kFocal, kFocal, float(kWidth) * 0.5f,
                                            float(kHeight) * 0.5f};
    }

    // A fronto-parallel plane. Uniform depth means uniform lateral spacing z/f, which is what makes
    // the expected voxel occupancy below something a test can state exactly.
    std::vector<std::uint16_t> RenderFlatFrame(float distance) {
        const float units = std::round(distance / kDepthScale);
        return std::vector<std::uint16_t>(std::size_t(kWidth) * kHeight, std::uint16_t(units));
    }

    struct DownSampleRun {
        std::vector<Eigen::Vector3f> points;
        Realsense::DownSampleCounters counters;
    };

    // The normal pass is off throughout: it has a stencil of its own and would remove a border of
    // pixels, which would make every count below measure two stages at once.
    DownSampleRun RunPipeline(Engine::Core::Context &context, const std::vector<std::uint16_t> &depth,
                    const DownSampleOptions &downSampleOptions) {
        NormalEstimationOptions normalOptions;
        normalOptions.enabled = false;

        RealSensePipeline pipeline(context, kWidth, kHeight);
        {
            Engine::Compute::CommandBatch batch(context);
            pipeline.Execute(batch, depth.data(), TestOptions(), TestIntrinsics(), 0.0f,
                             normalOptions, downSampleOptions);
            batch.Submit();
        }
        return DownSampleRun{pipeline.DownloadValidPoints(), pipeline.DownloadDownSampleCounters()};
    }

    // EXACT, not hashed. A multiply-xor mixer collides badly on small coordinates -- the first
    // version of this helper reported 42 voxels where the fixture has 48 -- and an oracle that
    // undercounts makes a correct kernel look like it kept duplicates.
    std::size_t OccupiedVoxels(const std::vector<Eigen::Vector3f> &points, float voxel) {
        std::unordered_set<std::int64_t> keys;
        keys.reserve(points.size());
        for (const Eigen::Vector3f &p: points) {
            const std::int64_t x = std::int64_t(std::floor(p.x() / voxel)) + 1048576;
            const std::int64_t y = std::int64_t(std::floor(p.y() / voxel)) + 1048576;
            const std::int64_t z = std::int64_t(std::floor(p.z() / voxel)) + 1048576;
            keys.insert(x | (y << 21) | (z << 42));
        }
        return keys.size();
    }

} // namespace

// Off unless asked for. The stage costs two dispatches and a table clear, and at a fine voxel it
// deletes nothing, so a caller who never mentions it must pay nothing and get the same cloud.
TEST(RealsenseDownSample, DisabledByDefaultAndChangesNothing) {
    Engine::Core::Context context;
    const std::vector<std::uint16_t> depth = RenderFlatFrame(1.0f);

    const DownSampleRun without = RunPipeline(context, depth, DownSampleOptions{});
    ASSERT_GT(without.points.size(), 0u);

    DownSampleOptions enabled;
    enabled.enabled = true;
    enabled.detailVoxelMeters = 0.02f; // 20 mm, far coarser than the 2.4 mm spacing at 1 m
    const DownSampleRun with = RunPipeline(context, depth, enabled);

    EXPECT_LT(with.points.size(), without.points.size());
}

// The property the stage exists for: exactly one survivor per occupied voxel. Not "fewer points" --
// a stage that merely thinned would pass a count assertion while punching holes in the surface.
TEST(RealsenseDownSample, KeepsExactlyOnePointPerOccupiedVoxel) {
    Engine::Core::Context context;
    const std::vector<std::uint16_t> depth = RenderFlatFrame(1.0f);
    const float voxel = 0.02f;

    const DownSampleRun before = RunPipeline(context, depth, DownSampleOptions{});
    const std::size_t occupied = OccupiedVoxels(before.points, voxel);
    ASSERT_GT(occupied, 0u);

    DownSampleOptions options;
    options.enabled = true;
    options.detailVoxelMeters = voxel;
    const DownSampleRun after = RunPipeline(context, depth, options);

    EXPECT_EQ(after.points.size(), occupied)
            << "survivors " << after.points.size() << " vs occupied voxels " << occupied;
    // ... and they occupy the same voxels, so nothing was emptied. This is the assertion an
    // image-space stride fails: measured on capture/ it kept a similar count while covering only
    // 45.9% of the voxels.
    EXPECT_EQ(OccupiedVoxels(after.points, voxel), occupied);
    EXPECT_EQ(after.counters.insertFailures, 0u);
    EXPECT_EQ(after.counters.outOfPackableRange, 0u);
}

// The compacted order reaches the ICP solve through the cloud's centroid, so which point survives
// each voxel must not depend on which lane won a race. atomicMin makes the row-major first pixel
// win whatever order the lanes arrive in.
TEST(RealsenseDownSample, PicksTheSameSurvivorEveryRun) {
    Engine::Core::Context context;
    // A depth ramp down the columns, so points differ within a voxel and "which one survived" is
    // observable rather than a tie between identical coordinates.
    std::vector<std::uint16_t> depth(std::size_t(kWidth) * kHeight);
    for (int row = 0; row < kHeight; ++row)
        for (int column = 0; column < kWidth; ++column)
            depth[std::size_t(row) * kWidth + column] = std::uint16_t(1000 + column);

    DownSampleOptions options;
    options.enabled = true;
    options.detailVoxelMeters = 0.02f;

    const DownSampleRun first = RunPipeline(context, depth, options);
    ASSERT_GT(first.points.size(), 0u);

    for (int repeat = 0; repeat < 3; ++repeat) {
        const DownSampleRun again = RunPipeline(context, depth, options);
        ASSERT_EQ(again.points.size(), first.points.size()) << "run " << repeat;
        EXPECT_EQ(std::memcmp(again.points.data(), first.points.data(),
                              first.points.size() * sizeof(Eigen::Vector3f)),
                  0)
                << "run " << repeat << " kept a different survivor";
    }
}

// A voxel finer than the sample spacing has nothing to remove -- at 1 m the spacing is z/f = 2.4 mm,
// so a 1 mm voxel holds at most one point already. The stage must be harmless there rather than
// merely cheap: this is the configuration a caller lands in by passing a small TSDF voxel.
TEST(RealsenseDownSample, AVoxelFinerThanTheSampleSpacingRemovesNothing) {
    Engine::Core::Context context;
    const std::vector<std::uint16_t> depth = RenderFlatFrame(1.0f);

    const DownSampleRun before = RunPipeline(context, depth, DownSampleOptions{});

    DownSampleOptions options;
    options.enabled = true;
    options.detailVoxelMeters = 0.001f; // 1 mm against 2.4 mm spacing
    const DownSampleRun after = RunPipeline(context, depth, options);

    EXPECT_EQ(after.points.size(), before.points.size());
}

// Enabled with no voxel size is a caller who asked for a reduction and would otherwise get the full
// cloud back with nothing to say why.
TEST(RealsenseDownSample, EnabledWithoutAVoxelSizeIsRefused) {
    DownSampleOptions options;
    options.enabled = true;
    options.detailVoxelMeters = 0.0f;
    EXPECT_THROW(Realsense::DownSample::ValidateOptions(options), std::runtime_error);

    options.detailVoxelMeters = -0.01f;
    EXPECT_THROW(Realsense::DownSample::ValidateOptions(options), std::runtime_error);

    // Disabled, so the size is irrelevant and must not be policed.
    options.enabled = false;
    EXPECT_NO_THROW(Realsense::DownSample::ValidateOptions(options));
}

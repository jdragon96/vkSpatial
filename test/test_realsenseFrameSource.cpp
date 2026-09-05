#include "Pipeline/Acquisition/RealsenseFrameSource.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

using Pipeline::RealsenseFrameSource;

namespace {

    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    constexpr float kFocal = 425.0f;

    Pipeline::CameraIntrinsics TestIntrinsics() {
        Pipeline::CameraIntrinsics intrinsics;
        intrinsics.fx = kFocal;
        intrinsics.fy = kFocal;
        intrinsics.cx = float(kWidth) * 0.5f;
        intrinsics.cy = float(kHeight) * 0.5f;
        intrinsics.width = kWidth;
        intrinsics.height = kHeight;
        return intrinsics;
    }

    // The smallest provider that satisfies IDepthProvider: hands out a fixed list of depth images
    // and then reports the stream as ended. No camera, no recording on disk.
    class ScriptedDepthProvider : public Pipeline::IDepthProvider {
    public:
        explicit ScriptedDepthProvider(std::vector<std::vector<float>> frames)
            : m_intrinsics(TestIntrinsics()), m_frames(std::move(frames)) {}

        const Pipeline::CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

        bool Grab(Pipeline::DepthFrame &out) override {
            if (m_next >= m_frames.size()) return false;
            out.depth = m_frames[m_next++];
            return true;
        }

    private:
        Pipeline::CameraIntrinsics m_intrinsics;
        std::vector<std::vector<float>> m_frames;
        std::size_t m_next = 0;
    };

    // A plane through (0,0,distance) with unit normal `normal`, in metres. The ray through (u,v) is
    // d = ((u-cx)/fx, (v-cy)/fy, 1), and it meets the plane at t = (n . p0)/(n . d), which IS the
    // depth because d.z == 1.
    std::vector<float> RenderPlane(const Eigen::Vector3f &normal, float distance) {
        const Pipeline::CameraIntrinsics k = TestIntrinsics();
        const float numerator = normal.dot(Eigen::Vector3f(0.0f, 0.0f, distance));

        std::vector<float> depth(std::size_t(kWidth) * kHeight, 0.0f);
        for (int row = 0; row < kHeight; ++row)
            for (int column = 0; column < kWidth; ++column) {
                const Eigen::Vector3f ray((float(column) - k.cx) / k.fx,
                                          (float(row) - k.cy) / k.fy, 1.0f);
                const float denominator = normal.dot(ray);
                if (std::abs(denominator) < 1e-6f) continue;
                const float z = numerator / denominator;
                if (z > 0.0f) depth[std::size_t(row) * kWidth + column] = z;
            }
        return depth;
    }

    Realsense::ValidationScoreOptions TestScoreOptions() {
        Realsense::ValidationScoreOptions options;
        options.focalLengthPixels = kFocal;
        options.baselineMeters = 0.05f;
        options.depthScale = 0.001f;
        options.nearFadeStart = 0.1f;
        options.nearFadeEnd = 0.2f;
        options.farFadeStart = 5.0f;
        options.farFadeEnd = 6.0f;
        return options;
    }

    float MedianAngleDegrees(const std::vector<Eigen::Vector3f> &normals,
                             const Eigen::Vector3f &truth) {
        std::vector<float> angles;
        for (const Eigen::Vector3f &normal: normals)
            angles.push_back(std::acos(std::clamp(normal.dot(truth), -1.0f, 1.0f)) * 180.0f /
                             float(M_PI));
        if (angles.empty()) return 180.0f;
        std::nth_element(angles.begin(), angles.begin() + angles.size() / 2, angles.end());
        return angles[angles.size() / 2];
    }

} // namespace

// The whole point of the source: a Pipeline::Frame comes out with points and normals that agree
// with the surface the provider described. Pins that the GPU front end is actually driven and that
// the camera convention survives the hand-off -- Realsense works in the sensor frame and so does
// Pipeline, so nothing should be flipped on the way through.
TEST(RealsenseFrameSource, ProducesAFrameWithPointsAndNormals) {
    const Eigen::Vector3f truth = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();

    RealsenseFrameSource source(std::make_unique<ScriptedDepthProvider>(
                                       std::vector<std::vector<float>>{RenderPlane(truth, 1.5f)}),
                               TestScoreOptions());
    source.Open();

    Pipeline::Frame frame;
    ASSERT_TRUE(source.Next(frame));
    ASSERT_GT(frame.pts.size(), 0u);
    ASSERT_EQ(frame.pts.size(), frame.nrm.size());

    for (const Eigen::Vector3f &normal: frame.nrm) EXPECT_NEAR(normal.norm(), 1.0f, 1e-4f);
    EXPECT_LT(MedianAngleDegrees(frame.nrm, truth), 2.0f);
}

// The one lossy step this path adds. The pipeline's providers hand out float metres while the GPU
// front end samples Z16, so the source re-quantises -- and the loss has to stay at the quantum
// rather than drifting, which a wrong scale or a rounding-to-truncation slip would do.
TEST(RealsenseFrameSource, RequantisingCostsAtMostOneDepthQuantum) {
    const Eigen::Vector3f truth(0.0f, 0.0f, -1.0f);
    const float distance = 1.2345f; // deliberately not a whole number of millimetres

    const Realsense::ValidationScoreOptions options = TestScoreOptions();
    RealsenseFrameSource source(std::make_unique<ScriptedDepthProvider>(
                                       std::vector<std::vector<float>>{RenderPlane(truth, distance)}),
                               options);
    source.Open();

    Pipeline::Frame frame;
    ASSERT_TRUE(source.Next(frame));
    ASSERT_GT(frame.pts.size(), 0u);
    for (const Eigen::Vector3f &point: frame.pts)
        EXPECT_NEAR(point.z(), distance, options.depthScale)
                << "the re-quantised depth drifted further than one quantum";
}

// The GPU downsample replaces AcquisitionThread's CPU voxel reduce, and it runs BEFORE the
// readback rather than after -- so the saving is in the transfer too, not only in what the pipeline
// then carries.
TEST(RealsenseFrameSource, TheDownsampleThinsTheFrameAndKeepsNormals) {
    const Eigen::Vector3f truth(0.0f, 0.0f, -1.0f);

    const auto run = [&](const Realsense::DownSampleOptions &downSample) {
        RealsenseFrameSource source(std::make_unique<ScriptedDepthProvider>(
                                           std::vector<std::vector<float>>{RenderPlane(truth, 1.0f)}),
                                   TestScoreOptions(), {}, downSample);
        source.Open();
        Pipeline::Frame frame;
        EXPECT_TRUE(source.Next(frame));
        return frame;
    };

    const Pipeline::Frame dense = run({});

    Realsense::DownSampleOptions downSample;
    downSample.enabled = true;
    downSample.detailVoxelMeters = 0.050f;
    const Pipeline::Frame thinned = run(downSample);

    ASSERT_GT(dense.pts.size(), 0u);
    EXPECT_LT(thinned.pts.size(), dense.pts.size() / 4)
            << "thinned " << thinned.pts.size() << " of " << dense.pts.size();
    ASSERT_EQ(thinned.pts.size(), thinned.nrm.size());
    for (const Eigen::Vector3f &normal: thinned.nrm) EXPECT_NEAR(normal.norm(), 1.0f, 1e-4f);
}

// An exhausted provider has to end the stream rather than hand back the last frame again, or the
// pipeline never stops.
TEST(RealsenseFrameSource, ReportsTheEndOfTheStream) {
    const Eigen::Vector3f truth(0.0f, 0.0f, -1.0f);

    RealsenseFrameSource source(std::make_unique<ScriptedDepthProvider>(
                                       std::vector<std::vector<float>>{RenderPlane(truth, 1.0f)}),
                               TestScoreOptions());
    source.Open();

    Pipeline::Frame frame;
    EXPECT_TRUE(source.Next(frame));
    EXPECT_FALSE(source.Next(frame));
}

// focalLengthPixels is what sigma_z is built from, and a frame scored with the wrong one looks
// entirely plausible -- which is why ValidationScoreOptions defaults it to 0 and the score kernel
// refuses that. The source fills it from whatever it is reading, so a caller no longer has to know
// the number, and no longer has a way to get it silently wrong.
TEST(RealsenseFrameSource, TheFocalLengthIsTakenFromTheSource) {
    const Eigen::Vector3f truth(0.0f, 0.0f, -1.0f);

    Realsense::ValidationScoreOptions options = TestScoreOptions();
    options.focalLengthPixels = 0.0f; // the caller does not know it

    RealsenseFrameSource source(std::make_unique<ScriptedDepthProvider>(
                                        std::vector<std::vector<float>>{RenderPlane(truth, 1.0f)}),
                                options);
    source.Open();

    Pipeline::Frame frame;
    ASSERT_NO_THROW(source.Next(frame));
    EXPECT_GT(frame.pts.size(), 0u);
}

// What is still refused: a source that cannot say how big its frames are. The front end is sized
// from that, and a zero-sized one would allocate nothing and then read past it.
TEST(RealsenseFrameSource, ASourceWithNoFrameSizeIsRefused) {
    Pipeline::CameraIntrinsics empty; // width and height left at 0
    class SizelessProvider : public Pipeline::IDepthProvider {
    public:
        explicit SizelessProvider(Pipeline::CameraIntrinsics k) : m_intrinsics(k) {}
        const Pipeline::CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }
        bool Grab(Pipeline::DepthFrame &) override { return true; }

    private:
        Pipeline::CameraIntrinsics m_intrinsics;
    };

    RealsenseFrameSource source(std::make_unique<SizelessProvider>(empty), TestScoreOptions());
    EXPECT_THROW(source.Open(), std::runtime_error);
}

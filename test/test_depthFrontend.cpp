#include "Pipeline/Reconstruction/DepthCameraFrameSource.h"
#include "Pipeline/Reconstruction/DepthRecording.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

using Pipeline::BackprojectDepth;
using Pipeline::CameraIntrinsics;
using Pipeline::DepthFilterOptions;
using Pipeline::DepthFrame;

namespace {
    CameraIntrinsics MakeIntrinsics(int width, int height) {
        CameraIntrinsics k;
        k.width = width;
        k.height = height;
        k.fx = k.fy = 400.0f;
        k.cx = float(width) * 0.5f;
        k.cy = float(height) * 0.5f;
        return k;
    }
} // namespace

// A depth step is two different surfaces, one behind the other. Differencing across it produces a
// normal that belongs to neither -- the "flying pixel" a stereo sensor generates at every object
// boundary. No normal may span the step.
TEST(DepthFrontend, DepthStepProducesNoNormalAcrossIt) {
    const CameraIntrinsics k = MakeIntrinsics(64, 64);
    DepthFrame frame;
    frame.depth.assign(std::size_t(k.width) * k.height, 0.0f);
    for (int v = 0; v < k.height; ++v)
        for (int u = 0; u < k.width; ++u)
            frame.depth[std::size_t(v) * k.width + u] = u < 32 ? 1.0f : 2.0f;

    const Pipeline::Frame out = BackprojectDepth(frame, k, DepthFilterOptions{});

    ASSERT_GT(out.pts.size(), 0u) << "the two flat halves must still produce normals";
    for (std::size_t i = 0; i < out.pts.size(); ++i) {
        // Both halves face the camera, so every surviving normal must be ~(0,0,-1). A normal built
        // across the step tilts away from that by a large angle.
        EXPECT_GT(-out.nrm[i].z(), 0.9f)
                << "point " << i << " at z=" << out.pts[i].z() << " has a normal spanning the step";
    }
}

// The guard must not kill ordinary surfaces: a plane facing the camera keeps every pixel.
TEST(DepthFrontend, FlatPlaneKeepsEveryInteriorPixel) {
    const CameraIntrinsics k = MakeIntrinsics(32, 32);
    DepthFrame frame;
    frame.depth.assign(std::size_t(k.width) * k.height, 1.5f);

    const Pipeline::Frame out = BackprojectDepth(frame, k, DepthFilterOptions{});

    // BackprojectDepth walks [0,W-1) x [0,H-1) -- it needs the u+1 and v+1 neighbours.
    EXPECT_EQ(out.pts.size(), std::size_t(k.width - 1) * (k.height - 1));
    for (const Eigen::Vector3f &n: out.nrm) EXPECT_NEAR(n.z(), -1.0f, 1e-3f);
}

// A slanted plane is a real surface with a per-pixel depth change. The guard is relative to depth,
// so it must survive -- a fixed threshold would cut it.
TEST(DepthFrontend, SlantedPlaneSurvivesTheGuard) {
    const CameraIntrinsics k = MakeIntrinsics(64, 64);
    DepthFrame frame;
    frame.depth.assign(std::size_t(k.width) * k.height, 0.0f);
    for (int v = 0; v < k.height; ++v)
        for (int u = 0; u < k.width; ++u) // 1.0 m rising to 1.5 m across the image
            frame.depth[std::size_t(v) * k.width + u] = 1.0f + 0.5f * float(u) / float(k.width);

    const Pipeline::Frame out = BackprojectDepth(frame, k, DepthFilterOptions{});
    EXPECT_EQ(out.pts.size(), std::size_t(k.width - 1) * (k.height - 1))
            << "a slanted surface must not be mistaken for a discontinuity";
}

// SlantedPlaneSurvivesTheGuard proves the threshold is generous enough for a gentle slope, but not
// that it SCALES with depth: dropping the `* z` and using 0.02 as a flat constant passes it too.
// This fixture separates them. At 5 m the per-column step is 0.05 m -- above the unscaled 0.02
// constant, below the scaled 0.02 * 5.0 = 0.10 -- so only a genuinely relative guard keeps it.
TEST(DepthFrontend, FarSlantedPlaneNeedsTheDepthScaledThreshold) {
    const CameraIntrinsics k = MakeIntrinsics(64, 64);
    DepthFrame frame;
    frame.depth.assign(std::size_t(k.width) * k.height, 0.0f);
    for (int v = 0; v < k.height; ++v)
        for (int u = 0; u < k.width; ++u) // 5.0 m rising to 8.2 m -> 0.05 m per column
            frame.depth[std::size_t(v) * k.width + u] = 5.0f + 3.2f * float(u) / float(k.width);

    const Pipeline::Frame out = BackprojectDepth(frame, k, DepthFilterOptions{});
    EXPECT_EQ(out.pts.size(), std::size_t(k.width - 1) * (k.height - 1))
            << "a real surface 5 m out was cut: the jump threshold is not scaling with depth";
}

namespace {
    // A provider that hands out a fixed number of synthetic frames -- stands in for the device so
    // the round-trip test needs no hardware.
    class FakeDepthProvider : public Pipeline::IDepthProvider {
    public:
        FakeDepthProvider(int frames, Pipeline::CameraIntrinsics k) : m_left(frames), m_k(k) {}
        const Pipeline::CameraIntrinsics &Intrinsics() const override { return m_k; }
        bool Grab(Pipeline::DepthFrame &out) override {
            if (m_left-- <= 0) return false;
            out.depth.assign(std::size_t(m_k.width) * m_k.height, 0.0f);
            for (std::size_t i = 0; i < out.depth.size(); ++i)
                out.depth[i] = 1.0f + 0.001f * float(i % 97) + 0.01f * float(m_left);
            return true;
        }
    private:
        int m_left;
        Pipeline::CameraIntrinsics m_k;
    };
} // namespace

// Recording stores the RAW depth, before back-projection, so replay runs the same normal
// estimation the device path does. A lossy round trip would silently change every reconstruction
// made from a recording.
TEST(DepthFrontend, RecordingRoundTripsBitExact) {
    const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "vkbvh_depth_roundtrip";
    std::filesystem::remove_all(dir);

    const Pipeline::CameraIntrinsics k = MakeIntrinsics(16, 12);
    std::vector<Pipeline::DepthFrame> written;
    {
        Pipeline::DepthRecorder recorder(std::make_unique<FakeDepthProvider>(3, k), dir.string());
        Pipeline::DepthFrame frame;
        while (recorder.Grab(frame)) written.push_back(frame);
        EXPECT_EQ(recorder.RecordedFrameCount(), 3);
    }

    Pipeline::RecordedDepthProvider replay(dir.string());
    EXPECT_EQ(replay.FrameCount(), 3);
    EXPECT_EQ(replay.Intrinsics().width, k.width);
    EXPECT_FLOAT_EQ(replay.Intrinsics().fx, k.fx);

    std::size_t read = 0;
    Pipeline::DepthFrame frame;
    while (replay.Grab(frame)) {
        ASSERT_LT(read, written.size());
        EXPECT_EQ(frame.depth, written[read].depth) << "frame " << read << " changed on round trip";
        ++read;
    }
    EXPECT_EQ(read, written.size());
    std::filesystem::remove_all(dir);
}

TEST(DepthFrontend, RecordedProviderRejectsAMissingDirectory) {
    EXPECT_THROW(Pipeline::RecordedDepthProvider("/no/such/recording"), std::runtime_error);
}

// A short read that quietly yielded a partial depth image would corrupt a reconstruction with no
// symptom -- a truncated recording must be rejected up front, not handed back silently.
TEST(DepthFrontend, RecordedProviderRejectsATruncatedFrame) {
    const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "vkbvh_depth_truncated";
    std::filesystem::remove_all(dir);

    const Pipeline::CameraIntrinsics k = MakeIntrinsics(16, 12);
    {
        Pipeline::DepthRecorder recorder(std::make_unique<FakeDepthProvider>(1, k), dir.string());
        Pipeline::DepthFrame frame;
        recorder.Grab(frame);
    }

    // Chop depth_0000.bin down from width*height*4 = 768 bytes to 4.
    std::filesystem::resize_file(dir / "depth_0000.bin", 4);
    EXPECT_THROW(Pipeline::RecordedDepthProvider(dir.string()), std::runtime_error);
    std::filesystem::remove_all(dir);
}

// The camera sits at the frame origin, so a point P is seen along direction P and a camera-facing
// normal must satisfy n·P < 0. Testing n.z() alone is only equivalent ON the optical axis; the
// error grows with the ray angle. Every fixture above uses fx=400 on a 64 px grid -- a ~9 degree
// field of view, where the two rules are indistinguishable. A D435 is 87 degrees.
TEST(DepthFrontend, NormalsFaceTheCameraAcrossAWideFieldOfView) {
    CameraIntrinsics k;                 // D435 848x480 depth intrinsics
    k.width = 848; k.height = 480;
    k.fx = k.fy = 421.6f;
    k.cx = 424.0f; k.cy = 238.0f;

    // A plane receding toward the right edge: ordinary geometry, grazing only off-axis.
    DepthFrame frame;
    frame.depth.assign(std::size_t(k.width) * k.height, 0.0f);
    for (int v = 0; v < k.height; ++v)
        for (int u = 0; u < k.width; ++u) {
            const float x = (float(u) - k.cx) / k.fx;
            // Plane -0.8x + 0.6z = -0.4 solved for z along the ray (x = x_normalized * z).
            const float z = -0.4f / (0.6f - 0.8f * x);
            frame.depth[std::size_t(v) * k.width + u] = z > 0.0f ? z : 0.0f;
        }

    const Pipeline::Frame out = BackprojectDepth(frame, k, DepthFilterOptions{});
    ASSERT_GT(out.pts.size(), 1000u);

    std::size_t facingAway = 0;
    for (std::size_t i = 0; i < out.pts.size(); ++i)
        if (out.nrm[i].dot(out.pts[i]) > 0.0f) ++facingAway;
    EXPECT_EQ(facingAway, 0u)
            << facingAway << " of " << out.pts.size()
            << " normals point away from the camera: the flip tests n.z() instead of n.P";
}

// Recording into a directory that already holds a capture used to splice two takes: overwriting
// from index 0 leaves the previous take's higher-numbered frames, and RecordedDepthProvider globs
// every depth_*.bin -- so the replay is one continuous capture with a teleport in the middle.
// Same intrinsics, same file sizes, so the size check cannot catch it.
TEST(DepthFrontend, RecorderRefusesADirectoryThatAlreadyHoldsACapture) {
    const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "vkbvh_depth_rerecord";
    std::filesystem::remove_all(dir);

    const Pipeline::CameraIntrinsics k = MakeIntrinsics(8, 6);
    {
        Pipeline::DepthRecorder recorder(std::make_unique<FakeDepthProvider>(4, k), dir.string());
        Pipeline::DepthFrame frame;
        while (recorder.Grab(frame)) {}
        EXPECT_EQ(recorder.RecordedFrameCount(), 4);
    }

    // The refusal must land on construction, not on the first Grab(). A caller that opens a
    // window before it starts streaming -- depth_live_viewer does -- would otherwise stand up its
    // whole render stack and only then discover the directory is occupied.
    EXPECT_THROW((Pipeline::DepthRecorder(std::make_unique<FakeDepthProvider>(2, k), dir.string())),
                 std::runtime_error);

    // Refusing must not disturb the take already on disk.
    std::size_t remaining = 0;
    for (const auto &entry: std::filesystem::directory_iterator(dir))
        if (entry.path().filename().string().rfind("depth_", 0) == 0) ++remaining;
    EXPECT_EQ(remaining, 4u);

    std::filesystem::remove_all(dir);
}

// A recorder that refuses creates nothing, and one that is never grabbed from creates nothing
// either -- the directory and its intrinsics.txt wait for a frame to actually arrive.
TEST(DepthFrontend, RecorderCreatesNothingUntilAFrameArrives) {
    const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "vkbvh_depth_untouched";
    std::filesystem::remove_all(dir);

    {
        Pipeline::DepthRecorder recorder(std::make_unique<FakeDepthProvider>(3, MakeIntrinsics(8, 6)),
                                         dir.string());
        EXPECT_FALSE(std::filesystem::exists(dir));
    }
    EXPECT_FALSE(std::filesystem::exists(dir));
}

// A depth buffer whose rows are padded. Walking it linearly drifts one padding-width further into
// the next row on every row, so the image shears progressively -- it does not fail outright, which
// is why this needs a fixture rather than a crash to catch it.
TEST(DepthFrontend, PaddedRowsUnpackWithoutShearing) {
    constexpr int kWidth = 5, kHeight = 4;
    constexpr std::size_t kStride = kWidth * sizeof(std::uint16_t) + 6; // 6 bytes of padding

    std::vector<unsigned char> buffer(kStride * kHeight, 0xEE); // padding is NOT zero
    for (int row = 0; row < kHeight; ++row)
        for (int column = 0; column < kWidth; ++column) {
            const std::uint16_t raw = std::uint16_t(1000 + row * 100 + column);
            std::memcpy(buffer.data() + row * kStride + column * sizeof raw, &raw, sizeof raw);
        }

    std::vector<float> out;
    Pipeline::UnpackDepthRows(buffer.data(), kStride, kWidth, kHeight, 0.001f, out);

    ASSERT_EQ(out.size(), std::size_t(kWidth) * kHeight);
    for (int row = 0; row < kHeight; ++row)
        for (int column = 0; column < kWidth; ++column)
            EXPECT_NEAR(out[std::size_t(row) * kWidth + column],
                        float(1000 + row * 100 + column) * 0.001f, 1e-6f)
                    << "row " << row << " column " << column;
}

// The unpadded case must stay exact too -- the common path.
TEST(DepthFrontend, TightlyPackedRowsUnpackExactly) {
    constexpr int kWidth = 3, kHeight = 2;
    const std::uint16_t raw[kWidth * kHeight] = {1, 2, 3, 4, 5, 6};

    std::vector<float> out;
    Pipeline::UnpackDepthRows(reinterpret_cast<const unsigned char *>(raw),
                              kWidth * sizeof(std::uint16_t), kWidth, kHeight, 0.001f, out);

    ASSERT_EQ(out.size(), std::size_t(kWidth) * kHeight);
    for (std::size_t i = 0; i < out.size(); ++i) EXPECT_FLOAT_EQ(out[i], float(i + 1) * 0.001f);
}

#include "Pipeline/Reconstruction/DepthCameraFrameSource.h"

#include <gtest/gtest.h>

#include <cmath>
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

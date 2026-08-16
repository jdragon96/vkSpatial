#pragma once

// Compiles only where librealsense2 was found (src/Pipeline/CMakeLists.txt defines
// VKBVH_HAS_REALSENSE when find_package(realsense2) succeeds), so the build stays clean on a
// machine without the SDK. Guarding the header as well as RealSenseDepthProvider.cpp means an
// accidental unconditional #include elsewhere still compiles -- it just declares nothing.
#ifdef VKBVH_HAS_REALSENSE

#include "Pipeline/Reconstruction/DepthCameraFrameSource.h" // IDepthProvider, CameraIntrinsics, DepthFrame

#include <librealsense2/rs.hpp>

namespace Pipeline {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // IDepthProvider backed by a live Intel RealSense device (D435 defaults: 848x480 depth @ 30
    // fps). The constructor starts the rs2::pipeline immediately and throws std::runtime_error if
    // no device answers -- IDepthProvider has no separate Open(), so "constructed" must mean
    // "ready to Grab()". The depth scale (metres per raw uint16 unit) is read from the device
    // itself, never assumed: it varies by model and firmware, and a wrong scale silently
    // mis-scales the whole reconstruction with no symptom. Every rs2::error is caught at the
    // class boundary and rethrown as std::runtime_error, so callers need not know rs2 exception
    // types.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class RealSenseDepthProvider : public IDepthProvider {
    public:
        explicit RealSenseDepthProvider(int width = 848, int height = 480, int fps = 30);

        const CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }
        bool Grab(DepthFrame &out) override;

    private:
        rs2::pipeline m_pipeline;
        CameraIntrinsics m_intrinsics;
        float m_depthScale = 0.0f; // metres per raw depth unit; rs2::depth_sensor::get_depth_scale()
    };

} // namespace Pipeline

#endif // VKBVH_HAS_REALSENSE

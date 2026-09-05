#pragma once

#include "Pipeline/Acquisition/DepthCameraFrameSource.h" // IDepthProvider, CameraIntrinsics, DepthFrame

#include "Realsense/RealSenseD435.h"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Pipeline {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Realsense::RealSenseD435 as an IDepthProvider, for the consumers that want float metres:
    // the CPU front end, the recorder, and the live viewers.
    //
    // No VKBVH_HAS_REALSENSE guard is needed around it. RealSenseD435 compiles to throwing stubs
    // without the SDK, so this header builds on a machine with no camera and no librealsense2 --
    // which the provider it replaces could not do, being wrapped in the macro end to end.
    //
    // The GPU path does NOT come through here. RealsenseFrameSource takes the camera's Z16 straight
    // to the device, which is the point of that path: no unpack on the way out and no requantise on
    // the way back in.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class D435DepthProvider : public IDepthProvider {
    public:
        explicit D435DepthProvider(Realsense::D435StreamOptions options = {}) {
            m_camera.Open(options);
            const Realsense::D435Calibration &calibration = m_camera.Calibration();
            m_intrinsics.fx = calibration.fx;
            m_intrinsics.fy = calibration.fy;
            m_intrinsics.cx = calibration.cx;
            m_intrinsics.cy = calibration.cy;
            m_intrinsics.width = calibration.width;
            m_intrinsics.height = calibration.height;
            m_depthScale = calibration.depthScale;
        }

        const CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

        bool Grab(DepthFrame &out) override {
            Realsense::D435Frame frame;
            if (!m_camera.Grab(frame) || !frame.depthZ16) return false;

            const std::size_t pixels = std::size_t(m_intrinsics.width) * m_intrinsics.height;
            out.depth.assign(pixels, 0.0f);
            for (std::size_t i = 0; i < pixels; ++i)
                out.depth[i] = float(frame.depthZ16[i]) * m_depthScale;
            return true;
        }

        void Close() override { m_camera.Close(); }

        // What the device answered, for a caller that has to fill sigma_z's constants.
        const Realsense::D435Calibration &Calibration() const { return m_camera.Calibration(); }

    private:
        Realsense::RealSenseD435 m_camera;
        CameraIntrinsics m_intrinsics;
        float m_depthScale = 0.0f;
    };

} // namespace Pipeline

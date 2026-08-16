// Compiles only where librealsense2 was found -- see the guard note in RealSenseDepthProvider.h.
// src/Pipeline/CMakeLists.txt globs every .cpp under this directory unconditionally, so this file
// must still compile (to nothing) on a machine without the SDK.
#ifdef VKBVH_HAS_REALSENSE

#include "Pipeline/Reconstruction/RealSenseDepthProvider.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace Pipeline {

    // D435 defaults: 848x480 depth @ 30 fps. The depth scale comes from the device -- it varies by
    // model and firmware, so hard-coding 0.001 would silently mis-scale the whole reconstruction.
    RealSenseDepthProvider::RealSenseDepthProvider(int width, int height, int fps) {
        try {
            rs2::config config;
            config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
            const rs2::pipeline_profile profile = m_pipeline.start(config); // throws rs2::error if no device
            m_depthScale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();

            const rs2_intrinsics intrinsics =
                    profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
            m_intrinsics = {intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy,
                            intrinsics.width, intrinsics.height};
        } catch (const rs2::error &e) {
            // Two failures dominate here and neither is obvious from librealsense's own message:
            //   * RS2_USB_STATUS_ACCESS / "failed to set power state" -- macOS has not granted the
            //     terminal camera access, so libusb cannot claim the interface. `sudo
            //     rs-enumerate-devices` succeeding while a plain run fails is the signature.
            //   * an unsupported mode -- the depth modes on offer depend on the link. Over USB 2.1
            //     (a hub or dock will do that) 848x480 tops out at 10 Hz; 640x480 @ 30 is the
            //     highest mode both USB 2.1 and USB 3.x support. `rs-enumerate-devices` lists them.
            throw std::runtime_error(
                    std::string("RealSenseDepthProvider: ") + e.what() +
                    "  [requested " + std::to_string(width) + "x" + std::to_string(height) + " @ " +
                    std::to_string(fps) + " Hz Z16. If this is a permission error, grant the "
                    "terminal Camera access in System Settings > Privacy & Security. If it is an "
                    "unsupported mode, run rs-enumerate-devices to see what this link offers -- "
                    "over USB 2.1, 848x480 depth is limited to 10 Hz.]");
        }
    }

    bool RealSenseDepthProvider::Grab(DepthFrame &out) {
        static constexpr unsigned kFrameTimeoutMs = 1000;

        try {
            rs2::frameset frames;
            // A timeout is NOT end of stream -- Grab's contract reserves false for that. At 30 fps
            // a 1 s gap is a USB stall or a bandwidth renegotiation, and reporting it as the end
            // truncates a capture while the caller exits successfully.
            if (!m_pipeline.try_wait_for_frames(&frames, kFrameTimeoutMs))
                throw std::runtime_error("RealSenseDepthProvider::Grab: timed out waiting for a "
                                         "depth frame");

            const rs2::depth_frame depth = frames.get_depth_frame();
            const uint16_t *raw = static_cast<const uint16_t *>(depth.get_data());
            const std::size_t count = std::size_t(m_intrinsics.width) * std::size_t(m_intrinsics.height);
            out.depth.resize(count);
            for (std::size_t i = 0; i < count; ++i) out.depth[i] = float(raw[i]) * m_depthScale;
            return true;
        } catch (const rs2::error &e) {
            throw std::runtime_error(std::string("RealSenseDepthProvider: ") + e.what());
        }
    }

} // namespace Pipeline

#endif // VKBVH_HAS_REALSENSE

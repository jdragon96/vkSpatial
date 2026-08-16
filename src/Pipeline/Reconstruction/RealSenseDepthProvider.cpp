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
            throw std::runtime_error(std::string("RealSenseDepthProvider: ") + e.what());
        }
    }

    bool RealSenseDepthProvider::Grab(DepthFrame &out) {
        try {
            rs2::frameset frames;
            if (!m_pipeline.try_wait_for_frames(&frames, 1000)) return false;

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

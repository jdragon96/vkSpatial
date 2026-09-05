#pragma once

#include <string>

namespace Pipeline {
    struct RealSenseOptions {
        bool highAccuracyPreset = false;

        float minimumDepthMeters = 0.0f;
        float maximumDepthMeters = 0.0f;

        std::string advancedModeJsonPath;
    };

} // namespace Pipeline
#ifdef VKBVH_HAS_REALSENSE

#include "Pipeline/Acquisition/DepthCameraFrameSource.h" // IDepthProvider, CameraIntrinsics, DepthFrame

#include <librealsense2/rs.hpp>

namespace Pipeline {

    class RealSenseDepthProvider : public IDepthProvider {
    public:
        explicit RealSenseDepthProvider(int width = 640, int height = 480, int fps = 30,
                                        RealSenseOptions options = {});
        ~RealSenseDepthProvider() override;

        RealSenseDepthProvider(const RealSenseDepthProvider &) = delete;
        RealSenseDepthProvider &operator=(const RealSenseDepthProvider &) = delete;

        const CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }
        bool Grab(DepthFrame &out) override;
        void Close() override;

    private:
        void StartStream(int width, int height, int fps);

        // Applies RealSenseOptions::advancedModeJsonPath to the device the config resolves to.
        void LoadAdvancedModeJson(rs2::config &config);

        rs2::pipeline m_pipeline;
        RealSenseOptions m_options;
        rs2::threshold_filter m_thresholdFilter;
        bool m_useThresholdFilter = false;
        CameraIntrinsics m_intrinsics;
        bool m_streaming = false;  // guards Close(): rs2::pipeline::stop() throws if not started
        float m_depthScale = 0.0f; // metres per raw depth unit; rs2::depth_sensor::get_depth_scale()
    };

} // namespace Pipeline

#endif // VKBVH_HAS_REALSENSE

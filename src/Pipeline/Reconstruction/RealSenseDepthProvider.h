#pragma once

#include <string>

namespace Pipeline {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Device-side confidence controls. Every field is off by default, so a caller that sets none
    // gets exactly what this provider produced before they existed.
    //
    // These are the levers a D400 actually has. The device publishes no per-pixel confidence
    // channel -- that is the L515 -- so what it offers instead is control over the threshold at
    // which its own stereo matcher gives up and writes a zero. Turning that threshold up is the
    // only way to get a confidence judgement made with information this side of the USB cable
    // never sees: the raw left/right images.
    //
    // NONE of this can be measured on a replay. capture/ holds raw Z16 recorded after the matcher
    // already decided, so a gate that lives in the device leaves no trace a recording can carry.
    // The gates in DepthFilterOptions are the ones a replay can measure; these need the camera.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    struct RealSenseOptions {
        // Load the High Accuracy visual preset, which raises the matcher's rejection thresholds:
        // fewer points, and the survivors are the ones it was confident about. High Density is the
        // opposite trade. Worth it here because a TSDF accumulates many frames -- the holes this
        // opens get filled by later views, while a wrong point stays in the map.
        bool highAccuracyPreset = false;

        // rs2::threshold_filter bounds, in metres; 0 disables an end. Duplicated by
        // DepthFilterOptions on purpose -- see the note there. Applying it at the device saves
        // back-projecting the pixels at all.
        float minimumDepthMeters = 0.0f;
        float maximumDepthMeters = 0.0f;

        // An advanced-mode preset file (rs400::advanced_mode::load_json), for tuning the matcher
        // thresholds directly rather than through a named preset. Empty = untouched.
        std::string advancedModeJsonPath;
    };

} // namespace Pipeline

// The provider itself compiles only where librealsense2 was found (src/Pipeline/CMakeLists.txt defines
// VKBVH_HAS_REALSENSE when find_package(realsense2) succeeds), so the build stays clean on a
// machine without the SDK. Guarding the header as well as RealSenseDepthProvider.cpp means an
// accidental unconditional #include elsewhere still compiles -- it declares only RealSenseOptions
// above, which names no rs2 type precisely so a caller can build one either way.
#ifdef VKBVH_HAS_REALSENSE

#include "Pipeline/Reconstruction/DepthCameraFrameSource.h" // IDepthProvider, CameraIntrinsics, DepthFrame

#include <librealsense2/rs.hpp>

namespace Pipeline {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // IDepthProvider backed by a live Intel RealSense device (D435 defaults: 640x480 depth @ 30
    // fps -- the highest depth mode both USB 2.1 and USB 3.x offer). The constructor starts the
    // rs2::pipeline immediately and throws std::runtime_error if no device answers -- IDepthProvider
    // has no separate Open(), so "constructed" must mean "ready to Grab()". A device left streaming
    // by a process that died is power-cycled and retried once before that throw.
    //
    // The depth scale (metres per raw uint16 unit) is read from the device itself, never assumed:
    // it varies by model and firmware, and a wrong scale silently mis-scales the whole
    // reconstruction with no symptom. Every rs2::error is caught at the
    // class boundary and rethrown as std::runtime_error, so callers need not know rs2 exception
    // types.
    ///////////////////////////////////////////////////////////////////////////////////////////////
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
        // Starts the depth stream and fills m_intrinsics/m_depthScale, or throws rs2::error. Split
        // out because the constructor calls it twice: once directly, and once after resetting a
        // device left stuck by a previous run.
        void StartStream(int width, int height, int fps);

        // Applies RealSenseOptions::advancedModeJsonPath to the device the config resolves to.
        void LoadAdvancedModeJson(rs2::config &config);

        rs2::pipeline m_pipeline;
        RealSenseOptions m_options;
        rs2::threshold_filter m_thresholdFilter;
        bool m_useThresholdFilter = false;
        CameraIntrinsics m_intrinsics;
        bool m_streaming = false; // guards Close(): rs2::pipeline::stop() throws if not started
        float m_depthScale = 0.0f; // metres per raw depth unit; rs2::depth_sensor::get_depth_scale()
    };

} // namespace Pipeline

#endif // VKBVH_HAS_REALSENSE

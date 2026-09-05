#pragma once

// The option structs only -- not ValidationMask.h, which would drag ComputePipeline and all of
// Vulkan into every translation unit that just wants to open a camera.
#include "Realsense/RealSenseTypes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Realsense {

    // Which of the D4 VPU's tuning presets to load.
    //
    // Intel does not document the individual parameters and says it will not:
    //
    //   "The depth calculations in the D4 VPU are influenced by over 40 different parameters...
    //    we currently use machine learning to globally optimize for different usages. For this
    //    reason we do not plan on providing any future descriptions of 'what each parameter does'."
    //
    // So a preset is the only handle there is on the VPU's own confidence threshold, and it is the
    // one knob upstream of everything this module computes -- a pixel the VPU dropped never reaches
    // the score kernel at all.
    struct VisualPresetEntry {
        const char *name;
        int value; // rs2_rs400_visual_preset
    };

    // "high-accuracy" is the default rather than the device's own: a tighter confidence gate leaves
    // fewer pixels but nearly no flying pixels, and normal estimation is limited by pixel QUALITY,
    // not pixel count. "high-density" is the opposite trade and makes normals visibly noisier.
    const std::vector<VisualPresetEntry> &VisualPresets();
    std::vector<std::string> VisualPresetNames();

    struct D435StreamOptions {
        // 848x480 at 30 is the D435's native depth resolution -- anything else is resampled in the
        // VPU, which blurs exactly the discontinuities the score kernel is looking for.
        int width = 848;
        int height = 480;
        int fps = 30;

        // Opens the left IR stream alongside depth. It is the only stand-in a stereo camera has
        // for a time-of-flight amplitude, and it arrives at the depth stream's resolution and
        // timing, so a pixel maps one to one with no alignment step.
        bool enableInfrared = true;

        std::string visualPreset = "high-accuracy";
    };

    // Everything the score kernel needs that only the device can answer. Read once at Open():
    // guessing any of these silently rescales sigma_z, and a frame scored with the wrong sigma_z
    // still looks entirely plausible.
    struct D435Calibration {
        int width = 0;
        int height = 0;
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;

        float depthScale = 0.0f;     // metres per Z16 unit; rs2::depth_sensor::get_depth_scale()
        float baselineMeters = 0.0f; // measured from the IR1 -> IR2 extrinsics, never hardcoded
    };

    // Pointers into the driver's own frame memory, valid until the next Grab(). Nothing is copied.
    struct D435Frame {
        const std::uint16_t *depthZ16 = nullptr;
        const std::uint8_t *infraredY8 = nullptr; // null unless D435StreamOptions::enableInfrared
    };

    class RealSenseD435 {
    public:
        RealSenseD435();
        ~RealSenseD435();

        RealSenseD435(const RealSenseD435 &) = delete;
        RealSenseD435 &operator=(const RealSenseD435 &) = delete;

        void Open(const D435StreamOptions &options = {});

        void Close();

        bool IsOpen() const;

        const D435Calibration &Calibration() const { return m_calibration; }

        bool Grab(D435Frame &out);

        ValidationScoreOptions MakeScoreOptions() const;

        NormalEstimationOptions MakeNormalOptions() const;

        // The back-projection half of the calibration, in the shape the pipeline takes it.
        PinholeIntrinsics MakeIntrinsics() const {
            return PinholeIntrinsics{m_calibration.fx, m_calibration.fy, m_calibration.cx,
                                     m_calibration.cy};
        }

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
        D435Calibration m_calibration;
        bool m_infraredEnabled = false;
    };

} // namespace Realsense

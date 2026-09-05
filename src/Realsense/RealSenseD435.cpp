#include "Realsense/RealSenseD435.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#ifdef VKBVH_HAS_REALSENSE
#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#endif

namespace Realsense {

    const std::vector<VisualPresetEntry> &VisualPresets() {
#ifdef VKBVH_HAS_REALSENSE
        static const std::vector<VisualPresetEntry> presets = {
                {"default", RS2_RS400_VISUAL_PRESET_DEFAULT},
                {"high-accuracy", RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY},
                {"high-density", RS2_RS400_VISUAL_PRESET_HIGH_DENSITY},
                {"medium-density", RS2_RS400_VISUAL_PRESET_MEDIUM_DENSITY},
                {"hand", RS2_RS400_VISUAL_PRESET_HAND},
        };
#else
        // The names stay available without the SDK so a tool can list and validate them, and so a
        // configuration written on a machine with a camera still parses on one without.
        static const std::vector<VisualPresetEntry> presets = {
                {"default", 0},
                {"high-accuracy", 0},
                {"high-density", 0},
                {"hand", 0},
                {"medium-density", 0},
        };
#endif
        return presets;
    }

    std::vector<std::string> VisualPresetNames() {
        std::vector<std::string> names;
        names.reserve(VisualPresets().size());
        for (const VisualPresetEntry &entry: VisualPresets()) names.emplace_back(entry.name);
        return names;
    }

    namespace {

        const VisualPresetEntry &FindVisualPreset(const std::string &name) {
            for (const VisualPresetEntry &entry: VisualPresets())
                if (name == entry.name) return entry;

            std::string known;
            for (const VisualPresetEntry &entry: VisualPresets()) {
                if (!known.empty()) known += ", ";
                known += entry.name;
            }
            throw std::runtime_error("Realsense::RealSenseD435::Open: unknown visual preset '" +
                                     name + "'; registered names are " + known);
        }

    } // namespace

#ifdef VKBVH_HAS_REALSENSE

    struct RealSenseD435::Impl {
        rs2::pipeline pipeline;
        rs2::frameset frames; // holds the driver's buffers alive between Grab() calls
        rs2::frame depth;
        rs2::frame infrared;
        bool streaming = false;
    };

    RealSenseD435::RealSenseD435() : m_impl(std::make_unique<Impl>()) {}

    RealSenseD435::~RealSenseD435() {
        try {
            Close();
        } catch (...) {
            // A destructor that throws while the stack is already unwinding terminates the process.
        }
    }

    bool RealSenseD435::IsOpen() const { return m_impl && m_impl->streaming; }

    void RealSenseD435::Open(const D435StreamOptions &options) {
        if (IsOpen()) throw std::runtime_error("Realsense::RealSenseD435::Open: already streaming");
        const VisualPresetEntry &preset = FindVisualPreset(options.visualPreset);

        rs2::config config;
        config.enable_stream(RS2_STREAM_DEPTH, options.width, options.height, RS2_FORMAT_Z16,
                             options.fps);
        if (options.enableInfrared) {
            // Stream index 1 is the LEFT imager, which is the one depth is rectified against, so
            // its pixels line up with depth without any alignment step. Index 2 would not.
            config.enable_stream(RS2_STREAM_INFRARED, 1, options.width, options.height, RS2_FORMAT_Y8,
                                 options.fps);
        }

        rs2::pipeline_profile profile;
        try {
            profile = m_impl->pipeline.start(config);
        } catch (const rs2::error &error) {
            throw std::runtime_error(std::string("Realsense::RealSenseD435::Open: ") + error.what());
        }
        m_impl->streaming = true;
        m_infraredEnabled = options.enableInfrared;

        rs2::depth_sensor depthSensor = profile.get_device().first<rs2::depth_sensor>();

        // The preset is applied AFTER start(): a D400 rejects the option while the stream is being
        // configured, and applying it silently does nothing on some firmware revisions.
        if (depthSensor.supports(RS2_OPTION_VISUAL_PRESET))
            depthSensor.set_option(RS2_OPTION_VISUAL_PRESET, float(preset.value));

        const rs2::video_stream_profile depthProfile =
                profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        const rs2_intrinsics intrinsics = depthProfile.get_intrinsics();

        m_calibration.width = intrinsics.width;
        m_calibration.height = intrinsics.height;
        m_calibration.fx = intrinsics.fx;
        m_calibration.fy = intrinsics.fy;
        m_calibration.cx = intrinsics.ppx;
        m_calibration.cy = intrinsics.ppy;
        m_calibration.depthScale = depthSensor.get_depth_scale();

        // Baseline from the extrinsics between the two imagers rather than from a datasheet: it is
        // per-unit calibration data, it differs between D435 and D455, and it is the denominator of
        // sigma_z -- an error here scales every same-surface decision in the frame.
        try {
            const rs2::stream_profile left = profile.get_stream(RS2_STREAM_INFRARED, 1);
            const rs2::stream_profile right = profile.get_stream(RS2_STREAM_INFRARED, 2);
            const rs2_extrinsics extrinsics = left.get_extrinsics_to(right);
            m_calibration.baselineMeters =
                    std::sqrt(extrinsics.translation[0] * extrinsics.translation[0] +
                              extrinsics.translation[1] * extrinsics.translation[1] +
                              extrinsics.translation[2] * extrinsics.translation[2]);
        } catch (const rs2::error &) {
            // Only the left imager was enabled, so there is no second stream to measure against.
            m_calibration.baselineMeters = 0.0f;
        }

        if (m_calibration.baselineMeters <= 0.0f) {
            // Left as zero rather than filled with 0.05: ValidationMask::Validate refuses it, which
            // is the loud failure. A plausible default here would score every frame quietly wrong.
            m_calibration.baselineMeters = 0.0f;
        }
    }

    void RealSenseD435::Close() {
        if (!m_impl || !m_impl->streaming) return;
        m_impl->pipeline.stop();
        m_impl->streaming = false;
    }

    bool RealSenseD435::Grab(D435Frame &out) {
        out = D435Frame{};
        if (!IsOpen()) throw std::runtime_error("Realsense::RealSenseD435::Grab: not streaming");

        rs2::frameset frames;
        if (!m_impl->pipeline.try_wait_for_frames(&frames)) return false;

        // Held on the object, not on the stack: the returned pointers are the driver's buffers and
        // stay valid exactly as long as these frame handles do.
        m_impl->frames = frames;
        m_impl->depth = frames.get_depth_frame();
        if (!m_impl->depth) return false;
        out.depthZ16 = static_cast<const std::uint16_t *>(m_impl->depth.get_data());

        if (m_infraredEnabled) {
            m_impl->infrared = frames.get_infrared_frame(1);
            if (m_impl->infrared)
                out.infraredY8 = static_cast<const std::uint8_t *>(m_impl->infrared.get_data());
        }
        return true;
    }

#else // VKBVH_HAS_REALSENSE

    struct RealSenseD435::Impl {};

    RealSenseD435::RealSenseD435() = default;
    RealSenseD435::~RealSenseD435() = default;
    bool RealSenseD435::IsOpen() const { return false; }
    void RealSenseD435::Close() {}

    void RealSenseD435::Open(const D435StreamOptions &options) {
        FindVisualPreset(options.visualPreset); // still reject a bad name, SDK or no SDK
        throw std::runtime_error("Realsense::RealSenseD435::Open: this build has no librealsense2 "
                                 "(VKBVH_HAS_REALSENSE is not defined). Install the SDK and "
                                 "re-run cmake, or drive the score kernel from a recording.");
    }

    bool RealSenseD435::Grab(D435Frame &) {
        throw std::runtime_error("Realsense::RealSenseD435::Grab: this build has no librealsense2");
    }

#endif // VKBVH_HAS_REALSENSE

    ValidationScoreOptions RealSenseD435::MakeScoreOptions() const {
        ValidationScoreOptions options;
        options.depthScale = m_calibration.depthScale;
        options.focalLengthPixels = m_calibration.fx;
        options.baselineMeters = m_calibration.baselineMeters;
        options.useInfrared = m_infraredEnabled;
        // subpixelRms is left at its default on purpose: it is a property of the SCENE's texture,
        // not of the camera, and no device query reports it.
        return options;
    }

    NormalEstimationOptions RealSenseD435::MakeNormalOptions() const {
        NormalEstimationOptions options;

        // The reference the default radius was measured at: a D435 at its native 848x480, whose
        // focal length is about 425 px. Radius 2 there is a 5x5 window, matching the configuration
        // docs/DEPTH_NOISE_FILTERING.md reports 12.5 degrees of ground-truth error for.
        constexpr float referenceFocalPixels = 425.0f;

        // Not open, or a device that never reported a focal length: keep the default rather than
        // scale by a zero. A radius of 0 would fail NormalEstimation::ValidateOptions, which is a
        // worse failure than a window sized for the wrong camera.
        if (!(m_calibration.fx > 0.0f)) return options;

        const float scaled = float(options.planeFitRadius) * m_calibration.fx / referenceFocalPixels;
        options.planeFitRadius = std::max(1, int(std::lround(scaled)));
        return options;
    }

} // namespace Realsense

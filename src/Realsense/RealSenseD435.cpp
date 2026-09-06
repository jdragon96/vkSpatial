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
        rs2::context context;
        rs2::device device;
        std::unique_ptr<rs2::depth_sensor> sensor;
        rs2::syncer syncer{1};
        rs2::frameset frames;
        rs2::frame depth;
        rs2::frame infrared;
        bool streaming = false;
    };

    namespace {
        rs2::stream_profile FindProfile(const std::vector<rs2::stream_profile> &profiles,
                                        rs2_stream stream, int index, rs2_format format,
                                        const D435StreamOptions &options) {
            for (const rs2::stream_profile &profile: profiles) {
                if (profile.stream_type() != stream || profile.format() != format) continue;
                if (index >= 0 && profile.stream_index() != index) continue;
                const rs2::video_stream_profile video = profile.as<rs2::video_stream_profile>();
                if (!video) continue;
                if (video.width() == options.width && video.height() == options.height &&
                    video.fps() == options.fps)
                    return profile;
            }
            return rs2::stream_profile();
        }
    } // namespace

    RealSenseD435::RealSenseD435() : m_impl(std::make_unique<Impl>()) {}

    RealSenseD435::~RealSenseD435() {
        try {
            Close();
        } catch (...) {
        }
    }

    bool RealSenseD435::IsOpen() const { return m_impl && m_impl->streaming; }

    void RealSenseD435::Open(const D435StreamOptions &options) {
        if (IsOpen()) throw std::runtime_error("Realsense::RealSenseD435::Open: already streaming");
        const VisualPresetEntry &preset = FindVisualPreset(options.visualPreset);

        const rs2::device_list devices = m_impl->context.query_devices();
        if (devices.size() == 0)
            throw std::runtime_error("Realsense::RealSenseD435::Open: no RealSense device connected");
        m_impl->device = devices.front();

        try {
            m_impl->sensor = std::make_unique<rs2::depth_sensor>(
                    m_impl->device.first<rs2::depth_sensor>());
        } catch (const rs2::error &error) {
            throw std::runtime_error(std::string("Realsense::RealSenseD435::Open: this device has "
                                                 "no stereo module: ") +
                                     error.what());
        }

        const std::vector<rs2::stream_profile> available = m_impl->sensor->get_stream_profiles();
        const rs2::stream_profile depthProfile =
                FindProfile(available, RS2_STREAM_DEPTH, -1, RS2_FORMAT_Z16, options);
        if (!depthProfile)
            throw std::runtime_error("Realsense::RealSenseD435::Open: this device offers no " +
                                     std::to_string(options.width) + "x" +
                                     std::to_string(options.height) + " Z16 depth at " +
                                     std::to_string(options.fps) + " Hz");

        std::vector<rs2::stream_profile> opened{depthProfile};
        if (options.enableInfrared) {
            const rs2::stream_profile infraredProfile =
                    FindProfile(available, RS2_STREAM_INFRARED, 1, RS2_FORMAT_Y8, options);
            if (!infraredProfile)
                throw std::runtime_error("Realsense::RealSenseD435::Open: infrared was requested "
                                         "but this device offers no matching Y8 profile");
            opened.push_back(infraredProfile);
        }

        try {
            m_impl->sensor->open(opened);
            m_impl->sensor->start(m_impl->syncer);
        } catch (const rs2::error &error) {
            m_impl->sensor.reset();
            throw std::runtime_error(std::string("Realsense::RealSenseD435::Open: ") + error.what());
        }
        m_impl->streaming = true;
        m_infraredEnabled = options.enableInfrared;

        m_visualPresetRefusal.clear();
        if (!m_impl->sensor->supports(RS2_OPTION_VISUAL_PRESET)) {
            m_visualPresetRefusal = "this device does not expose RS2_OPTION_VISUAL_PRESET";
        } else {
            try {
                if (int(m_impl->sensor->get_option(RS2_OPTION_VISUAL_PRESET)) != preset.value)
                    m_impl->sensor->set_option(RS2_OPTION_VISUAL_PRESET, float(preset.value));
            } catch (const rs2::error &error) {
                m_visualPresetRefusal = "'" + options.visualPreset + "' was refused by the device: " +
                                        error.what();
            }
        }

        const rs2_intrinsics intrinsics =
                depthProfile.as<rs2::video_stream_profile>().get_intrinsics();
        m_intrinsics.width = intrinsics.width;
        m_intrinsics.height = intrinsics.height;
        m_intrinsics.fx = intrinsics.fx;
        m_intrinsics.fy = intrinsics.fy;
        m_intrinsics.cx = intrinsics.ppx;
        m_intrinsics.cy = intrinsics.ppy;
        m_intrinsics.depthScale = m_impl->sensor->get_depth_scale();

        m_intrinsics.stereoBaselineMeters = 0.0f;
        try {
            rs2::stream_profile left, right;
            for (const rs2::stream_profile &profile: available) {
                if (profile.stream_type() != RS2_STREAM_INFRARED) continue;
                if (profile.stream_index() == 1 && !left) left = profile;
                if (profile.stream_index() == 2 && !right) right = profile;
            }
            if (left && right) {
                const rs2_extrinsics extrinsics = left.get_extrinsics_to(right);
                m_intrinsics.stereoBaselineMeters =
                        std::sqrt(extrinsics.translation[0] * extrinsics.translation[0] +
                                  extrinsics.translation[1] * extrinsics.translation[1] +
                                  extrinsics.translation[2] * extrinsics.translation[2]);
            }
        } catch (const rs2::error &) {
        }
    }

    void RealSenseD435::Close() {
        if (!m_impl || !m_impl->streaming) return;

        m_impl->sensor->stop();
        m_impl->sensor->close();
        m_impl->frames = rs2::frameset();
        m_impl->depth = rs2::frame();
        m_impl->infrared = rs2::frame();
        m_impl->streaming = false;
    }

    bool RealSenseD435::Grab(DepthFrame &out) {
        out = DepthFrame{};
        if (!IsOpen()) throw std::runtime_error("Realsense::RealSenseD435::Grab: not streaming");

        rs2::frameset frames;
        if (!m_impl->syncer.try_wait_for_frames(&frames)) return false;

        m_impl->frames = frames;
        m_impl->depth = frames.get_depth_frame();
        if (!m_impl->depth) return false;

        out.rawZ16 = static_cast<const std::uint16_t *>(m_impl->depth.get_data());

        if (m_infraredEnabled) {
            m_impl->infrared = frames.get_infrared_frame(1);
            if (m_impl->infrared)
                out.infraredY8 = static_cast<const std::uint8_t *>(m_impl->infrared.get_data());
        }
        return true;
    }

#else
    struct RealSenseD435::Impl {};

    RealSenseD435::RealSenseD435() = default;
    RealSenseD435::~RealSenseD435() = default;
    bool RealSenseD435::IsOpen() const { return false; }
    void RealSenseD435::Close() {}

    void RealSenseD435::Open(const D435StreamOptions &options) {
        FindVisualPreset(options.visualPreset);
        throw std::runtime_error("Realsense::RealSenseD435::Open: this build has no librealsense2 "
                                 "(VKBVH_HAS_REALSENSE is not defined). Install the SDK and "
                                 "re-run cmake, or drive the score kernel from a recording.");
    }

    bool RealSenseD435::Grab(DepthFrame &) {
        throw std::runtime_error("Realsense::RealSenseD435::Grab: this build has no librealsense2");
    }

#endif
    ValidationScoreOptions RealSenseD435::MakeScoreOptions() const {
        ValidationScoreOptions options;
        options.depthScale = m_intrinsics.depthScale;
        options.focalLengthPixels = m_intrinsics.fx;
        options.baselineMeters = m_intrinsics.stereoBaselineMeters;
        options.useInfrared = m_infraredEnabled;
        return options;
    }

    NormalEstimationOptions RealSenseD435::MakeNormalOptions() const {
        NormalEstimationOptions options;

        constexpr float referenceFocalPixels = 425.0f;

        if (!(m_intrinsics.fx > 0.0f)) return options;

        const float scaled = float(options.planeFitRadius) * m_intrinsics.fx / referenceFocalPixels;
        options.planeFitRadius = std::max(1, int(std::lround(scaled)));
        return options;
    }
} // namespace Realsense

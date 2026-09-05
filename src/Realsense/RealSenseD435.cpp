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

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // The stereo module is opened DIRECTLY, not through rs2::pipeline.
    //
    // Measured on macOS 15 with a D435 (FW 5.15.1.55) over USB 2.1: rs2::pipeline::start resolves
    // its config against EVERY sensor on the device, so asking for depth alone still powers the RGB
    // camera -- and the RGB camera is a standard UVC device, which macOS's own driver has already
    // claimed. libusb cannot detach a kernel driver on macOS, so the claim fails with
    // RS2_USB_STATUS_ACCESS, surfacing as "failed to set power state", and sudo does not help.
    // The stereo module has a vendor-specific class the macOS driver ignores, so it opens fine.
    //
    // rs2::syncer, not a bare frame_queue: with infrared enabled two streams arrive from one sensor
    // as separate frames, and the syncer is what pairs them into the frameset Grab reads.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    struct RealSenseD435::Impl {
        rs2::context context;
        rs2::device device;
        std::unique_ptr<rs2::depth_sensor> sensor;
        rs2::syncer syncer{1};
        rs2::frameset frames; // holds the driver's buffers alive between Grab() calls
        rs2::frame depth;
        rs2::frame infrared;
        bool streaming = false;
    };

    namespace {

        // The profile the caller asked for, from the ones the sensor actually offers. A near miss
        // is refused rather than silently substituted: a run configured for 640x480 that quietly
        // got 480x270 reports every measurement at the wrong scale.
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
            // A destructor that throws while the stack is already unwinding terminates the process.
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
                                                 "no stereo module: ") + error.what());
        }

        // Read BEFORE open(): the profile list is what the sensor offers, and the intrinsics and
        // extrinsics on it are calibration data, not stream state.
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
            // Stream index 1 is the LEFT imager, which is the one depth is rectified against, so
            // its pixels line up with depth without any alignment step. Index 2 would not.
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

        // The preset is applied AFTER the stream starts: a D400 rejects the option while the stream
        // is being configured, and applying it silently does nothing on some firmware revisions.
        //
        // It is also the one thing here that can be REFUSED without the stream being wrong.
        // Applying a preset is a write through advanced mode, which configures the colour controls
        // alongside the depth ones -- so it reaches for the RGB sensor, whose UVC interface macOS's
        // own driver holds and libusb cannot detach. Measured on a D435 (FW 5.15.1.55, macOS 15):
        // every preset write throws "failed to set power state", including a write of the value the
        // device is ALREADY set to.
        //
        // Hence: read first and skip a write that would change nothing, which is what makes the
        // default preset work on that machine at all. A write that is genuinely needed and refused
        // leaves a stream that is entirely correct, only not tuned -- so it is recorded rather than
        // thrown, and VisualPresetRefusal() is how a caller sees that the knob did not take.
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
        m_calibration.width = intrinsics.width;
        m_calibration.height = intrinsics.height;
        m_calibration.fx = intrinsics.fx;
        m_calibration.fy = intrinsics.fy;
        m_calibration.cx = intrinsics.ppx;
        m_calibration.cy = intrinsics.ppy;
        m_calibration.depthScale = m_impl->sensor->get_depth_scale();

        // Baseline from the extrinsics between the two imagers rather than from a datasheet: it is
        // per-unit calibration data, it differs between D435 and D455, and it is the denominator of
        // sigma_z -- an error here scales every same-surface decision in the frame.
        //
        // Left as zero when it cannot be read: ValidationMask::ValidateOptions refuses that, which
        // is the loud failure. A plausible default here would score every frame quietly wrong.
        m_calibration.baselineMeters = 0.0f;
        try {
            rs2::stream_profile left, right;
            for (const rs2::stream_profile &profile: available) {
                if (profile.stream_type() != RS2_STREAM_INFRARED) continue;
                if (profile.stream_index() == 1 && !left) left = profile;
                if (profile.stream_index() == 2 && !right) right = profile;
            }
            if (left && right) {
                const rs2_extrinsics extrinsics = left.get_extrinsics_to(right);
                m_calibration.baselineMeters =
                        std::sqrt(extrinsics.translation[0] * extrinsics.translation[0] +
                                  extrinsics.translation[1] * extrinsics.translation[1] +
                                  extrinsics.translation[2] * extrinsics.translation[2]);
            }
        } catch (const rs2::error &) {
            // Leave it at zero; the gate above is what reports it.
        }
    }

    void RealSenseD435::Close() {
        if (!m_impl || !m_impl->streaming) return;
        // Both, and in this order: stop() ends delivery, close() releases the USB interface. Only
        // stopping leaves the device claimed and the next Open on this machine fails.
        m_impl->sensor->stop();
        m_impl->sensor->close();
        m_impl->frames = rs2::frameset();
        m_impl->depth = rs2::frame();
        m_impl->infrared = rs2::frame();
        m_impl->streaming = false;
    }

    bool RealSenseD435::Grab(D435Frame &out) {
        out = D435Frame{};
        if (!IsOpen()) throw std::runtime_error("Realsense::RealSenseD435::Grab: not streaming");

        rs2::frameset frames;
        if (!m_impl->syncer.try_wait_for_frames(&frames)) return false;

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
        return options;
    }

    NormalEstimationOptions RealSenseD435::MakeNormalOptions() const {
        NormalEstimationOptions options;

        constexpr float referenceFocalPixels = 425.0f;

        if (!(m_calibration.fx > 0.0f)) return options;

        const float scaled = float(options.planeFitRadius) * m_calibration.fx / referenceFocalPixels;
        options.planeFitRadius = std::max(1, int(std::lround(scaled)));
        return options;
    }

} // namespace Realsense

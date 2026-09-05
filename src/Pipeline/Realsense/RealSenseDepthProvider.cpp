// Compiles only where librealsense2 was found -- see the guard note in RealSenseDepthProvider.h.
// src/Pipeline/CMakeLists.txt globs every .cpp under this directory unconditionally, so this file
// must still compile (to nothing) on a machine without the SDK.
#ifdef VKBVH_HAS_REALSENSE

#include "Pipeline/Realsense/RealSenseDepthProvider.h"

#include <librealsense2/rs_advanced_mode.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>

namespace Pipeline {

    namespace {

        // A D435 whose host process died mid-stream keeps the sensor powered and the USB interface
        // claimed. Every later open then fails with "failed to set power state" until the cable is
        // physically re-seated -- which, during development, is every exception and every Ctrl-C.
        // A hardware reset is the software equivalent of that re-seat.
        //
        // The device disappears from the bus and re-enumerates a few seconds later, so returning
        // early would just hand the caller a device that is not back yet.
        bool ResetDeviceAndWaitForReenumeration() {
            // A reset drops the device off the bus and brings it back. Both halves have to be
            // waited for: polling only for presence returns immediately, because the device is
            // still enumerated for a moment after the reset command lands.
            static constexpr int kDisappearMilliseconds = 2000;
            static constexpr int kReappearMilliseconds = 10000;
            static constexpr int kPollMilliseconds = 250;

            auto deviceIsPresent = [] { return rs2::context().query_devices().size() > 0; };
            auto waitUntil = [](int budgetMilliseconds, auto &&condition) {
                for (int waited = 0; waited < budgetMilliseconds; waited += kPollMilliseconds) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(kPollMilliseconds));
                    if (condition()) return true;
                }
                return false;
            };

            rs2::context context;
            rs2::device_list devices = context.query_devices();
            if (devices.size() == 0) return false;

            std::fprintf(stderr, "[RealSense] device did not start; power-cycling it and retrying "
                                 "(a few seconds)\n");
            devices.front().hardware_reset();

            // A device we cannot claim is a device we cannot reset either -- the command never
            // reaches it and it never drops off the bus. Giving up here rather than waiting out the
            // full reappear budget is what keeps a permission failure fast, and permission is the
            // most common reason to be in this function at all.
            if (!waitUntil(kDisappearMilliseconds, [&] { return !deviceIsPresent(); })) {
                std::fprintf(stderr, "[RealSense] the reset was not delivered -- the device is "
                                     "held by something else\n");
                return false;
            }
            return waitUntil(kReappearMilliseconds, deviceIsPresent);
        }

    } // namespace

    void RealSenseDepthProvider::StartStream(int width, int height, int fps) {
        rs2::config config;
        config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);

        // Advanced-mode controls are written BEFORE the stream starts: the SDK documents them as
        // set while not streaming, and resolve() reaches the device without opening it.
        if (!m_options.advancedModeJsonPath.empty()) LoadAdvancedModeJson(config);

        const rs2::pipeline_profile profile = m_pipeline.start(config);

        rs2::depth_sensor sensor = profile.get_device().first<rs2::depth_sensor>();
        // The preset goes on FIRST, and the depth scale is read after it. A visual preset may change
        // RS2_OPTION_DEPTH_UNITS, so a scale read beforehand describes the previous configuration --
        // and a wrong scale mis-sizes the entire reconstruction with no symptom, which is the reason
        // this class reads the scale from the device at all.
        if (m_options.highAccuracyPreset && sensor.supports(RS2_OPTION_VISUAL_PRESET))
            sensor.set_option(RS2_OPTION_VISUAL_PRESET,
                              float(RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY));
        m_depthScale = sensor.get_depth_scale();

        m_useThresholdFilter =
                m_options.minimumDepthMeters > 0.0f || m_options.maximumDepthMeters > 0.0f;
        if (m_useThresholdFilter) {
            // Each end is set only when asked for, so leaving one at 0 keeps the filter's own
            // default for that end rather than clamping the stream to zero.
            if (m_options.minimumDepthMeters > 0.0f)
                m_thresholdFilter.set_option(RS2_OPTION_MIN_DISTANCE, m_options.minimumDepthMeters);
            if (m_options.maximumDepthMeters > 0.0f)
                m_thresholdFilter.set_option(RS2_OPTION_MAX_DISTANCE, m_options.maximumDepthMeters);
        }

        const rs2_intrinsics intrinsics =
                profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
        m_intrinsics = {intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy,
                        intrinsics.width, intrinsics.height};
        m_streaming = true;
    }

    // D435 defaults: 640x480 depth @ 30 fps -- the highest depth mode BOTH USB 2.1 and USB 3.x
    // offer, and a hub or dock will give you USB 2.1. The depth scale comes from the device: it
    // varies by model and firmware, so hard-coding 0.001 would silently mis-scale everything.
    // Reads the preset file and hands it to the device. Failing loudly matters more here than
    // elsewhere: a preset that silently did not load leaves the camera on its previous settings,
    // and the whole point of passing one is that the defaults were not good enough.
    void RealSenseDepthProvider::LoadAdvancedModeJson(rs2::config &config) {
        const std::string path = m_options.advancedModeJsonPath;
        std::ifstream file(path);
        if (!file)
            throw std::runtime_error(
                    "RealSenseDepthProvider::LoadAdvancedModeJson: could not open '" + path + "'");
        const std::string json((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());

        const rs2::pipeline_profile resolved = config.resolve(m_pipeline);
        rs2::device device = resolved.get_device();
        if (!device.is<rs400::advanced_mode>())
            throw std::runtime_error("RealSenseDepthProvider::LoadAdvancedModeJson: this device "
                                     "has no advanced mode, so '" +
                                     path + "' cannot be applied");

        rs400::advanced_mode advanced = device.as<rs400::advanced_mode>();
        if (!advanced.is_enabled()) advanced.toggle_advanced_mode(true);
        advanced.load_json(json);
    }

    RealSenseDepthProvider::RealSenseDepthProvider(int width, int height, int fps,
                                                   RealSenseOptions options)
        : m_options(std::move(options)) {
        try {
            StartStream(width, height, fps);
            return;
        } catch (const rs2::error &) {
            // Fall through to one reset-and-retry. Nothing has been handed to a caller yet, so a
            // second attempt costs only the seconds the reset takes.
        }

        try {
            if (ResetDeviceAndWaitForReenumeration()) {
                StartStream(width, height, fps);
                return;
            }
        } catch (const rs2::error &) {
            // The retry failed too. Report the original diagnosis below rather than the retry's
            // error, which is usually the same message one power-cycle later.
        }

        // Three failures dominate here and none is obvious from librealsense's own message:
        //   * a stuck device -- handled by the retry above; if it is still stuck, the reset itself
        //     could not be delivered, which usually means the interface is held by someone else.
        //   * RS2_USB_STATUS_ACCESS / "failed to set power state" -- macOS has not granted the
        //     terminal camera access, so libusb cannot claim the interface. `sudo
        //     rs-enumerate-devices` succeeding while a plain run fails is the signature.
        //   * an unsupported mode -- the depth modes on offer depend on the link. Over USB 2.1
        //     848x480 tops out at 10 Hz; 640x480 @ 30 is the highest mode both links support.
        throw std::runtime_error(
                "RealSenseDepthProvider: could not start the depth stream, and a device reset did "
                "not recover it.  [requested " +
                std::to_string(width) + "x" + std::to_string(height) + " @ " + std::to_string(fps) +
                " Hz Z16.\n"
                "  * Permission is the usual cause on macOS, and re-running under sudo is what "
                "tells you: if sudo works, grant the terminal Camera access in System Settings > "
                "Privacy & Security and restart the terminal.\n"
                "  * Otherwise check the mode with rs-enumerate-devices -- the depth modes on "
                "offer depend on the link, and over USB 2.1 (any hub or dock) 848x480 is limited "
                "to 10 Hz.\n"
                "  * If neither, unplug the camera and plug it back in.]");
    }

    // Stopping explicitly is what keeps the NEXT run from meeting a stuck device: a D435 left
    // streaming keeps its interface claimed, and every later open fails with "failed to set power
    // state" until the cable is re-seated.
    //
    // Called on the pipeline's own shutdown path, so the camera is released when acquisition ends
    // rather than at process teardown -- and idempotent, because that path and the destructor both
    // run. rs2::pipeline::stop() throws on a pipeline that was never started.
    void RealSenseDepthProvider::Close() {
        if (!m_streaming) return;
        m_streaming = false;
        try {
            m_pipeline.stop();
        } catch (const rs2::error &) {
            // The device is already gone. Nothing left to release.
        }
    }

    RealSenseDepthProvider::~RealSenseDepthProvider() { Close(); }

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

            // A frameset can arrive without a depth frame. get_data() on one returns null, and
            // dereferencing it segfaults -- inside a worker thread, where the process simply dies.
            rs2::depth_frame depth = frames.get_depth_frame();
            if (!depth)
                throw std::runtime_error("RealSenseDepthProvider::Grab: the frameset carried no "
                                         "depth frame");

            // Out-of-range pixels are zeroed at the device, so they never reach back-projection.
            // The filter does not resample, so the size check below still describes this frame.
            if (m_useThresholdFilter) depth = m_thresholdFilter.process(depth);

            // Size the read from the FRAME, never from the stored intrinsics. A device may
            // renegotiate its mode -- USB 2.1 bandwidth pressure is exactly when it does -- and
            // then width*height from the start-up profile over-reads the smaller buffer. The
            // frames that survived that would also be back-projected with the wrong focal length
            // and principal point, so the reconstruction comes out at the wrong scale.
            if (depth.get_width() != m_intrinsics.width || depth.get_height() != m_intrinsics.height)
                throw std::runtime_error(
                        "RealSenseDepthProvider::Grab: the device delivered " +
                        std::to_string(depth.get_width()) + "x" + std::to_string(depth.get_height()) +
                        " but the stream profile describes " + std::to_string(m_intrinsics.width) +
                        "x" + std::to_string(m_intrinsics.height) +
                        ". Back-projection would use the wrong focal length and principal point.");

            const auto *base = static_cast<const unsigned char *>(depth.get_data());
            if (!base)
                throw std::runtime_error("RealSenseDepthProvider::Grab: the depth frame has no "
                                         "pixel data");

            UnpackDepthRows(base, std::size_t(depth.get_stride_in_bytes()), m_intrinsics.width,
                            m_intrinsics.height, m_depthScale, out.depth);
            return true;
        } catch (const rs2::error &e) {
            throw std::runtime_error(std::string("RealSenseDepthProvider: ") + e.what());
        }
    }

} // namespace Pipeline

#endif // VKBVH_HAS_REALSENSE

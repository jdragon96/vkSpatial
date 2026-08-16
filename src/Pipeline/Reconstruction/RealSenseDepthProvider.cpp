// Compiles only where librealsense2 was found -- see the guard note in RealSenseDepthProvider.h.
// src/Pipeline/CMakeLists.txt globs every .cpp under this directory unconditionally, so this file
// must still compile (to nothing) on a machine without the SDK.
#ifdef VKBVH_HAS_REALSENSE

#include "Pipeline/Reconstruction/RealSenseDepthProvider.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
        const rs2::pipeline_profile profile = m_pipeline.start(config);
        m_depthScale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();

        const rs2_intrinsics intrinsics =
                profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
        m_intrinsics = {intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy,
                        intrinsics.width, intrinsics.height};
        m_streaming = true;
    }

    // D435 defaults: 640x480 depth @ 30 fps -- the highest depth mode BOTH USB 2.1 and USB 3.x
    // offer, and a hub or dock will give you USB 2.1. The depth scale comes from the device: it
    // varies by model and firmware, so hard-coding 0.001 would silently mis-scale everything.
    RealSenseDepthProvider::RealSenseDepthProvider(int width, int height, int fps) {
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
                " Hz Z16. If this is a permission error, grant the terminal Camera access in "
                "System Settings > Privacy & Security, then restart the terminal. If it is an "
                "unsupported mode, run rs-enumerate-devices to see what this link offers -- over "
                "USB 2.1, 848x480 depth is limited to 10 Hz. If neither, unplug the camera and "
                "plug it back in.]");
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

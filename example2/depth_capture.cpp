// RealSense capture / replay tool for the depth PROVIDER layer (src/Realsense).
// --record drives a live D435 through RealSenseD435Recorder to a directory; --replay reads a
// recording back through the same class and prints what each frame actually holds.
// --replay needs neither a camera nor the librealsense2 SDK -- it is the only way this layer is
// exercisable in a session where the device is not attached.
//
// It deliberately stops at the depth image. Turning one into points is the GPU front end's job and
// belongs to the tools that own it: validation_score_lab shows it a frame at a time, and
// realsense_scan runs it through the whole pipeline.
//
// Usage:
//   depth_capture --record <dir> [--frames N] [--width 640] [--height 480] [--fps 30]
//   depth_capture --replay <dir>

#include "Realsense/RealSenseD435.h"          // RealSenseD435
#include "Realsense/RealSenseD435Recorder.h" // RealSenseD435Recorder
#include "utilities/ArgParser.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

    // Replays a recording with no camera or SDK required. Reports the measured pixels per frame
    // and the depth range over them: a recording whose frames are mostly zero, or whose range is
    // nothing like the scene that was scanned, is the failure this tool exists to catch, and it
    // catches it without any front end in the way.
    int RunReplay(const std::string &directory) {
        Realsense::RealSenseD435Recorder provider(directory);
        const Realsense::CameraIntrinsics &intrinsics = provider.Intrinsics();
        std::printf("replaying %s (%d recorded frames, %dx%d, depth scale %g)\n", directory.c_str(),
                    provider.FrameCount(), intrinsics.width, intrinsics.height,
                    double(intrinsics.depthScale));

        const std::size_t pixels = std::size_t(intrinsics.width) * std::size_t(intrinsics.height);

        Realsense::DepthFrame depthFrame;
        int frameIndex = 0;
        std::size_t totalMeasured = 0;
        while (provider.Grab(depthFrame)) {
            std::size_t measured = 0;
            float nearest = 0.0f, farthest = 0.0f;
            for (std::size_t i = 0; i < pixels; ++i) {
                // Z16 is what the recording holds; metres are for the human reading this line.
                const float z = float(depthFrame.rawZ16[i]) * intrinsics.depthScale;
                if (!(z > 0.0f)) continue;
                if (measured == 0) nearest = farthest = z;
                nearest = std::min(nearest, z);
                farthest = std::max(farthest, z);
                ++measured;
            }
            std::printf("  frame %4d: %6zu measured of %zu, %.3f-%.3f m\n", frameIndex, measured,
                        pixels, nearest, farthest);
            totalMeasured += measured;
            ++frameIndex;
        }
        std::printf("replayed %d frames, %zu total measured pixels\n", frameIndex, totalMeasured);
        return 0;
    }

    // Drives a live device through DepthRecorder to `directory`. Compiled out (with a clear
    // explanation printed instead) when librealsense2 was not found at configure time -- RunReplay
    // above is the path that must work everywhere.
    int RunRecord(const std::string &directory, int frameCount, int width, int height, int fps) {
#ifdef VKBVH_HAS_REALSENSE
        auto device = std::make_unique<Realsense::RealSenseD435>();
        device->Open(Realsense::D435StreamOptions{width, height, fps, false, "high-accuracy"});
        // The preset is a quality request the device may refuse while still streaming correctly, so
        // it is said out loud rather than assumed -- a recording made without it is not the one the
        // caller asked for.
        if (!device->VisualPresetRefusal().empty())
            std::printf("  visual preset NOT applied: %s\n", device->VisualPresetRefusal().c_str());
        Realsense::RealSenseD435Recorder recorder(std::move(device), directory);
        Realsense::DepthFrame frame;
        for (int i = 0; i < frameCount && recorder.Grab(frame); ++i)
            std::printf("  captured frame %d/%d\n", i + 1, frameCount);
        std::printf("recorded %d frames to %s\n", recorder.FrameCount(), directory.c_str());
        return 0;
#else
        (void) directory;
        (void) frameCount;
        (void) width;
        (void) height;
        (void) fps;
        std::printf("depth_capture: built without librealsense2 (VKBVH_HAS_REALSENSE is not defined) -- "
                    "--record is unavailable. Install librealsense2 and re-run cmake to enable it.\n");
        return 1;
#endif
    }

} // namespace

int main(int argc, char **argv) {
    try {
        util::ArgParser arg = util::BuildArgParser(argc, argv)
                                       .Option("--record")
                                       .Option("--replay")
                                       .Option("--frames", 30)
                                       .Option("--width", 640)
                                       .Option("--height", 480)
                                       .Option("--fps", 30);

        if (arg.Has("--replay")) return RunReplay(arg.Value("--replay"));
        if (arg.Has("--record"))
            return RunRecord(arg.Value("--record"), arg.ValueInt("--frames"), arg.ValueInt("--width"),
                              arg.ValueInt("--height"), arg.ValueInt("--fps"));

        std::fprintf(stderr,
                     "usage: depth_capture --record <dir> [--frames N] [--width 640] [--height 480] [--fps 30]\n"
                     "       depth_capture --replay <dir>            replay a recording; prints "
                     "per-frame point/normal counts\n");
        return 2;
    } catch (const std::exception &e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
}

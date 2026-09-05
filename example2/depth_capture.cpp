// RealSense capture / replay tool for the depth-camera front end (DepthCameraFrameSource +
// RealSenseDepthProvider, see src/Pipeline/Reconstruction). --record drives a live D435 through
// DepthRecorder to a directory; --replay reads a recording back through RecordedDepthProvider and
// the same BackprojectDepth front end the live path uses, printing per-frame point/normal counts.
// --replay needs neither a camera nor the librealsense2 SDK -- it is the only way this front end
// is exercisable in a session where the device is not attached.
//
// Usage:
//   depth_capture --record <dir> [--frames N] [--width 640] [--height 480] [--fps 30]
//   depth_capture --replay <dir>

#include "Pipeline/Acquisition/DepthCameraFrameSource.h" // BackprojectDepth, DepthFrame, IDepthProvider
#include "Pipeline/Acquisition/DepthRecording.h"          // DepthRecorder, RecordedDepthProvider
#include "utilities/ArgParser.h"

#ifdef VKBVH_HAS_REALSENSE
#include "Pipeline/Realsense/RealSenseDepthProvider.h"
#endif

#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

    // Replays a recording with no camera or SDK required: reads back the raw depth via
    // RecordedDepthProvider and runs it through the same BackprojectDepth front end the live
    // device path uses, printing the per-frame point/normal counts it produced.
    int RunReplay(const std::string &directory) {
        Pipeline::RecordedDepthProvider provider(directory);
        std::printf("replaying %s (%d recorded frames, %dx%d)\n", directory.c_str(),
                    provider.FrameCount(), provider.Intrinsics().width, provider.Intrinsics().height);

        Pipeline::DepthFrame depthFrame;
        int frameIndex = 0;
        std::size_t totalPointCount = 0, totalNormalCount = 0;
        while (provider.Grab(depthFrame)) {
            const Pipeline::Frame frame = Pipeline::BackprojectDepth(depthFrame, provider.Intrinsics());
            std::printf("  frame %4d: %6zu points, %6zu normals\n", frameIndex, frame.pts.size(),
                        frame.nrm.size());
            totalPointCount += frame.pts.size();
            totalNormalCount += frame.nrm.size();
            ++frameIndex;
        }
        std::printf("replayed %d frames, %zu total points, %zu total normals\n", frameIndex,
                    totalPointCount, totalNormalCount);
        return 0;
    }

    // Drives a live device through DepthRecorder to `directory`. Compiled out (with a clear
    // explanation printed instead) when librealsense2 was not found at configure time -- RunReplay
    // above is the path that must work everywhere.
    int RunRecord(const std::string &directory, int frameCount, int width, int height, int fps) {
#ifdef VKBVH_HAS_REALSENSE
        Pipeline::DepthRecorder recorder(
                std::make_unique<Pipeline::RealSenseDepthProvider>(width, height, fps), directory);
        Pipeline::DepthFrame frame;
        for (int i = 0; i < frameCount && recorder.Grab(frame); ++i)
            std::printf("  captured frame %d/%d\n", i + 1, frameCount);
        std::printf("recorded %d frames to %s\n", recorder.RecordedFrameCount(), directory.c_str());
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

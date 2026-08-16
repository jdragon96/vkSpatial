#pragma once

#include "Pipeline/Reconstruction/DepthCameraFrameSource.h" // IDepthProvider, CameraIntrinsics, DepthFrame

#include <memory>
#include <string>
#include <vector>

namespace Pipeline {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Recording and replay for IDepthProvider, so a capture can be replayed without the camera
    // attached -- the only way this front end is debuggable in a session where the device is not
    // present.
    //
    // DepthRecorder stores the RAW depth, before back-projection: replay then runs through the
    // same BackprojectDepth / normal-estimation front end the live device path does, so a later
    // fix to that front end is exercised against real captures, not only synthetic ones. Storing
    // the back-projected Frame instead would freeze the front end at record time.
    //
    // Format -- no external dependency, debuggable by hand:
    //   <directory>/intrinsics.txt   "fx fy cx cy width height"   (one space-separated line)
    //   <directory>/depth_%04d.bin   float32 * width*height, row-major, metres
    // Not PNG: 16-bit would need a scale factor, and a scale mistake would silently corrupt every
    // reconstruction made from the recording. Float32 is lossless and inspectable.
    ///////////////////////////////////////////////////////////////////////////////////////////////

    // Decorator over a live IDepthProvider: forwards Grab to the wrapped device, writes what it
    // got to `directory`, and returns it unchanged -- recording is transparent to the pipeline, so
    // wrapping a device in a DepthRecorder is the only change needed to start capturing.
    class DepthRecorder : public IDepthProvider {
    public:
        DepthRecorder(std::unique_ptr<IDepthProvider> device, std::string directory);

        const CameraIntrinsics &Intrinsics() const override;
        bool Grab(DepthFrame &out) override;

        int RecordedFrameCount() const;

    private:
        std::unique_ptr<IDepthProvider> m_device;
        std::string m_directory;
        int m_recordedFrameCount = 0;
    };

    // Replays a directory written by DepthRecorder, standing in for the live device when no camera
    // is attached. The constructor throws std::runtime_error if `directory` has no readable
    // intrinsics.txt, or if a recorded depth_*.bin's size does not match width * height * 4 bytes
    // -- a truncated recording must fail loudly, not hand back a silently-partial depth image.
    class RecordedDepthProvider : public IDepthProvider {
    public:
        explicit RecordedDepthProvider(std::string directory);

        const CameraIntrinsics &Intrinsics() const override;
        bool Grab(DepthFrame &out) override; // false after the last recorded frame

        int FrameCount() const;

    private:
        CameraIntrinsics m_intrinsics;
        std::vector<std::string> m_framePaths; // sorted, one per recorded frame
        int m_nextFrame = 0;
    };

} // namespace Pipeline

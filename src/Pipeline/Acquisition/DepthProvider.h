#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace Pipeline {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // The depth device, and the ONLY interface acquisition switches on.
    //
    // There used to be a second layer above this -- IFrameSource -- whose implementations each
    // turned a device into a Frame. Two interfaces for one axis meant every tool picked a device
    // AND a source and could pair them wrongly, so the frame-making moved into AcquisitionThread
    // and this is what is left: a thing that hands over depth images.
    ///////////////////////////////////////////////////////////////////////////////////////////////

    struct CameraIntrinsics {
        float fx = 0, fy = 0, cx = 0, cy = 0;
        int width = 0, height = 0;
    };

    // What a provider produces. A provider fills whichever form it holds NATIVELY and leaves the
    // other empty -- it does not convert on the chance that somebody wants the other one.
    //
    // That is the whole point of carrying both. A D400 hands over Z16 and the GPU front end wants
    // Z16, so unpacking to metres on the way out and re-quantising on the way in would be two
    // full-image host passes per frame to arrive back where the driver started. A recording holds
    // float metres and has no Z16 to hand over. Consumers that need metres call EnsureMetres.
    struct DepthFrame {
        std::vector<float> depth;             // row-major width*height, metres; <=0 = invalid
        const std::uint16_t *rawZ16 = nullptr; // the device's own buffer, valid until the next Grab
        float depthScale = 0.0f;               // metres per rawZ16 unit; 0 when rawZ16 is null
    };

    // The actual device (RealSense / recorded stream). Add a subclass per device.
    class IDepthProvider {
    public:
        virtual ~IDepthProvider() = default;
        virtual const CameraIntrinsics &Intrinsics() const = 0;
        virtual bool Grab(DepthFrame &out) = 0; // false when the stream ends

        // Release the device. Called when acquisition ends -- not only when the object dies -- so
        // a sensor is freed the moment the pipeline stops rather than at process teardown. Must be
        // idempotent: the pipeline closes on both the exhausted-stream path and the interrupt
        // path, and a destructor closes again after that.
        virtual void Close() {}
    };

    // Unpack a row-padded 16-bit depth image into metres. Kept here, out of the sensor-specific
    // code, because the arithmetic is the part that can be wrong without hardware to prove it.
    //
    // rowStrideBytes is NOT width*2 in general: a device may pad each row for alignment, and
    // walking the buffer linearly then drifts one padding-width further into the next row on every
    // row -- the image shears progressively rather than failing outright.
    inline void UnpackDepthRows(const unsigned char *base, std::size_t rowStrideBytes, int width,
                                int height, float metresPerUnit, std::vector<float> &out) {
        out.assign(std::size_t(width) * std::size_t(height), 0.0f);
        for (int row = 0; row < height; ++row) {
            const unsigned char *rowBase = base + std::size_t(row) * rowStrideBytes;
            for (int column = 0; column < width; ++column) {
                std::uint16_t raw = 0;
                std::memcpy(&raw, rowBase + std::size_t(column) * sizeof(std::uint16_t), sizeof raw);
                out[std::size_t(row) * width + column] = float(raw) * metresPerUnit;
            }
        }
    }

    // Metres for a consumer that needs them, from whichever form the provider filled. A no-op when
    // `depth` is already there, so a recording costs nothing.
    //
    // Throws rather than returning an empty image: a silently empty depth frame reads downstream as
    // a scene with no measurements in it, which is indistinguishable from a sensor pointed at the
    // sky.
    inline void EnsureMetres(DepthFrame &frame, const CameraIntrinsics &intrinsics) {
        if (!frame.depth.empty()) return;
        if (!frame.rawZ16 || !(frame.depthScale > 0.0f))
            throw std::runtime_error("Pipeline::EnsureMetres: the frame holds neither metres nor a "
                                     "scaled Z16 image");
        UnpackDepthRows(reinterpret_cast<const unsigned char *>(frame.rawZ16),
                        std::size_t(intrinsics.width) * sizeof(std::uint16_t), intrinsics.width,
                        intrinsics.height, frame.depthScale, frame.depth);
    }

} // namespace Pipeline

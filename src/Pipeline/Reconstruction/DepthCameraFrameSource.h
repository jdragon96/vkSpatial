#pragma once

#include "Pipeline/Reconstruction/ReconstructionSource.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace Pipeline {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Depth-camera strategy — back-project a depth image into an oriented point cloud (in the camera
    // frame, camera at the origin; the ICP stage resolves the world pose). The device itself is a
    // pluggable IDepthProvider, so a new sensor (RealSense / Kinect / recorded stream) is a new
    // provider, not a new source. Its reconstruction core is the back-projection below.
    ///////////////////////////////////////////////////////////////////////////////////////////////

    struct CameraIntrinsics {
        float fx = 0, fy = 0, cx = 0, cy = 0;
        int width = 0, height = 0;
    };

    struct DepthFrame {
        std::vector<float> depth; // row-major width*height, metres; <=0 = invalid
    };

    // The actual device (RealSense / Kinect / recorded stream). Add a subclass per device.
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

    struct DepthFilterOptions {
        // Reject a neighbour whose depth differs by more than max(minimumDepthJump,
        // relativeDepthJump * z). Relative because stereo depth error grows as z^2/(f*baseline): a
        // fixed threshold over-rejects near the camera and under-rejects far from it.
        float relativeDepthJump = 0.02f;  // 2 % of range
        float minimumDepthJump = 0.005f;  // 5 mm floor, for the near field

        // Side of a square, discontinuity-aware mean applied to the depth image BEFORE
        // back-projection. 0 or 1 = off (the historical behaviour, and the default: exposing a knob
        // must not change what existing callers get).
        //
        // Why it exists: the normal below is a ONE-PIXEL forward difference, so its conditioning is
        // set entirely by per-pixel depth noise. Measured on capture/ (D435 640x480, 0.25-6.8 m),
        // adjacent normals disagree by a median of 24 degrees where a smooth surface should read
        // 1-3; a 5x5 window takes that to 2.9 (3x3 to 4.7). Point-to-plane integration depends on
        // the normal twice -- the SDF value is dot(voxelCentre - point, n) AND the truncation band is
        // marched along n -- so a noisy normal both mis-values and mis-places the band. The
        // projective form uses no normal in its value, which is why it tolerates raw depth.
        //
        // It reuses relativeDepthJump/minimumDepthJump as the inclusion gate rather than adding a
        // second threshold: "average only neighbours on the same surface" and "do not difference
        // across a step" are the same rule, and a plain box mean would undo the flying-pixel guard.
        int prefilterWindow = 0;
    };

    // Discontinuity-aware square mean over `depth`: each pixel averages only the neighbours that are
    // valid AND within the same depth-jump tolerance the normal estimator uses, so a step edge is
    // never averaged across. Invalid pixels stay invalid; border pixels average over whatever exists
    // (never dropped -- shrinking the usable region would silently crop the frame).
    inline std::vector<float> PrefilterDepth(const std::vector<float> &depth, int width, int height,
                                             int window, const DepthFilterOptions &filter) {
        if (window <= 1) return depth;
        const int radius = window / 2;
        std::vector<float> out(depth.size(), 0.0f);
        for (int v = 0; v < height; ++v)
            for (int u = 0; u < width; ++u) {
                const std::size_t centre = std::size_t(v) * width + u;
                const float z = depth[centre];
                if (z <= 0.0f) continue; // invalid stays invalid
                const float tolerance =
                        std::max(filter.minimumDepthJump, filter.relativeDepthJump * z);
                float sum = 0.0f;
                int count = 0;
                for (int dv = -radius; dv <= radius; ++dv) {
                    const int vv = v + dv;
                    if (vv < 0 || vv >= height) continue;
                    for (int du = -radius; du <= radius; ++du) {
                        const int uu = u + du;
                        if (uu < 0 || uu >= width) continue;
                        const float neighbour = depth[std::size_t(vv) * width + uu];
                        if (neighbour <= 0.0f || std::abs(neighbour - z) > tolerance) continue;
                        sum += neighbour;
                        ++count;
                    }
                }
                out[centre] = count > 0 ? sum / float(count) : z;
            }
        return out;
    }

    // Back-project a depth image to camera-frame points + normals (normals from the organized-grid
    // neighbours, oriented toward the camera). Reusable across any depth device.
    inline Frame BackprojectDepth(const DepthFrame &d, const CameraIntrinsics &k,
                                   const DepthFilterOptions &filter = {}) {
        Frame fr;
        const int W = k.width, H = k.height;
        if (W <= 0 || H <= 0 || int(d.depth.size()) < W * H) return fr;
        // Points AND normals come from the filtered depth. Filtering only for the normals would leave
        // the two describing different surfaces, which is precisely the inconsistency a
        // point-to-plane SDF punishes.
        const std::vector<float> depth =
                PrefilterDepth(d.depth, W, H, filter.prefilterWindow, filter);
        std::vector<Eigen::Vector3f> grid(std::size_t(W) * H, Eigen::Vector3f::Zero());
        std::vector<char> valid(std::size_t(W) * H, 0);
        for (int v = 0; v < H; ++v)
            for (int u = 0; u < W; ++u) {
                const float z = depth[std::size_t(v) * W + u];
                if (z <= 0.0f) continue;
                grid[std::size_t(v) * W + u] =
                        Eigen::Vector3f((u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z);
                valid[std::size_t(v) * W + u] = 1;
            }
        fr.pts.reserve(std::size_t(W) * H);
        fr.nrm.reserve(std::size_t(W) * H);
        for (int v = 0; v + 1 < H; ++v)
            for (int u = 0; u + 1 < W; ++u) {
                const std::size_t i = std::size_t(v) * W + u;
                if (!valid[i] || !valid[i + 1] || !valid[i + W]) continue;

                // A step in depth is two surfaces, not one: differencing across it yields a normal
                // belonging to neither. The point goes with the normal -- Frame's contract is
                // pts.size() == nrm.size(), and the whole pipeline assumes it.
                const float z = grid[i].z();
                const float maxJump = std::max(filter.minimumDepthJump, filter.relativeDepthJump * z);
                if (std::abs(grid[i + 1].z() - z) > maxJump) continue;
                if (std::abs(grid[i + W].z() - z) > maxJump) continue;

                Eigen::Vector3f n = (grid[i + 1] - grid[i]).cross(grid[i + W] - grid[i]);
                if (n.norm() < 1e-9f) continue;
                n.normalize();
                // The camera is at the frame origin, so P is also the view direction: a
                // camera-facing normal satisfies n.P < 0. Testing n.z() alone is the same thing
                // only ON the optical axis, and the two diverge with the ray angle -- across a
                // D435's 87 degree field of view an ordinary wall receding toward the image edge
                // inverts from roughly 40 degrees of incidence outward. An inverted normal flips
                // the sign of the TSDF update and of every point-to-plane ICP residual.
                if (n.dot(grid[i]) > 0.0f) n = -n;
                fr.pts.push_back(grid[i]);
                fr.nrm.push_back(n);
            }
        fr.cam = Eigen::Vector3f::Zero(); // camera at the origin of its own frame
        return fr;
    }

    class DepthCameraFrameSource : public IFrameSource {
    public:
        // The filter options travel with the source, not with AcquisitionConfig: makeSource is a
        // caller-supplied lambda, so a tool opts in here without every config gaining a depth field.
        explicit DepthCameraFrameSource(std::unique_ptr<IDepthProvider> device,
                                        DepthFilterOptions filter = {})
            : m_device(std::move(device)), m_filter(filter) {}

        EAcquisitionType Type() const override { return EAcquisitionType::DepthCamera; }
        const char *Name() const override { return "depth-camera"; }

        bool Next(Frame &out) override {
            DepthFrame d;
            if (!m_device || !m_device->Grab(d)) return false;
            out = BackprojectDepth(d, m_device->Intrinsics(), m_filter);
            return true;
        }

        void Close() override {
            if (m_device) m_device->Close();
        }

    private:
        std::unique_ptr<IDepthProvider> m_device;
        DepthFilterOptions m_filter;
    };

} // namespace Pipeline

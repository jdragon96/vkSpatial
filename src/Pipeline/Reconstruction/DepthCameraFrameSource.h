#pragma once

#include "Pipeline/Reconstruction/ReconstructionSource.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
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
    };

    struct DepthFilterOptions {
        // Reject a neighbour whose depth differs by more than max(minimumDepthJump,
        // relativeDepthJump * z). Relative because stereo depth error grows as z^2/(f*baseline): a
        // fixed threshold over-rejects near the camera and under-rejects far from it.
        float relativeDepthJump = 0.02f;  // 2 % of range
        float minimumDepthJump = 0.005f;  // 5 mm floor, for the near field
    };

    // Back-project a depth image to camera-frame points + normals (normals from the organized-grid
    // neighbours, oriented toward the camera). Reusable across any depth device.
    inline Frame BackprojectDepth(const DepthFrame &d, const CameraIntrinsics &k,
                                   const DepthFilterOptions &filter = {}) {
        Frame fr;
        const int W = k.width, H = k.height;
        if (W <= 0 || H <= 0 || int(d.depth.size()) < W * H) return fr;
        std::vector<Eigen::Vector3f> grid(std::size_t(W) * H, Eigen::Vector3f::Zero());
        std::vector<char> valid(std::size_t(W) * H, 0);
        for (int v = 0; v < H; ++v)
            for (int u = 0; u < W; ++u) {
                const float z = d.depth[std::size_t(v) * W + u];
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
                if (n.z() > 0.0f) n = -n; // face the camera (camera looks +Z)
                fr.pts.push_back(grid[i]);
                fr.nrm.push_back(n);
            }
        fr.cam = Eigen::Vector3f::Zero(); // camera at the origin of its own frame
        return fr;
    }

    class DepthCameraFrameSource : public IFrameSource {
    public:
        explicit DepthCameraFrameSource(std::unique_ptr<IDepthProvider> device)
            : m_device(std::move(device)) {}

        EAcquisitionType Type() const override { return EAcquisitionType::DepthCamera; }
        const char *Name() const override { return "depth-camera"; }

        bool Next(Frame &out) override {
            DepthFrame d;
            if (!m_device || !m_device->Grab(d)) return false;
            out = BackprojectDepth(d, m_device->Intrinsics());
            return true;
        }

    private:
        std::unique_ptr<IDepthProvider> m_device;
    };

} // namespace Pipeline

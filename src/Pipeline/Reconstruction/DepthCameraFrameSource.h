#pragma once

#include "Pipeline/Reconstruction/ReconstructionSource.h"

#include <Eigen/Core>

#include <algorithm>
#include <atomic>
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

        // Range gate, in metres; 0 disables an end. A stereo sensor reports depth outside the range
        // it can actually measure -- below its minimum the disparity search has nothing to match,
        // and above it the z^2/(f*baseline) error growth turns a reading into a guess. Neither end
        // announces itself: the pixel arrives with a value like any other.
        //
        // This duplicates rs2::threshold_filter on purpose. The recorded frames in capture/ are raw
        // Z16 taken before any SDK filter ran, so a gate that lives only in the device cannot be
        // measured on a replay -- and a replay is where every A/B in this repo is taken.
        float minimumDepthMeters = 0.0f;
        float maximumDepthMeters = 0.0f;

        // How many of the eight neighbours must be valid AND on the same surface for a pixel to be
        // emitted. 0 (default) disables it; 8 keeps only pixels with a complete neighbourhood.
        //
        // This is the closest thing a D400 gives to a per-pixel confidence. The device publishes no
        // confidence channel (that is L515), so what is left is how the pixel sits in its
        // neighbourhood: real surface arrives in sheets, and a mismatch arrives as a small island
        // in a field of pixels the matcher rejected. The depth-jump guard cannot see the
        // difference, because it looks at two neighbours and an island's interior agrees with both.
        //
        // "Same surface" reuses the jump tolerance rather than counting mere validity: a neighbour
        // across a step belongs to another surface, and counting it would admit the flying pixels
        // at every object boundary -- the ones this is for.
        int minimumValidNeighbours = 0;

        // Reject a point whose surface is turned further than this from its own view ray, in
        // degrees. 0 (default) disables it; 90 would admit everything.
        //
        // Stereo triangulation degrades as a surface turns edge-on -- the same disparity error
        // displaces the point further along the ray -- and the reading stays perfectly continuous
        // while it happens, so no discontinuity guard can see it. Point-to-plane fusion is the part
        // that pays: the normal is both the SDF value and the direction the truncation band is
        // marched along, so a grazing patch mis-values AND mis-places what it writes.
        float maximumIncidenceDegrees = 0.0f;

        // Test the point against ALL EIGHT neighbours for a depth step, not only the two the
        // normal is differenced from. false (default) keeps the historical forward-only test.
        //
        // The forward-only test is not a bug -- it protects the normal, which is built from u+1
        // and v+1, and it does that exactly. Admitting the point is a separate question, and the
        // two answers differ on the TRAILING edge of every step: a pixel whose right and down
        // neighbours are its own surface but whose left or up neighbour is half a metre behind it
        // is a stereo interpolation between two surfaces, and it is emitted today. Measured on
        // capture/ (12 frames, prefilter 3) those are 0.17% of emitted points, sitting a median
        // 403 mm from the neighbour nobody tested -- the streaks along the view direction.
        bool symmetricDepthJumpGuard = false;
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

    // What the confidence gates removed, split by cause, plus what survived.
    //
    // The gates are the only place in the depth front end that discards measurements silently: a
    // point that was never emitted looks exactly like surface the sensor never saw, so a gate set
    // too tight presents as "the reconstruction is a bit thin" and nothing else. Counting per cause
    // rather than in total is what makes it diagnosable -- the three gates fail for opposite
    // reasons and want opposite corrections.
    //
    // rejectedByRange counts PIXELS (the gate runs before back-projection); the other two count
    // candidate POINTS, which the depth-jump guard has already thinned. The three are not
    // populations of the same thing and must not be summed.
    //
    // Atomic because a caller may read them from its own thread while acquisition runs.
    struct DepthFilterStats {
        std::atomic<std::uint64_t> emittedPoints{0};
        std::atomic<std::uint64_t> rejectedByRange{0};
        std::atomic<std::uint64_t> rejectedByNeighbourSupport{0};
        std::atomic<std::uint64_t> rejectedByIncidence{0};
    };

    // True when any of the eight neighbours of (u,v) exists and lies further than `tolerance` in
    // depth -- i.e. this pixel straddles a surface boundary. A neighbour outside the image is
    // absent, not a step, so the image border is never rejected for having one.
    inline bool StraddlesADepthStep(const std::vector<float> &depth, const std::vector<char> &valid,
                                    int width, int height, int u, int v, float tolerance) {
        const float z = depth[std::size_t(v) * width + u];
        for (int dv = -1; dv <= 1; ++dv) {
            const int vv = v + dv;
            if (vv < 0 || vv >= height) continue;
            for (int du = -1; du <= 1; ++du) {
                if (du == 0 && dv == 0) continue;
                const int uu = u + du;
                if (uu < 0 || uu >= width) continue;
                const std::size_t j = std::size_t(vv) * width + uu;
                if (!valid[j]) continue;
                if (std::abs(depth[j] - z) > tolerance) return true;
            }
        }
        return false;
    }

    // How many of the eight neighbours of (u,v) are valid AND within `tolerance` of its depth --
    // the neighbourhood support behind DepthFilterOptions::minimumValidNeighbours.
    //
    // A neighbour outside the image does not exist and is not counted, so the one-pixel image
    // border can never reach eight. That is deliberate: a pixel whose neighbourhood leaves the
    // sensor is exactly as unsupported as one whose neighbourhood was never matched.
    inline int CountSameSurfaceNeighbours(const std::vector<float> &depth,
                                          const std::vector<char> &valid, int width, int height,
                                          int u, int v, float tolerance) {
        const float z = depth[std::size_t(v) * width + u];
        int count = 0;
        for (int dv = -1; dv <= 1; ++dv) {
            const int vv = v + dv;
            if (vv < 0 || vv >= height) continue;
            for (int du = -1; du <= 1; ++du) {
                if (du == 0 && dv == 0) continue;
                const int uu = u + du;
                if (uu < 0 || uu >= width) continue;
                const std::size_t j = std::size_t(vv) * width + uu;
                if (!valid[j]) continue;
                if (std::abs(depth[j] - z) > tolerance) continue;
                ++count;
            }
        }
        return count;
    }

    // Back-project a depth image to camera-frame points + normals (normals from the organized-grid
    // neighbours, oriented toward the camera). Reusable across any depth device.
    inline Frame BackprojectDepth(const DepthFrame &d, const CameraIntrinsics &k,
                                  const DepthFilterOptions &filter = {},
                                  DepthFilterStats *stats = nullptr) {
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
                // Out of range makes the pixel INVALID rather than merely unemitted: a reading the
                // sensor could not have measured must not be a neighbour either, or it still sets
                // the normal of the pixel next to it.
                if ((filter.minimumDepthMeters > 0.0f && z < filter.minimumDepthMeters) ||
                    (filter.maximumDepthMeters > 0.0f && z > filter.maximumDepthMeters)) {
                    if (stats) stats->rejectedByRange.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                grid[std::size_t(v) * W + u] =
                        Eigen::Vector3f((u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z);
                valid[std::size_t(v) * W + u] = 1;
            }
        // Hoisted: the gate is a comparison against one cosine, and computing it per pixel would
        // put a trigonometric call in the inner loop of every frame for a value that never changes.
        const float minimumIncidenceCosine =
                filter.maximumIncidenceDegrees > 0.0f
                        ? std::cos(filter.maximumIncidenceDegrees * float(M_PI) / 180.0f)
                        : 0.0f;

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

                // The forward test above protects the NORMAL, which is differenced from exactly
                // those two neighbours. Whether the POINT is trustworthy is a different question,
                // and it is answered on the trailing edge of the step the forward test cannot see.
                if (filter.symmetricDepthJumpGuard &&
                    StraddlesADepthStep(depth, valid, W, H, u, v, maxJump))
                    continue;

                // Neighbourhood support: the two neighbours above agree with this pixel, which is
                // just as true of a mismatched island as of real surface. How much of the
                // neighbourhood exists at all is what separates them.
                if (filter.minimumValidNeighbours > 0 &&
                    CountSameSurfaceNeighbours(depth, valid, W, H, u, v, maxJump) <
                            filter.minimumValidNeighbours) {
                    if (stats) stats->rejectedByNeighbourSupport.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

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

                // Incidence against the pixel's OWN ray, for the same reason the orientation flip
                // above uses it: on a wide field of view the optical axis is not the direction this
                // pixel is looking. n is now camera-facing, so -n.rayDirection is cos(incidence)
                // and falls toward 0 as the surface turns edge-on.
                if (minimumIncidenceCosine > 0.0f &&
                    -n.dot(grid[i].normalized()) < minimumIncidenceCosine) {
                    if (stats) stats->rejectedByIncidence.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                fr.pts.push_back(grid[i]);
                fr.nrm.push_back(n);
            }
        if (stats) stats->emittedPoints.fetch_add(fr.pts.size(), std::memory_order_relaxed);
        fr.cam = Eigen::Vector3f::Zero(); // camera at the origin of its own frame
        return fr;
    }

    class DepthCameraFrameSource : public IFrameSource {
    public:
        // The filter options travel with the source, not with AcquisitionConfig: makeSource is a
        // caller-supplied lambda, so a tool opts in here without every config gaining a depth field.
        //
        // `stats` is shared rather than owned because the source is built inside makeSource, on the
        // acquisition thread, and nothing hands it back -- a caller that wants to read the counters
        // has to have created them before the pipeline existed. nullptr disables the counting.
        explicit DepthCameraFrameSource(std::unique_ptr<IDepthProvider> device,
                                        DepthFilterOptions filter = {},
                                        std::shared_ptr<DepthFilterStats> stats = nullptr)
            : m_device(std::move(device)), m_filter(filter), m_stats(std::move(stats)) {}

        EAcquisitionType Type() const override { return EAcquisitionType::DepthCamera; }
        const char *Name() const override { return "depth-camera"; }

        bool Next(Frame &out) override {
            DepthFrame d;
            if (!m_device || !m_device->Grab(d)) return false;
            out = BackprojectDepth(d, m_device->Intrinsics(), m_filter, m_stats.get());
            return true;
        }

        void Close() override {
            if (m_device) m_device->Close();
        }

    private:
        std::unique_ptr<IDepthProvider> m_device;
        DepthFilterOptions m_filter;
        std::shared_ptr<DepthFilterStats> m_stats;
    };

} // namespace Pipeline

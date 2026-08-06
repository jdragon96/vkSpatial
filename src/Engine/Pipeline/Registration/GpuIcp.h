#pragma once
#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Registration/Icp.h" // RegistrationParam
#include "Engine/Registration/RegistrationTypes.h"
#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Pipeline {

    // Dense uniform grid over a point set's AABB (cell = correspondence radius). Buckets are stored as
    // a CSR-style pair (bucketStart prefix-sum + bucketIdx grouped indices) so the SAME arrays upload
    // straight to the GPU. Built once per ICP solve over the CROPPED (local) target -> small + cheap.
    class LocalGrid {
    public:
        // NOTE: `pts` is stored by reference (see m_pts below) -> the caller must keep the vector
        // alive for the LocalGrid's lifetime.
        LocalGrid(const std::vector<Eigen::Vector3f> &pts, float cell);

        // Nearest neighbour of `q` within `radius`, or -1 if none. PRECONDITION: correct only for
        // radius <= m_cell -- the 27-cell (3x3x3) neighbourhood scan cannot see farther than one cell
        // beyond q's own cell. Planned callers always pass radius == cell.
        int Nearest(const Eigen::Vector3f &q, float radius) const;

        Eigen::Vector3f m_origin = Eigen::Vector3f::Zero(); // AABB min (zeroed: left indeterminate on
                                                             // empty input otherwise, and this is a
                                                             // public GPU-upload member)
        Eigen::Vector3i m_dims{1, 1, 1};         // cells per axis
        float m_cell = 1.0f;
        std::vector<uint32_t> m_bucketStart;     // size dims.prod()+1 (prefix sum)
        std::vector<uint32_t> m_bucketIdx;       // size pts.size() (pt indices grouped by cell)
        const std::vector<Eigen::Vector3f> &m_pts; // caller-owned; must outlive this LocalGrid

    private:
        int cellIndex(const Eigen::Vector3i &c) const {
            return (c.z() * m_dims.y() + c.y()) * m_dims.x() + c.x();
        }
        Eigen::Vector3i cellOf(const Eigen::Vector3f &p) const {
            return Eigen::Vector3i(int(std::floor((p.x() - m_origin.x()) / m_cell)),
                                   int(std::floor((p.y() - m_origin.y()) / m_cell)),
                                   int(std::floor((p.z() - m_origin.z()) / m_cell)));
        }
    };

    // One GPU dispatch of point-to-plane ICP accumulation: transforms `src` by `T`, finds grid-NN
    // correspondences against `tgt`, and reduces the 6x6 normal equations H,b (+ inlier count) via a
    // per-workgroup fixed-point shared-memory reduction (MoltenVK has no GPU float atomics). Internally
    // centres source and target on the target centroid for fixed-point numerical safety. IMPORTANT: the
    // returned H,b are the TARGET-CENTROID-CENTRED-frame normal equations, NOT the un-centred (world)
    // frame's -- p x n (the Jacobian's rotational block) is not translation-invariant, so centring
    // changes H,b's values even though it leaves the point-to-plane residual itself invariant. See
    // icp_iterate.comp.glsl header for the same statement on the GPU side.
    class GpuPointToPlaneIcp {
    public:
        struct IterOut { Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b; int inliers; };
        explicit GpuPointToPlaneIcp(Engine::Core::Context &ctx);

        // Thin wrapper: computes the target centroid `c` from `tgt` and delegates to AccumulateCentred.
        // `T` here is used AS-IS in the centred frame (it is NOT re-centred internally) -- see
        // AccumulateCentred's doc comment.
        IterOut Accumulate(const std::vector<Eigen::Vector3f> &src,
                           const Engine::Registration::PointCloud &tgt,
                           const Eigen::Matrix4f &T, float maxCorrDist);

        // Full ICP iterate loop: repeatedly calls AccumulateCentred, solves the 6x6 normal equations
        // with Eigen LDLT, composes the incremental twist onto a centred working pose, and converges --
        // producing a RegistrationResult equivalent to Engine::Registration::AlignPointToPlaneIcp. Seeds
        // and un-centres around `priorT`/`tgt`'s centroid once, up front/at the end respectively (see
        // .cpp for the exact Tc * priorT * Tc^-1 / Tc^-1 * T * Tc composition).
        Engine::Registration::RegistrationResult Solve(
                const std::vector<Eigen::Vector3f> &src, const Engine::Registration::PointCloud &tgt,
                const Eigen::Matrix4f &priorT, const Engine::Registration::RegistrationParam &params);

    private:
        static constexpr uint32_t kLocal = 256;
        static constexpr float kScale = 10000.0f;
        Engine::Core::Context *m_ctx;
        std::unique_ptr<Engine::Core::ComputePipeline> m_kernel;
        std::unique_ptr<Engine::Core::Buffer> m_src, m_tgtPts, m_tgtNrm, m_bucketStart, m_bucketIdx, m_partials;

        // One GPU dispatch in the CENTRED frame: `c` (precomputed target centroid) and `T` (the CENTRED
        // working pose, i.e. NOT re-centred here -- callers that seed/iterate in the centred frame, like
        // Solve, must pass an already-centred T) are both used as-is. Builds sc = src - c, tc =
        // tgt.points - c, uploads, dispatches with T, and returns the centred-frame H,b (+ inliers).
        // Guards: zero IterOut if src is empty, tgt has < 3 points, or tgt.normals.size() !=
        // tgt.points.size() (mismatched normals would read out of bounds on the GPU).
        IterOut AccumulateCentred(const std::vector<Eigen::Vector3f> &src,
                                  const Engine::Registration::PointCloud &tgt, const Eigen::Vector3f &c,
                                  const Eigen::Matrix4f &T, float maxCorrDist);

        // Upload phase (called ONCE per Solve): centres src/tgt on `c`, builds the LocalGrid, uploads all
        // five GPU buffers, allocates the partials readback, and binds everything to the kernel. Returns
        // false on the same guards as AccumulateCentred (empty src / <3 tgt / mismatched normals). None of
        // the uploaded data depends on the pose, so Solve does this once and then only re-dispatches.
        bool prepareCentred(const std::vector<Eigen::Vector3f> &src,
                            const Engine::Registration::PointCloud &tgt, const Eigen::Vector3f &c,
                            float maxCorrDist);

        // Dispatch phase: runs ONE GPU accumulation for the centred pose `T` against the buffers already
        // uploaded + bound by prepareCentred, and reduces the readback into centred-frame H,b (+ inliers).
        // Only the push-constant T changes between iterations -- no grid rebuild, no buffer re-upload.
        IterOut dispatchCentred(const Eigen::Matrix4f &T);

        // Prepared-solve state (set by prepareCentred, consumed by dispatchCentred): grid metadata + counts.
        Eigen::Vector3f m_pOrigin = Eigen::Vector3f::Zero();
        Eigen::Vector3i m_pDims{1, 1, 1};
        float m_pCell = 1.0f;
        float m_pMaxCorr = 0.0f;
        uint32_t m_pNumSrc = 0;
        uint32_t m_pNumWG = 0;
    };

} // namespace Engine::Pipeline

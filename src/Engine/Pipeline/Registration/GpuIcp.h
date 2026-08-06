#pragma once
#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
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
    // centres source and target on the target centroid for fixed-point numerical safety; since a common
    // translation leaves the point-to-plane residual and Jacobian invariant (see icp_iterate.comp.glsl
    // header), the returned H,b equal the CPU reference computed on the SAME (uncentred) points and T.
    class GpuPointToPlaneIcp {
    public:
        struct IterOut { Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b; int inliers; };
        explicit GpuPointToPlaneIcp(Engine::Core::Context &ctx);
        IterOut Accumulate(const std::vector<Eigen::Vector3f> &src,
                           const Engine::Registration::PointCloud &tgt,
                           const Eigen::Matrix4f &T, float maxCorrDist);
    private:
        static constexpr uint32_t kLocal = 256;
        static constexpr float kScale = 10000.0f;
        Engine::Core::Context *m_ctx;
        std::unique_ptr<Engine::Core::ComputePipeline> m_kernel;
        std::unique_ptr<Engine::Core::Buffer> m_src, m_tgtPts, m_tgtNrm, m_bucketStart, m_bucketIdx, m_partials;
    };

} // namespace Engine::Pipeline

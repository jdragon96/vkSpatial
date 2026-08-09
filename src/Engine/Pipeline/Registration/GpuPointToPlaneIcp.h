#pragma once
#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Pipeline/Registration/RegistrationTypes.h"
#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Pipeline {

    class LocalGrid {
    public:
        LocalGrid(const std::vector<Eigen::Vector3f> &pts, float cell);

        int Nearest(const Eigen::Vector3f &q, float radius) const;

        Eigen::Vector3f m_origin = Eigen::Vector3f::Zero(); // AABB min (zeroed: left indeterminate on
                                                            // empty input otherwise, and this is a
                                                            // public GPU-upload member)
        Eigen::Vector3i m_dims{1, 1, 1};                    // cells per axis
        float m_cell = 1.0f;
        std::vector<uint32_t> m_bucketStart;       // size dims.prod()+1 (prefix sum)
        std::vector<uint32_t> m_bucketIdx;         // size pts.size() (pt indices grouped by cell)
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

    class GpuPointToPlaneIcp {
    public:
        struct IterOut {
            Eigen::Matrix<double, 6, 6> H;
            Eigen::Matrix<double, 6, 1> b;
            int inliers;
            double sumOfSquaredResiduals = 0.0;
        };
        explicit GpuPointToPlaneIcp(Engine::Core::Context &ctx);

        IterOut Accumulate(const std::vector<Eigen::Vector3f> &src,
                           const Engine::Registration::PointCloud &tgt,
                           const Eigen::Matrix4f &T, float maxCorrDist);

        // `sourceNormals` (sensor/source-frame, pre-pose) may be empty to skip the normal-compatibility
        // rejection entirely, matching AlignPointToPlaneIcp's CPU semantics; when non-empty it MUST be
        // index-aligned with `src`.
        Engine::Registration::RegistrationResult Solve(
                const std::vector<Eigen::Vector3f> &src, const std::vector<Eigen::Vector3f> &sourceNormals,
                const Engine::Registration::PointCloud &tgt,
                const Eigen::Matrix4f &priorT, const Engine::Registration::RegistrationParam &params);

    private:
        static constexpr uint32_t kLocal = 256;
        static constexpr float kScale = 10000.0f;
        // Sentinel normal-compatibility cosine used when the caller supplies no source normals: real
        // cosines lie in [-1,1], so this threshold is never crossed and the rejection is effectively off
        // (mirrors the CPU's `sourceNormals.empty()` early-out).
        static constexpr float kNoNormalRejectionCosine = -2.0f;
        // Sentinel Huber scale used by the legacy Accumulate() convenience path (no RegistrationParam,
        // so no caller-chosen knee): large enough that robustWeight == 1 for any real-world residual,
        // reproducing pre-Tier-2 (unweighted) accumulation exactly.
        static constexpr float kNoRobustWeightingHuberScale = 1e30f;
        Engine::Core::Context *m_ctx;
        std::unique_ptr<Engine::Core::ComputePipeline> m_kernel;
        std::unique_ptr<Engine::Core::Buffer> m_src, m_tgtPts, m_tgtNrm, m_bucketStart, m_bucketIdx, m_partials,
                m_sourceNormals;

        IterOut AccumulateCentred(const std::vector<Eigen::Vector3f> &src,
                                  const Engine::Registration::PointCloud &tgt, const Eigen::Vector3f &c,
                                  const Eigen::Matrix4f &T, float maxCorrDist);

        bool prepareCentred(const std::vector<Eigen::Vector3f> &src,
                            const std::vector<Eigen::Vector3f> &sourceNormals,
                            const Engine::Registration::PointCloud &tgt, const Eigen::Vector3f &c,
                            float maxCorrDist);

        IterOut dispatchCentred(const Eigen::Matrix4f &T, float huberScale, float normalCompatibilityCosine);

        Eigen::Vector3f m_pOrigin = Eigen::Vector3f::Zero();
        Eigen::Vector3i m_pDims{1, 1, 1};
        float m_pCell = 1.0f;
        float m_pMaxCorr = 0.0f;
        uint32_t m_pNumSrc = 0;
        uint32_t m_pNumWG = 0;
    };

} // namespace Engine::Pipeline

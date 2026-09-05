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

namespace Pipeline {

    class LocalGrid {
    public:
        LocalGrid(const std::vector<Eigen::Vector3f> &points, float cell);

        int Nearest(const Eigen::Vector3f &query, float radius) const;

        Eigen::Vector3f m_origin = Eigen::Vector3f::Zero(); // AABB min (zeroed: left indeterminate on
                                                            // empty input otherwise, and this is a
                                                            // public GPU-upload member)
        Eigen::Vector3i m_dims{1, 1, 1};                    // cells per axis
        float m_cell = 1.0f;
        std::vector<uint32_t> m_bucketStart;       // size dims.prod()+1 (prefix sum)
        std::vector<uint32_t> m_bucketIdx;         // size pts.size() (pt indices grouped by cell)
        const std::vector<Eigen::Vector3f> &m_pts; // caller-owned; must outlive this LocalGrid

    private:
        int cellIndex(const Eigen::Vector3i &cellCoordinate) const {
            return (cellCoordinate.z() * m_dims.y() + cellCoordinate.y()) * m_dims.x() + cellCoordinate.x();
        }
        Eigen::Vector3i cellOf(const Eigen::Vector3f &point) const {
            return Eigen::Vector3i(int(std::floor((point.x() - m_origin.x()) / m_cell)),
                                   int(std::floor((point.y() - m_origin.y()) / m_cell)),
                                   int(std::floor((point.z() - m_origin.z()) / m_cell)));
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
        std::unique_ptr<Engine::Core::Buffer> m_src;
        std::unique_ptr<Engine::Core::Buffer> m_tgtPts;
        std::unique_ptr<Engine::Core::Buffer> m_tgtNrm;
        std::unique_ptr<Engine::Core::Buffer> m_bucketStart;
        std::unique_ptr<Engine::Core::Buffer> m_bucketIdx;
        std::unique_ptr<Engine::Core::Buffer> m_partials;
        std::unique_ptr<Engine::Core::Buffer> m_sourceNormals;

        IterOut AccumulateCentred(const std::vector<Eigen::Vector3f> &src,
                                  const Engine::Registration::PointCloud &tgt, const Eigen::Vector3f &c,
                                  const Eigen::Matrix4f &T, float maxCorrDist);

        bool prepareCentred(const std::vector<Eigen::Vector3f> &src,
                            const std::vector<Eigen::Vector3f> &sourceNormals,
                            const Engine::Registration::PointCloud &tgt, const Eigen::Vector3f &c,
                            float maxCorrDist);

        // `currentMaxCorrespondenceDistance` is the per-ITERATION distance filter (pc.maxCorr); it may
        // be narrower than the grid's build-time cell width (m_pCell, set once in prepareCentred at
        // the WIDEST annealed distance) -- buffers/grid stay bound from prepareCentred (hoist intact),
        // only the push constant changes per dispatch. See RegistrationParam::minCorrespondenceDistance.
        IterOut dispatchCentred(const Eigen::Matrix4f &T, float huberScale, float normalCompatibilityCosine,
                                float currentMaxCorrespondenceDistance);

        Eigen::Vector3f m_pOrigin = Eigen::Vector3f::Zero();
        Eigen::Vector3i m_pDims{1, 1, 1};
        float m_pCell = 1.0f;
        float m_pMaxCorr = 0.0f;
        uint32_t m_pNumSrc = 0;
        uint32_t m_pNumWorkerGroup = 0;
    };

} // namespace Pipeline

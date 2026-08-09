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
        };
        explicit GpuPointToPlaneIcp(Engine::Core::Context &ctx);

        IterOut Accumulate(const std::vector<Eigen::Vector3f> &src,
                           const Engine::Registration::PointCloud &tgt,
                           const Eigen::Matrix4f &T, float maxCorrDist);

        Engine::Registration::RegistrationResult Solve(
                const std::vector<Eigen::Vector3f> &src, const Engine::Registration::PointCloud &tgt,
                const Eigen::Matrix4f &priorT, const Engine::Registration::RegistrationParam &params);

    private:
        static constexpr uint32_t kLocal = 256;
        static constexpr float kScale = 10000.0f;
        Engine::Core::Context *m_ctx;
        std::unique_ptr<Engine::Core::ComputePipeline> m_kernel;
        std::unique_ptr<Engine::Core::Buffer> m_src, m_tgtPts, m_tgtNrm, m_bucketStart, m_bucketIdx, m_partials;

        IterOut AccumulateCentred(const std::vector<Eigen::Vector3f> &src,
                                  const Engine::Registration::PointCloud &tgt, const Eigen::Vector3f &c,
                                  const Eigen::Matrix4f &T, float maxCorrDist);

        bool prepareCentred(const std::vector<Eigen::Vector3f> &src,
                            const Engine::Registration::PointCloud &tgt, const Eigen::Vector3f &c,
                            float maxCorrDist);

        IterOut dispatchCentred(const Eigen::Matrix4f &T);

        Eigen::Vector3f m_pOrigin = Eigen::Vector3f::Zero();
        Eigen::Vector3i m_pDims{1, 1, 1};
        float m_pCell = 1.0f;
        float m_pMaxCorr = 0.0f;
        uint32_t m_pNumSrc = 0;
        uint32_t m_pNumWG = 0;
    };

} // namespace Engine::Pipeline

#pragma once
#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Registration/RegistrationTypes.h"
#include "Common/PointCloud.h"
#include "Registration/RegistrationParam.h"
#include "Registration/RegistrationResult.h"
#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <vector>

namespace Pipeline {

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
                           const Common::PointCloud &tgt,
                           const Eigen::Matrix4f &T, float maxCorrDist);

        Registration::RegistrationResult Solve(
                const std::vector<Eigen::Vector3f> &src, const std::vector<Eigen::Vector3f> &sourceNormals,
                const Common::PointCloud &tgt,
                const Eigen::Matrix4f &priorT, const Registration::RegistrationParam &params);

    private:
        static constexpr uint32_t kLocal = 256;
        static constexpr float kScale = 10000.0f;
        static constexpr float kNoNormalRejectionCosine = -2.0f;
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
                                  const Common::PointCloud &tgt, const Eigen::Vector3f &c,
                                  const Eigen::Matrix4f &T, float maxCorrDist);

        bool prepareCentred(const std::vector<Eigen::Vector3f> &src,
                            const std::vector<Eigen::Vector3f> &sourceNormals,
                            const Common::PointCloud &tgt, const Eigen::Vector3f &c,
                            float maxCorrDist);

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

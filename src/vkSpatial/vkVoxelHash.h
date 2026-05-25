#pragma once

#include "vkSpatial/common/vkComputeBase.h"
#include "vkSpatial/common/vkContext.h"
#include "vkSpatial/common/vkGPUMemory.h"

#include <Eigen/Core>
#include <string>
#include <vector>

namespace vkSpatial {

    enum class EVoxelHashType {
        WANG,
    };

    struct VoxelEntry {
        uint32_t key;
        uint32_t count;
    };
    class vkVoxelHash {
    public:
        explicit vkVoxelHash();

        void SetAlgorithmType(EVoxelHashType type);

        void Build(vkCommon::VkContext *ctx,
                   float voxelSize = 0.1f,
                   uint32_t hashCapacity = 1u << 20,
                   uint32_t maxPoints = 1u << 12);

        void Integrate(const std::vector<Eigen::Vector3f> &points);

        void Reset();

        void ExportMesh(const std::string &path) const;

        void ExportMC(const std::string &path, uint32_t maxTris = 500000u) const;

        uint32_t FilledCount() const;

    private:
        vkCommon::VkContext *m_ctx = nullptr;
        float m_voxelSize = 0.1f;
        uint32_t m_hashCapacity = 0;
        uint32_t m_maxPoints = 0;
        EVoxelHashType m_type = EVoxelHashType::WANG;

        vkCommon::vkGPUMemory::SharedPtr m_hashBuffer;
        vkCommon::vkGPUMemory::SharedPtr m_pointBuffer;
        vkCommon::vkGPUMemory::SharedPtr m_statBuffer;
        vkCommon::vkComputeBase::SharedPtr m_kernel;
    };

} // namespace vkSpatial

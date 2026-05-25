#include "vkSpatial/common/vkComputeBase.h"
#include "vkSpatial/common/vkGPUMemory.h"
#include "vkSpatial/vkBVH.h"

#include <algorithm>
#include <stdexcept>

namespace vkbvh {

    namespace {
        constexpr uint32_t MAX_K = 64;
        constexpr uint32_t INVALID_IDX = 0xFFFFFFFFu;

        struct KNNPC {
            float cx, cy, cz;
            uint32_t k;
        };

        vkComputeBase makeKernel(VkContext *ctx) {
            return vkComputeBase(ctx->device, ctx->physDevice,
                                 ctx->computeQueue, ctx->cmdPool);
        }
    } // namespace

    std::vector<uint32_t> vkBVH::KNN(float cx, float cy, float cz, int k) {
        if (!m_built)
            throw std::runtime_error("vkBVH: Build() must be called first");
        if (k <= 0 || static_cast<uint32_t>(k) > MAX_K)
            throw std::runtime_error("vkBVH: KNN k must be in [1, 64]");

        const uint32_t uk = static_cast<uint32_t>(k);

        vkGPUMemory resultBuf(m_ctx->device, m_ctx->physDevice);
        vkGPUMemory distBuf(m_ctx->device, m_ctx->physDevice);

        if (!resultBuf.Allocate(uk * sizeof(uint32_t)))
            throw std::runtime_error("vkBVH: KNN resultBuf alloc failed");
        if (!distBuf.Allocate(uk * sizeof(float)))
            throw std::runtime_error("vkBVH: KNN distBuf alloc failed");

        KNNPC pc{cx, cy, cz, uk};

        auto kernel = makeKernel(m_ctx);
        kernel.Build(m_shaderDir + "/cmd_knn.comp")
                .Bind(0, *m_nodeBuf)
                .Bind(1, resultBuf)
                .Bind(2, distBuf)
                .Args(pc)
                .Dispatch(1);
        kernel.Sync();

        std::vector<uint32_t> indices(uk);
        if (!resultBuf.Download(indices.data(), uk * sizeof(uint32_t),
                                m_ctx->computeQueue, m_ctx->cmdPool))
            throw std::runtime_error("vkBVH: KNN result download failed");

        // k보다 이웃이 적을 때 INVALID_IDX 슬롯 제거
        indices.erase(
                std::remove(indices.begin(), indices.end(), INVALID_IDX),
                indices.end());

        return indices;
    }

} // namespace vkbvh

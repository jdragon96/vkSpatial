#include "vkSpatial/common/vkComputeBase.h"
#include "vkSpatial/common/vkGPUMemory.h"
#include "vkSpatial/vkBVH.h"

#include <algorithm>
#include <stdexcept>

namespace vkbvh {

    namespace {
        struct RadiusSearchPC {
            float cx, cy, cz, r;
            uint32_t maxResults;
        };

        vkComputeBase makeKernel(VkContext *ctx) {
            return vkComputeBase(ctx->device, ctx->physDevice,
                                 ctx->computeQueue, ctx->cmdPool);
        }
    } // namespace

    std::vector<uint32_t> vkBVH::RadiusSearch(float cx, float cy, float cz, float r) {
        if (!m_built) throw std::runtime_error("vkBVH: Build() must be called first");

        const uint32_t maxResults = Length();

        vkGPUMemory queryBuffer(m_ctx->device, m_ctx->physDevice);
        vkGPUMemory countBuffer(m_ctx->device, m_ctx->physDevice);

        if (!queryBuffer.Allocate(maxResults * sizeof(uint32_t)))
            throw std::runtime_error("vkBVH: RadiusSearch resultBuf alloc failed");
        if (!countBuffer.Allocate(sizeof(uint32_t)))
            throw std::runtime_error("vkBVH: RadiusSearch countBuf alloc failed");
        const uint32_t zero = 0;
        if (!countBuffer.Upload(&zero, sizeof(uint32_t), m_ctx->computeQueue, m_ctx->cmdPool))
            throw std::runtime_error("vkBVH: RadiusSearch countBuf zero-init failed");

        RadiusSearchPC pc{cx, cy, cz, r, maxResults};

        auto kernel = makeKernel(m_ctx);
        kernel.Build(m_shaderDir + "/cmd_radiusSearch.comp")
                .Bind(0, *m_nodeBuf)
                .Bind(1, queryBuffer)
                .Bind(2, countBuffer)
                .Args(pc)
                .Dispatch(1);
        kernel.Sync();

        uint32_t count = 0;
        if (!countBuffer.Download(&count, sizeof(uint32_t), m_ctx->computeQueue, m_ctx->cmdPool))
            throw std::runtime_error("vkBVH: RadiusSearch count download failed");

        if (count == 0) return {};

        uint32_t downloadCount = std::min(count, maxResults);
        std::vector<uint32_t> results(downloadCount);
        if (!queryBuffer.Download(results.data(), downloadCount * sizeof(uint32_t),
                                  m_ctx->computeQueue, m_ctx->cmdPool))
            throw std::runtime_error("vkBVH: RadiusSearch result download failed");

        return results;
    }

} // namespace vkbvh

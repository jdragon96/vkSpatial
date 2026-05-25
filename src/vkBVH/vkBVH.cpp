#include "vkBVH/vkBVH.h"
#include "vkBVH/common/vkComputeBase.h"

#include <limits>
#include <stdexcept>

namespace {
    vkComputeBase makeKernel(VkContext *ctx) {
        return vkComputeBase(ctx->device, ctx->physDevice,
                             ctx->computeQueue, ctx->cmdPool);
    }

} // namespace

vkBVH::vkBVH(VkContext *ctx, const std::string &shaderDir)
    : m_ctx(ctx), m_shaderDir(shaderDir) {}

vkBVH::~vkBVH() = default;


void vkBVH::buildBVH(const std::vector<Primitive> &prims) {
    m_count = static_cast<uint32_t>(prims.size());
    if (m_count < 2) throw std::runtime_error("vkBVH: need at least 2 primitives");

    stepAllocateBuffers();
    stepUploadPrimitives(prims);
    stepComputeMortonCodes(prims);
    stepSortMortonCodes();
    stepBuildHierarchy();
    stepComputeBoundingBoxes();

    m_built = true;
}

void vkBVH::stepAllocateBuffers() {
    const uint32_t N = m_count;
    const uint32_t NODES = N + N - 1;

    auto alloc = [&](std::unique_ptr<vkGPUMemory> &buf,
                     uint32_t bytes, const char *name) {
        buf = std::make_unique<vkGPUMemory>(m_ctx->device, m_ctx->physDevice);
        if (!buf->Allocate(bytes))
            throw std::runtime_error(std::string("vkBVH: failed to allocate ") + name);
    };

    alloc(m_primBuf, N * sizeof(Primitive), "primBuf");
    alloc(m_mortonBuf, N * sizeof(MortonCode), "mortonBuf");
    alloc(m_mortonPingBuf, N * sizeof(MortonCode), "mortonPingBuf");
    alloc(m_nodeBuf, NODES * 36, "nodeBuf");
    alloc(m_constructionBuf, NODES * 8, "constrBuf");
}

void vkBVH::stepUploadPrimitives(const std::vector<Primitive> &prims) {
    if (!m_primBuf->Upload(prims.data(),
                           m_count * sizeof(Primitive),
                           m_ctx->computeQueue,
                           m_ctx->cmdPool))
        throw std::runtime_error("vkBVH: primitive upload failed");
}

void vkBVH::stepComputeMortonCodes(const std::vector<Primitive> &prims) {

    MortonConstant pc;
    pc.Extend(prims);

    auto kernel = makeKernel(m_ctx);
    kernel.Build(m_shaderDir + "/bvh_mortonCode.comp")
            .Bind(0, *m_mortonBuf)
            .Bind(1, *m_primBuf)
            .Args(pc)
            .DispatchElements(m_count);
    kernel.Sync();
}

void vkBVH::stepSortMortonCodes() {
    constexpr uint32_t WG_SIZE = 256;
    constexpr uint32_t RADIX = 16; // 4-bit radix
    constexpr uint32_t PASSES = 8; // 8 × 4 = 32 bit

    const uint32_t numWGs = (m_count + WG_SIZE - 1) / WG_SIZE;

    m_histBuf = std::make_unique<vkGPUMemory>(m_ctx->device, m_ctx->physDevice);
    if (!m_histBuf->Allocate(RADIX * numWGs * sizeof(uint32_t)))
        throw std::runtime_error("vkBVH: histBuf alloc failed");

    vkGPUMemory *ping = m_mortonBuf.get();
    vkGPUMemory *pong = m_mortonPingBuf.get();

    auto histogramkernel = makeKernel(m_ctx);
    auto prefixScanKernel = makeKernel(m_ctx);
    auto reorderKernel = makeKernel(m_ctx);

    histogramkernel.Build(m_shaderDir + "/bvh_radixSort_histogram.comp");
    prefixScanKernel.Build(m_shaderDir + "/bvh_radixSort_prefixScan.comp");
    reorderKernel.Build(m_shaderDir + "/bvh_radixSort_reorder.comp");

    for (uint32_t pass = 0; pass < PASSES; pass++) {
        RadixSortPC pcSort{m_count, pass * 4};
        RadixScanPC pcScan{numWGs};

        histogramkernel
                .Bind(0, *ping)
                .Bind(1, *m_histBuf)
                .Args(pcSort)
                .Dispatch(numWGs);
        histogramkernel.Sync();

        prefixScanKernel
                .Bind(0, *m_histBuf)
                .Args(pcScan)
                .Dispatch(1);
        prefixScanKernel.Sync();

        reorderKernel
                .Bind(0, *ping)
                .Bind(1, *m_histBuf)
                .Bind(2, *pong)
                .Args(pcSort)
                .Dispatch(numWGs);
        reorderKernel.Sync();

        std::swap(ping, pong);
    }
}

void vkBVH::stepBuildHierarchy() {
    HierarchyPC pc{m_count, 1 /*absolute pointers*/};

    auto kernel = makeKernel(m_ctx);
    kernel.Build(m_shaderDir + "/bvh_hierarchy.comp")
            .Bind(0, *m_mortonBuf)
            .Bind(1, *m_primBuf)
            .Bind(2, *m_nodeBuf)
            .Bind(3, *m_constructionBuf)
            .Args(pc)
            .DispatchElements(m_count);
    kernel.Sync();
}

void vkBVH::stepComputeBoundingBoxes() {
    HierarchyPC pc{m_count, 1};

    auto kernel = makeKernel(m_ctx);
    kernel.Build(m_shaderDir + "/bvh_boundingBox.comp")
            .Bind(0, *m_nodeBuf)
            .Bind(1, *m_constructionBuf)
            .Args(pc)
            .DispatchElements(m_count);
    kernel.Sync();
}

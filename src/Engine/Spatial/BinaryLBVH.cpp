#include "Engine/Spatial/BinaryLBVH.h"

#include "Engine/Core/ComputePipeline.h"

#include <algorithm>
#include <stdexcept>

namespace Engine::Spatial {

    using Engine::Core::Buffer;
    using Engine::Core::ComputePipeline;

    namespace {
        constexpr uint32_t MAX_K = 64;
        constexpr uint32_t INVALID_IDX = 0xFFFFFFFFu;

        struct RadixSortPC {
            uint32_t g_count;
            uint32_t g_shift;
        };
        struct RadixScanPC {
            uint32_t g_numWGs;
        };
        struct HierarchyPC {
            uint32_t g_count;
            uint32_t g_absolutePointers;
        };
        struct KNNPC {
            float cx, cy, cz;
            uint32_t k;
        };
        struct RadiusPC {
            float cx, cy, cz, r;
            uint32_t maxResults;
        };
    } // namespace

    BinaryLBVH::BinaryLBVH(Engine::Core::Context &ctx) : m_ctx(&ctx) {}
    BinaryLBVH::~BinaryLBVH() = default;

    void BinaryLBVH::BuildFromPrimitives(const std::vector<Primitive> &prims) {
        m_count = static_cast<uint32_t>(prims.size());
        if (m_count < 2)
            throw std::runtime_error("BinaryLBVH: need at least 2 primitives");

        const uint32_t N = m_count;
        const uint32_t NODES = N + N - 1;

        m_primBuf = std::make_unique<Buffer>(*m_ctx);
        m_mortonBuf = std::make_unique<Buffer>(*m_ctx);
        m_mortonPingBuf = std::make_unique<Buffer>(*m_ctx);
        m_nodeBuf = std::make_unique<Buffer>(*m_ctx);
        m_constructionBuf = std::make_unique<Buffer>(*m_ctx);

        m_primBuf->Allocate(N * static_cast<uint32_t>(sizeof(Primitive)));
        m_mortonBuf->Allocate(N * static_cast<uint32_t>(sizeof(MortonCode)));
        m_mortonPingBuf->Allocate(N * static_cast<uint32_t>(sizeof(MortonCode)));
        m_nodeBuf->Allocate(NODES * 36u);
        m_constructionBuf->Allocate(NODES * 8u);

        m_primBuf->Upload(prims.data(), N * static_cast<uint32_t>(sizeof(Primitive)));

        MortonConstant mpc;
        mpc.Extend(prims);
        ComputePipeline(*m_ctx)
                .Build("bvh_mortonCode.comp")
                .Bind(0, *m_mortonBuf)
                .Bind(1, *m_primBuf)
                .Args(mpc)
                .DispatchElements(N);

        stepSortMortonCodes(); // sorted MortonCode[] ends up in m_mortonBuf

        const HierarchyPC hpc{N, 1u};
        ComputePipeline(*m_ctx)
                .Build("bvh_hierarchy.comp")
                .Bind(0, *m_mortonBuf)
                .Bind(1, *m_primBuf)
                .Bind(2, *m_nodeBuf)
                .Bind(3, *m_constructionBuf)
                .Args(hpc)
                .DispatchElements(N);

        ComputePipeline(*m_ctx)
                .Build("bvh_boundingBox.comp")
                .Bind(0, *m_nodeBuf)
                .Bind(1, *m_constructionBuf)
                .Args(hpc)
                .DispatchElements(N);

        setupQueryKernels();
        m_built = true;
    }

    void BinaryLBVH::stepSortMortonCodes() {
        constexpr uint32_t WG_SIZE = 256;
        constexpr uint32_t RADIX = 16; // 4-bit radix
        constexpr uint32_t PASSES = 8; // 8 x 4 = 32 bits

        const uint32_t numWGs = (m_count + WG_SIZE - 1) / WG_SIZE;

        m_histBuf = std::make_unique<Buffer>(*m_ctx);
        m_histBuf->Allocate(RADIX * numWGs * static_cast<uint32_t>(sizeof(uint32_t)));

        Buffer *ping = m_mortonBuf.get();
        Buffer *pong = m_mortonPingBuf.get();

        ComputePipeline histogram(*m_ctx), prefixScan(*m_ctx), reorder(*m_ctx);
        histogram.Build("bvh_radixSort_histogram.comp");
        prefixScan.Build("bvh_radixSort_prefixScan.comp");
        reorder.Build("bvh_radixSort_reorder.comp");

        for (uint32_t pass = 0; pass < PASSES; ++pass) {
            const RadixSortPC pcSort{m_count, pass * 4u};
            const RadixScanPC pcScan{numWGs};

            histogram.Bind(0, *ping).Bind(1, *m_histBuf).Args(pcSort).Dispatch(numWGs);
            prefixScan.Bind(0, *m_histBuf).Args(pcScan).Dispatch(1);
            reorder.Bind(0, *ping).Bind(1, *m_histBuf).Bind(2, *pong).Args(pcSort).Dispatch(numWGs);

            std::swap(ping, pong);
        }
        // PASSES is even, so after the final swap `ping == m_mortonBuf` and it holds the
        // fully sorted MortonCode[]. bvh_hierarchy.comp reads from m_mortonBuf.
    }

    void BinaryLBVH::setupQueryKernels() {
        m_knnResultBuf = std::make_unique<Buffer>(*m_ctx);
        m_knnDistBuf = std::make_unique<Buffer>(*m_ctx);
        m_radiusResultBuf = std::make_unique<Buffer>(*m_ctx);
        m_radiusCountBuf = std::make_unique<Buffer>(*m_ctx);

        m_knnResultBuf->Allocate(MAX_K * static_cast<uint32_t>(sizeof(uint32_t)));
        m_knnDistBuf->Allocate(MAX_K * static_cast<uint32_t>(sizeof(float)));
        m_radiusResultBuf->Allocate(m_count * static_cast<uint32_t>(sizeof(uint32_t)));
        m_radiusCountBuf->Allocate(static_cast<uint32_t>(sizeof(uint32_t)));

        m_knnKernel = std::make_unique<ComputePipeline>(*m_ctx);
        m_knnKernel->Build("cmd_knn.comp")
                .Bind(0, *m_nodeBuf)
                .Bind(1, *m_knnResultBuf)
                .Bind(2, *m_knnDistBuf);

        m_radiusKernel = std::make_unique<ComputePipeline>(*m_ctx);
        m_radiusKernel->Build("cmd_radiusSearch.comp")
                .Bind(0, *m_nodeBuf)
                .Bind(1, *m_radiusResultBuf)
                .Bind(2, *m_radiusCountBuf);
    }

    uint32_t BinaryLBVH::MemoryBytes() const {
        return m_nodeBuf ? m_nodeBuf->Size() : 0u;
    }

    std::vector<uint32_t> BinaryLBVH::RadiusSearch(float, float, float, float) {
        return {}; // implemented in Task 3
    }

    std::vector<uint32_t> BinaryLBVH::KNN(float, float, float, int) {
        return {}; // implemented in Task 4
    }

} // namespace Engine::Spatial

#include "Engine/Spatial/WideBVH.h"

#include "Engine/Core/ComputePipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace Engine::Spatial {

    using Engine::Core::Buffer;
    using Engine::Core::ComputePipeline;

    namespace {
        constexpr uint32_t RADIX = 16;
        constexpr uint32_t RADIX_PASSES = 8;
        constexpr uint32_t RADIX_WORKGROUP_SIZE = 256;
        constexpr uint32_t MAX_K = 64;
        constexpr uint32_t INVALID_IDX = 0xFFFFFFFFu;

        struct CountPC {
            uint32_t count;
        };
        struct WideBuildPC {
            uint32_t binaryNodeCount;
            uint32_t maxLeafPrimitives;
        };
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
        struct RadiusPC {
            float cx, cy, cz, radius;
            uint32_t maxResults;
        };
        struct KNNPC {
            float cx, cy, cz;
            uint32_t k;
        };

        struct QueryState {
            uint32_t count;
            uint32_t status;
        };
        struct WideBuildState {
            uint32_t nodeCount;
            uint32_t leafCount;
            uint32_t status;
            uint32_t reserved;
        };

        struct QuantizedWideNode {
            float originX, originY, originZ;
            float scaleX, scaleY, scaleZ;
            uint32_t child[8];
            uint32_t qBounds[12];
            uint32_t childCount;
            uint32_t leafMask;
        };
        struct LeafRange {
            uint32_t firstPrimitive;
            uint32_t primitiveCount;
        };
        struct BinaryRange {
            uint32_t firstPrimitive;
            uint32_t primitiveCount;
            uint32_t visitationCount;
            uint32_t reserved;
        };

        static_assert(std::is_standard_layout_v<QuantizedWideNode>);
        static_assert(sizeof(QuantizedWideNode) == 112);
        static_assert(offsetof(QuantizedWideNode, child) == 24);
        static_assert(offsetof(QuantizedWideNode, qBounds) == 56);
        static_assert(offsetof(QuantizedWideNode, childCount) == 104);
        static_assert(offsetof(QuantizedWideNode, leafMask) == 108);
        static_assert(sizeof(LeafRange) == 8);
        static_assert(sizeof(BinaryRange) == 16);
        static_assert(sizeof(WideBuildState) == 16);
        static_assert(sizeof(QueryState) == 8);

        uint32_t checkedBytes(uint64_t count, uint64_t elementSize, const char *name) {
            const uint64_t bytes = count * elementSize;
            if (count == 0 || bytes > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error(std::string("WideBVH: invalid buffer size for ") + name);
            return static_cast<uint32_t>(bytes);
        }

        std::unique_ptr<Buffer> makeBuffer(Engine::Core::Context &ctx, uint32_t bytes) {
            auto b = std::make_unique<Buffer>(ctx);
            b->Allocate(bytes);
            return b;
        }

        void validatePrimitive(const Primitive &p) {
            const std::array<float, 6> v{p.aabbMinX, p.aabbMinY, p.aabbMinZ,
                                         p.aabbMaxX, p.aabbMaxY, p.aabbMaxZ};
            for (float f : v)
                if (!std::isfinite(f))
                    throw std::runtime_error("WideBVH: primitive bounds must be finite");
            if (p.aabbMinX > p.aabbMaxX || p.aabbMinY > p.aabbMaxY || p.aabbMinZ > p.aabbMaxZ)
                throw std::runtime_error("WideBVH: primitive AABB min exceeds max");
        }
    } // namespace

    WideBVH::WideBVH(Engine::Core::Context &ctx, uint32_t maxLeafPrimitives)
        : m_ctx(&ctx), m_maxLeafPrimitives(maxLeafPrimitives) {
        if (maxLeafPrimitives == 0 || maxLeafPrimitives > 64)
            throw std::runtime_error("WideBVH: maxLeafPrimitives must be in [1, 64]");
        m_name = "WideBVH(leaf=" + std::to_string(maxLeafPrimitives) + ")";
    }

    WideBVH::~WideBVH() = default;

    void WideBVH::BuildFromPrimitives(const std::vector<Primitive> &prims) {
        if (prims.empty())
            throw std::runtime_error("WideBVH: need at least one primitive");
        for (const Primitive &p : prims) validatePrimitive(p);

        const uint32_t count = static_cast<uint32_t>(prims.size());
        const uint64_t binaryNodeCount64 = static_cast<uint64_t>(count) * 2u - 1u;
        if (binaryNodeCount64 > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("WideBVH: binary node count exceeds uint32");
        const uint32_t binaryNodeCount = static_cast<uint32_t>(binaryNodeCount64);
        const uint32_t radixWGs = (count + RADIX_WORKGROUP_SIZE - 1u) / RADIX_WORKGROUP_SIZE;

        auto primitiveBuffer = makeBuffer(*m_ctx, checkedBytes(count, sizeof(Primitive), "primitiveBuffer"));
        auto mortonBuffer = makeBuffer(*m_ctx, checkedBytes(count, sizeof(MortonCode), "mortonBuffer"));
        auto mortonPingBuffer = makeBuffer(*m_ctx, checkedBytes(count, sizeof(MortonCode), "mortonPingBuffer"));
        auto histogramBuffer = makeBuffer(*m_ctx, checkedBytes(static_cast<uint64_t>(RADIX) * radixWGs, sizeof(uint32_t), "histogramBuffer"));
        auto sceneBoundsBuffer = makeBuffer(*m_ctx, checkedBytes(6, sizeof(uint32_t), "sceneBoundsBuffer"));
        auto binaryNodeBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, 36, "binaryNodeBuffer"));
        auto constructionBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, 8, "constructionBuffer"));
        auto rangeBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, sizeof(BinaryRange), "rangeBuffer"));
        auto queueBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, sizeof(uint32_t), "queueBuffer"));
        auto buildStateBuffer = makeBuffer(*m_ctx, sizeof(WideBuildState));
        auto wideNodeBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, sizeof(QuantizedWideNode), "wideNodeBuffer"));
        auto leafBuffer = makeBuffer(*m_ctx, checkedBytes(binaryNodeCount, sizeof(LeafRange), "leafBuffer"));

        primitiveBuffer->Upload(prims.data(), checkedBytes(count, sizeof(Primitive), "primitiveBuffer"));

        // One pipeline object per shader, created once and reused across dispatches.
        auto mk = [&](const char *shader) {
            auto p = std::make_unique<ComputePipeline>(*m_ctx);
            p->Build(shader);
            return p;
        };
        auto boundsInit = mk("bvh_wide_bounds_init.comp");
        auto boundsReduce = mk("bvh_wide_bounds_reduce.comp");
        auto morton = mk("bvh_wide_morton.comp");
        auto histogram = mk("bvh_radixSort_histogram.comp");
        auto prefixScan = mk("bvh_radixSort_prefixScan.comp");
        auto reorder = mk("bvh_radixSort_reorder.comp");
        auto hierarchy = mk("bvh_hierarchy.comp");
        auto boundingBox = mk("bvh_boundingBox.comp");
        auto rangeInit = mk("bvh_wide_range_init.comp");
        auto range = mk("bvh_wide_range.comp");
        auto wideInit = mk("bvh_wide_build_init.comp");
        auto wideBuild = mk("bvh_wide_build.comp");

        boundsInit->Bind(0, *sceneBoundsBuffer).Dispatch(1);

        const CountPC countPC{count};
        boundsReduce->Bind(0, *primitiveBuffer).Bind(1, *sceneBoundsBuffer).Args(countPC).DispatchElements(count);
        morton->Bind(0, *mortonBuffer).Bind(1, *primitiveBuffer).Bind(2, *sceneBoundsBuffer).Args(countPC).DispatchElements(count);

        Buffer *radixInput = mortonBuffer.get();
        Buffer *radixOutput = mortonPingBuffer.get();
        for (uint32_t pass = 0; pass < RADIX_PASSES; ++pass) {
            const RadixSortPC sortPC{count, pass * 4u};
            const RadixScanPC scanPC{radixWGs};
            histogram->Bind(0, *radixInput).Bind(1, *histogramBuffer).Args(sortPC).Dispatch(radixWGs);
            prefixScan->Bind(0, *histogramBuffer).Args(scanPC).Dispatch(1);
            reorder->Bind(0, *radixInput).Bind(1, *histogramBuffer).Bind(2, *radixOutput).Args(sortPC).Dispatch(radixWGs);
            std::swap(radixInput, radixOutput);
        }

        const HierarchyPC hierarchyPC{count, 1u};
        hierarchy->Bind(0, *radixInput).Bind(1, *primitiveBuffer).Bind(2, *binaryNodeBuffer).Bind(3, *constructionBuffer).Args(hierarchyPC).DispatchElements(count);
        boundingBox->Bind(0, *binaryNodeBuffer).Bind(1, *constructionBuffer).Args(hierarchyPC).DispatchElements(count);

        const CountPC binaryNodeCountPC{binaryNodeCount};
        rangeInit->Bind(0, *rangeBuffer).Args(binaryNodeCountPC).DispatchElements(binaryNodeCount);
        range->Bind(0, *binaryNodeBuffer).Bind(1, *constructionBuffer).Bind(2, *rangeBuffer).Args(countPC).DispatchElements(count);

        wideInit->Bind(0, *queueBuffer).Bind(1, *buildStateBuffer).Dispatch(1);

        const WideBuildPC wideBuildPC{binaryNodeCount, m_maxLeafPrimitives};
        wideBuild->Bind(0, *binaryNodeBuffer).Bind(1, *rangeBuffer).Bind(2, *queueBuffer).Bind(3, *buildStateBuffer).Bind(4, *wideNodeBuffer).Bind(5, *leafBuffer).Args(wideBuildPC).Dispatch(1);

        WideBuildState state{};
        buildStateBuffer->Download(&state, sizeof(state));
        if (state.status != 0u)
            throw std::runtime_error("WideBVH: GPU wide collapse exceeded buffer capacity");
        if (state.nodeCount == 0u || state.nodeCount > binaryNodeCount ||
            state.leafCount == 0u || state.leafCount > binaryNodeCount)
            throw std::runtime_error("WideBVH: GPU wide collapse produced invalid counts");

        if (radixInput == mortonBuffer.get())
            m_sortedMortonBuf = std::move(mortonBuffer);
        else
            m_sortedMortonBuf = std::move(mortonPingBuffer);

        m_primitiveBuf = std::move(primitiveBuffer);
        m_nodeBuf = std::move(wideNodeBuffer);
        m_leafBuf = std::move(leafBuffer);
        m_count = count;
        m_nodeCount = state.nodeCount;

        setupQueryKernels();
        m_built = true;
    }

    void WideBVH::setupQueryKernels() {
        m_radiusResultBuf = makeBuffer(*m_ctx, checkedBytes(m_count, sizeof(uint32_t), "radiusResult"));
        m_radiusStateBuf = makeBuffer(*m_ctx, sizeof(QueryState));
        m_knnResultBuf = makeBuffer(*m_ctx, MAX_K * static_cast<uint32_t>(sizeof(uint32_t)));
        m_knnDistBuf = makeBuffer(*m_ctx, MAX_K * static_cast<uint32_t>(sizeof(float)));
        m_knnStateBuf = makeBuffer(*m_ctx, sizeof(QueryState));

        m_radiusKernel = std::make_unique<ComputePipeline>(*m_ctx);
        m_radiusKernel->Build("cmd_radiusSearch_wide.comp")
                .Bind(0, *m_nodeBuf).Bind(1, *m_leafBuf).Bind(2, *m_sortedMortonBuf)
                .Bind(3, *m_primitiveBuf).Bind(4, *m_radiusResultBuf).Bind(5, *m_radiusStateBuf);

        m_knnKernel = std::make_unique<ComputePipeline>(*m_ctx);
        m_knnKernel->Build("cmd_knn_wide.comp")
                .Bind(0, *m_nodeBuf).Bind(1, *m_leafBuf).Bind(2, *m_sortedMortonBuf)
                .Bind(3, *m_primitiveBuf).Bind(4, *m_knnResultBuf).Bind(5, *m_knnDistBuf)
                .Bind(6, *m_knnStateBuf);
    }

    uint32_t WideBVH::MemoryBytes() const {
        uint32_t bytes = 0;
        if (m_nodeBuf) bytes += m_nodeBuf->Size();
        if (m_leafBuf) bytes += m_leafBuf->Size();
        return bytes;
    }

    std::vector<uint32_t> WideBVH::RadiusSearch(float, float, float, float) {
        return {}; // implemented in Task 7
    }

    std::vector<uint32_t> WideBVH::KNN(float, float, float, int) {
        return {}; // implemented in Task 8
    }

} // namespace Engine::Spatial

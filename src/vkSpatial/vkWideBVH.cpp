#include "vkSpatial/vkWideBVH.h"

#include "vkCommon/vkComputeBase.h"
#include "vkCommon/vkGPUMemory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkSpatial {

    namespace {

        constexpr uint32_t RADIX = 16;
        constexpr uint32_t RADIX_PASSES = 8;
        constexpr uint32_t RADIX_WORKGROUP_SIZE = 256;
        constexpr uint32_t MAX_K = 64;

        struct CountPC {
            uint32_t count;
        };

        struct WideBuildPC {
            uint32_t binaryNodeCount;
            uint32_t maxLeafPrimitives;
        };

        struct RadiusPC {
            float cx;
            float cy;
            float cz;
            float radius;
            uint32_t maxResults;
        };

        struct KNNPC {
            float cx;
            float cy;
            float cz;
            uint32_t k;
        };

        struct RayTracePC {
            float originX, originY, originZ;
            float rightX, rightY, rightZ;
            float upX, upY, upZ;
            float forwardX, forwardY, forwardZ;
            float tanHalfFovY;
            float aspect;
            uint32_t width, height, shadeMode;
            float tMin, tMax;
            float lightDirX, lightDirY, lightDirZ;
            uint32_t groundStartIndex;
            uint32_t outputSwapRedBlue;
            float lightPosX, lightPosY, lightPosZ;
            uint32_t lightType;
        };

        struct PathTracePC {
            float originX, originY, originZ;
            float rightX, rightY, rightZ;
            float upX, upY, upZ;
            float forwardX, forwardY, forwardZ;
            float tanHalfFovY;
            float aspect;
            uint32_t width, height, frameIndex, maxBounces;
            uint32_t samplesPerPixel, outputSwapRedBlue;
            float tMin, tMax;
            float lightCenterX, lightCenterY, lightCenterZ;
            float environmentStrength;
            float lightUX, lightUY, lightUZ, lightPad0;
            float lightVX, lightVY, lightVZ, lightPad1;
            float lightEmissionX, lightEmissionY, lightEmissionZ, lightPad2;
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
            float originX;
            float originY;
            float originZ;
            float scaleX;
            float scaleY;
            float scaleZ;
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
        static_assert(sizeof(PathTracePC) <= 256);

        uint32_t checkedBytes(
                uint64_t count, uint64_t elementSize, const char *name) {
            const uint64_t bytes = count * elementSize;
            if (count == 0 || bytes > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error(
                        std::string("vkWideBVH: invalid buffer size for ") +
                        name);
            return static_cast<uint32_t>(bytes);
        }

        std::unique_ptr<vkGPUMemory> makeBuffer(
                VkContext *ctx, uint32_t bytes, const char *name) {
            auto buffer = std::make_unique<vkGPUMemory>(
                    ctx->device, ctx->physDevice);
            if (!buffer->Allocate(bytes))
                throw std::runtime_error(
                        std::string("vkWideBVH: failed to allocate ") + name);
            return buffer;
        }

        void validatePrimitive(const Primitive &primitive) {
            const std::array<float, 6> values{
                    primitive.aabbMinX,
                    primitive.aabbMinY,
                    primitive.aabbMinZ,
                    primitive.aabbMaxX,
                    primitive.aabbMaxY,
                    primitive.aabbMaxZ};
            for (float value: values) {
                if (!std::isfinite(value))
                    throw std::runtime_error(
                            "vkWideBVH: primitive bounds must be finite");
            }

            if (primitive.aabbMinX > primitive.aabbMaxX ||
                primitive.aabbMinY > primitive.aabbMaxY ||
                primitive.aabbMinZ > primitive.aabbMaxZ)
                throw std::runtime_error(
                        "vkWideBVH: primitive AABB min exceeds max");
        }

    } // namespace

    struct vkWideBVH::Impl {
        Impl(VkContext *context, uint32_t leafSize)
            : ctx(context), maxLeafPrimitives(leafSize) {
            if (ctx == nullptr)
                throw std::runtime_error(
                        "vkWideBVH: context must not be null");
            if (maxLeafPrimitives == 0 || maxLeafPrimitives > 64)
                throw std::runtime_error(
                        "vkWideBVH: maxLeafPrimitives must be in [1, 64]");
        }

        void ensureBuildKernels() {
            if (boundsInitKernel)
                return;

            boundsInitKernel = makeKernel("bvh_wide_bounds_init.comp");
            boundsReduceKernel = makeKernel("bvh_wide_bounds_reduce.comp");
            mortonKernel = makeKernel("bvh_wide_morton.comp");
            histogramKernel = makeKernel("bvh_radixSort_histogram.comp");
            prefixScanKernel = makeKernel("bvh_radixSort_prefixScan.comp");
            reorderKernel = makeKernel("bvh_radixSort_reorder.comp");
            hierarchyKernel = makeKernel("bvh_hierarchy.comp");
            boundingBoxKernel = makeKernel("bvh_boundingBox.comp");
            rangeInitKernel = makeKernel("bvh_wide_range_init.comp");
            rangeKernel = makeKernel("bvh_wide_range.comp");
            wideInitKernel = makeKernel("bvh_wide_build_init.comp");
            wideBuildKernel = makeKernel("bvh_wide_build.comp");
        }

        void ensureRadiusKernel() {
            if (!radiusKernel)
                radiusKernel = makeKernel("cmd_radiusSearch_wide.comp");
        }

        void ensureKNNKernel() {
            if (!knnKernel)
                knnKernel = makeKernel("cmd_knn_wide.comp");
        }

        void ensureRayTraceKernel() {
            if (!rayTraceKernel)
                rayTraceKernel = makeKernel("cmd_raytrace_wide.comp");
        }

        void ensurePathTraceKernel() {
            if (!pathTraceKernel)
                pathTraceKernel = makeKernel("cmd_pathtrace_wide.comp");
        }

        std::unique_ptr<vkComputeBase> makeKernel(const char *shader) const {
            auto kernel = std::make_unique<vkComputeBase>(
                    ctx->device,
                    ctx->physDevice,
                    ctx->computeQueue,
                    ctx->cmdPool);
            kernel->Build(shader);
            return kernel;
        }

        VkContext *ctx;
        uint32_t maxLeafPrimitives;
        uint32_t count = 0;
        uint32_t nodeCount = 0;
        bool built = false;

        std::unique_ptr<vkGPUMemory> primitiveBuffer;
        std::unique_ptr<vkGPUMemory> sortedMortonBuffer;
        std::unique_ptr<vkGPUMemory> nodeBuffer;
        std::unique_ptr<vkGPUMemory> leafBuffer;

        std::unique_ptr<vkComputeBase> boundsInitKernel;
        std::unique_ptr<vkComputeBase> boundsReduceKernel;
        std::unique_ptr<vkComputeBase> mortonKernel;
        std::unique_ptr<vkComputeBase> histogramKernel;
        std::unique_ptr<vkComputeBase> prefixScanKernel;
        std::unique_ptr<vkComputeBase> reorderKernel;
        std::unique_ptr<vkComputeBase> hierarchyKernel;
        std::unique_ptr<vkComputeBase> boundingBoxKernel;
        std::unique_ptr<vkComputeBase> rangeInitKernel;
        std::unique_ptr<vkComputeBase> rangeKernel;
        std::unique_ptr<vkComputeBase> wideInitKernel;
        std::unique_ptr<vkComputeBase> wideBuildKernel;
        std::unique_ptr<vkComputeBase> radiusKernel;
        std::unique_ptr<vkComputeBase> knnKernel;
        std::unique_ptr<vkComputeBase> rayTraceKernel;
        std::unique_ptr<vkComputeBase> pathTraceKernel;
    };

    vkWideBVH::vkWideBVH(
            VkContext *ctx, uint32_t maxLeafPrimitives)
        : m_impl(std::make_unique<Impl>(ctx, maxLeafPrimitives)) {}

    vkWideBVH::~vkWideBVH() = default;

    void vkWideBVH::buildWideBVH(
            const std::vector<Primitive> &primitives) {
        if (primitives.empty())
            throw std::runtime_error(
                    "vkWideBVH: need at least one primitive");
        if (primitives.size() > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error(
                    "vkWideBVH: primitive count exceeds uint32");

        for (const Primitive &primitive: primitives)
            validatePrimitive(primitive);

        const uint32_t count = static_cast<uint32_t>(primitives.size());
        const uint64_t binaryNodeCount64 = static_cast<uint64_t>(count) * 2u - 1u;
        if (binaryNodeCount64 > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("vkWideBVH: binary node count exceeds uint32");

        const uint32_t binaryNodeCount = static_cast<uint32_t>(binaryNodeCount64);
        const uint32_t radixWorkgroupCount = (count + RADIX_WORKGROUP_SIZE - 1u) / RADIX_WORKGROUP_SIZE;

        auto primitiveBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(count, sizeof(Primitive), "primitiveBuffer"),
                "primitiveBuffer");
        auto mortonBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(count, sizeof(MortonCode), "mortonBuffer"),
                "mortonBuffer");
        auto mortonPingBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(count, sizeof(MortonCode), "mortonPingBuffer"),
                "mortonPingBuffer");
        auto histogramBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(
                        static_cast<uint64_t>(RADIX) * radixWorkgroupCount,
                        sizeof(uint32_t),
                        "histogramBuffer"),
                "histogramBuffer");
        auto sceneBoundsBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(6, sizeof(uint32_t), "sceneBoundsBuffer"),
                "sceneBoundsBuffer");
        auto binaryNodeBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(binaryNodeCount, 36, "binaryNodeBuffer"),
                "binaryNodeBuffer");
        auto constructionBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(binaryNodeCount, 8, "constructionBuffer"),
                "constructionBuffer");
        auto rangeBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(
                        binaryNodeCount,
                        sizeof(BinaryRange),
                        "rangeBuffer"),
                "rangeBuffer");
        auto queueBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(
                        binaryNodeCount, sizeof(uint32_t), "queueBuffer"),
                "queueBuffer");
        auto buildStateBuffer = makeBuffer(
                m_impl->ctx,
                sizeof(WideBuildState),
                "buildStateBuffer");
        auto wideNodeBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(
                        binaryNodeCount,
                        sizeof(QuantizedWideNode),
                        "wideNodeBuffer"),
                "wideNodeBuffer");
        auto leafBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(
                        binaryNodeCount,
                        sizeof(LeafRange),
                        "leafBuffer"),
                "leafBuffer");

        if (!primitiveBuffer->Upload(
                    primitives.data(),
                    checkedBytes(count, sizeof(Primitive), "primitiveBuffer"),
                    m_impl->ctx->computeQueue,
                    m_impl->ctx->cmdPool))
            throw std::runtime_error("vkWideBVH: primitive upload failed");

        m_impl->ensureBuildKernels();

        m_impl->boundsInitKernel
                ->Bind(0, *sceneBoundsBuffer)
                .Dispatch(1);

        const CountPC countPC{count};
        m_impl->boundsReduceKernel
                ->Bind(0, *primitiveBuffer)
                .Bind(1, *sceneBoundsBuffer)
                .Args(countPC)
                .DispatchElements(count);

        m_impl->mortonKernel
                ->Bind(0, *mortonBuffer)
                .Bind(1, *primitiveBuffer)
                .Bind(2, *sceneBoundsBuffer)
                .Args(countPC)
                .DispatchElements(count);

        vkGPUMemory *radixInput = mortonBuffer.get();
        vkGPUMemory *radixOutput = mortonPingBuffer.get();
        for (uint32_t pass = 0; pass < RADIX_PASSES; ++pass) {
            const RadixSortPC sortPC{count, pass * 4u};
            const RadixScanPC scanPC{radixWorkgroupCount};

            m_impl->histogramKernel
                    ->Bind(0, *radixInput)
                    .Bind(1, *histogramBuffer)
                    .Args(sortPC)
                    .Dispatch(radixWorkgroupCount);

            m_impl->prefixScanKernel
                    ->Bind(0, *histogramBuffer)
                    .Args(scanPC)
                    .Dispatch(1);

            m_impl->reorderKernel
                    ->Bind(0, *radixInput)
                    .Bind(1, *histogramBuffer)
                    .Bind(2, *radixOutput)
                    .Args(sortPC)
                    .Dispatch(radixWorkgroupCount);

            std::swap(radixInput, radixOutput);
        }

        const HierarchyPC hierarchyPC{count, 1u};
        m_impl->hierarchyKernel
                ->Bind(0, *radixInput)
                .Bind(1, *primitiveBuffer)
                .Bind(2, *binaryNodeBuffer)
                .Bind(3, *constructionBuffer)
                .Args(hierarchyPC)
                .DispatchElements(count);

        m_impl->boundingBoxKernel
                ->Bind(0, *binaryNodeBuffer)
                .Bind(1, *constructionBuffer)
                .Args(hierarchyPC)
                .DispatchElements(count);

        const CountPC binaryNodeCountPC{binaryNodeCount};
        m_impl->rangeInitKernel
                ->Bind(0, *rangeBuffer)
                .Args(binaryNodeCountPC)
                .DispatchElements(binaryNodeCount);

        m_impl->rangeKernel
                ->Bind(0, *binaryNodeBuffer)
                .Bind(1, *constructionBuffer)
                .Bind(2, *rangeBuffer)
                .Args(countPC)
                .DispatchElements(count);

        m_impl->wideInitKernel
                ->Bind(0, *queueBuffer)
                .Bind(1, *buildStateBuffer)
                .Dispatch(1);

        const WideBuildPC wideBuildPC{
                binaryNodeCount,
                m_impl->maxLeafPrimitives};
        m_impl->wideBuildKernel
                ->Bind(0, *binaryNodeBuffer)
                .Bind(1, *rangeBuffer)
                .Bind(2, *queueBuffer)
                .Bind(3, *buildStateBuffer)
                .Bind(4, *wideNodeBuffer)
                .Bind(5, *leafBuffer)
                .Args(wideBuildPC)
                .Dispatch(1);

        WideBuildState state{};
        if (!buildStateBuffer->Download(
                    &state,
                    sizeof(state),
                    m_impl->ctx->computeQueue,
                    m_impl->ctx->cmdPool))
            throw std::runtime_error(
                    "vkWideBVH: build state download failed");
        if (state.status != 0u)
            throw std::runtime_error(
                    "vkWideBVH: GPU wide collapse exceeded buffer capacity");
        if (state.nodeCount == 0u ||
            state.nodeCount > binaryNodeCount ||
            state.leafCount == 0u ||
            state.leafCount > binaryNodeCount)
            throw std::runtime_error(
                    "vkWideBVH: GPU wide collapse produced invalid counts");

        std::unique_ptr<vkGPUMemory> sortedMortonBuffer;
        if (radixInput == mortonBuffer.get())
            sortedMortonBuffer = std::move(mortonBuffer);
        else
            sortedMortonBuffer = std::move(mortonPingBuffer);

        m_impl->primitiveBuffer = std::move(primitiveBuffer);
        m_impl->sortedMortonBuffer = std::move(sortedMortonBuffer);
        m_impl->nodeBuffer = std::move(wideNodeBuffer);
        m_impl->leafBuffer = std::move(leafBuffer);
        m_impl->count = count;
        m_impl->nodeCount = state.nodeCount;
        m_impl->built = true;
    }

    std::vector<uint32_t> vkWideBVH::RadiusSearch(
            float cx, float cy, float cz, float radius) {
        if (!m_impl->built)
            throw std::runtime_error(
                    "vkWideBVH: Build() must be called first");
        if (!std::isfinite(cx) ||
            !std::isfinite(cy) ||
            !std::isfinite(cz) ||
            !std::isfinite(radius) ||
            radius < 0.0f)
            throw std::runtime_error(
                    "vkWideBVH: RadiusSearch arguments are invalid");

        auto resultBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(
                        m_impl->count,
                        sizeof(uint32_t),
                        "radiusResultBuffer"),
                "radiusResultBuffer");
        auto stateBuffer = makeBuffer(
                m_impl->ctx,
                sizeof(QueryState),
                "radiusStateBuffer");

        m_impl->ensureRadiusKernel();
        const RadiusPC pc{
                cx, cy, cz, radius, m_impl->count};
        m_impl->radiusKernel
                ->Bind(0, *m_impl->nodeBuffer)
                .Bind(1, *m_impl->leafBuffer)
                .Bind(2, *m_impl->sortedMortonBuffer)
                .Bind(3, *m_impl->primitiveBuffer)
                .Bind(4, *resultBuffer)
                .Bind(5, *stateBuffer)
                .Args(pc)
                .Dispatch(1);

        QueryState state{};
        if (!stateBuffer->Download(
                    &state,
                    sizeof(state),
                    m_impl->ctx->computeQueue,
                    m_impl->ctx->cmdPool))
            throw std::runtime_error(
                    "vkWideBVH: RadiusSearch state download failed");
        if (state.status != 0u)
            throw std::runtime_error(
                    "vkWideBVH: RadiusSearch traversal stack overflow");
        if (state.count > m_impl->count)
            throw std::runtime_error(
                    "vkWideBVH: RadiusSearch result count is corrupt");
        if (state.count == 0u)
            return {};

        std::vector<uint32_t> results(state.count);
        if (!resultBuffer->Download(
                    results.data(),
                    checkedBytes(
                            results.size(),
                            sizeof(uint32_t),
                            "radiusResults"),
                    m_impl->ctx->computeQueue,
                    m_impl->ctx->cmdPool))
            throw std::runtime_error(
                    "vkWideBVH: RadiusSearch result download failed");
        return results;
    }

    std::vector<uint32_t> vkWideBVH::KNN(
            float cx, float cy, float cz, int k) {
        if (!m_impl->built)
            throw std::runtime_error(
                    "vkWideBVH: Build() must be called first");
        if (!std::isfinite(cx) ||
            !std::isfinite(cy) ||
            !std::isfinite(cz))
            throw std::runtime_error(
                    "vkWideBVH: KNN arguments are invalid");
        if (k <= 0 || static_cast<uint32_t>(k) > MAX_K)
            throw std::runtime_error(
                    "vkWideBVH: KNN k must be in [1, 64]");

        const uint32_t queryK = static_cast<uint32_t>(k);
        auto resultBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(
                        queryK, sizeof(uint32_t), "knnResultBuffer"),
                "knnResultBuffer");
        auto distanceBuffer = makeBuffer(
                m_impl->ctx,
                checkedBytes(
                        queryK, sizeof(float), "knnDistanceBuffer"),
                "knnDistanceBuffer");
        auto stateBuffer = makeBuffer(
                m_impl->ctx,
                sizeof(QueryState),
                "knnStateBuffer");

        m_impl->ensureKNNKernel();
        const KNNPC pc{cx, cy, cz, queryK};
        m_impl->knnKernel
                ->Bind(0, *m_impl->nodeBuffer)
                .Bind(1, *m_impl->leafBuffer)
                .Bind(2, *m_impl->sortedMortonBuffer)
                .Bind(3, *m_impl->primitiveBuffer)
                .Bind(4, *resultBuffer)
                .Bind(5, *distanceBuffer)
                .Bind(6, *stateBuffer)
                .Args(pc)
                .Dispatch(1);

        QueryState state{};
        if (!stateBuffer->Download(
                    &state,
                    sizeof(state),
                    m_impl->ctx->computeQueue,
                    m_impl->ctx->cmdPool))
            throw std::runtime_error(
                    "vkWideBVH: KNN state download failed");
        if (state.status != 0u)
            throw std::runtime_error(
                    "vkWideBVH: KNN traversal stack overflow");

        std::vector<uint32_t> indices(queryK);
        if (!resultBuffer->Download(
                    indices.data(),
                    checkedBytes(
                            indices.size(),
                            sizeof(uint32_t),
                            "knnResults"),
                    m_impl->ctx->computeQueue,
                    m_impl->ctx->cmdPool))
            throw std::runtime_error(
                    "vkWideBVH: KNN result download failed");

        constexpr uint32_t INVALID_INDEX = 0xFFFFFFFFu;
        indices.erase(
                std::remove(
                        indices.begin(),
                        indices.end(),
                        INVALID_INDEX),
                indices.end());
        return indices;
    }

    uint32_t vkWideBVH::Length() const {
        return m_impl->count;
    }

    uint32_t vkWideBVH::NodeCount() const {
        return m_impl->nodeCount;
    }

    uint32_t vkWideBVH::MaxLeafPrimitives() const {
        return m_impl->maxLeafPrimitives;
    }

    vkGPUMemory &vkWideBVH::NodeBuffer() const {
        if (!m_impl->built)
            throw std::runtime_error("vkWideBVH: Build() must be called first");
        return *m_impl->nodeBuffer;
    }

    vkGPUMemory &vkWideBVH::LeafBuffer() const {
        if (!m_impl->built)
            throw std::runtime_error("vkWideBVH: Build() must be called first");
        return *m_impl->leafBuffer;
    }

    vkGPUMemory &vkWideBVH::SortedMortonBuffer() const {
        if (!m_impl->built)
            throw std::runtime_error("vkWideBVH: Build() must be called first");
        return *m_impl->sortedMortonBuffer;
    }

    vkGPUMemory &vkWideBVH::PrimitiveBuffer() const {
        if (!m_impl->built)
            throw std::runtime_error("vkWideBVH: Build() must be called first");
        return *m_impl->primitiveBuffer;
    }

    void vkWideBVH::TraceRays(
            const RayTraceCamera &camera,
            uint32_t width, uint32_t height,
            float tMin, float tMax, uint32_t shadeMode,
            const RayTraceLighting &lighting,
            bool outputSwapRedBlue,
            vkGPUMemory &triangleVertexBuffer,
            vkGPUMemory &outputPixelBuffer) {
        if (!m_impl->built)
            throw std::runtime_error("vkWideBVH: Build() must be called first");
        if (width == 0 || height == 0)
            throw std::runtime_error("vkWideBVH: TraceRays width/height must be > 0");

        m_impl->ensureRayTraceKernel();

        const RayTracePC pc{
                camera.originX, camera.originY, camera.originZ,
                camera.rightX, camera.rightY, camera.rightZ,
                camera.upX, camera.upY, camera.upZ,
                camera.forwardX, camera.forwardY, camera.forwardZ,
                camera.tanHalfFovY, camera.aspect,
                width, height, shadeMode,
                tMin, tMax,
                lighting.lightDirX, lighting.lightDirY, lighting.lightDirZ,
                lighting.groundStartIndex,
                outputSwapRedBlue ? 1u : 0u,
                lighting.lightPosX, lighting.lightPosY, lighting.lightPosZ,
                lighting.lightType};

        m_impl->rayTraceKernel
                ->Bind(0, *m_impl->nodeBuffer)
                .Bind(1, *m_impl->leafBuffer)
                .Bind(2, *m_impl->sortedMortonBuffer)
                .Bind(3, *m_impl->primitiveBuffer)
                .Bind(4, triangleVertexBuffer)
                .Bind(5, outputPixelBuffer)
                .Args(pc)
                .Dispatch((width + 7u) / 8u, (height + 7u) / 8u, 1u);
    }

    void vkWideBVH::TracePath(
            const RayTraceCamera &camera,
            uint32_t width, uint32_t height,
            float tMin, float tMax,
            uint32_t frameIndex,
            uint32_t maxBounces,
            uint32_t samplesPerPixel,
            const PathTraceLight &light,
            bool outputSwapRedBlue,
            vkGPUMemory &triangleVertexBuffer,
            vkGPUMemory &materialIdBuffer,
            vkGPUMemory &materialBuffer,
            vkGPUMemory &accumulationBuffer,
            vkGPUMemory &outputPixelBuffer) {
        if (!m_impl->built)
            throw std::runtime_error("vkWideBVH: Build() must be called first");
        if (width == 0 || height == 0)
            throw std::runtime_error("vkWideBVH: TracePath width/height must be > 0");
        if (maxBounces == 0)
            throw std::runtime_error("vkWideBVH: TracePath maxBounces must be > 0");
        if (samplesPerPixel == 0)
            throw std::runtime_error("vkWideBVH: TracePath samplesPerPixel must be > 0");

        m_impl->ensurePathTraceKernel();

        const PathTracePC pc{
                camera.originX, camera.originY, camera.originZ,
                camera.rightX, camera.rightY, camera.rightZ,
                camera.upX, camera.upY, camera.upZ,
                camera.forwardX, camera.forwardY, camera.forwardZ,
                camera.tanHalfFovY, camera.aspect,
                width, height, frameIndex, maxBounces,
                samplesPerPixel, outputSwapRedBlue ? 1u : 0u,
                tMin, tMax,
                light.centerX, light.centerY, light.centerZ,
                light.environmentStrength,
                light.uX, light.uY, light.uZ, light.pad0,
                light.vX, light.vY, light.vZ, light.pad1,
                light.emissionX, light.emissionY, light.emissionZ, light.pad2};

        m_impl->pathTraceKernel
                ->Bind(0, *m_impl->nodeBuffer)
                .Bind(1, *m_impl->leafBuffer)
                .Bind(2, *m_impl->sortedMortonBuffer)
                .Bind(3, *m_impl->primitiveBuffer)
                .Bind(4, triangleVertexBuffer)
                .Bind(5, materialIdBuffer)
                .Bind(6, materialBuffer)
                .Bind(7, accumulationBuffer)
                .Bind(8, outputPixelBuffer)
                .Args(pc)
                .Dispatch((width + 7u) / 8u, (height + 7u) / 8u, 1u);
    }

} // namespace vkSpatial

#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/SpatialIndex.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Engine::Core {
    class ComputePipeline;
}

namespace Engine::Spatial {

    // N-wide BVH: builds a binary LBVH, then GPU-collapses it into a quantized wide
    // hierarchy (up to maxLeafPrimitives per leaf). Query kernels cached at Build().
    // Reuses src/shader/bvh_wide_*.comp + cmd_knn_wide/cmd_radiusSearch_wide.
    // Ray/path tracing is NOT implemented (future RayTraceable capability).
    // KNOWN ISSUE (this HW): RadiusSearch returns empty on Apple/MoltenVK
    // (cmd_radiusSearch_wide.comp); build/KNN/memory are unaffected.
    class WideBVH : public SpatialIndex {
    public:
        explicit WideBVH(Engine::Core::Context &ctx, uint32_t maxLeafPrimitives = 4);
        ~WideBVH() override;

        std::vector<uint32_t> RadiusSearch(float cx, float cy, float cz, float r) override;
        std::vector<uint32_t> KNN(float cx, float cy, float cz, int k) override;

        uint32_t Length() const override { return m_count; }
        uint32_t NodeCount() const override { return m_nodeCount; }
        uint32_t MemoryBytes() const override;
        const char *Name() const override { return m_name.c_str(); }
        uint32_t MaxLeafPrimitives() const { return m_maxLeafPrimitives; }

    protected:
        void BuildFromPrimitives(const std::vector<Primitive> &prims) override;

    private:
        void setupQueryKernels();

        Engine::Core::Context *m_ctx = nullptr;
        uint32_t m_maxLeafPrimitives = 4;
        std::string m_name;
        uint32_t m_count = 0;
        uint32_t m_nodeCount = 0;
        uint32_t m_leafCount = 0;
        bool m_built = false;

        std::unique_ptr<Engine::Core::Buffer> m_primitiveBuf;  // Primitive[]
        std::unique_ptr<Engine::Core::Buffer> m_sortedMortonBuf;
        std::unique_ptr<Engine::Core::Buffer> m_nodeBuf; // QuantizedWideNode[]
        std::unique_ptr<Engine::Core::Buffer> m_leafBuf; // LeafRange[]

        std::unique_ptr<Engine::Core::ComputePipeline> m_radiusKernel;
        std::unique_ptr<Engine::Core::ComputePipeline> m_knnKernel;
        std::unique_ptr<Engine::Core::Buffer> m_radiusResultBuf; // uint[N]
        std::unique_ptr<Engine::Core::Buffer> m_radiusStateBuf;  // QueryState
        std::unique_ptr<Engine::Core::Buffer> m_knnResultBuf;    // uint[MAX_K]
        std::unique_ptr<Engine::Core::Buffer> m_knnDistBuf;      // float[MAX_K]
        std::unique_ptr<Engine::Core::Buffer> m_knnStateBuf;     // QueryState
    };

} // namespace Engine::Spatial

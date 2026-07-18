#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/SpatialIndex.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Core {
    class ComputePipeline;
}

namespace Engine::Spatial {

    // Binary LBVH (Karras-style: Morton codes -> radix sort -> hierarchy -> bounds),
    // built on Engine::Core. Query kernels are created once at Build() and re-dispatched
    // per query (cache pattern). Reuses src/shader/bvh_*.comp + cmd_knn/cmd_radiusSearch.
    class BinaryLBVH : public SpatialIndex {
    public:
        explicit BinaryLBVH(Engine::Core::Context &ctx);
        ~BinaryLBVH() override;

        std::vector<uint32_t> RadiusSearch(float cx, float cy, float cz, float r) override;
        std::vector<uint32_t> KNN(float cx, float cy, float cz, int k) override;

        uint32_t Length() const override { return m_count; }
        uint32_t NodeCount() const override { return m_count ? (2u * m_count - 1u) : 0u; }
        uint32_t MemoryBytes() const override;
        const char *Name() const override { return "BinaryLBVH"; }

    protected:
        void BuildFromPrimitives(const std::vector<Primitive> &prims) override;

    private:
        void stepSortMortonCodes();
        void setupQueryKernels();

        Engine::Core::Context *m_ctx = nullptr;
        uint32_t m_count = 0;
        bool m_built = false;

        std::unique_ptr<Engine::Core::Buffer> m_primBuf;
        std::unique_ptr<Engine::Core::Buffer> m_mortonBuf;     // sorted result ends here
        std::unique_ptr<Engine::Core::Buffer> m_mortonPingBuf; // radix pong
        std::unique_ptr<Engine::Core::Buffer> m_histBuf;
        std::unique_ptr<Engine::Core::Buffer> m_nodeBuf;
        std::unique_ptr<Engine::Core::Buffer> m_constructionBuf;

        std::unique_ptr<Engine::Core::ComputePipeline> m_knnKernel;
        std::unique_ptr<Engine::Core::ComputePipeline> m_radiusKernel;
        std::unique_ptr<Engine::Core::Buffer> m_knnResultBuf;    // uint[MAX_K]
        std::unique_ptr<Engine::Core::Buffer> m_knnDistBuf;      // float[MAX_K]
        std::unique_ptr<Engine::Core::Buffer> m_radiusResultBuf; // uint[N]
        std::unique_ptr<Engine::Core::Buffer> m_radiusCountBuf;  // uint[1]
    };

} // namespace Engine::Spatial

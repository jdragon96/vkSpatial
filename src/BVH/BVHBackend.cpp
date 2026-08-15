#include "BVH/BVHBackend.h"

#include "BVH/BinaryLBVH/BinaryLBVH.h"
#include "BVH/WideBVH/WideBVH.h"
#include "Engine/Core/Context.h"

namespace {

    using Engine::Spatial::Primitive;

    // Both builders take the device in their constructor, so the backend holds them by pointer and
    // creates one on Build.
    template<typename Concrete>
    class ConcreteBackend : public BVHBackend {
    public:
        void BuildFromPrimitives(const std::vector<Primitive> &primitives) override {
            if (m_index) m_index->BuildFromPrimitives(primitives);
        }

        std::vector<uint32_t> RadiusSearch(float x, float y, float z, float radius) override {
            return m_index ? m_index->RadiusSearch(x, y, z, radius) : std::vector<uint32_t>{};
        }

        std::vector<uint32_t> KNN(float x, float y, float z, int k) override {
            return m_index ? m_index->KNN(x, y, z, k) : std::vector<uint32_t>{};
        }

        BVHBackendStats Stats() const override {
            if (!m_index) return {};
            return {m_index->Length(), m_index->NodeCount(), m_index->MemoryBytes()};
        }

        const char *Name() const override { return m_index ? m_index->Name() : Concrete::kName; }

    protected:
        std::unique_ptr<Concrete> m_index;
    };

    class BinaryBackend final : public ConcreteBackend<Engine::Spatial::BinaryLBVH> {
    public:
        void Build(Engine::Core::Context &context, const BVHBackendConfig &) override {
            m_index = std::make_unique<Engine::Spatial::BinaryLBVH>(context);
        }
    };

    class WideBackend final : public ConcreteBackend<Engine::Spatial::WideBVH> {
    public:
        void Build(Engine::Core::Context &context, const BVHBackendConfig &config) override {
            m_index = std::make_unique<Engine::Spatial::WideBVH>(context, config.maxLeafPrimitives);
        }
    };

} // namespace

std::unique_ptr<BVHBackend> MakeBVHBackend(const std::string &name) {
    if (name == "binary") return std::make_unique<BinaryBackend>();
    if (name == "wide") return std::make_unique<WideBackend>();
    return nullptr;
}

std::vector<std::string> BVHBackendNames() {
    return {"binary", "wide"};
}

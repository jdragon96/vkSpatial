#pragma once

#include <memory>
#include <string>
#include <vector>

#include "BVH/BVHBackend.h"
#include "Engine/Core/Context.h"

struct BVHConfiguration {
    // Switchable by name. See BVHBackendNames().
    std::string backend = "binary";

    BVHBackendConfig backendConfig{};
};

// An acceleration structure over a primitive set, with the structure itself swappable.
class BVH {
public:
    void Build(Engine::Core::Context &context, BVHConfiguration config);

    // T is Primitive, PointPrim, TrianglePrim, or any type with a PrimitiveConverter<T>.
    template<typename T>
    void Insert(const std::vector<T> &primitives) {
        if (!m_backend) return;
        std::vector<Engine::Spatial::Primitive> converted;
        converted.reserve(primitives.size());
        for (uint32_t i = 0; i < uint32_t(primitives.size()); ++i)
            converted.push_back(Engine::Spatial::PrimitiveConverter<T>::convert(primitives[i], i));
        m_backend->BuildFromPrimitives(converted);
    }

    std::vector<uint32_t> RadiusSearch(float x, float y, float z, float radius) {
        return m_backend ? m_backend->RadiusSearch(x, y, z, radius) : std::vector<uint32_t>{};
    }

    std::vector<uint32_t> KNN(float x, float y, float z, int k) {
        return m_backend ? m_backend->KNN(x, y, z, k) : std::vector<uint32_t>{};
    }

    BVHBackendStats Stats() const { return m_backend ? m_backend->Stats() : BVHBackendStats{}; }

    const char *BackendName() const { return m_backend ? m_backend->Name() : "none"; }

    const BVHConfiguration &Config() const { return m_config; }

private:
    Engine::Core::Context *m_context = nullptr;
    BVHConfiguration m_config;
    std::unique_ptr<BVHBackend> m_backend;
};

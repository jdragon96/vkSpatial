#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "BVH/BVHTypes.h"

namespace Engine::Core {
    class Context;
}

struct BVHBackendConfig {
    // Wide only; BinaryLBVH ignores it.
    uint32_t maxLeafPrimitives = 4;
};

struct BVHBackendStats {
    uint32_t primitiveCount = 0;
    uint32_t nodeCount = 0;
    uint32_t memoryBytes = 0;
};

// One acceleration structure over a primitive set. Implementations live entirely in
// BVHBackend.cpp and wrap the concrete builders in namespace Engine::Spatial.
class BVHBackend {
public:
    virtual ~BVHBackend() = default;

    virtual void Build(Engine::Core::Context &context, const BVHBackendConfig &config) = 0;

    virtual void BuildFromPrimitives(const std::vector<Engine::Spatial::Primitive> &primitives) = 0;

    // Primitives whose centre lies within `radius` of the query point. Unordered.
    virtual std::vector<uint32_t> RadiusSearch(float x, float y, float z, float radius) = 0;

    // The k nearest primitives; k in [1, 64]. The correct SET, in unspecified order -- compare as
    // a set, never by position.
    virtual std::vector<uint32_t> KNN(float x, float y, float z, int k) = 0;

    virtual BVHBackendStats Stats() const = 0;

    virtual const char *Name() const = 0;
};

std::unique_ptr<BVHBackend> MakeBVHBackend(const std::string &name);

std::vector<std::string> BVHBackendNames();

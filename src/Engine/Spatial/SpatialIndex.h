#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Spatial/BVHTypes.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Spatial {

    // Common swappable surface every acceleration structure implements.
    class SpatialIndex {
    public:
        virtual ~SpatialIndex() = default;

        SpatialIndex(const SpatialIndex &) = delete;
        SpatialIndex &operator=(const SpatialIndex &) = delete;

        // Ergonomic build. T is Primitive, PointPrim, TrianglePrim, or any type with a
        // PrimitiveConverter<T> specialisation. Converts to Primitive then delegates to
        // the virtual BuildFromPrimitives so every backend shares this entry point.
        template<typename T>
        void Build(const std::vector<T> &primitives) {
            std::vector<Primitive> prims;
            prims.reserve(primitives.size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(primitives.size()); ++i)
                prims.push_back(PrimitiveConverter<T>::convert(primitives[i], i));
            BuildFromPrimitives(prims);
        }

        // Indices of primitives whose centre lies within radius r of (cx,cy,cz). Unordered.
        virtual std::vector<uint32_t> RadiusSearch(float cx, float cy, float cz, float r) = 0;
        // The k nearest primitive indices to (cx,cy,cz); k in [1,64]. Returns the correct
        // k-nearest SET, but the result ORDER is unspecified (backends may return heap or
        // traversal order, not sorted by distance). Compare as a set, not by position.
        virtual std::vector<uint32_t> KNN(float cx, float cy, float cz, int k) = 0;

        virtual uint32_t Length() const = 0;       // primitive count
        virtual uint32_t NodeCount() const = 0;    // structural node count
        virtual uint32_t MemoryBytes() const = 0;  // structural GPU buffer bytes
        virtual const char *Name() const = 0;      // benchmark label

    protected:
        SpatialIndex() = default;
        virtual void BuildFromPrimitives(const std::vector<Primitive> &prims) = 0;
    };

    // Optional capability — only algorithms that can trace rays implement it.
    // Declared-only this pass; WideBVH will inherit + implement it in future work.
    class RayTraceable {
    public:
        virtual ~RayTraceable() = default;
        // TODO(future): TraceRays / TracePath. See old vkSpatial::vkWideBVH for signatures.
    };

    enum class BVHKind {
        BinaryLBVH,
        Wide,
    };

    struct BVHParams {
        uint32_t maxLeafPrimitives = 4; // Wide only; ignored by BinaryLBVH.
    };

    std::unique_ptr<SpatialIndex>
    MakeSpatialIndex(Engine::Core::Context &ctx, BVHKind kind, const BVHParams &params = {});

} // namespace Engine::Spatial

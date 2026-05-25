#pragma once

#include "vkBVH/common/vkContext.h"
#include "vkBVH/common/vkGPUMemory.h"
#include "vkBVH/types.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
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
} // namespace


struct PointPrim {
    float x, y, z;
};

struct TrianglePrim {
    float v0[3], v1[3], v2[3];
};

template<typename T>
struct PrimitiveConverter;

template<>
struct PrimitiveConverter<Primitive> {
    static Primitive convert(const Primitive &p, uint32_t) { return p; }
};

template<>
struct PrimitiveConverter<PointPrim> {
    static Primitive convert(const PointPrim &p, uint32_t idx) {
        return Primitive{idx, p.x, p.y, p.z, p.x, p.y, p.z};
    }
};

template<>
struct PrimitiveConverter<TrianglePrim> {
    static Primitive convert(const TrianglePrim &t, uint32_t idx) {
        float minX = std::min({t.v0[0], t.v1[0], t.v2[0]});
        float minY = std::min({t.v0[1], t.v1[1], t.v2[1]});
        float minZ = std::min({t.v0[2], t.v1[2], t.v2[2]});
        float maxX = std::max({t.v0[0], t.v1[0], t.v2[0]});
        float maxY = std::max({t.v0[1], t.v1[1], t.v2[1]});
        float maxZ = std::max({t.v0[2], t.v1[2], t.v2[2]});
        return Primitive{idx, minX, minY, minZ, maxX, maxY, maxZ};
    }
};

class vkBVH {
public:
    explicit vkBVH(VkContext *ctx, const std::string &shaderDir);

    ~vkBVH();

    template<typename T>
    void Build(const std::vector<T> &primitives) {
        std::vector<Primitive> prims;
        prims.reserve(primitives.size());
        for (uint32_t i = 0; i < static_cast<uint32_t>(primitives.size()); i++)
            prims.push_back(PrimitiveConverter<T>::convert(primitives[i], i));
        buildBVH(prims);
    }

    std::vector<uint32_t> RadiusSearch(float cx, float cy, float cz, float r);

    std::vector<uint32_t> KNN(float cx, float cy, float cz, int k);

    inline uint32_t Length() { return m_count; }

private:
    void buildBVH(const std::vector<Primitive> &prims);
    void stepAllocateBuffers();
    void stepUploadPrimitives(const std::vector<Primitive> &prims);
    void stepComputeMortonCodes(const std::vector<Primitive> &prims);
    void stepSortMortonCodes(); // 8 pass radix sort
    void stepBuildHierarchy();
    void stepComputeBoundingBoxes();

    VkContext *m_ctx = nullptr;

    std::unique_ptr<vkGPUMemory> m_primBuf;         // Primitive[]
    std::unique_ptr<vkGPUMemory> m_mortonBuf;       // MortonCode[] (ping)
    std::unique_ptr<vkGPUMemory> m_mortonPingBuf;   // MortonCode[] (pong)
    std::unique_ptr<vkGPUMemory> m_histBuf;         // radix sort histogram
    std::unique_ptr<vkGPUMemory> m_nodeBuf;         // Node[] (BVH output)
    std::unique_ptr<vkGPUMemory> m_constructionBuf; // LBVHConstructionInfo[]

    std::string m_shaderDir;

    uint32_t m_count = 0;

    bool m_built = false;
};

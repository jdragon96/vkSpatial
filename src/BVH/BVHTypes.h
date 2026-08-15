#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace Engine::Spatial {

    // GPU-shared primitive (AABB + source index). Byte layout matches the GLSL std430
    // structs in src/shader/bvh_*.comp — do not reorder without updating the shaders.
    struct Primitive {
        uint32_t index;
        float aabbMinX, aabbMinY, aabbMinZ;
        float aabbMaxX, aabbMaxY, aabbMaxZ;

        Eigen::Vector3f GetCenter() const {
            return Eigen::Vector3f((aabbMaxX + aabbMinX) * 0.5f,
                                   (aabbMaxY + aabbMinY) * 0.5f,
                                   (aabbMaxZ + aabbMinZ) * 0.5f);
        }

        static std::vector<Primitive> RandomPoints(int numberOfPoints) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
            std::vector<Primitive> points(numberOfPoints);
            for (auto &point : points) {
                point.aabbMinX = dist(gen);
                point.aabbMinY = dist(gen);
                point.aabbMinZ = dist(gen);
                point.aabbMaxX = dist(gen);
                point.aabbMaxY = dist(gen);
                point.aabbMaxZ = dist(gen);
            }
            return points;
        }
    };

    struct MortonCode {
        uint32_t code;
        uint32_t index;
    };

    // Scene-bounds accumulator passed to bvh_mortonCode.comp as a push constant.
    struct MortonConstant {
        uint32_t g_numberOfPrimitive = 0;
        float g_minX = std::numeric_limits<float>::max();
        float g_minY = std::numeric_limits<float>::max();
        float g_minZ = std::numeric_limits<float>::max();
        float g_maxX = std::numeric_limits<float>::lowest();
        float g_maxY = std::numeric_limits<float>::lowest();
        float g_maxZ = std::numeric_limits<float>::lowest();

        void Extend(const std::vector<Primitive> &primitives) {
            for (const auto &prim : primitives) Extend(prim);
        }

        void Extend(const Primitive &prim) {
            g_minX = std::min(g_minX, prim.aabbMinX);
            g_maxX = std::max(g_maxX, prim.aabbMaxX);
            g_minY = std::min(g_minY, prim.aabbMinY);
            g_maxY = std::max(g_maxY, prim.aabbMaxY);
            g_minZ = std::min(g_minZ, prim.aabbMinZ);
            g_maxZ = std::max(g_maxZ, prim.aabbMaxZ);
            g_numberOfPrimitive++;
        }
    };

    // ── Convenience primitive inputs + converters to Primitive ────────────────────

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

    // ── GLSL std430 layout verification (offsetof needs standard-layout) ──────────

    static_assert(std::is_standard_layout_v<Primitive>, "Primitive must be standard-layout");
    static_assert(std::is_standard_layout_v<MortonCode>, "MortonCode must be standard-layout");
    static_assert(std::is_standard_layout_v<MortonConstant>, "MortonConstant must be standard-layout");

    static_assert(sizeof(Primitive) == 28, "Primitive size mismatch with GLSL");
    static_assert(offsetof(Primitive, index) == 0);
    static_assert(offsetof(Primitive, aabbMinX) == 4);
    static_assert(offsetof(Primitive, aabbMaxZ) == 24);

    static_assert(sizeof(MortonCode) == 8, "MortonCode size mismatch with GLSL");
    static_assert(offsetof(MortonCode, index) == 4);

    static_assert(sizeof(MortonConstant) == 28, "MortonConstant size mismatch");
    static_assert(offsetof(MortonConstant, g_minX) == 4);
    static_assert(offsetof(MortonConstant, g_maxZ) == 24);

} // namespace Engine::Spatial

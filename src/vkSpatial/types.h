#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>

#include <Eigen/Core>

namespace vkCommon {

    struct Primitive {
        uint32_t index;
        float aabbMinX, aabbMinY, aabbMinZ;
        float aabbMaxX, aabbMaxY, aabbMaxZ;

        Eigen::Vector3f GetCenter() {
            float x = (aabbMaxX + aabbMinX) * 0.5f;
            float y = (aabbMaxY + aabbMinY) * 0.5f;
            float z = (aabbMaxZ + aabbMinZ) * 0.5f;
            return Eigen::Vector3f(x, y, z);
        }

        static std::vector<Primitive> RadomPoint(int numberOfPoint) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
            std::vector<Primitive> points(numberOfPoint);
            for (auto &point: points) {
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

    struct MortonConstant {
        uint32_t g_numberOfPrimitive = 0;
        float g_minX = std::numeric_limits<float>::max();
        float g_minY = std::numeric_limits<float>::max();
        float g_minZ = std::numeric_limits<float>::max();
        float g_maxX = std::numeric_limits<float>::max();
        float g_maxY = std::numeric_limits<float>::max();
        float g_maxZ = std::numeric_limits<float>::max();

        void Extend(const std::vector<Primitive> &primitives) {
            for (auto &prim: primitives) {
                Extend(prim);
            }
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

    // ── 메모리 레이아웃 검증 ──────────────────────────────────────────────────────
    // GLSL std430: 모든 스칼라는 4바이트, 패딩 없음.
    // offsetof 는 standard-layout 타입에서만 정의됨.

    static_assert(std::is_standard_layout_v<Primitive>, "Primitive must be standard-layout");
    static_assert(std::is_standard_layout_v<MortonCode>, "MortonCode must be standard-layout");
    static_assert(std::is_standard_layout_v<MortonConstant>, "MortonConstant must be standard-layout");

    // Primitive (28 bytes)
    static_assert(sizeof(Primitive) == 28, "Primitive size mismatch with GLSL");
    static_assert(offsetof(Primitive, index) == 0, "Primitive.index offset mismatch");
    static_assert(offsetof(Primitive, aabbMinX) == 4, "Primitive.aabbMinX offset mismatch");
    static_assert(offsetof(Primitive, aabbMinY) == 8, "Primitive.aabbMinY offset mismatch");
    static_assert(offsetof(Primitive, aabbMinZ) == 12, "Primitive.aabbMinZ offset mismatch");
    static_assert(offsetof(Primitive, aabbMaxX) == 16, "Primitive.aabbMaxX offset mismatch");
    static_assert(offsetof(Primitive, aabbMaxY) == 20, "Primitive.aabbMaxY offset mismatch");
    static_assert(offsetof(Primitive, aabbMaxZ) == 24, "Primitive.aabbMaxZ offset mismatch");

    // MortonCode (8 bytes)
    static_assert(sizeof(MortonCode) == 8, "MortonCode size mismatch with GLSL");
    static_assert(offsetof(MortonCode, code) == 0, "MortonCode.code offset mismatch");
    static_assert(offsetof(MortonCode, index) == 4, "MortonCode.index offset mismatch");

    // MortonConstant (push constant, 28 bytes)
    static_assert(sizeof(MortonConstant) == 28, "MortonConstant size mismatch");
    static_assert(offsetof(MortonConstant, g_numberOfPrimitive) == 0, "MortonConstant.g_numberOfPrimitive offset mismatch");
    static_assert(offsetof(MortonConstant, g_minX) == 4, "MortonConstant.g_minX offset mismatch");
    static_assert(offsetof(MortonConstant, g_minY) == 8, "MortonConstant.g_minY offset mismatch");
    static_assert(offsetof(MortonConstant, g_minZ) == 12, "MortonConstant.g_minZ offset mismatch");
    static_assert(offsetof(MortonConstant, g_maxX) == 16, "MortonConstant.g_maxX offset mismatch");
    static_assert(offsetof(MortonConstant, g_maxY) == 20, "MortonConstant.g_maxY offset mismatch");
    static_assert(offsetof(MortonConstant, g_maxZ) == 24, "MortonConstant.g_maxZ offset mismatch");

} // namespace vkCommon

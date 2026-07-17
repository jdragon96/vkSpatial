#pragma once

#include "vkSpatial/vkBVH.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace vkSpatial {

    class vkWideBVH {
    public:
        explicit vkWideBVH(VkContext *ctx, uint32_t maxLeafPrimitives = 4);
        ~vkWideBVH();

        vkWideBVH(const vkWideBVH &) = delete;
        vkWideBVH &operator=(const vkWideBVH &) = delete;

        template<typename T>
        void Build(const std::vector<T> &primitives) {
            if (primitives.size() > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error(
                        "vkWideBVH: primitive count exceeds uint32");

            std::vector<Primitive> prims;
            prims.reserve(primitives.size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(primitives.size()); ++i)
                prims.push_back(PrimitiveConverter<T>::convert(primitives[i], i));
            buildWideBVH(prims);
        }

        std::vector<uint32_t> RadiusSearch(float cx, float cy, float cz, float r);
        std::vector<uint32_t> KNN(float cx, float cy, float cz, int k);

        uint32_t Length() const;
        uint32_t NodeCount() const;
        uint32_t MaxLeafPrimitives() const;

        // 순회에 필요한 내부 GPU 버퍼 접근자. Build()가 호출된 이후에만 유효하며,
        // 외부 ray-trace 등 커스텀 컴퓨트 파이프라인이 이 버퍼들을 SSBO로 바인딩할 때 사용한다.
        vkCommon::vkGPUMemory &NodeBuffer() const;
        vkCommon::vkGPUMemory &LeafBuffer() const;
        vkCommon::vkGPUMemory &SortedMortonBuffer() const;
        vkCommon::vkGPUMemory &PrimitiveBuffer() const;

        struct RayTraceCamera {
            float originX, originY, originZ;
            float rightX, rightY, rightZ;
            float upX, upY, upZ;
            float forwardX, forwardY, forwardZ;
            float tanHalfFovY;
            float aspect;
        };

        // 기본은 방향광(directional light)이며, lightType=1이면 lightPos*를 점광원 위치로 사용한다.
        // groundStartIndex 이상의 primitive index는 바닥(ground) 재질로, 미만은 구체 재질로 취급한다.
        struct RayTraceLighting {
            float lightDirX, lightDirY, lightDirZ;
            uint32_t groundStartIndex;
            float lightPosX = 0.0f;
            float lightPosY = 0.0f;
            float lightPosZ = 0.0f;
            uint32_t lightType = 0;
        };

        enum class PathTraceMaterialType : uint32_t {
            Diffuse = 0,
            Metal = 1,
            Dielectric = 2,
            Emissive = 3,
        };

        struct PathTraceMaterial {
            float albedoX = 1.0f;
            float albedoY = 1.0f;
            float albedoZ = 1.0f;
            float roughness = 0.0f;
            float emissionX = 0.0f;
            float emissionY = 0.0f;
            float emissionZ = 0.0f;
            float ior = 1.5f;
            uint32_t type = static_cast<uint32_t>(PathTraceMaterialType::Diffuse);
            float metallic = 0.0f;
            float transmission = 0.0f;
            float padding = 0.0f;
        };

        struct PathTraceLight {
            float centerX = 0.0f;
            float centerY = 0.0f;
            float centerZ = 0.0f;
            float environmentStrength = 0.02f;
            float uX = 1.0f;
            float uY = 0.0f;
            float uZ = 0.0f;
            float pad0 = 0.0f;
            float vX = 0.0f;
            float vY = 0.0f;
            float vZ = 1.0f;
            float pad1 = 0.0f;
            float emissionX = 12.0f;
            float emissionY = 12.0f;
            float emissionZ = 12.0f;
            float pad2 = 0.0f;
        };

        // KNN/RadiusSearch와 달리 매 프레임 호출되는 것을 전제로 하며 결과를 CPU로
        // 내려받지 않는다. 따라서 자체 버퍼를 할당하지 않고, 호출자가 프레임 동안
        // 계속 소유하는 정점/출력 버퍼를 그대로 받아 GPU 상에서만 결과를 남긴다.
        // outputSwapRedBlue: 출력 버퍼가 최종적으로 복사될 대상 이미지가
        // VK_FORMAT_B8G8R8A8_* (예: 대부분의 macOS/MoltenVK 스왑체인)이면 true를
        // 넘겨야 한다. packUnorm4x8은 R8G8B8A8 바이트 순서로 채우므로, 대상이
        // BGRA면 셰이더에서 R/B 채널을 미리 바꿔 써야 색이 뒤집히지 않는다.
        void TraceRays(const RayTraceCamera &camera,
                       uint32_t width, uint32_t height,
                       float tMin, float tMax, uint32_t shadeMode,
                       const RayTraceLighting &lighting,
                       bool outputSwapRedBlue,
                       vkCommon::vkGPUMemory &triangleVertexBuffer,
                       vkCommon::vkGPUMemory &outputPixelBuffer);

        void TracePath(const RayTraceCamera &camera,
                       uint32_t width, uint32_t height,
                       float tMin, float tMax,
                       uint32_t frameIndex,
                       uint32_t maxBounces,
                       uint32_t samplesPerPixel,
                       const PathTraceLight &light,
                       bool outputSwapRedBlue,
                       vkCommon::vkGPUMemory &triangleVertexBuffer,
                       vkCommon::vkGPUMemory &materialIdBuffer,
                       vkCommon::vkGPUMemory &materialBuffer,
                       vkCommon::vkGPUMemory &accumulationBuffer,
                       vkCommon::vkGPUMemory &outputPixelBuffer);

    private:
        struct Impl;

        void buildWideBVH(const std::vector<Primitive> &prims);

        std::unique_ptr<Impl> m_impl;
    };

} // namespace vkSpatial

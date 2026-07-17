#include "vkRender/vkRender.h"
#include "vkSpatial/vkWideBVH.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

namespace {

    constexpr float kPi = 3.14159265358979323846f;

    vkSpatial::TrianglePrim MakeTri(const float a[3], const float b[3], const float c[3]) {
        vkSpatial::TrianglePrim t{};
        t.v0[0] = a[0];
        t.v0[1] = a[1];
        t.v0[2] = a[2];
        t.v1[0] = b[0];
        t.v1[1] = b[1];
        t.v1[2] = b[2];
        t.v2[0] = c[0];
        t.v2[1] = c[1];
        t.v2[2] = c[2];
        return t;
    }

    std::vector<vkSpatial::TrianglePrim> MakeSphereTriangles(
            float radius, int lonSegments, int latSegments,
            float centerX, float centerY, float centerZ) {
        auto vertexAt = [&](int lat, int lon) {
            const float theta = static_cast<float>(lat) /
                                static_cast<float>(latSegments) * kPi;
            const float phi = static_cast<float>(lon) /
                              static_cast<float>(lonSegments) * 2.0f * kPi;
            const float y = std::cos(theta) * radius;
            const float ringRadius = std::sin(theta) * radius;
            const float x = ringRadius * std::cos(phi);
            const float z = ringRadius * std::sin(phi);
            return std::array<float, 3>{
                    centerX + x,
                    centerY + y,
                    centerZ + z};
        };

        std::vector<vkSpatial::TrianglePrim> tris;
        tris.reserve(static_cast<size_t>(lonSegments) * latSegments * 2);
        for (int lat = 0; lat < latSegments; ++lat) {
            for (int lon = 0; lon < lonSegments; ++lon) {
                const auto p00 = vertexAt(lat, lon);
                const auto p10 = vertexAt(lat + 1, lon);
                const auto p01 = vertexAt(lat, lon + 1);
                const auto p11 = vertexAt(lat + 1, lon + 1);
                if (lat == 0) {
                    tris.push_back(MakeTri(p00.data(), p10.data(), p11.data()));
                } else if (lat == latSegments - 1) {
                    tris.push_back(MakeTri(p00.data(), p10.data(), p01.data()));
                } else {
                    tris.push_back(MakeTri(p00.data(), p10.data(), p11.data()));
                    tris.push_back(MakeTri(p00.data(), p11.data(), p01.data()));
                }
            }
        }
        return tris;
    }

    std::vector<vkSpatial::TrianglePrim> MakeRoomTriangles(
            float halfX, float halfZ, float floorY, float ceilY) {
        std::vector<vkSpatial::TrianglePrim> tris;
        auto quad = [&](const float a[3], const float b[3],
                        const float c[3], const float d[3]) {
            tris.push_back(MakeTri(a, b, c));
            tris.push_back(MakeTri(a, c, d));
        };

        const float f00[3] = {-halfX, floorY, -halfZ};
        const float f10[3] = {halfX, floorY, -halfZ};
        const float f11[3] = {halfX, floorY, halfZ};
        const float f01[3] = {-halfX, floorY, halfZ};
        quad(f00, f10, f11, f01);

        const float c00[3] = {-halfX, ceilY, -halfZ};
        const float c10[3] = {halfX, ceilY, -halfZ};
        const float c11[3] = {halfX, ceilY, halfZ};
        const float c01[3] = {-halfX, ceilY, halfZ};
        quad(c00, c01, c11, c10);

        quad(f00, c00, c10, f10);
        quad(f11, c11, c01, f01);
        quad(f01, c01, c00, f00);
        quad(f10, c10, c11, f11);
        return tris;
    }

    struct OrbitCamera {
        float yaw = 0.0f;
        float pitch = -0.12f;
        float distance = 2.65f;
        float targetX = 0.0f;
        float targetY = -0.15f;
        float targetZ = 0.0f;
    };

    vkSpatial::vkWideBVH::RayTraceCamera BuildCamera(
            const OrbitCamera &orbit, float aspect) {
        const float cp = std::cos(orbit.pitch);
        const float sp = std::sin(orbit.pitch);
        const float cy = std::cos(orbit.yaw);
        const float sy = std::sin(orbit.yaw);

        const float dir[3] = {cp * sy, sp, cp * cy};
        const float eye[3] = {
                orbit.targetX + dir[0] * orbit.distance,
                orbit.targetY + dir[1] * orbit.distance,
                orbit.targetZ + dir[2] * orbit.distance};
        const float forward[3] = {-dir[0], -dir[1], -dir[2]};
        const float worldUp[3] = {0.0f, 1.0f, 0.0f};

        float right[3] = {
                forward[1] * worldUp[2] - forward[2] * worldUp[1],
                forward[2] * worldUp[0] - forward[0] * worldUp[2],
                forward[0] * worldUp[1] - forward[1] * worldUp[0]};
        float rightLen = std::sqrt(right[0] * right[0] +
                                   right[1] * right[1] +
                                   right[2] * right[2]);
        if (rightLen < 1e-6f) {
            right[0] = 1.0f;
            right[1] = 0.0f;
            right[2] = 0.0f;
            rightLen = 1.0f;
        }
        right[0] /= rightLen;
        right[1] /= rightLen;
        right[2] /= rightLen;

        const float up[3] = {
                right[1] * forward[2] - right[2] * forward[1],
                right[2] * forward[0] - right[0] * forward[2],
                right[0] * forward[1] - right[1] * forward[0]};

        vkSpatial::vkWideBVH::RayTraceCamera camera{};
        camera.originX = eye[0];
        camera.originY = eye[1];
        camera.originZ = eye[2];
        camera.rightX = right[0];
        camera.rightY = right[1];
        camera.rightZ = right[2];
        camera.upX = up[0];
        camera.upY = up[1];
        camera.upZ = up[2];
        camera.forwardX = forward[0];
        camera.forwardY = forward[1];
        camera.forwardZ = forward[2];
        camera.tanHalfFovY = std::tan(0.5f * (62.0f * kPi / 180.0f));
        camera.aspect = aspect;
        return camera;
    }

    void EnsurePixelBuffer(vkCommon::vkGPUMemory &buffer,
                           VkExtent2D extent,
                           VkExtent2D &allocatedExtent) {
        if (extent.width == allocatedExtent.width &&
            extent.height == allocatedExtent.height)
            return;

        const uint32_t bytes = extent.width * extent.height * 4u;
        if (!buffer.Allocate(bytes))
            throw std::runtime_error("bvh_shadow_room: failed to allocate pixel buffer");
        allocatedExtent = extent;
    }

} // namespace

int main() {
    if (!glfwInit()) {
        std::cerr << "bvh_shadow_room: failed to initialize GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(
            900, 700, "vkBVH Ray Traced Shadow Room", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        std::cerr << "bvh_shadow_room: failed to create window\n";
        return 1;
    }

    vkCommon::VkContext context;

    try {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions =
                glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions || glfwExtensionCount == 0)
            throw std::runtime_error("bvh_shadow_room: GLFW did not provide Vulkan extensions");

        std::vector<const char *> instanceExtensions(
                glfwExtensions, glfwExtensions + glfwExtensionCount);

        context.init(true, instanceExtensions,
                     [&](VkInstance instance) {
                         VkSurfaceKHR surface = VK_NULL_HANDLE;
                         if (glfwCreateWindowSurface(instance, window,
                                                     nullptr, &surface) != VK_SUCCESS)
                             throw std::runtime_error("bvh_shadow_room: failed to create window surface");
                         return surface;
                     });

        auto engine = vkRender::Engine::Create(&context);

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

        vkRender::SwapChainDescriptor swapChainDescriptor{};
        swapChainDescriptor.width = static_cast<uint32_t>(framebufferWidth);
        swapChainDescriptor.height = static_cast<uint32_t>(framebufferHeight);
        auto swapChain = engine->CreateSwapChain(swapChainDescriptor);
        auto renderer = engine->CreateRenderer();

        constexpr float sphereRadius = 0.9f;
        constexpr float floorY = -1.05f;
        constexpr float ceilY = 2.65f;
        constexpr float roomHalfX = 3.0f;
        constexpr float roomHalfZ = 3.0f;
        constexpr float sphereCenterY = floorY + sphereRadius;

        std::vector<vkSpatial::TrianglePrim> triangles =
                MakeSphereTriangles(sphereRadius, 12, 6,
                                    0.0f, sphereCenterY, 0.0f);
        const uint32_t roomStartIndex = static_cast<uint32_t>(triangles.size());
        std::vector<vkSpatial::TrianglePrim> room =
                MakeRoomTriangles(roomHalfX, roomHalfZ, floorY, ceilY);
        triangles.insert(triangles.end(), room.begin(), room.end());

        vkSpatial::vkWideBVH bvh(&context, 64);
        bvh.Build(triangles);

        vkCommon::vkGPUMemory triangleVertexBuffer(context.device, context.physDevice);
        const uint32_t triangleBytes =
                static_cast<uint32_t>(triangles.size() * sizeof(vkSpatial::TrianglePrim));
        if (!triangleVertexBuffer.Allocate(triangleBytes) ||
            !triangleVertexBuffer.Upload(triangles.data(), triangleBytes,
                                         context.computeQueue, context.cmdPool))
            throw std::runtime_error("bvh_shadow_room: failed to upload triangle buffer");

        vkCommon::vkGPUMemory outputPixelBuffer(context.device, context.physDevice);
        VkExtent2D outputExtent{};
        OrbitCamera orbit{};
        orbit.targetY = sphereCenterY;

        std::cout << "[bvh_shadow_room] triangles: " << bvh.Length()
                  << "  wide nodes: " << bvh.NodeCount() << "\n";

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
            if (framebufferWidth == 0 || framebufferHeight == 0) {
                glfwWaitEvents();
                continue;
            }

            const VkExtent2D swapExtent = swapChain->Extent();
            if (swapExtent.width != static_cast<uint32_t>(framebufferWidth) ||
                swapExtent.height != static_cast<uint32_t>(framebufferHeight)) {
                vkDeviceWaitIdle(context.device);
                swapChain->Recreate(static_cast<uint32_t>(framebufferWidth),
                                    static_cast<uint32_t>(framebufferHeight));
                renderer->ClearSwapChainRecreateFlag();
                continue;
            }

            EnsurePixelBuffer(outputPixelBuffer, swapExtent, outputExtent);

            const auto camera = BuildCamera(
                    orbit,
                    static_cast<float>(swapExtent.width) /
                            static_cast<float>(swapExtent.height));

            vkSpatial::vkWideBVH::RayTraceLighting lighting{};
            lighting.lightDirX = 0.0f;
            lighting.lightDirY = 1.0f;
            lighting.lightDirZ = 0.0f;
            lighting.groundStartIndex = roomStartIndex;
            lighting.lightPosX = 0.0f;
            lighting.lightPosY = 2.25f;
            lighting.lightPosZ = 0.15f;
            lighting.lightType = 1u;

            bvh.TraceRays(camera,
                          swapExtent.width, swapExtent.height,
                          0.001f, 40.0f, 2u,
                          lighting,
                          swapChain->RequiresRedBlueSwap(),
                          triangleVertexBuffer,
                          outputPixelBuffer);

            if (!renderer->BeginFrame(*swapChain)) {
                vkDeviceWaitIdle(context.device);
                swapChain->Recreate(static_cast<uint32_t>(framebufferWidth),
                                    static_cast<uint32_t>(framebufferHeight));
                renderer->ClearSwapChainRecreateFlag();
                continue;
            }

            renderer->CopyBufferToSwapChain(outputPixelBuffer);
            renderer->EndFrame();

            if (renderer->NeedsSwapChainRecreate()) {
                vkDeviceWaitIdle(context.device);
                swapChain->Recreate(static_cast<uint32_t>(framebufferWidth),
                                    static_cast<uint32_t>(framebufferHeight));
                renderer->ClearSwapChainRecreateFlag();
            }
        }

        vkDeviceWaitIdle(context.device);

        renderer.reset();
        swapChain.reset();
        engine.reset();
        context.shutdown();
    } catch (const std::exception &e) {
        if (context.device != VK_NULL_HANDLE)
            vkDeviceWaitIdle(context.device);
        context.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        std::cerr << e.what() << "\n";
        return 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

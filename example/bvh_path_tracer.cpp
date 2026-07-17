#include "utilities/Math.h"
#include "vkRender/vkRender.h"
#include "vkSpatial/vkWideBVH.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

namespace {

    constexpr float kPi = 3.14159265358979323846f;

    vkRender::MouseButton ToMouseButton(int button) {
        switch (button) {
            case GLFW_MOUSE_BUTTON_LEFT:
                return vkRender::MouseButton::Left;
            case GLFW_MOUSE_BUTTON_RIGHT:
                return vkRender::MouseButton::Right;
            case GLFW_MOUSE_BUTTON_MIDDLE:
                return vkRender::MouseButton::Middle;
            case GLFW_MOUSE_BUTTON_4:
                return vkRender::MouseButton::Button4;
            case GLFW_MOUSE_BUTTON_5:
                return vkRender::MouseButton::Button5;
            default:
                return vkRender::MouseButton::Other;
        }
    }

    vkRender::MouseModifierFlags ToMouseModifiers(int mods) {
        vkRender::MouseModifierFlags flags = 0;
        if ((mods & GLFW_MOD_SHIFT) != 0)
            flags |= vkRender::MouseModifierShift;
        if ((mods & GLFW_MOD_CONTROL) != 0)
            flags |= vkRender::MouseModifierControl;
        if ((mods & GLFW_MOD_ALT) != 0)
            flags |= vkRender::MouseModifierAlt;
        if ((mods & GLFW_MOD_SUPER) != 0)
            flags |= vkRender::MouseModifierSuper;
        return flags;
    }

    vkRender::MouseModifierFlags QueryMouseModifiers(GLFWwindow *window) {
        vkRender::MouseModifierFlags flags = 0;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
            flags |= vkRender::MouseModifierShift;
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS)
            flags |= vkRender::MouseModifierControl;
        if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS)
            flags |= vkRender::MouseModifierAlt;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS)
            flags |= vkRender::MouseModifierSuper;
        return flags;
    }

    vkRender::MouseInput *WindowMouseInput(GLFWwindow *window) {
        return static_cast<vkRender::MouseInput *>(glfwGetWindowUserPointer(window));
    }

    struct TrackballCamera {
        vkMath::Quat orientation = vkMath::Quat::Identity();
        vkMath::Vec3 target{0.0f, 0.0f, 0.15f};
        float distance = 5.3f;
        bool dragging = false;
        vkMath::Vec3 lastBall = vkMath::Vec3::Zero();

        void BeginDrag(const vkRender::MouseEvent &event, int width, int height) {
            if (width <= 0 || height <= 0)
                return;
            dragging = true;
            lastBall = vkMath::MapToArcball(event.x, event.y, width, height);
        }

        bool Drag(const vkRender::MouseEvent &event, int width, int height) {
            if (width <= 0 || height <= 0)
                return false;

            vkMath::Vec3 current = vkMath::MapToArcball(event.x, event.y, width, height);
            if (!dragging) {
                dragging = true;
                lastBall = current;
                return false;
            }

            vkMath::Vec3 axis = lastBall.cross(current);
            const float axisLen2 = axis.dot(axis);
            bool changed = false;
            if (axisLen2 > 1e-8f) {
                const float w = std::clamp(lastBall.dot(current), -1.0f, 1.0f);
                vkMath::Quat delta = vkMath::Quat(w, axis.x(), axis.y(), axis.z()).normalized();
                orientation = (delta * orientation).normalized();
                changed = true;
            }
            lastBall = current;
            return changed;
        }

        void EndDrag() {
            dragging = false;
        }

        bool Scroll(const vkRender::MouseEvent &event) {
            if (std::abs(event.scrollY) <= 1e-6)
                return false;

            distance *= std::exp(static_cast<float>(-event.scrollY) * 0.08f);
            distance = std::clamp(distance, 2.6f, 8.0f);
            return true;
        }
    };

    vkSpatial::vkWideBVH::RayTraceCamera BuildCamera(
            const TrackballCamera &trackball, float aspect) {
        const vkMath::Vec3 eye = trackball.target +
                                 trackball.orientation * vkMath::Vec3(0.0f, 0.0f, trackball.distance);
        const vkMath::Vec3 forward = (trackball.target - eye).normalized();

        const vkMath::Vec3 rawRight = forward.cross(vkMath::Vec3(0.0f, 1.0f, 0.0f));
        vkMath::Vec3 right = rawRight.norm() < 1e-6f
                                     ? trackball.orientation * vkMath::Vec3(1.0f, 0.0f, 0.0f)
                                     : rawRight.normalized();
        vkMath::Vec3 up = right.cross(forward);

        vkSpatial::vkWideBVH::RayTraceCamera camera{};
        camera.originX = eye.x();
        camera.originY = eye.y();
        camera.originZ = eye.z();
        camera.rightX = right.x();
        camera.rightY = right.y();
        camera.rightZ = right.z();
        camera.upX = up.x();
        camera.upY = up.y();
        camera.upZ = up.z();
        camera.forwardX = forward.x();
        camera.forwardY = forward.y();
        camera.forwardZ = forward.z();
        camera.tanHalfFovY = std::tan(0.5f * (48.0f * kPi / 180.0f));
        camera.aspect = aspect;
        return camera;
    }

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

    struct SceneData {
        std::vector<vkSpatial::TrianglePrim> triangles;
        std::vector<uint32_t> materialIds;
        std::vector<vkSpatial::vkWideBVH::PathTraceMaterial> materials;
        vkSpatial::vkWideBVH::PathTraceLight light;
    };

    uint32_t AddMaterial(
            SceneData &scene,
            vkSpatial::vkWideBVH::PathTraceMaterialType type,
            vkMath::Vec3 albedo,
            float roughness = 0.0f,
            vkMath::Vec3 emission = vkMath::Vec3::Zero(),
            float ior = 1.5f) {
        vkSpatial::vkWideBVH::PathTraceMaterial material{};
        material.albedoX = albedo.x();
        material.albedoY = albedo.y();
        material.albedoZ = albedo.z();
        material.roughness = roughness;
        material.emissionX = emission.x();
        material.emissionY = emission.y();
        material.emissionZ = emission.z();
        material.ior = ior;
        material.type = static_cast<uint32_t>(type);
        scene.materials.push_back(material);
        return static_cast<uint32_t>(scene.materials.size() - 1);
    }

    void AddTriangle(SceneData &scene,
                     const float a[3],
                     const float b[3],
                     const float c[3],
                     uint32_t materialId) {
        scene.triangles.push_back(MakeTri(a, b, c));
        scene.materialIds.push_back(materialId);
    }

    void AddQuad(SceneData &scene,
                 const float a[3],
                 const float b[3],
                 const float c[3],
                 const float d[3],
                 uint32_t materialId) {
        AddTriangle(scene, a, b, c, materialId);
        AddTriangle(scene, a, c, d, materialId);
    }

    void AddSphere(SceneData &scene,
                   vkMath::Vec3 center,
                   float radius,
                   int lonSegments,
                   int latSegments,
                   uint32_t materialId) {
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
                    center.x() + x,
                    center.y() + y,
                    center.z() + z};
        };

        for (int lat = 0; lat < latSegments; ++lat) {
            for (int lon = 0; lon < lonSegments; ++lon) {
                const auto p00 = vertexAt(lat, lon);
                const auto p10 = vertexAt(lat + 1, lon);
                const auto p01 = vertexAt(lat, lon + 1);
                const auto p11 = vertexAt(lat + 1, lon + 1);
                if (lat == 0) {
                    AddTriangle(scene, p00.data(), p10.data(), p11.data(), materialId);
                } else if (lat == latSegments - 1) {
                    AddTriangle(scene, p00.data(), p10.data(), p01.data(), materialId);
                } else {
                    AddTriangle(scene, p00.data(), p10.data(), p11.data(), materialId);
                    AddTriangle(scene, p00.data(), p11.data(), p01.data(), materialId);
                }
            }
        }
    }

    SceneData MakePathTraceScene() {
        SceneData scene;

        const uint32_t gray = AddMaterial(
                scene,
                vkSpatial::vkWideBVH::PathTraceMaterialType::Diffuse,
                {0.55f, 0.55f, 0.56f},
                0.6f);
        const uint32_t green = AddMaterial(
                scene,
                vkSpatial::vkWideBVH::PathTraceMaterialType::Diffuse,
                {0.38f, 0.55f, 0.44f},
                0.9f);
        const uint32_t gold = AddMaterial(
                scene,
                vkSpatial::vkWideBVH::PathTraceMaterialType::Metal,
                {0.95f, 0.78f, 0.22f},
                0.18f);
        const uint32_t glass = AddMaterial(
                scene,
                vkSpatial::vkWideBVH::PathTraceMaterialType::Dielectric,
                {0.96f, 0.98f, 1.0f},
                0.0f,
                {0.0f, 0.0f, 0.0f},
                1.5f);
        const uint32_t lightMaterial = AddMaterial(
                scene,
                vkSpatial::vkWideBVH::PathTraceMaterialType::Emissive,
                {1.0f, 1.0f, 1.0f},
                0.0f,
                {16.0f, 15.0f, 13.0f});

        constexpr float floorY = -1.0f;
        constexpr float ceilY = 3.0f;
        constexpr float leftX = -4.0f;
        constexpr float rightX = 4.0f;
        constexpr float backZ = -3.6f;
        constexpr float frontZ = 5.4f;

        const float f00[3] = {leftX, floorY, backZ};
        const float f10[3] = {rightX, floorY, backZ};
        const float f11[3] = {rightX, floorY, frontZ};
        const float f01[3] = {leftX, floorY, frontZ};
        AddQuad(scene, f00, f10, f11, f01, gray);

        const float c00[3] = {leftX, ceilY, backZ};
        const float c10[3] = {rightX, ceilY, backZ};
        const float c11[3] = {rightX, ceilY, frontZ};
        const float c01[3] = {leftX, ceilY, frontZ};
        AddQuad(scene, c00, c01, c11, c10, gray);
        AddQuad(scene, f00, c00, c10, f10, gray);
        AddQuad(scene, f01, c01, c00, f00, gray);
        AddQuad(scene, f10, c10, c11, f11, gray);

        AddSphere(scene, {-1.7f, floorY + 1.18f, 0.1f}, 1.18f, 14, 7, green);
        AddSphere(scene, {0.55f, floorY + 0.82f, -0.05f}, 0.82f, 16, 8, glass);
        AddSphere(scene, {2.1f, floorY + 0.44f, 0.65f}, 0.44f, 12, 6, gold);

        scene.light.centerX = 0.0f;
        scene.light.centerY = 2.72f;
        scene.light.centerZ = 0.55f;
        scene.light.environmentStrength = 0.04f;
        scene.light.uX = 1.55f;
        scene.light.uY = 0.0f;
        scene.light.uZ = 0.0f;
        scene.light.vX = 0.0f;
        scene.light.vY = 0.0f;
        scene.light.vZ = 1.15f;
        scene.light.emissionX = 16.0f;
        scene.light.emissionY = 15.0f;
        scene.light.emissionZ = 13.0f;

        const vkMath::Vec3 lc{scene.light.centerX, scene.light.centerY, scene.light.centerZ};
        const vkMath::Vec3 hu{scene.light.uX * 0.5f, scene.light.uY * 0.5f, scene.light.uZ * 0.5f};
        const vkMath::Vec3 hv{scene.light.vX * 0.5f, scene.light.vY * 0.5f, scene.light.vZ * 0.5f};
        const vkMath::Vec3 a = lc - hu - hv;
        const vkMath::Vec3 b = lc + hu - hv;
        const vkMath::Vec3 c = lc + hu + hv;
        const vkMath::Vec3 d = lc - hu + hv;
        const float la[3] = {a.x(), a.y(), a.z()};
        const float lb[3] = {b.x(), b.y(), b.z()};
        const float lcpt[3] = {c.x(), c.y(), c.z()};
        const float ld[3] = {d.x(), d.y(), d.z()};
        AddQuad(scene, la, lb, lcpt, ld, lightMaterial);

        return scene;
    }

    void EnsureRenderBuffers(vkCommon::vkGPUMemory &output,
                             vkCommon::vkGPUMemory &accum,
                             VkExtent2D extent,
                             VkExtent2D &allocatedExtent) {
        if (extent.width == allocatedExtent.width &&
            extent.height == allocatedExtent.height)
            return;

        const uint32_t pixelBytes = extent.width * extent.height * 4u;
        const uint32_t accumBytes = extent.width * extent.height * 4u * sizeof(float);
        if (!output.Allocate(pixelBytes))
            throw std::runtime_error("bvh_path_tracer: failed to allocate output buffer");
        if (!accum.Allocate(accumBytes))
            throw std::runtime_error("bvh_path_tracer: failed to allocate accumulation buffer");
        allocatedExtent = extent;
    }

} // namespace

int main() {
    if (!glfwInit()) {
        std::cerr << "bvh_path_tracer: failed to initialize GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(
            960, 720, "vkBVH Path Tracer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        std::cerr << "bvh_path_tracer: failed to create window\n";
        return 1;
    }

    vkCommon::VkContext context;

    try {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions =
                glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions || glfwExtensionCount == 0)
            throw std::runtime_error("bvh_path_tracer: GLFW did not provide Vulkan extensions");

        std::vector<const char *> instanceExtensions(
                glfwExtensions, glfwExtensions + glfwExtensionCount);

        context.init(true, instanceExtensions,
                     [&](VkInstance instance) {
                         VkSurfaceKHR surface = VK_NULL_HANDLE;
                         if (glfwCreateWindowSurface(instance, window,
                                                     nullptr, &surface) != VK_SUCCESS)
                             throw std::runtime_error("bvh_path_tracer: failed to create window surface");
                         return surface;
                     });

        auto engine = vkRender::Engine::Create(&context);
        auto mouseInput = engine->CreateMouseInput();
        glfwSetWindowUserPointer(window, mouseInput.get());
        glfwSetCursorPosCallback(window, [](GLFWwindow *callbackWindow,
                                            double x,
                                            double y) {
            vkRender::MouseInput *mouse = WindowMouseInput(callbackWindow);
            if (!mouse)
                return;
            mouse->OnMouseMove(x, y, QueryMouseModifiers(callbackWindow), glfwGetTime());
        });
        glfwSetMouseButtonCallback(window, [](GLFWwindow *callbackWindow,
                                              int button,
                                              int action,
                                              int mods) {
            vkRender::MouseInput *mouse = WindowMouseInput(callbackWindow);
            if (!mouse)
                return;

            double x = 0.0;
            double y = 0.0;
            glfwGetCursorPos(callbackWindow, &x, &y);
            mouse->OnButton(ToMouseButton(button),
                            action == GLFW_PRESS,
                            x,
                            y,
                            ToMouseModifiers(mods),
                            glfwGetTime());
        });
        glfwSetScrollCallback(window, [](GLFWwindow *callbackWindow,
                                         double xoffset,
                                         double yoffset) {
            vkRender::MouseInput *mouse = WindowMouseInput(callbackWindow);
            if (!mouse)
                return;

            double x = 0.0;
            double y = 0.0;
            glfwGetCursorPos(callbackWindow, &x, &y);
            mouse->OnScroll(x,
                            y,
                            xoffset,
                            yoffset,
                            QueryMouseModifiers(callbackWindow),
                            glfwGetTime());
        });
        glfwSetCursorEnterCallback(window, [](GLFWwindow *callbackWindow, int entered) {
            vkRender::MouseInput *mouse = WindowMouseInput(callbackWindow);
            if (!mouse)
                return;

            double x = 0.0;
            double y = 0.0;
            glfwGetCursorPos(callbackWindow, &x, &y);
            if (entered) {
                mouse->OnCursorEnter(
                        x, y, QueryMouseModifiers(callbackWindow), glfwGetTime());
            } else {
                mouse->OnCursorLeave(
                        x, y, QueryMouseModifiers(callbackWindow), glfwGetTime());
            }
        });

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

        vkRender::SwapChainDescriptor swapChainDescriptor{};
        swapChainDescriptor.width = static_cast<uint32_t>(framebufferWidth);
        swapChainDescriptor.height = static_cast<uint32_t>(framebufferHeight);
        auto swapChain = engine->CreateSwapChain(swapChainDescriptor);
        auto renderer = engine->CreateRenderer();

        SceneData scene = MakePathTraceScene();

        vkSpatial::vkWideBVH bvh(&context, 64);
        bvh.Build(scene.triangles);

        vkCommon::vkGPUMemory triangleVertexBuffer(context.device, context.physDevice);
        const uint32_t triangleBytes =
                static_cast<uint32_t>(scene.triangles.size() * sizeof(vkSpatial::TrianglePrim));
        if (!triangleVertexBuffer.Allocate(triangleBytes) ||
            !triangleVertexBuffer.Upload(scene.triangles.data(), triangleBytes,
                                         context.computeQueue, context.cmdPool))
            throw std::runtime_error("bvh_path_tracer: failed to upload triangle buffer");

        vkCommon::vkGPUMemory materialIdBuffer(context.device, context.physDevice);
        const uint32_t materialIdBytes =
                static_cast<uint32_t>(scene.materialIds.size() * sizeof(uint32_t));
        if (!materialIdBuffer.Allocate(materialIdBytes) ||
            !materialIdBuffer.Upload(scene.materialIds.data(), materialIdBytes,
                                     context.computeQueue, context.cmdPool))
            throw std::runtime_error("bvh_path_tracer: failed to upload material id buffer");

        vkCommon::vkGPUMemory materialBuffer(context.device, context.physDevice);
        const uint32_t materialBytes =
                static_cast<uint32_t>(
                        scene.materials.size() * sizeof(vkSpatial::vkWideBVH::PathTraceMaterial));
        if (!materialBuffer.Allocate(materialBytes) ||
            !materialBuffer.Upload(scene.materials.data(), materialBytes,
                                   context.computeQueue, context.cmdPool))
            throw std::runtime_error("bvh_path_tracer: failed to upload material buffer");

        vkCommon::vkGPUMemory outputBuffer(context.device, context.physDevice);
        vkCommon::vkGPUMemory accumulationBuffer(context.device, context.physDevice);
        VkExtent2D allocatedExtent{};

        TrackballCamera trackball;
        bool cameraDirty = true;
        uint32_t frameIndex = 0;
        vkRender::MouseListenerGroup mouseBindings(*mouseInput);

        mouseBindings.Add(
                vkRender::MouseEventType::DragBegin,
                [&](vkRender::MouseEvent &event) {
                    if (event.button != vkRender::MouseButton::Left)
                        return;
                    trackball.BeginDrag(event, framebufferWidth, framebufferHeight);
                    event.handled = true;
                },
                10);
        mouseBindings.Add(
                vkRender::MouseEventType::Drag,
                [&](vkRender::MouseEvent &event) {
                    if (event.button != vkRender::MouseButton::Left)
                        return;
                    if (trackball.Drag(event, framebufferWidth, framebufferHeight))
                        cameraDirty = true;
                    event.handled = true;
                },
                10);
        mouseBindings.Add(
                vkRender::MouseEventType::DragEnd,
                [&](vkRender::MouseEvent &event) {
                    if (event.button != vkRender::MouseButton::Left)
                        return;
                    trackball.EndDrag();
                    event.handled = true;
                },
                10);
        mouseBindings.Add(
                vkRender::MouseEventType::Scroll,
                [&](vkRender::MouseEvent &event) {
                    if (trackball.Scroll(event))
                        cameraDirty = true;
                    event.handled = true;
                },
                10);

        std::cout << "[bvh_path_tracer] triangles: " << bvh.Length()
                  << "  wide nodes: " << bvh.NodeCount()
                  << "  materials: " << scene.materials.size() << "\n";

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
            if (framebufferWidth == 0 || framebufferHeight == 0) {
                glfwWaitEvents();
                continue;
            }

            bool resetAccumulation = false;
            const VkExtent2D swapExtent = swapChain->Extent();
            if (swapExtent.width != static_cast<uint32_t>(framebufferWidth) ||
                swapExtent.height != static_cast<uint32_t>(framebufferHeight)) {
                vkDeviceWaitIdle(context.device);
                swapChain->Recreate(static_cast<uint32_t>(framebufferWidth),
                                    static_cast<uint32_t>(framebufferHeight));
                renderer->ClearSwapChainRecreateFlag();
                allocatedExtent = {};
                frameIndex = 0;
                resetAccumulation = true;
                continue;
            }

            resetAccumulation = cameraDirty;
            cameraDirty = false;

            const VkExtent2D extent = swapChain->Extent();
            if (extent.width != allocatedExtent.width ||
                extent.height != allocatedExtent.height) {
                EnsureRenderBuffers(outputBuffer, accumulationBuffer, extent, allocatedExtent);
                resetAccumulation = true;
            }

            if (resetAccumulation)
                frameIndex = 0;

            const auto camera = BuildCamera(
                    trackball,
                    static_cast<float>(extent.width) /
                            static_cast<float>(extent.height));

            bvh.TracePath(camera,
                          extent.width, extent.height,
                          0.001f, 100.0f,
                          frameIndex,
                          8u,
                          1u,
                          scene.light,
                          swapChain->RequiresRedBlueSwap(),
                          triangleVertexBuffer,
                          materialIdBuffer,
                          materialBuffer,
                          accumulationBuffer,
                          outputBuffer);

            if (!renderer->BeginFrame(*swapChain)) {
                vkDeviceWaitIdle(context.device);
                swapChain->Recreate(static_cast<uint32_t>(framebufferWidth),
                                    static_cast<uint32_t>(framebufferHeight));
                renderer->ClearSwapChainRecreateFlag();
                allocatedExtent = {};
                frameIndex = 0;
                continue;
            }

            renderer->CopyBufferToSwapChain(outputBuffer);
            renderer->EndFrame();

            if (renderer->NeedsSwapChainRecreate()) {
                vkDeviceWaitIdle(context.device);
                swapChain->Recreate(static_cast<uint32_t>(framebufferWidth),
                                    static_cast<uint32_t>(framebufferHeight));
                renderer->ClearSwapChainRecreateFlag();
                allocatedExtent = {};
                frameIndex = 0;
                continue;
            }

            ++frameIndex;
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

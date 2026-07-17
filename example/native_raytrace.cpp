#include "vkCommon/vkContext.h"
#include "vkCommon/vkGPUMemory.h"
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

    // 스크롤(줌) 입력 누적값. GLFW 콜백은 캡처 있는 람다를 받지 않으므로
    // 파일 스코프 변수로 보관했다가 메인 루프에서 매 프레임 소비한다.
    double g_scrollAccum = 0.0;

    void scrollCallback(GLFWwindow *, double, double yoffset) {
        g_scrollAccum += yoffset;
    }

    // ── Swapchain ────────────────────────────────────────────────────────────

    struct Swapchain {
        VkSwapchainKHR handle = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{};
        std::vector<VkImage> images;
        std::vector<VkImageView> views;
    };

    void destroySwapchain(VkDevice device, Swapchain &sc) {
        for (VkImageView view: sc.views)
            vkDestroyImageView(device, view, nullptr);
        sc.views.clear();
        sc.images.clear();
        if (sc.handle != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(device, sc.handle, nullptr);
        sc.handle = VK_NULL_HANDLE;
    }

    Swapchain createSwapchain(vkCommon::VkContext &ctx, GLFWwindow *window,
                              VkSwapchainKHR oldSwapchain) {
        VkSurfaceCapabilitiesKHR caps{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.physDevice, ctx.surface, &caps);

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physDevice, ctx.surface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physDevice, ctx.surface, &formatCount, formats.data());

        VkSurfaceFormatKHR chosenFormat = formats[0];
        for (const auto &f: formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosenFormat = f;
                break;
            }
        }

        VkExtent2D extent = caps.currentExtent;
        if (extent.width == 0xFFFFFFFFu) {
            int fbWidth = 0, fbHeight = 0;
            glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
            extent.width = std::clamp(static_cast<uint32_t>(fbWidth),
                                      caps.minImageExtent.width, caps.maxImageExtent.width);
            extent.height = std::clamp(static_cast<uint32_t>(fbHeight),
                                       caps.minImageExtent.height, caps.maxImageExtent.height);
        }

        uint32_t imageCount = caps.minImageCount + 1;
        if (caps.maxImageCount > 0)
            imageCount = std::min(imageCount, caps.maxImageCount);

        VkSwapchainCreateInfoKHR sci{};
        sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        sci.surface = ctx.surface;
        sci.minImageCount = imageCount;
        sci.imageFormat = chosenFormat.format;
        sci.imageColorSpace = chosenFormat.colorSpace;
        sci.imageExtent = extent;
        sci.imageArrayLayers = 1;
        // 컴퓨트로 만든 픽셀 버퍼를 스왑체인 이미지에 직접 복사하므로
        // COLOR_ATTACHMENT 대신 TRANSFER_DST 사용만 있으면 된다.
        sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = caps.currentTransform;
        sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        sci.clipped = VK_TRUE;
        sci.oldSwapchain = oldSwapchain;

        Swapchain sc;
        sc.format = chosenFormat.format;
        sc.extent = extent;
        if (vkCreateSwapchainKHR(ctx.device, &sci, nullptr, &sc.handle) != VK_SUCCESS)
            throw std::runtime_error("native_raytrace: failed to create swapchain");

        uint32_t actualCount = 0;
        vkGetSwapchainImagesKHR(ctx.device, sc.handle, &actualCount, nullptr);
        sc.images.resize(actualCount);
        vkGetSwapchainImagesKHR(ctx.device, sc.handle, &actualCount, sc.images.data());

        sc.views.resize(actualCount);
        for (uint32_t i = 0; i < actualCount; i++) {
            VkImageViewCreateInfo ivci{};
            ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ivci.image = sc.images[i];
            ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ivci.format = sc.format;
            ivci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                               VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
            ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (vkCreateImageView(ctx.device, &ivci, nullptr, &sc.views[i]) != VK_SUCCESS)
                throw std::runtime_error("native_raytrace: failed to create swapchain image view");
        }
        return sc;
    }

    // ── 소소한 Vulkan 헬퍼 ───────────────────────────────────────────────────────

    void transitionImage(VkCommandBuffer cmd, VkImage image,
                         VkImageLayout oldLayout, VkImageLayout newLayout,
                         VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                         VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    // ── 궤도 카메라 (좌클릭 드래그로 회전, 스크롤로 줌 — 순수 GLFW 폴링) ────────────

    struct OrbitCamera {
        float yaw = 0.6f;
        float pitch = 0.35f;
        float distance = 2.3f;
        float targetX = 0.0f;
        float targetY = 0.0f;
        float targetZ = 0.0f;
    };

    vkSpatial::vkWideBVH::RayTraceCamera buildCamera(const OrbitCamera &orbit, float aspect) {
        const float cp = std::cos(orbit.pitch), sp = std::sin(orbit.pitch);
        const float cy = std::cos(orbit.yaw), sy = std::sin(orbit.yaw);

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
        float rightLen = std::sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
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

        vkSpatial::vkWideBVH::RayTraceCamera cam{};
        cam.originX = eye[0];
        cam.originY = eye[1];
        cam.originZ = eye[2];
        cam.rightX = right[0];
        cam.rightY = right[1];
        cam.rightZ = right[2];
        cam.upX = up[0];
        cam.upY = up[1];
        cam.upZ = up[2];
        cam.forwardX = forward[0];
        cam.forwardY = forward[1];
        cam.forwardZ = forward[2];
        cam.tanHalfFovY = std::tan(0.5f * (55.0f * 3.14159265f / 180.0f));
        cam.aspect = aspect;
        return cam;
    }

    // ── 프로시저럴 지오메트리: 구체 + 방(바닥/천장/벽 4개) ─────────────────────────

    vkSpatial::TrianglePrim makeTri(const float a[3], const float b[3], const float c[3]) {
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

    std::vector<vkSpatial::TrianglePrim> makeSphereTriangles(
            float radius, int lonSegments, int latSegments,
            float centerX = 0.0f, float centerY = 0.0f, float centerZ = 0.0f) {
        auto vertexAt = [&](int lat, int lon) {
            const float theta = static_cast<float>(lat) / static_cast<float>(latSegments) * 3.14159265f;
            const float phi = static_cast<float>(lon) / static_cast<float>(lonSegments) * 2.0f * 3.14159265f;
            const float y = std::cos(theta) * radius;
            const float ringRadius = std::sin(theta) * radius;
            const float x = ringRadius * std::cos(phi);
            const float z = ringRadius * std::sin(phi);
            return std::array<float, 3>{centerX + x, centerY + y, centerZ + z};
        };

        std::vector<vkSpatial::TrianglePrim> tris;
        tris.reserve(static_cast<size_t>(lonSegments) * latSegments * 2);
        for (int lat = 0; lat < latSegments; ++lat) {
            for (int lon = 0; lon < lonSegments; ++lon) {
                const auto p00 = vertexAt(lat, lon);
                const auto p10 = vertexAt(lat + 1, lon);
                const auto p01 = vertexAt(lat, lon + 1);
                const auto p11 = vertexAt(lat + 1, lon + 1);
                // 극점(lat==0 또는 lat==latSegments-1)에서는 사각형의 한 변이
                // 극점 하나로 완전히 축퇴되므로, 면적이 0인 삼각형을 만들지 않도록
                // 팬(fan) 삼각형 하나만 생성한다.
                if (lat == 0) {
                    tris.push_back(makeTri(p00.data(), p10.data(), p11.data()));
                } else if (lat == latSegments - 1) {
                    tris.push_back(makeTri(p00.data(), p10.data(), p01.data()));
                } else {
                    tris.push_back(makeTri(p00.data(), p10.data(), p11.data()));
                    tris.push_back(makeTri(p00.data(), p11.data(), p01.data()));
                }
            }
        }
        return tris;
    }

    // halfX/halfZ x floorY..ceilY 범위의 속이 빈 상자(바닥+천장+벽 4개, 12삼각형).
    // 카메라가 상자 안쪽에 있으므로 별도 문/컬링 없이 안쪽 면만 보이면 충분하다.
    std::vector<vkSpatial::TrianglePrim> makeRoomTriangles(
            float halfX, float halfZ, float floorY, float ceilY) {
        std::vector<vkSpatial::TrianglePrim> tris;
        auto quad = [&](const float a[3], const float b[3], const float c[3], const float d[3]) {
            tris.push_back(makeTri(a, b, c));
            tris.push_back(makeTri(a, c, d));
        };

        const float f00[3] = {-halfX, floorY, -halfZ};
        const float f10[3] = {halfX, floorY, -halfZ};
        const float f11[3] = {halfX, floorY, halfZ};
        const float f01[3] = {-halfX, floorY, halfZ};
        quad(f00, f10, f11, f01); // 바닥

        const float c00[3] = {-halfX, ceilY, -halfZ};
        const float c10[3] = {halfX, ceilY, -halfZ};
        const float c11[3] = {halfX, ceilY, halfZ};
        const float c01[3] = {-halfX, ceilY, halfZ};
        quad(c00, c01, c11, c10); // 천장

        quad(f00, c00, c10, f10); // -Z 벽
        quad(f11, c11, c01, f01); // +Z 벽
        quad(f01, c01, c00, f00); // -X 벽
        quad(f10, c10, c11, f11); // +X 벽

        return tris;
    }

    // ── 메인 애플리케이션 (모든 GPU 자원은 이 함수 스코프 안에서 생성/소멸되어,
    //    호출자가 이후 ctx.shutdown()을 안전하게 호출할 수 있게 한다) ──────────────

    void runApp(vkCommon::VkContext &ctx, GLFWwindow *window) {
        Swapchain swapchain = createSwapchain(ctx, window, VK_NULL_HANDLE);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = ctx.graphicsFamily;
        VkCommandPool graphicsPool = VK_NULL_HANDLE;
        if (vkCreateCommandPool(ctx.device, &cpci, nullptr, &graphicsPool) != VK_SUCCESS)
            throw std::runtime_error("native_raytrace: failed to create graphics command pool");

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = graphicsPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(ctx.device, &cbai, &cmd);

        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        vkCreateSemaphore(ctx.device, &semInfo, nullptr, &imageAvailable);
        vkCreateSemaphore(ctx.device, &semInfo, nullptr, &renderFinished);

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VkFence inFlightFence = VK_NULL_HANDLE;
        vkCreateFence(ctx.device, &fenceInfo, nullptr, &inFlightFence);

        // ── 지오메트리 + Wide BVH: 방 안에 놓인 구체 ───────────────────────────────
        // groundStartIndex 이상의 primitive index는 방(바닥/천장/벽) 재질, 미만은
        // 구체 재질로 셰이더에서 구분한다.
        //
        // vkWideBVH 빌드에는 primitive 구성에 따라 트리가 깨지는 기존(uncommitted)
        // 버그가 있다. wide 노드 수나 maxLeafPrimitives 값만으로는 안전 여부를 예측할
        // 수 없었다(실측 결과 노드 수가 적어도 깨지는 조합, 많아도 되는 조합이 둘 다
        // 있었음). 아래 (구체 12x6 세그먼트 + 이 방 크기 + maxLeafPrimitives=64) 조합은
        // 실제로 정상 렌더링을 확인했으므로 이 조합을 그대로 유지한다.
        constexpr float kSphereRadius = 1.0f;
        constexpr float kFloorY = -1.2f;
        constexpr float kCeilY = 2.8f;
        constexpr float kRoomHalfX = 3.0f;
        constexpr float kRoomHalfZ = 3.0f;
        constexpr float kSphereCenterY = kFloorY + kSphereRadius; // 바닥에 닿게 배치

        std::vector<vkSpatial::TrianglePrim> triangles =
                makeSphereTriangles(kSphereRadius, 12, 6, 0.0f, kSphereCenterY, 0.0f);
        const uint32_t groundStartIndex = static_cast<uint32_t>(triangles.size());
        std::vector<vkSpatial::TrianglePrim> room =
                makeRoomTriangles(kRoomHalfX, kRoomHalfZ, kFloorY, kCeilY);
        triangles.insert(triangles.end(), room.begin(), room.end());

        // maxLeafPrimitives=64: 이 primitive 구성(구체+방)에서는 wide 노드가 여러
        // 레벨로 나뉘면(기본값 4일 때 확인된 wide 노드 13개) 트리가 깨지는 기존
        // (uncommitted) 버그가 실측 확인되어, 노드가 1개로 collapse되는 64로 우회한다.
        vkSpatial::vkWideBVH bvh(&ctx, 64);
        bvh.Build(triangles);

        vkCommon::vkGPUMemory triVertBuf(ctx.device, ctx.physDevice);
        const uint32_t triBytes = static_cast<uint32_t>(triangles.size() * sizeof(vkSpatial::TrianglePrim));
        if (!triVertBuf.Allocate(triBytes) ||
            !triVertBuf.Upload(triangles.data(), triBytes, ctx.computeQueue, ctx.cmdPool))
            throw std::runtime_error("native_raytrace: failed to upload triangle vertex buffer");

        // 레이트레이스 출력 버퍼는 스왑체인 해상도에 맞춰 (재)할당한다 —
        // ImGui 없이 바로 스왑체인 이미지로 복사하므로 별도 표시용 이미지가 불필요하다.
        vkCommon::vkGPUMemory outputPixelBuf(ctx.device, ctx.physDevice);
        VkExtent2D outputExtent{};
        auto ensureOutputBuffer = [&](VkExtent2D extent) {
            if (extent.width == outputExtent.width && extent.height == outputExtent.height)
                return;
            if (!outputPixelBuf.Allocate(extent.width * extent.height * 4u))
                throw std::runtime_error("native_raytrace: failed to allocate output pixel buffer");
            outputExtent = extent;
        };
        ensureOutputBuffer(swapchain.extent);

        // packUnorm4x8은 R8G8B8A8 바이트 순서를 가정하므로, 스왑체인이
        // B8G8R8A8 계열(대부분의 macOS/MoltenVK 환경)이면 셰이더에서 R/B를 바꿔 써야 한다.
        const bool outputSwapRedBlue =
                swapchain.format == VK_FORMAT_B8G8R8A8_UNORM ||
                swapchain.format == VK_FORMAT_B8G8R8A8_SRGB;

        // ── 메인 루프 ────────────────────────────────────────────────────────────
        glfwSetScrollCallback(window, scrollCallback);

        OrbitCamera orbit;
        orbit.targetY = kSphereCenterY;
        constexpr float kOrbitMinDistance = 1.6f;
        constexpr float kOrbitMaxDistance = 2.6f; // 방 halfX/halfZ(3.0)보다 작게 유지해 벽을 뚫지 않게 함
        constexpr uint32_t kShadeModeRealistic = 2u;
        constexpr float kTMin = 0.001f;
        constexpr float kTMax = 50.0f;
        constexpr float kLightElevation = 0.85f;     // radians, 고정
        constexpr float kLightRotationSpeed = 0.35f; // radians/sec — 라이트가 방을 회전하며 그림자를 움직인다

        double lastCursorX = 0.0, lastCursorY = 0.0;
        glfwGetCursorPos(window, &lastCursorX, &lastCursorY);
        bool dragging = false;
        double lastFpsPrintTime = glfwGetTime();
        int frameCount = 0;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            int fbWidth = 0, fbHeight = 0;
            glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
            if (fbWidth == 0 || fbHeight == 0) {
                glfwWaitEvents();
                continue;
            }
            if (static_cast<uint32_t>(fbWidth) != swapchain.extent.width ||
                static_cast<uint32_t>(fbHeight) != swapchain.extent.height) {
                vkDeviceWaitIdle(ctx.device);
                Swapchain newSwapchain = createSwapchain(ctx, window, swapchain.handle);
                destroySwapchain(ctx.device, swapchain);
                swapchain = newSwapchain;
                ensureOutputBuffer(swapchain.extent);
            }

            // 좌클릭 드래그로 오빗, 스크롤로 줌 (순수 GLFW 폴링, ImGui 불필요)
            double cursorX = 0.0, cursorY = 0.0;
            glfwGetCursorPos(window, &cursorX, &cursorY);
            const bool leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            if (leftDown) {
                if (dragging) {
                    orbit.yaw -= static_cast<float>(cursorX - lastCursorX) * 0.008f;
                    orbit.pitch -= static_cast<float>(cursorY - lastCursorY) * 0.008f;
                    orbit.pitch = std::clamp(orbit.pitch, -1.5f, 1.5f);
                }
                dragging = true;
            } else {
                dragging = false;
            }
            lastCursorX = cursorX;
            lastCursorY = cursorY;

            orbit.distance -= static_cast<float>(g_scrollAccum) * 0.15f;
            orbit.distance = std::clamp(orbit.distance, kOrbitMinDistance, kOrbitMaxDistance);
            g_scrollAccum = 0.0;

            const vkSpatial::vkWideBVH::RayTraceCamera camera = buildCamera(
                    orbit, static_cast<float>(swapchain.extent.width) /
                                   static_cast<float>(swapchain.extent.height));

            // 라이트가 시간에 따라 방 둘레를 회전하며 그림자가 움직인다.
            const float lightAzimuth = static_cast<float>(glfwGetTime()) * kLightRotationSpeed;
            const float lce = std::cos(kLightElevation), lse = std::sin(kLightElevation);
            const float lca = std::cos(lightAzimuth), lsa = std::sin(lightAzimuth);
            const vkSpatial::vkWideBVH::RayTraceLighting lighting{
                    lce * lsa, lse, lce * lca, groundStartIndex};

            bvh.TraceRays(camera, swapchain.extent.width, swapchain.extent.height,
                          kTMin, kTMax, kShadeModeRealistic, lighting, outputSwapRedBlue,
                          triVertBuf, outputPixelBuf);

            ++frameCount;
            const double now = glfwGetTime();
            if (now - lastFpsPrintTime >= 1.0) {
                std::cout << "[native_raytrace] FPS: " << frameCount
                          << "  triangles: " << bvh.Length()
                          << "  wide nodes: " << bvh.NodeCount() << "\n";
                frameCount = 0;
                lastFpsPrintTime = now;
            }

            vkWaitForFences(ctx.device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);

            uint32_t imageIndex = 0;
            VkResult acquireResult = vkAcquireNextImageKHR(
                    ctx.device, swapchain.handle, UINT64_MAX, imageAvailable, VK_NULL_HANDLE, &imageIndex);
            if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
                vkDeviceWaitIdle(ctx.device);
                Swapchain newSwapchain = createSwapchain(ctx, window, swapchain.handle);
                destroySwapchain(ctx.device, swapchain);
                swapchain = newSwapchain;
                ensureOutputBuffer(swapchain.extent);
                continue;
            }
            if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR)
                throw std::runtime_error("native_raytrace: vkAcquireNextImageKHR failed");

            vkResetFences(ctx.device, 1, &inFlightFence);
            vkResetCommandBuffer(cmd, 0);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            // TraceRays()의 compute dispatch는 vkComputeBase 내부에서 완전히
            // 동기적으로(vkQueueWaitIdle까지) 처리되므로, 여기서는 단순한
            // (같은 큐 패밀리 내) 동기화 배리어만으로 충분하다.
            VkBufferMemoryBarrier bufBarrier{};
            bufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            bufBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bufBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarrier.buffer = outputPixelBuf.GetBuffer();
            bufBarrier.offset = 0;
            bufBarrier.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 1, &bufBarrier, 0, nullptr);

            transitionImage(cmd, swapchain.images[imageIndex],
                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, VK_ACCESS_TRANSFER_WRITE_BIT);

            VkBufferImageCopy copyRegion{};
            copyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copyRegion.imageExtent = {swapchain.extent.width, swapchain.extent.height, 1};
            vkCmdCopyBufferToImage(cmd, outputPixelBuf.GetBuffer(), swapchain.images[imageIndex],
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

            transitionImage(cmd, swapchain.images[imageIndex],
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            VK_ACCESS_TRANSFER_WRITE_BIT, 0);

            vkEndCommandBuffer(cmd);

            const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &imageAvailable;
            submitInfo.pWaitDstStageMask = &waitStage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &renderFinished;
            if (vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS)
                throw std::runtime_error("native_raytrace: vkQueueSubmit failed");

            VkPresentInfoKHR presentInfo{};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = &renderFinished;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &swapchain.handle;
            presentInfo.pImageIndices = &imageIndex;
            VkResult presentResult = vkQueuePresentKHR(ctx.graphicsQueue, &presentInfo);
            if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
                vkDeviceWaitIdle(ctx.device);
                Swapchain newSwapchain = createSwapchain(ctx, window, swapchain.handle);
                destroySwapchain(ctx.device, swapchain);
                swapchain = newSwapchain;
                ensureOutputBuffer(swapchain.extent);
            } else if (presentResult != VK_SUCCESS) {
                throw std::runtime_error("native_raytrace: vkQueuePresentKHR failed");
            }
        }

        vkDeviceWaitIdle(ctx.device);

        vkDestroyFence(ctx.device, inFlightFence, nullptr);
        vkDestroySemaphore(ctx.device, renderFinished, nullptr);
        vkDestroySemaphore(ctx.device, imageAvailable, nullptr);
        vkDestroyCommandPool(ctx.device, graphicsPool, nullptr);

        destroySwapchain(ctx.device, swapchain);
    }

} // namespace

int main() {
    vkCommon::VkContext ctx;
    GLFWwindow *window = nullptr;
    bool ctxInitialized = false;

    try {
        if (!glfwInit())
            throw std::runtime_error("native_raytrace: glfwInit failed");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(1280, 800, "VkLBVH - Wide BVH Ray Trace (Native)", nullptr, nullptr);
        if (!window)
            throw std::runtime_error("native_raytrace: glfwCreateWindow failed");

        uint32_t glfwExtCount = 0;
        const char **glfwExtRaw = glfwGetRequiredInstanceExtensions(&glfwExtCount);
        if (!glfwExtRaw)
            throw std::runtime_error("native_raytrace: Vulkan not available via GLFW");
        std::vector<const char *> glfwExts(glfwExtRaw, glfwExtRaw + glfwExtCount);

        ctx.init(true, glfwExts, [window](VkInstance instance) {
            VkSurfaceKHR s = VK_NULL_HANDLE;
            if (glfwCreateWindowSurface(instance, window, nullptr, &s) != VK_SUCCESS)
                throw std::runtime_error("native_raytrace: glfwCreateWindowSurface failed");
            return s;
        });
        ctxInitialized = true;

        runApp(ctx, window);

        ctx.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "native_raytrace failed: " << e.what() << "\n";
        if (ctxInitialized) ctx.shutdown();
        if (window) glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
}

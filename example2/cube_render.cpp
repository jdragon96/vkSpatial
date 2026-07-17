#include "utilities/Math.h"
#include "utilities/SimpleResource.h"

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/Image.h"
#include "Engine/Render/GraphicsPipeline.h"
#include "Engine/Render/Rendering.h"
#include "Engine/Render/SwapChain.h"

#include <GLFW/glfw3.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    constexpr float kPi = 3.14159265358979323846f;

    struct Vertex {
        float position[3];
        float color[3];
    };

    struct PushConstants {
        vkMath::Mat4 mvp;
    };

    vkMath::Mat4 BuildViewProjection(VkExtent2D extent) {
        const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        const vkMath::Mat4 projection = vkMath::Perspective(60.0f * kPi / 180.0f, aspect, 0.1f, 100.0f);
        const vkMath::Mat4 view = vkMath::Translation(0.0f, 0.0f, -4.5f);
        return projection * view;
    }

    vkMath::Mat4 AnimatedModel(float timeSeconds) {
        return vkMath::RotationY(timeSeconds * 0.8f) * vkMath::RotationX(timeSeconds * 0.55f);
    }

} // namespace

int main() {
    if (!glfwInit()) {
        std::cerr << "cube_render2: failed to initialize GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(900, 700, "Engine::Render Cube", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        std::cerr << "cube_render2: failed to create window\n";
        return 1;
    }

    try {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions || glfwExtensionCount == 0)
            throw std::runtime_error("cube_render2: GLFW did not provide Vulkan extensions");
        std::vector<const char *> instanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        Engine::Core::Context context(
                true, instanceExtensions,
                [&](VkInstance instance) {
                    VkSurfaceKHR surface = VK_NULL_HANDLE;
                    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
                        throw std::runtime_error("cube_render2: failed to create window surface");
                    return surface;
                });

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

        Engine::Render::SwapChainDescriptor swapChainDescriptor{};
        swapChainDescriptor.width = static_cast<uint32_t>(framebufferWidth);
        swapChainDescriptor.height = static_cast<uint32_t>(framebufferHeight);
        Engine::Render::SwapChain swapChain(context, swapChainDescriptor);

        // Cube geometry
        Primitives cube = SimpleResource::CreateCube(1.0f);
        std::vector<Vertex> vertices;
        vertices.reserve(cube.vertices.size());
        for (size_t i = 0; i < cube.vertices.size(); ++i) {
            const Eigen::Vector3f color =
                    i < cube.colors.size() ? cube.colors[i] : Eigen::Vector3f(1.0f, 1.0f, 1.0f);
            vertices.push_back({
                    {cube.vertices[i].x(), cube.vertices[i].y(), cube.vertices[i].z()},
                    {color.x(), color.y(), color.z()},
            });
        }
        const uint32_t indexCount = static_cast<uint32_t>(cube.indices.size());

        Engine::Core::Buffer vertexBuffer(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        const uint32_t vertexBytes = static_cast<uint32_t>(vertices.size() * sizeof(Vertex));
        vertexBuffer.Allocate(vertexBytes);
        vertexBuffer.Upload(vertices.data(), vertexBytes, Engine::Core::QueueRole::Graphics);

        Engine::Core::Buffer indexBuffer(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        const uint32_t indexBytes = static_cast<uint32_t>(cube.indices.size() * sizeof(uint32_t));
        indexBuffer.Allocate(indexBytes);
        indexBuffer.Upload(cube.indices.data(), indexBytes, Engine::Core::QueueRole::Graphics);

        Engine::Core::Image depthImage(context);
        depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(swapChain.Extent()));

        const std::string shaderDir = CUBE_RENDER2_SHADER_DIR;
        Engine::Render::GraphicsPipelineDescriptor pipelineDescriptor;
        pipelineDescriptor
                .VertexShader(shaderDir + "/cube.vert.spv")
                .FragmentShader(shaderDir + "/cube.frag.spv")
                .VertexBinding<Vertex>()
                .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position))
                .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
                .ColorTarget(swapChain.Format())
                .DepthTarget(VK_FORMAT_D32_SFLOAT)
                .PushConstant<PushConstants>(VK_SHADER_STAGE_VERTEX_BIT);

        Engine::Render::GraphicsPipeline pipeline(context);
        pipeline.Build(pipelineDescriptor);

        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlightFence = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &imageAvailable) != VK_SUCCESS ||
            vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &renderFinished) != VK_SUCCESS)
            throw std::runtime_error("cube_render2: failed to create semaphores");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(context.device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS)
            throw std::runtime_error("cube_render2: failed to create fence");

        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = context.graphicsCmdPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(context.device, &cmdAllocInfo, &cmd) != VK_SUCCESS)
            throw std::runtime_error("cube_render2: failed to allocate command buffer");

        // The whole frame loop is wrapped so that, if anything throws mid-loop (after a
        // vkQueueSubmit put GPU work in flight), we wait for the device to go idle before
        // unwinding into `context`'s destructor — Context::~Context() does not itself call
        // vkDeviceWaitIdle, so the caller must guarantee no in-flight work before it runs.
        try {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();

                glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
                if (framebufferWidth == 0 || framebufferHeight == 0) {
                    glfwWaitEvents();
                    continue;
                }

                const VkExtent2D currentExtent = swapChain.Extent();
                if (currentExtent.width != static_cast<uint32_t>(framebufferWidth) ||
                    currentExtent.height != static_cast<uint32_t>(framebufferHeight)) {
                    vkDeviceWaitIdle(context.device);
                    swapChain.Recreate(static_cast<uint32_t>(framebufferWidth),
                                       static_cast<uint32_t>(framebufferHeight));
                    depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(swapChain.Extent()));
                    continue;
                }

                vkWaitForFences(context.device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);

                uint32_t imageIndex = 0;
                VkResult acquireResult = swapChain.AcquireNextImage(imageAvailable, VK_NULL_HANDLE, &imageIndex);
                if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
                    vkDeviceWaitIdle(context.device);
                    swapChain.Recreate(static_cast<uint32_t>(framebufferWidth),
                                       static_cast<uint32_t>(framebufferHeight));
                    depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(swapChain.Extent()));
                    continue;
                }
                if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR)
                    throw std::runtime_error("cube_render2: failed to acquire swapchain image");

                vkResetFences(context.device, 1, &inFlightFence);
                vkResetCommandBuffer(cmd, 0);

                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                vkBeginCommandBuffer(cmd, &beginInfo);

                VkImage swapImage = swapChain.Image(imageIndex);
                Engine::Core::Image::TransitionLayout(
                        cmd, swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
                depthImage.TransitionLayout(
                        cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                        0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

                Engine::Render::ClearOptions clear{};
                clear.color[0] = 0.025f;
                clear.color[1] = 0.027f;
                clear.color[2] = 0.032f;
                clear.color[3] = 1.0f;
                auto renderingDescriptor = Engine::Render::RenderingDescriptor::ColorDepth(
                        swapChain.Extent(), swapChain.ImageView(imageIndex), depthImage.View(), clear);

                {
                    Engine::Render::RenderingScope scope(cmd, renderingDescriptor);

                    pipeline.Bind(cmd);
                    VkBuffer vertexHandle = vertexBuffer.Handle();
                    VkDeviceSize vertexOffset = 0;
                    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexHandle, &vertexOffset);
                    vkCmdBindIndexBuffer(cmd, indexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

                    PushConstants push{
                            BuildViewProjection(swapChain.Extent()) *
                            AnimatedModel(static_cast<float>(glfwGetTime()))};
                    pipeline.PushConstants(cmd, VK_SHADER_STAGE_VERTEX_BIT, push);
                    vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
                }

                Engine::Core::Image::TransitionLayout(
                        cmd, swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0);

                vkEndCommandBuffer(cmd);

                VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                VkSubmitInfo submitInfo{};
                submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submitInfo.waitSemaphoreCount = 1;
                submitInfo.pWaitSemaphores = &imageAvailable;
                submitInfo.pWaitDstStageMask = &waitStage;
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &cmd;
                submitInfo.signalSemaphoreCount = 1;
                submitInfo.pSignalSemaphores = &renderFinished;
                if (vkQueueSubmit(context.graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS)
                    throw std::runtime_error("cube_render2: failed to submit frame");

                VkResult presentResult = swapChain.Present(imageIndex, renderFinished);
                if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
                    vkDeviceWaitIdle(context.device);
                    swapChain.Recreate(static_cast<uint32_t>(framebufferWidth),
                                       static_cast<uint32_t>(framebufferHeight));
                    depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(swapChain.Extent()));
                } else if (presentResult != VK_SUCCESS) {
                    throw std::runtime_error("cube_render2: failed to present frame");
                }
            }
        } catch (...) {
            vkDeviceWaitIdle(context.device);
            throw;
        }

        vkDeviceWaitIdle(context.device);

        vkDestroyFence(context.device, inFlightFence, nullptr);
        vkDestroySemaphore(context.device, renderFinished, nullptr);
        vkDestroySemaphore(context.device, imageAvailable, nullptr);
        // pipeline, depthImage, indexBuffer, vertexBuffer, swapChain, context all clean up
        // via their own destructors (RAII) in reverse declaration order as this scope ends.
    } catch (const std::exception &e) {
        glfwDestroyWindow(window);
        glfwTerminate();
        std::cerr << e.what() << "\n";
        return 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

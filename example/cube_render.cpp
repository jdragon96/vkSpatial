#include "utilities/Math.h"
#include "utilities/SimpleResource.h"
#include "vkRender/vkRender.h"

#include <GLFW/glfw3.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
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

    struct Object {
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        vkMath::Mat4 model = vkMath::Mat4::Identity();
    };

    class CubePass final : public vkRender::RenderPass {
    public:
        explicit CubePass(vkCommon::VkContext *context)
            : m_context(context) {
            if (!m_context)
                throw std::runtime_error("CubePass requires a valid VkContext");
            CreateBuffers();
        }

        ~CubePass() override {
            DestroyPipeline();
            DestroyDepthResources();
        }

        const char *Name() const override { return "CubePass"; }

        void SetTime(float seconds) { m_timeSeconds = seconds; }

        void Execute(vkRender::RenderContext &renderContext) override {
            if (!renderContext.swapChain)
                throw std::runtime_error("CubePass requires an active swapchain");

            const VkExtent2D extent = renderContext.swapChain->Extent();
            EnsureDepthResources(extent);
            EnsurePipeline(renderContext.swapChain->Format());

            VkCommandBuffer cmd = renderContext.commandBuffer;
            VkImage swapImage = renderContext.swapChain->Image(renderContext.imageIndex);

            const vkRender::ClearOptions clear = renderContext.view ? renderContext.view->GetClearOptions() : vkRender::ClearOptions{};
            auto desc = vkRender::RenderingDescriptor::ColorDepth(
                    extent,
                    renderContext.swapChain->ImageView(renderContext.imageIndex),
                    m_depthTarget->View(),
                    clear);

            vkRender::RenderingScope rendering(cmd, desc);

            vkRender::Image::TransitionLayout(cmd,
                                              swapImage,
                                              VK_IMAGE_ASPECT_COLOR_BIT,
                                              VK_IMAGE_LAYOUT_UNDEFINED,
                                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                              0,
                                              VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

            m_depthTarget->TransitionLayout(cmd,
                                            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                                            0,
                                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            DrawObjects(cmd, extent);

            vkRender::Image::TransitionLayout(cmd,
                                              swapImage,
                                              VK_IMAGE_ASPECT_COLOR_BIT,
                                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                              VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                              VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                              VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                              0);
        }

    private:
        vkCommon::VkContext *m_context = nullptr;
        std::unique_ptr<vkCommon::vkGPUMemory> m_vertexBuffer;
        std::unique_ptr<vkCommon::vkGPUMemory> m_indexBuffer;
        std::unique_ptr<vkRender::GraphicsPipeline> m_pipeline;
        std::unique_ptr<vkRender::Image> m_depthTarget;
        std::vector<Object> m_objects;
        float m_timeSeconds = 0.0f;

        void CreateBuffers() {
            Primitives cube = SimpleResource::CreateCube(1.0f);
            std::vector<Vertex> vertices;
            vertices.reserve(cube.vertices.size());
            for (size_t i = 0; i < cube.vertices.size(); ++i) {
                const Eigen::Vector3f color =
                        i < cube.colors.size()
                                ? cube.colors[i]
                                : Eigen::Vector3f(1.0f, 1.0f, 1.0f);
                vertices.push_back({
                        {cube.vertices[i].x(), cube.vertices[i].y(), cube.vertices[i].z()},
                        {color.x(), color.y(), color.z()},
                });
            }
            m_objects = {
                    {0, static_cast<uint32_t>(cube.indices.size()), vkMath::Mat4::Identity()},
            };

            VkCommandPool uploadPool = VK_NULL_HANDLE;
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            poolInfo.queueFamilyIndex = m_context->graphicsFamily;
            if (vkCreateCommandPool(m_context->device, &poolInfo, nullptr, &uploadPool) != VK_SUCCESS)
                throw std::runtime_error("cube_render: failed to create upload command pool");

            const uint32_t vertexBytes = static_cast<uint32_t>(vertices.size() * sizeof(Vertex));
            m_vertexBuffer = std::make_unique<vkCommon::vkGPUMemory>(m_context->device, m_context->physDevice, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            if (!m_vertexBuffer->Allocate(vertexBytes) ||
                !m_vertexBuffer->Upload(vertices.data(),
                                        vertexBytes,
                                        m_context->graphicsQueue,
                                        uploadPool))
                throw std::runtime_error("cube_render: failed to upload vertex buffer");

            const uint32_t indexBytes = static_cast<uint32_t>(cube.indices.size() * sizeof(uint32_t));
            m_indexBuffer = std::make_unique<vkCommon::vkGPUMemory>(m_context->device, m_context->physDevice, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            if (!m_indexBuffer->Allocate(indexBytes) ||
                !m_indexBuffer->Upload(cube.indices.data(),
                                       indexBytes,
                                       m_context->graphicsQueue,
                                       uploadPool))
                throw std::runtime_error("cube_render: failed to upload index buffer");

            vkDestroyCommandPool(m_context->device, uploadPool, nullptr);
        }

        void EnsurePipeline(VkFormat colorFormat) {
            if (m_pipeline && m_pipeline->MatchesColorTarget(0, colorFormat))
                return;

            DestroyPipeline();

            const std::string shaderDir = CUBE_RENDER_SHADER_DIR;
            vkRender::GraphicsPipelineDescriptor descriptor;
            descriptor
                    .VertexShader(shaderDir + "/cube.vert.spv")
                    .FragmentShader(shaderDir + "/cube.frag.spv")
                    .VertexBinding<Vertex>()
                    .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position))
                    .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
                    .ColorTarget(colorFormat)
                    .DepthTarget(VK_FORMAT_D32_SFLOAT)
                    .PushConstant<PushConstants>(VK_SHADER_STAGE_VERTEX_BIT);

            m_pipeline = std::make_unique<vkRender::GraphicsPipeline>(m_context->device);
            m_pipeline->Build(descriptor);
        }

        void DestroyPipeline() {
            m_pipeline.reset();
        }

        void EnsureDepthResources(VkExtent2D extent) {
            if (m_depthTarget && m_depthTarget->Matches(extent, VK_FORMAT_D32_SFLOAT))
                return;

            DestroyDepthResources();

            m_depthTarget = std::make_unique<vkRender::Image>(
                    m_context,
                    vkRender::ImageDescriptor::Depth2D(extent, VK_FORMAT_D32_SFLOAT));
        }

        void DestroyDepthResources() {
            m_depthTarget.reset();
        }

        void DrawObjects(VkCommandBuffer cmd, VkExtent2D extent) {
            const VkDeviceSize vertexOffset = 0;
            VkBuffer vertexBuffer = m_vertexBuffer->GetBuffer();
            m_pipeline->Bind(cmd);
            vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &vertexOffset);
            vkCmdBindIndexBuffer(cmd, m_indexBuffer->GetBuffer(), 0, VK_INDEX_TYPE_UINT32);

            const vkMath::Mat4 viewProj = BuildViewProjection(extent);
            for (const Object &object: m_objects) {
                const PushConstants push{viewProj * AnimatedModel(object)};
                m_pipeline->PushConstants(cmd, VK_SHADER_STAGE_VERTEX_BIT, push);
                vkCmdDrawIndexed(cmd, object.indexCount, 1, object.firstIndex, 0, 0);
            }
        }

        vkMath::Mat4 BuildViewProjection(VkExtent2D extent) const {
            const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
            const vkMath::Mat4 projection = vkMath::Perspective(60.0f * kPi / 180.0f, aspect, 0.1f, 100.0f);
            const vkMath::Mat4 view = vkMath::Translation(0.0f, 0.0f, -4.5f);
            return projection * view;
        }

        vkMath::Mat4 AnimatedModel(const Object &object) const {
            const vkMath::Mat4 rotation = vkMath::RotationY(m_timeSeconds * 0.8f) *
                                          vkMath::RotationX(m_timeSeconds * 0.55f);
            return object.model * rotation;
        }
    };

    vkRender::KeyInput *WindowKeyInput(GLFWwindow *window) {
        return static_cast<vkRender::KeyInput *>(glfwGetWindowUserPointer(window));
    }
} // namespace

int main() {
    if (!glfwInit()) {
        std::cerr << "cube_render: failed to initialize GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(900, 700, "vkRender Cube", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        std::cerr << "cube_render: failed to create window\n";
        return 1;
    }

    vkCommon::VkContext context;

    try {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions || glfwExtensionCount == 0)
            throw std::runtime_error("cube_render: GLFW did not provide Vulkan extensions");

        std::vector<const char *> instanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        context.init(true, instanceExtensions,
                     [&](VkInstance instance) {
                         VkSurfaceKHR surface = VK_NULL_HANDLE;
                         if (glfwCreateWindowSurface(instance, window,
                                                     nullptr, &surface) != VK_SUCCESS)
                             throw std::runtime_error("cube_render: failed to create window surface");
                         return surface;
                     });

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

        auto engine = vkRender::Engine::Create(&context);

        vkRender::SwapChainDescriptor swapChainDescriptor{};
        swapChainDescriptor.width = static_cast<uint32_t>(framebufferWidth);
        swapChainDescriptor.height = static_cast<uint32_t>(framebufferHeight);
        swapChainDescriptor.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        auto swapChain = engine->CreateSwapChain(swapChainDescriptor);
        auto renderer = engine->CreateRenderer();
        auto scene = engine->CreateScene();
        auto camera = engine->CreateCamera();
        auto view = engine->CreateView();
        auto graph = engine->CreateRenderGraph();
        auto keyInput = engine->CreateKeyInput();

        glfwSetWindowUserPointer(window, keyInput.get());
        glfwSetKeyCallback(window, [](GLFWwindow *callbackWindow, int key, int, int action, int mods) {
            vkRender::KeyInput *keys = WindowKeyInput(callbackWindow);
            if (!keys)
                return;

            vkRender::KeyEventType type;
            if (action == GLFW_PRESS)
                type = vkRender::KeyEventType::Press;
            else if (action == GLFW_RELEASE)
                type = vkRender::KeyEventType::Release;
            else
                type = vkRender::KeyEventType::Repeat;

            keys->OnKey(key, type, static_cast<uint32_t>(mods), glfwGetTime());
        });

        bool captureRequested = false;
        keyInput->AddListener(vkRender::KeyEventType::Press,
                              [&captureRequested](vkRender::KeyEvent &event) {
                                  if (event.keyCode == GLFW_KEY_ENTER)
                                      captureRequested = true;
                              });

        auto cubePass = std::make_unique<CubePass>(&context);
        CubePass *cubePassPtr = cubePass.get();
        graph->AddPass(std::move(cubePass));

        vkRender::ClearOptions clearOptions{};
        clearOptions.color[0] = 0.025f;
        clearOptions.color[1] = 0.027f;
        clearOptions.color[2] = 0.032f;
        clearOptions.color[3] = 1.0f;
        view->SetClearOptions(clearOptions);
        view->SetScene(scene.get());
        view->SetCamera(camera.get());
        view->SetRenderGraph(graph.get());

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
            if (framebufferWidth == 0 || framebufferHeight == 0) {
                glfwWaitEvents();
                continue;
            }

            const VkExtent2D swapExtent = swapChain->Extent();
            if (swapExtent.width != static_cast<uint32_t>(framebufferWidth) || swapExtent.height != static_cast<uint32_t>(framebufferHeight)) {
                vkDeviceWaitIdle(context.device);
                swapChain->Recreate(static_cast<uint32_t>(framebufferWidth),
                                    static_cast<uint32_t>(framebufferHeight));
                renderer->ClearSwapChainRecreateFlag();
                continue;
            }

            cubePassPtr->SetTime(static_cast<float>(glfwGetTime()));

            if (!renderer->BeginFrame(*swapChain)) {
                vkDeviceWaitIdle(context.device);
                swapChain->Recreate(static_cast<uint32_t>(framebufferWidth),
                                    static_cast<uint32_t>(framebufferHeight));
                renderer->ClearSwapChainRecreateFlag();
                continue;
            }

            renderer->Render(*view);
            renderer->EndFrame();

            if (captureRequested) {
                captureRequested = false;
                vkRender::Capture::CaptureToPNG(
                        &context,
                        swapChain->Image(renderer->CurrentImageIndex()),
                        swapChain->Format(),
                        swapChain->Extent(),
                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        vkRender::Capture::TimestampedPath());
                std::cout << "cube_render: screenshot saved\n";
            }

            if (renderer->NeedsSwapChainRecreate()) {
                vkDeviceWaitIdle(context.device);
                swapChain->Recreate(static_cast<uint32_t>(framebufferWidth),
                                    static_cast<uint32_t>(framebufferHeight));
                renderer->ClearSwapChainRecreateFlag();
            }
        }

        vkDeviceWaitIdle(context.device);

        graph.reset();
        view.reset();
        camera.reset();
        scene.reset();
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

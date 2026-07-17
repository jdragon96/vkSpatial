#include "utilities/Math.h"
#include "vkRender/vkRender.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    constexpr float kPi = 3.14159265358979323846f;
    constexpr uint32_t kShadowMapSize = 2048;

    struct Vertex {
        float position[3];
        float normal[3];
        float color[3];
    };

    struct PushConstants {
        vkMath::Mat4 model = vkMath::Mat4::Identity();
        vkMath::Mat4 viewProj = vkMath::Mat4::Identity();
        vkMath::Mat4 lightViewProj = vkMath::Mat4::Identity();
        float lightDir[4] = {};
    };

    struct Object {
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        vkMath::Mat4 model = vkMath::Mat4::Identity();
    };

    class RealtimeShadowPass final : public vkRender::RenderPass {
    public:
        explicit RealtimeShadowPass(vkCommon::VkContext *context)
            : m_context(context) {
            CreateGeometry();
            CreateShadowResources();
            CreateDescriptorResources();
        }

        ~RealtimeShadowPass() override {
            DestroyPipelines();
            DestroyDescriptorResources();
            DestroyShadowResources();
            DestroyCaptureBuffer();
        }

        const char *Name() const override { return "RealtimeShadowPass"; }

        void SetTime(float seconds) { m_timeSeconds = seconds; }

        void Execute(vkRender::RenderContext &renderContext) override {
            if (!renderContext.swapChain)
                throw std::runtime_error("RealtimeShadowPass requires an active swapchain");

            const VkExtent2D extent = renderContext.swapChain->Extent();
            EnsureSceneDepthResources(extent);
            EnsurePipelines(renderContext.swapChain->Format());

            VkCommandBuffer cmd = renderContext.commandBuffer;
            const uint32_t imageIndex = renderContext.imageIndex;

            const vkMath::Vec3 cameraEye{0.0f, 3.0f, 7.0f};
            const vkMath::Vec3 cameraTarget{0.0f, 0.75f, 0.0f};
            const vkMath::Vec3 worldUp{0.0f, 1.0f, 0.0f};
            const vkMath::Mat4 view = vkMath::LookAt(cameraEye, cameraTarget, worldUp);
            const vkMath::Mat4 proj = vkMath::Perspective(
                    55.0f * kPi / 180.0f,
                    static_cast<float>(extent.width) / static_cast<float>(extent.height),
                    0.1f, 80.0f);
            const vkMath::Mat4 viewProj = proj * view;

            const vkMath::Vec3 lightPos{
                    std::sin(m_timeSeconds * 0.55f) * 3.8f,
                    5.2f,
                    std::cos(m_timeSeconds * 0.55f) * 2.6f};
            const vkMath::Vec3 lightTarget{0.0f, 0.55f, 0.0f};
            const vkMath::Vec3 lightDir = (lightTarget - lightPos).normalized();
            const vkMath::Mat4 lightView = vkMath::LookAt(lightPos, lightTarget, worldUp);
            const vkMath::Mat4 lightProj = vkMath::Orthographic(-6.2f, 6.2f, -6.2f, 6.2f, 0.1f, 15.0f);
            const vkMath::Mat4 lightViewProj = lightProj * lightView;

            UpdateDrawModels(m_timeSeconds);

            RecordShadowPass(cmd, lightViewProj, lightDir);
            RecordScenePass(cmd, *renderContext.swapChain, imageIndex, viewProj, lightViewProj, lightDir);
        }

        void RequestCapture() {
            m_captureRequested = true;
        }

        bool HasCaptureToSave() const {
            return m_captureReady;
        }

        std::string SaveCapturedImage() {
            if (!m_captureReady)
                return {};
            if (m_captureBuffer == VK_NULL_HANDLE ||
                m_captureMemory == VK_NULL_HANDLE ||
                m_captureExtent.width == 0 ||
                m_captureExtent.height == 0)
                throw std::runtime_error("realtime_shadow: no capture buffer to save");

            void *rawMapped = nullptr;
            if (vkMapMemory(m_context->device,
                            m_captureMemory,
                            0,
                            m_captureBufferBytes,
                            0,
                            &rawMapped) != VK_SUCCESS)
                throw std::runtime_error("realtime_shadow: failed to map capture buffer");

            const uint8_t *mapped = static_cast<const uint8_t *>(rawMapped);
            const size_t pixelCount =
                    static_cast<size_t>(m_captureExtent.width) *
                    static_cast<size_t>(m_captureExtent.height);
            std::vector<uint8_t> rgb(pixelCount * 3u);

            for (size_t i = 0; i < pixelCount; ++i) {
                const uint8_t *src = mapped + i * 4u;
                uint8_t *dst = rgb.data() + i * 3u;

                switch (m_captureFormat) {
                    case VK_FORMAT_B8G8R8A8_UNORM:
                    case VK_FORMAT_B8G8R8A8_SRGB:
                        dst[0] = src[2];
                        dst[1] = src[1];
                        dst[2] = src[0];
                        break;
                    case VK_FORMAT_R8G8B8A8_UNORM:
                    case VK_FORMAT_R8G8B8A8_SRGB:
                        dst[0] = src[0];
                        dst[1] = src[1];
                        dst[2] = src[2];
                        break;
                    default:
                        vkUnmapMemory(m_context->device, m_captureMemory);
                        throw std::runtime_error("realtime_shadow: unsupported capture format");
                }
            }

            vkUnmapMemory(m_context->device, m_captureMemory);

            std::ostringstream filename;
            filename << "realtime_shadow_capture_"
                     << std::setw(4) << std::setfill('0') << m_captureSequence++
                     << ".ppm";

            std::ofstream file(filename.str(), std::ios::binary);
            if (!file)
                throw std::runtime_error("realtime_shadow: failed to open capture file");

            file << "P6\n"
                 << m_captureExtent.width << " " << m_captureExtent.height
                 << "\n255\n";
            file.write(reinterpret_cast<const char *>(rgb.data()),
                       static_cast<std::streamsize>(rgb.size()));
            if (!file)
                throw std::runtime_error("realtime_shadow: failed to write capture file");

            m_captureReady = false;
            return filename.str();
        }

    private:
        vkCommon::VkContext *m_context = nullptr;
        std::unique_ptr<vkCommon::vkGPUMemory> m_vertexBuffer;
        std::unique_ptr<vkCommon::vkGPUMemory> m_indexBuffer;
        std::vector<Object> m_objects;

        std::unique_ptr<vkRender::Image> m_shadowTarget;
        VkSampler m_shadowSampler = VK_NULL_HANDLE;

        std::unique_ptr<vkRender::Image> m_sceneDepthTarget;

        VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;

        std::unique_ptr<vkRender::GraphicsPipeline> m_shadowPipeline;
        std::unique_ptr<vkRender::GraphicsPipeline> m_scenePipeline;

        VkBuffer m_captureBuffer = VK_NULL_HANDLE;
        VkDeviceMemory m_captureMemory = VK_NULL_HANDLE;
        VkDeviceSize m_captureBufferBytes = 0;
        VkExtent2D m_captureExtent{};
        VkFormat m_captureFormat = VK_FORMAT_UNDEFINED;
        bool m_captureRequested = false;
        bool m_captureReady = false;
        float m_timeSeconds = 0.0f;
        uint32_t m_captureSequence = 1;

        void CreateGeometry() {
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;

            auto addVertex = [&](vkMath::Vec3 p, vkMath::Vec3 n, vkMath::Vec3 c) {
                vertices.push_back({{p.x(), p.y(), p.z()},
                                    {n.x(), n.y(), n.z()},
                                    {c.x(), c.y(), c.z()}});
            };

            auto addPlane = [&]() {
                const uint32_t firstVertex = static_cast<uint32_t>(vertices.size());
                const uint32_t firstIndex = static_cast<uint32_t>(indices.size());
                const vkMath::Vec3 color{0.58f, 0.58f, 0.60f};
                addVertex({-6.0f, 0.0f, -6.0f}, {0.0f, 1.0f, 0.0f}, color);
                addVertex({6.0f, 0.0f, -6.0f}, {0.0f, 1.0f, 0.0f}, color);
                addVertex({6.0f, 0.0f, 6.0f}, {0.0f, 1.0f, 0.0f}, color);
                addVertex({-6.0f, 0.0f, 6.0f}, {0.0f, 1.0f, 0.0f}, color);
                indices.insert(indices.end(), {firstVertex + 0, firstVertex + 1, firstVertex + 2,
                                               firstVertex + 2, firstVertex + 3, firstVertex + 0});
                m_objects.push_back({firstIndex, 6, vkMath::Mat4::Identity()});
            };

            auto addCube = [&]() {
                const uint32_t firstVertex = static_cast<uint32_t>(vertices.size());
                const uint32_t firstIndex = static_cast<uint32_t>(indices.size());
                const vkMath::Vec3 color{0.35f, 0.47f, 0.92f};

                const std::array<vkMath::Vec3, 8> p = {{
                        {-0.75f, -0.75f, -0.75f},
                        {0.75f, -0.75f, -0.75f},
                        {0.75f, 0.75f, -0.75f},
                        {-0.75f, 0.75f, -0.75f},
                        {-0.75f, -0.75f, 0.75f},
                        {0.75f, -0.75f, 0.75f},
                        {0.75f, 0.75f, 0.75f},
                        {-0.75f, 0.75f, 0.75f},
                }};

                auto face = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d, vkMath::Vec3 n) {
                    const uint32_t base = static_cast<uint32_t>(vertices.size());
                    addVertex(p[a], n, color);
                    addVertex(p[b], n, color);
                    addVertex(p[c], n, color);
                    addVertex(p[d], n, color);
                    indices.insert(indices.end(), {base + 0, base + 1, base + 2,
                                                   base + 2, base + 3, base + 0});
                };

                face(4, 5, 6, 7, {0.0f, 0.0f, 1.0f});
                face(1, 0, 3, 2, {0.0f, 0.0f, -1.0f});
                face(0, 4, 7, 3, {-1.0f, 0.0f, 0.0f});
                face(5, 1, 2, 6, {1.0f, 0.0f, 0.0f});
                face(3, 7, 6, 2, {0.0f, 1.0f, 0.0f});
                face(0, 1, 5, 4, {0.0f, -1.0f, 0.0f});

                (void) firstVertex;
                m_objects.push_back({firstIndex, 36, vkMath::Translation(0.0f, 0.75f, 0.0f)});
            };

            addPlane();
            addCube();

            VkCommandPool uploadPool = VK_NULL_HANDLE;
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            poolInfo.queueFamilyIndex = m_context->graphicsFamily;
            if (vkCreateCommandPool(m_context->device, &poolInfo, nullptr, &uploadPool) != VK_SUCCESS)
                throw std::runtime_error("realtime_shadow: failed to create upload command pool");

            m_vertexBuffer = std::make_unique<vkCommon::vkGPUMemory>(m_context->device, m_context->physDevice, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            const uint32_t vertexBytes = static_cast<uint32_t>(vertices.size() * sizeof(Vertex));
            if (!m_vertexBuffer->Allocate(vertexBytes) ||
                !m_vertexBuffer->Upload(vertices.data(), vertexBytes, m_context->graphicsQueue, uploadPool))
                throw std::runtime_error("realtime_shadow: failed to upload vertices");

            m_indexBuffer = std::make_unique<vkCommon::vkGPUMemory>(m_context->device, m_context->physDevice, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            const uint32_t indexBytes =
                    static_cast<uint32_t>(indices.size() * sizeof(uint32_t));
            if (!m_indexBuffer->Allocate(indexBytes) ||
                !m_indexBuffer->Upload(indices.data(), indexBytes, m_context->graphicsQueue, uploadPool))
                throw std::runtime_error("realtime_shadow: failed to upload indices");

            vkDestroyCommandPool(m_context->device, uploadPool, nullptr);
        }

        void UpdateDrawModels(float timeSeconds) {
            if (m_objects.size() < 2)
                return;

            const vkMath::Mat4 rotation = vkMath::RotationY(timeSeconds * 0.7f);
            const vkMath::Mat4 translate = vkMath::Translation(0.0f, 0.78f, 0.0f);
            m_objects[1].model = translate * rotation;
        }

        void CreateShadowPipeline() {
            const std::string shaderDir = REALTIME_SHADOW_SHADER_DIR;
            vkRender::GraphicsPipelineDescriptor descriptor;
            descriptor
                    .VertexShader(shaderDir + "/realtime_shadow_depth.vert.spv")
                    .VertexBinding<Vertex>()
                    .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position))
                    .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
                    .VertexAttribute(2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
                    .DepthTarget(VK_FORMAT_D32_SFLOAT)
                    .DepthBias(1.2f, 1.8f)
                    .PushConstant<PushConstants>(
                            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

            m_shadowPipeline = std::make_unique<vkRender::GraphicsPipeline>(m_context->device);
            m_shadowPipeline->Build(descriptor);
        }

        void CreateScenePipeline(VkFormat colorFormat) {
            const std::string shaderDir = REALTIME_SHADOW_SHADER_DIR;
            vkRender::GraphicsPipelineDescriptor descriptor;
            descriptor.descriptorSetLayouts.push_back(m_descriptorSetLayout);
            descriptor
                    .VertexShader(shaderDir + "/realtime_shadow_scene.vert.spv")
                    .FragmentShader(shaderDir + "/realtime_shadow_scene.frag.spv")
                    .VertexBinding<Vertex>()
                    .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position))
                    .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
                    .VertexAttribute(2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
                    .ColorTarget(colorFormat)
                    .DepthTarget(VK_FORMAT_D32_SFLOAT)
                    .PushConstant<PushConstants>(
                            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

            m_scenePipeline = std::make_unique<vkRender::GraphicsPipeline>(m_context->device);
            m_scenePipeline->Build(descriptor);
        }

        void DestroyPipelines() {
            m_scenePipeline.reset();
            m_shadowPipeline.reset();
        }

        void CreateDescriptorResources() {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 1;
            layoutInfo.pBindings = &binding;
            if (vkCreateDescriptorSetLayout(m_context->device, &layoutInfo,
                                            nullptr, &m_descriptorSetLayout) != VK_SUCCESS)
                throw std::runtime_error("realtime_shadow: failed to create descriptor set layout");

            VkDescriptorPoolSize poolSize{};
            poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            poolSize.descriptorCount = 1;

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            if (vkCreateDescriptorPool(m_context->device, &poolInfo,
                                       nullptr, &m_descriptorPool) != VK_SUCCESS)
                throw std::runtime_error("realtime_shadow: failed to create descriptor pool");

            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = m_descriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &m_descriptorSetLayout;
            if (vkAllocateDescriptorSets(m_context->device, &allocInfo,
                                         &m_descriptorSet) != VK_SUCCESS)
                throw std::runtime_error("realtime_shadow: failed to allocate descriptor set");

            VkDescriptorImageInfo imageInfo{};
            imageInfo.sampler = m_shadowSampler;
            imageInfo.imageView = m_shadowTarget->View();
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = m_descriptorSet;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(m_context->device, 1, &write, 0, nullptr);
        }

        void DestroyDescriptorResources() {
            if (m_descriptorPool != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(m_context->device, m_descriptorPool, nullptr);
            if (m_descriptorSetLayout != VK_NULL_HANDLE)
                vkDestroyDescriptorSetLayout(m_context->device, m_descriptorSetLayout, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
            m_descriptorSetLayout = VK_NULL_HANDLE;
            m_descriptorSet = VK_NULL_HANDLE;
        }

        void EnsureCaptureBuffer(VkExtent2D extent) {
            const VkDeviceSize bytes =
                    static_cast<VkDeviceSize>(extent.width) *
                    static_cast<VkDeviceSize>(extent.height) * 4u;
            if (m_captureBuffer != VK_NULL_HANDLE &&
                m_captureExtent.width == extent.width &&
                m_captureExtent.height == extent.height &&
                m_captureBufferBytes == bytes)
                return;

            DestroyCaptureBuffer();

            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = bytes;
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(m_context->device, &bufferInfo,
                               nullptr, &m_captureBuffer) != VK_SUCCESS)
                throw std::runtime_error("realtime_shadow: failed to create capture buffer");

            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(m_context->device,
                                          m_captureBuffer,
                                          &requirements);

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = requirements.size;
            allocInfo.memoryTypeIndex = FindMemoryType(
                    requirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (vkAllocateMemory(m_context->device, &allocInfo,
                                 nullptr, &m_captureMemory) != VK_SUCCESS) {
                DestroyCaptureBuffer();
                throw std::runtime_error("realtime_shadow: failed to allocate capture memory");
            }

            vkBindBufferMemory(m_context->device, m_captureBuffer, m_captureMemory, 0);
            m_captureExtent = extent;
            m_captureBufferBytes = bytes;
            m_captureReady = false;
        }

        void DestroyCaptureBuffer() {
            if (m_captureBuffer != VK_NULL_HANDLE)
                vkDestroyBuffer(m_context->device, m_captureBuffer, nullptr);
            if (m_captureMemory != VK_NULL_HANDLE)
                vkFreeMemory(m_context->device, m_captureMemory, nullptr);
            m_captureBuffer = VK_NULL_HANDLE;
            m_captureMemory = VK_NULL_HANDLE;
            m_captureBufferBytes = 0;
            m_captureExtent = {};
            m_captureFormat = VK_FORMAT_UNDEFINED;
            m_captureReady = false;
        }

        void CreateShadowResources() {
            vkRender::ImageDescriptor descriptor = vkRender::ImageDescriptor::Depth2D(
                    {kShadowMapSize, kShadowMapSize}, VK_FORMAT_D32_SFLOAT);
            descriptor.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
            m_shadowTarget = std::make_unique<vkRender::Image>(m_context, descriptor);

            VkSamplerCreateInfo samplerInfo{};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_NEAREST;
            samplerInfo.minFilter = VK_FILTER_NEAREST;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.maxLod = 1.0f;
            if (vkCreateSampler(m_context->device, &samplerInfo,
                                nullptr, &m_shadowSampler) != VK_SUCCESS)
                throw std::runtime_error("realtime_shadow: failed to create shadow sampler");
        }

        void DestroyShadowResources() {
            if (m_shadowSampler != VK_NULL_HANDLE)
                vkDestroySampler(m_context->device, m_shadowSampler, nullptr);
            m_shadowSampler = VK_NULL_HANDLE;
            m_shadowTarget.reset();
        }

        void EnsurePipelines(VkFormat colorFormat) {
            /*
            그림자, 씬 렌더 파이프라인의 픽셀 포맷이 달라지면, 파이프라인을 재생성한다.
            */
            if (m_scenePipeline &&
                m_shadowPipeline &&
                m_scenePipeline->MatchesColorTarget(0, colorFormat))
                return;

            DestroyPipelines();
            CreateShadowPipeline();
            CreateScenePipeline(colorFormat);
        }

        void EnsureSceneDepthResources(VkExtent2D extent) {
            /*
            화면의 크기가 바뀌었을 때, Depth buffer를 새로 생성해준다.'

            Vulkan에서는 이미지 하나를 만들 때 
            - 저장 공간, 
            - 이미지 객체, 
            - 이미지를 바라보는 방식
            을 각각 분리해서 관리해. 그래서 이 세 개가 보통 한 세트로 움직여.

            VkDeviceMemory
                └─ 실제 GPU 메모리 공간
                    ↑ 바인딩
            VkImage
                └─ 이미지의 크기, 포맷, 사용 목적을 정의한 객체
                    ↑ 참조
            VkImageView
                └─ VkImage를 어떤 방식으로 해석해서 사용할지 정의
            */
            if (m_sceneDepthTarget &&
                m_sceneDepthTarget->Matches(extent, VK_FORMAT_D32_SFLOAT))
                return;

            m_sceneDepthTarget = std::make_unique<vkRender::Image>(
                    m_context,
                    vkRender::ImageDescriptor::Depth2D(extent, VK_FORMAT_D32_SFLOAT));
        }

        uint32_t FindMemoryType(uint32_t typeFilter,
                                VkMemoryPropertyFlags properties) const {
            VkPhysicalDeviceMemoryProperties memoryProperties{};
            vkGetPhysicalDeviceMemoryProperties(m_context->physDevice, &memoryProperties);
            for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
                if ((typeFilter & (1u << i)) &&
                    (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                    return i;
            }
            throw std::runtime_error("realtime_shadow: no suitable memory type");
        }

        void RecordShadowPass(VkCommandBuffer cmd,
                              const vkMath::Mat4 &lightViewProj,
                              vkMath::Vec3 lightDir) {
            m_shadowTarget->TransitionLayout(cmd,
                                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                                             0,
                                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            vkRender::RenderingDescriptor descriptor({kShadowMapSize, kShadowMapSize});
            descriptor.SetDepthAttachment(
                    vkRender::DepthAttachment(m_shadowTarget->View())
                            .Clear(1.0f)
                            .Store(true)
                            .Build());
            vkRender::RenderingScope rendering(cmd, descriptor);

            BindGeometry(cmd);
            m_shadowPipeline->Bind(cmd);
            for (const Object &object: m_objects) {
                PushConstants pc{};
                pc.model = object.model;
                pc.lightViewProj = lightViewProj;
                pc.lightDir[0] = lightDir.x();
                pc.lightDir[1] = lightDir.y();
                pc.lightDir[2] = lightDir.z();
                m_shadowPipeline->PushConstants(
                        cmd,
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        pc);
                vkCmdDrawIndexed(cmd, object.indexCount, 1, object.firstIndex, 0, 0);
            }
            rendering.End();

            m_shadowTarget->TransitionLayout(cmd,
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                             VK_ACCESS_SHADER_READ_BIT);
        }

        void RecordScenePass(VkCommandBuffer cmd,
                             vkRender::SwapChain &swapChain,
                             uint32_t imageIndex,
                             const vkMath::Mat4 &viewProj,
                             const vkMath::Mat4 &lightViewProj,
                             vkMath::Vec3 lightDir) {
            VkImage swapImage = swapChain.Image(imageIndex);
            vkRender::Image::TransitionLayout(cmd,
                                              swapImage,
                                              VK_IMAGE_ASPECT_COLOR_BIT,
                                              VK_IMAGE_LAYOUT_UNDEFINED,
                                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                              0,
                                              VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
            m_sceneDepthTarget->TransitionLayout(cmd,
                                                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                                                 0,
                                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            vkRender::ClearOptions clear{};
            clear.color[0] = 0.045f;
            clear.color[1] = 0.047f;
            clear.color[2] = 0.055f;
            clear.color[3] = 1.0f;
            auto descriptor = vkRender::RenderingDescriptor::ColorDepth(
                    swapChain.Extent(),
                    swapChain.ImageView(imageIndex),
                    m_sceneDepthTarget->View(),
                    clear);
            descriptor.depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            vkRender::RenderingScope rendering(cmd, descriptor);

            BindGeometry(cmd);
            m_scenePipeline->Bind(cmd);
            vkCmdBindDescriptorSets(cmd,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_scenePipeline->Layout(),
                                    0,
                                    1,
                                    &m_descriptorSet,
                                    0,
                                    nullptr);

            for (const Object &object: m_objects) {
                PushConstants pc{};
                pc.model = object.model;
                pc.viewProj = viewProj;
                pc.lightViewProj = lightViewProj;
                pc.lightDir[0] = lightDir.x();
                pc.lightDir[1] = lightDir.y();
                pc.lightDir[2] = lightDir.z();
                m_scenePipeline->PushConstants(
                        cmd,
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        pc);
                vkCmdDrawIndexed(cmd, object.indexCount, 1, object.firstIndex, 0, 0);
            }
            rendering.End();

            if (m_captureRequested) {
                RecordCaptureCopy(cmd, swapChain, imageIndex);
                m_captureRequested = false;
                m_captureReady = true;
            } else {
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
        }

        void RecordCaptureCopy(VkCommandBuffer cmd, vkRender::SwapChain &swapChain, uint32_t imageIndex) {
            const VkExtent2D extent = swapChain.Extent();
            EnsureCaptureBuffer(extent);
            m_captureFormat = swapChain.Format();

            VkImage swapImage = swapChain.Image(imageIndex);
            vkRender::Capture::RecordImageToBufferCopy(
                    cmd,
                    swapImage,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    m_captureBuffer,
                    m_captureBufferBytes,
                    extent);
        }

        void BindGeometry(VkCommandBuffer cmd) {
            const VkDeviceSize vertexOffset = 0;
            VkBuffer vertexBuffer = m_vertexBuffer->GetBuffer();
            vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &vertexOffset);
            vkCmdBindIndexBuffer(cmd, m_indexBuffer->GetBuffer(), 0, VK_INDEX_TYPE_UINT32);
        }

        void CaptureImage(
                VkCommandBuffer cmd,
                VkImage srcImage,
                VkExtent2D extent,
                VkFormat format) {
            EnsureCaptureBuffer(extent);

            auto GetBytes = [&](VkExtent2D extent, VkFormat fmt) -> unsigned int {
                unsigned int numberOfPixel = static_cast<unsigned int>(extent.width) * static_cast<unsigned int>(extent.height);
                unsigned int channel = 0;

                switch (fmt) {
                    case VK_FORMAT_R8G8B8A8_UNORM:
                    case VK_FORMAT_B8G8R8A8_UNORM:
                    case VK_FORMAT_R8G8B8A8_SRGB:
                    case VK_FORMAT_B8G8R8A8_SRGB:
                        return 4;
                    case VK_FORMAT_R8G8B8_UNORM:
                    case VK_FORMAT_B8G8R8_UNORM:
                        return 3;
                    default:
                        throw std::runtime_error("realtime_shadow: unsupported capture format");
                }
            };

            auto CreateBuffer = [&](VkExtent2D extent, VkFormat fmt) -> VkBuffer {
                VkBufferCreateInfo bufferInfo{};
                bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bufferInfo.size = GetBytes(extent, fmt);
                bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                VkBuffer buffer;
                if (vkCreateBuffer(
                            m_context->device,
                            &bufferInfo,
                            nullptr, &buffer) != VK_SUCCESS)
                    throw std::runtime_error("realtime_shadow: failed to create capture buffer");
                return buffer;
            };

            unsigned int bytes = GetBytes(extent, format);
            VkBuffer captureBuffer = CreateBuffer(extent, format);

            vkRender::Capture::RecordImageToBufferCopy(
                    cmd,
                    srcImage,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    m_captureBuffer,
                    m_captureBufferBytes,
                    extent);
        }
    };

    vkRender::KeyInput *WindowKeyInput(GLFWwindow *window) {
        return static_cast<vkRender::KeyInput *>(glfwGetWindowUserPointer(window));
    }

} // namespace

int main() {
    if (!glfwInit()) {
        std::cerr << "realtime_shadow: failed to initialize GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(960, 720, "vkRender Realtime Shadow Map", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        std::cerr << "realtime_shadow: failed to create window\n";
        return 1;
    }

    vkCommon::VkContext context;

    try {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions || glfwExtensionCount == 0)
            throw std::runtime_error("realtime_shadow: GLFW did not provide Vulkan extensions");
        std::vector<const char *> instanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        context.init(true, instanceExtensions,
                     [&](VkInstance instance) {
                         VkSurfaceKHR surface = VK_NULL_HANDLE;
                         if (glfwCreateWindowSurface(instance, window,
                                                     nullptr, &surface) != VK_SUCCESS)
                             throw std::runtime_error("realtime_shadow: failed to create window surface");
                         return surface;
                     });

        auto engine = vkRender::Engine::Create(&context);

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

        vkRender::SwapChainDescriptor descriptor{};
        descriptor.width = static_cast<uint32_t>(framebufferWidth);
        descriptor.height = static_cast<uint32_t>(framebufferHeight);
        descriptor.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        auto swapChain = engine->CreateSwapChain(descriptor);
        auto renderer = engine->CreateRenderer();
        auto scene = engine->CreateScene();
        auto camera = engine->CreateCamera();
        auto view = engine->CreateView();
        auto graph = engine->CreateRenderGraph();
        auto keyInput = engine->CreateKeyInput();

        auto realtimeShadowPass = std::make_unique<RealtimeShadowPass>(&context);
        RealtimeShadowPass *realtimeShadow = realtimeShadowPass.get();
        graph->AddPass(std::move(realtimeShadowPass));

        view->SetScene(scene.get());
        view->SetCamera(camera.get());
        view->SetRenderGraph(graph.get());

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

        keyInput->AddListener(vkRender::KeyEventType::Press,
                              [realtimeShadow](vkRender::KeyEvent &event) {
                                  if (event.keyCode == GLFW_KEY_ENTER || event.keyCode == GLFW_KEY_KP_ENTER) {
                                      realtimeShadow->RequestCapture();
                                      std::cout << "[realtime_shadow] capture requested\n";
                                  }
                              });

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
            if (framebufferWidth == 0 || framebufferHeight == 0) {
                glfwWaitEvents();
                continue;
            }

            const VkExtent2D extent = swapChain->Extent();
            if (extent.width != static_cast<uint32_t>(framebufferWidth) || extent.height != static_cast<uint32_t>(framebufferHeight)) {
                vkDeviceWaitIdle(context.device);
                swapChain->Recreate(static_cast<uint32_t>(framebufferWidth), static_cast<uint32_t>(framebufferHeight));
                renderer->ClearSwapChainRecreateFlag();
                continue;
            }

            if (!renderer->BeginFrame(*swapChain)) {
                vkDeviceWaitIdle(context.device);
                swapChain->Recreate(static_cast<uint32_t>(framebufferWidth), static_cast<uint32_t>(framebufferHeight));
                renderer->ClearSwapChainRecreateFlag();
                continue;
            }

            realtimeShadow->SetTime(static_cast<float>(glfwGetTime()));
            renderer->Render(*view);
            renderer->EndFrame();

            if (realtimeShadow->HasCaptureToSave()) {
                vkDeviceWaitIdle(context.device);
                const std::string filename = realtimeShadow->SaveCapturedImage();
                std::cout << "[realtime_shadow] saved " << filename << "\n";
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

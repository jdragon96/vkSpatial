#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace vkRender {

    struct ShaderStageDescriptor {
        VkShaderStageFlagBits stage = VK_SHADER_STAGE_VERTEX_BIT;
        std::string path;
        std::string entryPoint = "main";
    };

    struct PipelineColorTarget {
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkPipelineColorBlendAttachmentState blend{};

        static PipelineColorTarget Opaque(VkFormat format);
    };

    struct GraphicsPipelineDescriptor {
        std::vector<ShaderStageDescriptor> shaderStages;
        std::vector<VkVertexInputBindingDescription> vertexBindings;
        std::vector<VkVertexInputAttributeDescription> vertexAttributes;
        std::vector<PipelineColorTarget> colorTargets;
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
        std::vector<VkPushConstantRange> pushConstantRanges;
        std::vector<VkDynamicState> dynamicStates = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR,
        };

        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;
        VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
        VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        bool depthBiasEnable = false;
        float depthBiasConstantFactor = 0.0f;
        float depthBiasClamp = 0.0f;
        float depthBiasSlopeFactor = 0.0f;

        VkFormat depthFormat = VK_FORMAT_UNDEFINED;
        bool depthTestEnable = false;
        bool depthWriteEnable = false;
        VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS;

        GraphicsPipelineDescriptor &Shader(VkShaderStageFlagBits stage,
                                           const std::string &path,
                                           const std::string &entryPoint = "main");
        GraphicsPipelineDescriptor &VertexShader(const std::string &path,
                                                 const std::string &entryPoint = "main");
        GraphicsPipelineDescriptor &FragmentShader(const std::string &path,
                                                   const std::string &entryPoint = "main");

        GraphicsPipelineDescriptor &VertexBinding(uint32_t binding,
                                                  uint32_t stride,
                                                  VkVertexInputRate inputRate =
                                                          VK_VERTEX_INPUT_RATE_VERTEX);
        template<typename VertexT>
        GraphicsPipelineDescriptor &VertexBinding(uint32_t binding = 0,
                                                  VkVertexInputRate inputRate =
                                                          VK_VERTEX_INPUT_RATE_VERTEX) {
            return VertexBinding(binding, static_cast<uint32_t>(sizeof(VertexT)), inputRate);
        }

        GraphicsPipelineDescriptor &VertexAttribute(uint32_t location,
                                                    uint32_t binding,
                                                    VkFormat format,
                                                    uint32_t offset);
        GraphicsPipelineDescriptor &ColorTarget(VkFormat format);
        GraphicsPipelineDescriptor &ColorTarget(const PipelineColorTarget &target);
        GraphicsPipelineDescriptor &DepthTarget(VkFormat format,
                                                bool writeDepth = true,
                                                VkCompareOp compareOp = VK_COMPARE_OP_LESS);
        GraphicsPipelineDescriptor &DepthBias(float constantFactor,
                                              float slopeFactor,
                                              float clamp = 0.0f);
        GraphicsPipelineDescriptor &PushConstant(VkShaderStageFlags stageFlags,
                                                 uint32_t size,
                                                 uint32_t offset = 0);
        template<typename PushT>
        GraphicsPipelineDescriptor &PushConstant(VkShaderStageFlags stageFlags,
                                                 uint32_t offset = 0) {
            return PushConstant(stageFlags, static_cast<uint32_t>(sizeof(PushT)), offset);
        }
    };

    class GraphicsPipeline {
    public:
        using UniquePtr = std::unique_ptr<GraphicsPipeline>;

        explicit GraphicsPipeline(VkDevice device);
        ~GraphicsPipeline();

        GraphicsPipeline(const GraphicsPipeline &) = delete;
        GraphicsPipeline &operator=(const GraphicsPipeline &) = delete;

        GraphicsPipeline &Build(const GraphicsPipelineDescriptor &descriptor);
        void Destroy();

        void Bind(VkCommandBuffer commandBuffer) const;

        template<typename T>
        void PushConstants(VkCommandBuffer commandBuffer,
                           VkShaderStageFlags stageFlags,
                           const T &value,
                           uint32_t offset = 0) const {
            vkCmdPushConstants(commandBuffer, m_layout, stageFlags, offset,
                               static_cast<uint32_t>(sizeof(T)), &value);
        }

        VkPipeline Pipeline() const { return m_pipeline; }
        VkPipelineLayout Layout() const { return m_layout; }
        VkFormat ColorFormat(uint32_t index = 0) const;
        VkFormat DepthFormat() const { return m_depthFormat; }
        bool MatchesColorTarget(uint32_t index, VkFormat format) const;

    private:
        VkDevice m_device = VK_NULL_HANDLE;
        VkPipelineLayout m_layout = VK_NULL_HANDLE;
        VkPipeline m_pipeline = VK_NULL_HANDLE;
        std::vector<VkFormat> m_colorFormats;
        VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;

        static std::vector<uint32_t> LoadSPIRV(const std::string &path);
        VkShaderModule CreateShaderModule(const std::string &path) const;
    };

} // namespace vkRender

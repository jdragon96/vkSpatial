#include "vkRender/GraphicsPipeline.h"

#include <fstream>
#include <stdexcept>

namespace vkRender {

    PipelineColorTarget PipelineColorTarget::Opaque(VkFormat format) {
        PipelineColorTarget target{};
        target.format = format;
        target.blend.blendEnable = VK_FALSE;
        target.blend.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT |
                VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;
        return target;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::Shader(
            VkShaderStageFlagBits stage,
            const std::string &path,
            const std::string &entryPoint) {
        shaderStages.push_back({stage, path, entryPoint});
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::VertexShader(
            const std::string &path,
            const std::string &entryPoint) {
        return Shader(VK_SHADER_STAGE_VERTEX_BIT, path, entryPoint);
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::FragmentShader(
            const std::string &path,
            const std::string &entryPoint) {
        return Shader(VK_SHADER_STAGE_FRAGMENT_BIT, path, entryPoint);
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::VertexBinding(
            uint32_t binding,
            uint32_t stride,
            VkVertexInputRate inputRate) {
        VkVertexInputBindingDescription description{};
        description.binding = binding;
        description.stride = stride;
        description.inputRate = inputRate;
        vertexBindings.push_back(description);
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::VertexAttribute(
            uint32_t location,
            uint32_t binding,
            VkFormat format,
            uint32_t offset) {
        VkVertexInputAttributeDescription description{};
        description.location = location;
        description.binding = binding;
        description.format = format;
        description.offset = offset;
        vertexAttributes.push_back(description);
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::ColorTarget(VkFormat format) {
        return ColorTarget(PipelineColorTarget::Opaque(format));
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::ColorTarget(
            const PipelineColorTarget &target) {
        colorTargets.push_back(target);
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::DepthTarget(
            VkFormat format,
            bool writeDepth,
            VkCompareOp compareOp) {
        depthFormat = format;
        depthTestEnable = format != VK_FORMAT_UNDEFINED;
        depthWriteEnable = writeDepth;
        depthCompareOp = compareOp;
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::DepthBias(
            float constantFactor,
            float slopeFactor,
            float clamp) {
        depthBiasEnable = true;
        depthBiasConstantFactor = constantFactor;
        depthBiasSlopeFactor = slopeFactor;
        depthBiasClamp = clamp;
        return *this;
    }

    GraphicsPipelineDescriptor &GraphicsPipelineDescriptor::PushConstant(
            VkShaderStageFlags stageFlags,
            uint32_t size,
            uint32_t offset) {
        VkPushConstantRange range{};
        range.stageFlags = stageFlags;
        range.offset = offset;
        range.size = size;
        pushConstantRanges.push_back(range);
        return *this;
    }

    GraphicsPipeline::GraphicsPipeline(VkDevice device)
        : m_device(device) {
        if (m_device == VK_NULL_HANDLE)
            throw std::runtime_error("GraphicsPipeline requires a valid VkDevice");
    }

    GraphicsPipeline::~GraphicsPipeline() {
        Destroy();
    }

    GraphicsPipeline &GraphicsPipeline::Build(
            const GraphicsPipelineDescriptor &descriptor) {
        if (descriptor.shaderStages.empty())
            throw std::runtime_error("GraphicsPipeline::Build requires at least one shader");

        Destroy();

        std::vector<VkShaderModule> shaderModules;
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
        shaderModules.reserve(descriptor.shaderStages.size());
        shaderStages.reserve(descriptor.shaderStages.size());

        auto destroyShaderModules = [&]() {
            for (VkShaderModule module: shaderModules)
                vkDestroyShaderModule(m_device, module, nullptr);
            shaderModules.clear();
        };

        try {
            for (const ShaderStageDescriptor &shader: descriptor.shaderStages) {
                VkShaderModule module = CreateShaderModule(shader.path);
                shaderModules.push_back(module);

                VkPipelineShaderStageCreateInfo stage{};
                stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                stage.stage = shader.stage;
                stage.module = module;
                stage.pName = shader.entryPoint.c_str();
                shaderStages.push_back(stage);
            }

            VkPipelineVertexInputStateCreateInfo vertexInput{};
            vertexInput.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInput.vertexBindingDescriptionCount =
                    static_cast<uint32_t>(descriptor.vertexBindings.size());
            vertexInput.pVertexBindingDescriptions =
                    descriptor.vertexBindings.empty() ? nullptr
                                                      : descriptor.vertexBindings.data();
            vertexInput.vertexAttributeDescriptionCount =
                    static_cast<uint32_t>(descriptor.vertexAttributes.size());
            vertexInput.pVertexAttributeDescriptions =
                    descriptor.vertexAttributes.empty() ? nullptr
                                                        : descriptor.vertexAttributes.data();

            VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
            inputAssembly.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.topology = descriptor.topology;

            VkPipelineViewportStateCreateInfo viewportState{};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rasterizer{};
            rasterizer.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.polygonMode = descriptor.polygonMode;
            rasterizer.cullMode = descriptor.cullMode;
            rasterizer.frontFace = descriptor.frontFace;
            rasterizer.lineWidth = 1.0f;
            rasterizer.depthBiasEnable = descriptor.depthBiasEnable ? VK_TRUE : VK_FALSE;
            rasterizer.depthBiasConstantFactor = descriptor.depthBiasConstantFactor;
            rasterizer.depthBiasClamp = descriptor.depthBiasClamp;
            rasterizer.depthBiasSlopeFactor = descriptor.depthBiasSlopeFactor;

            VkPipelineMultisampleStateCreateInfo multisampling{};
            multisampling.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.rasterizationSamples = descriptor.samples;

            VkPipelineDepthStencilStateCreateInfo depthStencil{};
            depthStencil.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = descriptor.depthTestEnable ? VK_TRUE : VK_FALSE;
            depthStencil.depthWriteEnable = descriptor.depthWriteEnable ? VK_TRUE : VK_FALSE;
            depthStencil.depthCompareOp = descriptor.depthCompareOp;

            std::vector<VkPipelineColorBlendAttachmentState> blendAttachments;
            blendAttachments.reserve(descriptor.colorTargets.size());
            m_colorFormats.clear();
            m_colorFormats.reserve(descriptor.colorTargets.size());
            for (const PipelineColorTarget &target: descriptor.colorTargets) {
                blendAttachments.push_back(target.blend);
                m_colorFormats.push_back(target.format);
            }

            VkPipelineColorBlendStateCreateInfo colorBlending{};
            colorBlending.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.attachmentCount =
                    static_cast<uint32_t>(blendAttachments.size());
            colorBlending.pAttachments =
                    blendAttachments.empty() ? nullptr : blendAttachments.data();

            VkPipelineDynamicStateCreateInfo dynamicState{};
            dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamicState.dynamicStateCount =
                    static_cast<uint32_t>(descriptor.dynamicStates.size());
            dynamicState.pDynamicStates =
                    descriptor.dynamicStates.empty() ? nullptr
                                                     : descriptor.dynamicStates.data();

            VkPipelineLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layoutInfo.setLayoutCount =
                    static_cast<uint32_t>(descriptor.descriptorSetLayouts.size());
            layoutInfo.pSetLayouts =
                    descriptor.descriptorSetLayouts.empty()
                            ? nullptr
                            : descriptor.descriptorSetLayouts.data();
            layoutInfo.pushConstantRangeCount =
                    static_cast<uint32_t>(descriptor.pushConstantRanges.size());
            layoutInfo.pPushConstantRanges =
                    descriptor.pushConstantRanges.empty()
                            ? nullptr
                            : descriptor.pushConstantRanges.data();
            if (vkCreatePipelineLayout(m_device, &layoutInfo,
                                       nullptr, &m_layout) != VK_SUCCESS)
                throw std::runtime_error("GraphicsPipeline: failed to create pipeline layout");

            VkPipelineRenderingCreateInfo renderingInfo{};
            renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            renderingInfo.colorAttachmentCount =
                    static_cast<uint32_t>(m_colorFormats.size());
            renderingInfo.pColorAttachmentFormats =
                    m_colorFormats.empty() ? nullptr : m_colorFormats.data();
            renderingInfo.depthAttachmentFormat = descriptor.depthFormat;

            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.pNext = &renderingInfo;
            pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
            pipelineInfo.pStages = shaderStages.data();
            pipelineInfo.pVertexInputState = &vertexInput;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = &depthStencil;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = m_layout;

            if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1,
                                          &pipelineInfo, nullptr,
                                          &m_pipeline) != VK_SUCCESS)
                throw std::runtime_error("GraphicsPipeline: failed to create graphics pipeline");

            m_depthFormat = descriptor.depthFormat;
            destroyShaderModules();
        } catch (...) {
            destroyShaderModules();
            Destroy();
            throw;
        }

        return *this;
    }

    void GraphicsPipeline::Destroy() {
        if (m_pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
        if (m_layout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(m_device, m_layout, nullptr);
        m_pipeline = VK_NULL_HANDLE;
        m_layout = VK_NULL_HANDLE;
        m_colorFormats.clear();
        m_depthFormat = VK_FORMAT_UNDEFINED;
    }

    void GraphicsPipeline::Bind(VkCommandBuffer commandBuffer) const {
        if (m_pipeline == VK_NULL_HANDLE)
            throw std::runtime_error("GraphicsPipeline::Bind called before Build");
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    }

    VkFormat GraphicsPipeline::ColorFormat(uint32_t index) const {
        if (index >= m_colorFormats.size())
            return VK_FORMAT_UNDEFINED;
        return m_colorFormats[index];
    }

    bool GraphicsPipeline::MatchesColorTarget(uint32_t index, VkFormat format) const {
        return ColorFormat(index) == format;
    }

    std::vector<uint32_t> GraphicsPipeline::LoadSPIRV(const std::string &path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file)
            throw std::runtime_error("GraphicsPipeline: failed to open shader " + path);

        const std::streamsize size = file.tellg();
        if (size <= 0 || size % 4 != 0)
            throw std::runtime_error("GraphicsPipeline: invalid shader bytecode " + path);

        std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(code.data()), size);
        return code;
    }

    VkShaderModule GraphicsPipeline::CreateShaderModule(
            const std::string &path) const {
        const std::vector<uint32_t> code = LoadSPIRV(path);

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size() * sizeof(uint32_t);
        createInfo.pCode = code.data();

        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(m_device, &createInfo, nullptr, &module) != VK_SUCCESS)
            throw std::runtime_error("GraphicsPipeline: failed to create shader module " + path);
        return module;
    }

} // namespace vkRender

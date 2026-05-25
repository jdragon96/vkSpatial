#include "vkComputeBase.h"

#include "SPIRV-Reflect/spirv_reflect.h"
#include <fstream>
#include <shaderc/shaderc.hpp>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace vkCommon {

    // ── 파일시스템 include 처리기 ─────────────────────────────────────────────────
    // BuildFromFile() 전용: lbvh/glslc 방식과 동일하게
    // #include "other.glsl" 을 지정된 디렉토리에서 읽어 해결

    class FilesystemIncluder : public shaderc::CompileOptions::IncluderInterface {
    public:
        explicit FilesystemIncluder(std::string dir) : m_dir(std::move(dir)) {}

        shaderc_include_result *GetInclude(const char *requested,
                                           shaderc_include_type,
                                           const char * /*requesting*/,
                                           size_t) override {
            std::string fullPath = m_dir + "/" + requested;
            std::ifstream f(fullPath, std::ios::binary);

            auto *r = new shaderc_include_result{};
            if (f.is_open()) {
                auto *content = new std::string(
                        std::istreambuf_iterator<char>(f),
                        std::istreambuf_iterator<char>());
                auto *name = new std::string(fullPath);
                r->source_name = name->c_str();
                r->source_name_length = name->size();
                r->content = content->c_str();
                r->content_length = content->size();
                r->user_data = new std::pair<std::string *, std::string *>(name, content);
            } else {
                static const char kErr[] = "file not found";
                r->source_name = "";
                r->source_name_length = 0;
                r->content = kErr;
                r->content_length = sizeof(kErr) - 1;
                r->user_data = nullptr;
            }
            return r;
        }

        void ReleaseInclude(shaderc_include_result *r) override {
            if (r->user_data) {
                auto *p = static_cast<std::pair<std::string *, std::string *> *>(r->user_data);
                delete p->first;
                delete p->second;
                delete p;
            }
            delete r;
        }

    private:
        std::string m_dir;
    };

    // ── 인메모리 include 처리기 ────────────────────────────────────────────────────
    // AddInclude()로 등록된 소스를 shaderc가 #include 지시어로 참조할 수 있게 해줌
    class InMemoryIncluder : public shaderc::CompileOptions::IncluderInterface {
    public:
        explicit InMemoryIncluder(const std::unordered_map<std::string, std::string> &srcs)
            : m_srcs(srcs) {}

        shaderc_include_result *GetInclude(const char *requested,
                                           shaderc_include_type,
                                           const char * /*requesting*/,
                                           size_t) override {
            auto it = m_srcs.find(requested);
            auto *r = new shaderc_include_result{};
            if (it != m_srcs.end()) {
                r->source_name = it->first.c_str();
                r->source_name_length = it->first.size();
                r->content = it->second.c_str();
                r->content_length = it->second.size();
            } else {
                static const char kErr[] = "include not found";
                r->source_name = "";
                r->source_name_length = 0;
                r->content = kErr;
                r->content_length = sizeof(kErr) - 1;
            }
            return r;
        }

        void ReleaseInclude(shaderc_include_result *r) override { delete r; }

    private:
        const std::unordered_map<std::string, std::string> &m_srcs;
    };

    // ── AddInclude ────────────────────────────────────────────────────────────────

    vkComputeBase &vkComputeBase::AddInclude(const std::string &name,
                                             const std::string &src) {
        m_includes[name] = src;
        return *this;
    }

    // ── compileGlslToSpv (m_includes 를 includer로 전달) ─────────────────────────

    std::vector<uint32_t> vkComputeBase::compileGlslToSpv(const std::string &src) const {
        shaderc::Compiler compiler;
        shaderc::CompileOptions opts;
        opts.SetOptimizationLevel(shaderc_optimization_level_performance);
        opts.SetIncluder(std::make_unique<InMemoryIncluder>(m_includes));

        auto result = compiler.CompileGlslToSpv(
                src, shaderc_compute_shader, "inline", opts);
        if (result.GetCompilationStatus() != shaderc_compilation_status_success)
            throw std::runtime_error("GLSL compile: " + result.GetErrorMessage());

        return std::vector<uint32_t>(result.cbegin(), result.cend());
    }

    vkComputeBase::vkComputeBase(VkDevice device,
                                 VkPhysicalDevice physicalDevice,
                                 VkQueue computeQueue,
                                 VkCommandPool commandPool)
        : m_device(device), m_physDevice(physicalDevice),
          m_queue(computeQueue), m_cmdPool(commandPool) {}

    void vkComputeBase::destroyShaderResources() {
        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
            m_pipeline = VK_NULL_HANDLE;
        }
        if (m_pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
            m_pipelineLayout = VK_NULL_HANDLE;
        }
        if (m_descPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
            m_descPool = VK_NULL_HANDLE;
            m_descSet = VK_NULL_HANDLE;
        }
        if (m_descLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_descLayout, nullptr);
            m_descLayout = VK_NULL_HANDLE;
        }
        if (m_shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, m_shaderModule, nullptr);
            m_shaderModule = VK_NULL_HANDLE;
        }
        m_dirty = true;
    }

    vkComputeBase &vkComputeBase::Build(const std::string &path) {
        destroyShaderResources();

        std::ifstream f(path, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("vkComputeBase::BuildFromFile: cannot open " + path);
        std::ostringstream ss;
        ss << f.rdbuf();
        std::string src = ss.str();

        auto lastSlash = path.rfind('/');
        std::string dir = (lastSlash != std::string::npos) ? path.substr(0, lastSlash) : ".";

        shaderc::Compiler compiler;
        shaderc::CompileOptions opts;
        opts.SetOptimizationLevel(shaderc_optimization_level_performance);
        opts.SetIncluder(std::make_unique<FilesystemIncluder>(dir));

        auto result = compiler.CompileGlslToSpv(
                src, shaderc_compute_shader, path.c_str(), opts);
        if (result.GetCompilationStatus() != shaderc_compilation_status_success)
            throw std::runtime_error("vkComputeBase::BuildFromFile: " + result.GetErrorMessage());

        std::vector<uint32_t> spv(result.cbegin(), result.cend());

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size() * sizeof(uint32_t);
        smci.pCode = spv.data();
        if (vkCreateShaderModule(m_device, &smci, nullptr, &m_shaderModule) != VK_SUCCESS)
            throw std::runtime_error("vkComputeBase::BuildFromFile: vkCreateShaderModule failed");

        reflectLocalSize(spv);
        return *this;
    }

    vkComputeBase &vkComputeBase::Build(const std::string &source, ShaderInput inputType) {
        destroyShaderResources();

        std::vector<uint32_t> spv = (inputType == ShaderInput::GlslSrc)
                                            ? compileGlslToSpv(source)
                                            : loadSPIRV(source);

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size() * sizeof(uint32_t);
        smci.pCode = spv.data();
        if (vkCreateShaderModule(m_device, &smci, nullptr, &m_shaderModule) != VK_SUCCESS)
            throw std::runtime_error("vkComputeBase::Build: vkCreateShaderModule failed");

        reflectLocalSize(spv);

        return *this;
    }

    void vkComputeBase::reflectLocalSize(const std::vector<uint32_t> &spv) {
        SpvReflectShaderModule module{};
        SpvReflectResult r = spvReflectCreateShaderModule(
                spv.size() * sizeof(uint32_t), spv.data(), &module);
        if (r != SPV_REFLECT_RESULT_SUCCESS) {
            fprintf(stderr, "[vkComputeBase] SPIRV-Reflect failed (code=%d), "
                            "local_size will be 0 — DispatchElements will throw\n",
                    static_cast<int>(r));
            return;
        }

        const SpvReflectEntryPoint *ep = spvReflectGetEntryPoint(&module, "main");
        if (ep) {
            m_localSize = {ep->local_size.x, ep->local_size.y, ep->local_size.z};
        } else {
            fprintf(stderr, "[vkComputeBase] SPIRV-Reflect: no 'main' entry point found\n");
        }

        spvReflectDestroyShaderModule(&module);
    }

    vkComputeBase::~vkComputeBase() {
        destroyShaderResources();
    }

    vkComputeBase &vkComputeBase::Bind(uint32_t binding,
                                       VkBuffer buffer, VkDeviceSize sizeBytes) {
        // 이미 같은 binding이 있으면 교체
        for (auto &b: m_bindings) {
            if (b.binding == binding) {
                b.buffer = buffer;
                b.size = sizeBytes;
                m_dirty = true;
                return *this;
            }
        }
        m_bindings.push_back({binding, buffer, sizeBytes});
        m_dirty = true;
        return *this;
    }

    void vkComputeBase::Dispatch(VkExtent3D grid) {
        Dispatch(grid.width, grid.height, grid.depth);
    }

    void vkComputeBase::Dispatch(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        ensurePipeline();
        if (m_dirty) updateDescriptors();
        submit(gridX, gridY, gridZ);
    }

    void vkComputeBase::DispatchElements(uint32_t numElements) {
        if (m_localSize.width == 0)
            throw std::runtime_error(
                    "vkComputeBase::DispatchElements: local_size.width=0 "
                    "(SPIR-V reflection failed — check Build() was called with valid shader)");
        uint32_t gridX = (numElements + m_localSize.width - 1) / m_localSize.width;
        uint32_t gridY = 1;
        uint32_t gridZ = 1;
        Dispatch(gridX, gridY, gridZ);
    }

    void vkComputeBase::Sync() {
        vkQueueWaitIdle(m_queue);
    }

    void vkComputeBase::ensurePipeline() {
        if (m_pipeline != VK_NULL_HANDLE) return;

        // ① 바인딩별 descriptor set layout 생성
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (auto &b: m_bindings) {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = b.binding;
            lb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            lb.descriptorCount = 1;
            lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(lb);
        }

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(m_device, &dlci, nullptr, &m_descLayout) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreateDescriptorSetLayout failed");

        // ② descriptor pool
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(bindings.size());

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(m_device, &dpci, nullptr, &m_descPool) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreateDescriptorPool failed");

        // ③ descriptor set 할당
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = m_descPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &m_descLayout;
        if (vkAllocateDescriptorSets(m_device, &dsai, &m_descSet) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkAllocateDescriptorSets failed");

        // ④ push constant range (비어 있으면 크기 0)
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = m_pushData.empty() ? 0
                                          : static_cast<uint32_t>(m_pushData.size());

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &m_descLayout;
        plci.pushConstantRangeCount = pcRange.size > 0 ? 1 : 0;
        plci.pPushConstantRanges = pcRange.size > 0 ? &pcRange : nullptr;
        if (vkCreatePipelineLayout(m_device, &plci, nullptr, &m_pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreatePipelineLayout failed");

        // ⑤ compute pipeline
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = m_shaderModule;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = m_pipelineLayout;
        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreateComputePipelines failed");
    }

    void vkComputeBase::updateDescriptors() {
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorBufferInfo> bufInfos(m_bindings.size());

        for (size_t i = 0; i < m_bindings.size(); i++) {
            bufInfos[i].buffer = m_bindings[i].buffer;
            bufInfos[i].offset = 0;
            bufInfos[i].range = m_bindings[i].size;

            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = m_descSet;
            w.dstBinding = m_bindings[i].binding;
            w.dstArrayElement = 0;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &bufInfos[i];
            writes.push_back(w);
        }

        vkUpdateDescriptorSets(m_device,
                               static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
        m_dirty = false;
    }

    void vkComputeBase::submit(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = m_cmdPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;

        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(m_device, &ai, &cmd);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);

        if (!m_pushData.empty()) {
            vkCmdPushConstants(cmd, m_pipelineLayout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, static_cast<uint32_t>(m_pushData.size()),
                               m_pushData.data());
        }

        vkCmdDispatch(cmd, gridX, gridY, gridZ);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;

        vkQueueSubmit(m_queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_queue);
        vkFreeCommandBuffers(m_device, m_cmdPool, 1, &cmd);
    }

    std::vector<uint32_t> vkComputeBase::loadSPIRV(const std::string &path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("vkComputeBase: cannot open SPIR-V: " + path);

        size_t byteSize = static_cast<size_t>(file.tellg());
        if (byteSize % 4 != 0)
            throw std::runtime_error("vkComputeBase: SPIR-V size not 4-byte aligned");

        std::vector<uint32_t> code(byteSize / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char *>(code.data()), static_cast<std::streamsize>(byteSize));
        return code;
    }

} // namespace vkCommon

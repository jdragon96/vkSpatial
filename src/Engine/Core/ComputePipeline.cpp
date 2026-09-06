#include "Engine/Core/ComputePipeline.h"

#include "SPIRV-Reflect/spirv_reflect.h"
#include <fstream>
#include <map>
#include <mutex>
#include <shaderc/shaderc.hpp>
#include <sstream>
#include <stdexcept>

#ifndef VKBVH_SHADER_DIR
#define VKBVH_SHADER_DIR "."
#endif

#ifndef VKBVH_SRC_DIR
#define VKBVH_SRC_DIR "."
#endif

namespace Engine::Core {

    class FilesystemIncluder : public shaderc::CompileOptions::IncluderInterface {
    public:
        explicit FilesystemIncluder(std::vector<std::string> dirs) : m_dirs(std::move(dirs)) {}

        shaderc_include_result *GetInclude(const char *requested,
                                           shaderc_include_type,
                                           const char * /*requesting*/,
                                           size_t) override {
            for (const std::string &dir: m_dirs) {
                std::string fullPath = dir + "/" + requested;
                std::ifstream f(fullPath, std::ios::binary);
                if (!f.is_open()) continue;
                auto *r = new shaderc_include_result{};
                auto *content = new std::string(std::istreambuf_iterator<char>(f),
                                                std::istreambuf_iterator<char>());
                auto *name = new std::string(fullPath);
                r->source_name = name->c_str();
                r->source_name_length = name->size();
                r->content = content->c_str();
                r->content_length = content->size();
                r->user_data = new std::pair<std::string *, std::string *>(name, content);
                return r;
            }
            auto *r = new shaderc_include_result{};
            static const char kErr[] = "file not found";
            r->source_name = "";
            r->source_name_length = 0;
            r->content = kErr;
            r->content_length = sizeof(kErr) - 1;
            r->user_data = nullptr;
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
        std::vector<std::string> m_dirs;
    };

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

    ComputePipeline &ComputePipeline::AddInclude(const std::string &name, const std::string &src) {
        m_includes[name] = src;
        return *this;
    }

    ComputePipeline &ComputePipeline::Define(const std::string &name, const std::string &value) {
        m_defines[name] = value;
        return *this;
    }

    std::vector<uint32_t> ComputePipeline::compileGlslToSpv(const std::string &src) const {
        shaderc::Compiler compiler;
        shaderc::CompileOptions opts;
        opts.SetOptimizationLevel(shaderc_optimization_level_performance);
        opts.SetIncluder(std::make_unique<InMemoryIncluder>(m_includes));

        auto result = compiler.CompileGlslToSpv(src, shaderc_compute_shader, "inline", opts);
        if (result.GetCompilationStatus() != shaderc_compilation_status_success)
            throw std::runtime_error("GLSL compile: " + result.GetErrorMessage());

        return std::vector<uint32_t>(result.cbegin(), result.cend());
    }

    ComputePipeline::ComputePipeline(Context &context) : m_context(context) {}

    void ComputePipeline::destroyShaderResources() {
        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_context.device, m_pipeline, nullptr);
            m_pipeline = VK_NULL_HANDLE;
        }
        if (m_pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_context.device, m_pipelineLayout, nullptr);
            m_pipelineLayout = VK_NULL_HANDLE;
        }
        if (m_descPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_context.device, m_descPool, nullptr);
            m_descPool = VK_NULL_HANDLE;
            m_descSet = VK_NULL_HANDLE;
        }
        if (m_descLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_context.device, m_descLayout, nullptr);
            m_descLayout = VK_NULL_HANDLE;
        }
        if (m_shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_context.device, m_shaderModule, nullptr);
            m_shaderModule = VK_NULL_HANDLE;
        }
        m_dirty = true;
    }

    namespace {
        // Ordered roots for resolving a kernel path and an #include. The shader directory comes
        // first so every pre-existing call site -- which passes a bare filename -- resolves exactly
        // as before; the source root lets a kernel that lives beside its calling .cpp be named by
        // its path from src/.
        const std::vector<std::string> &shaderRoots() {
            static const std::vector<std::string> roots = {VKBVH_SHADER_DIR, VKBVH_SRC_DIR};
            return roots;
        }

        // First root that actually holds the file wins. Returns the untouched relative path when
        // none do, so the caller's error message still names what was asked for.
        std::string resolveShaderPath(const std::string &relative) {
            for (const std::string &root: shaderRoots()) {
                std::string candidate = root + "/" + relative;
                std::ifstream probe(candidate);
                if (probe.good()) return candidate;
            }
            return relative;
        }

        // Process-wide SPIR-V cache keyed by shader path: compile each shader file ONCE, then reuse
        // across every ComputePipeline that Builds it. Without this, a fine-voxel TiledAdvancedTSDF
        // pays hundreds of redundant shaderc compilations on its first frame (each of its ~100 tiles
        // builds the same integrate + compact kernels). SPIR-V is device-independent, so the cache is
        // shared across Contexts; only the (cheap) VkShaderModule is per-pipeline.
        std::vector<uint32_t> compileFileCached(const std::string &fullPath,
                                                const std::map<std::string, std::string> &defines) {
            static std::unordered_map<std::string, std::vector<uint32_t>> cache;
            static std::mutex mutex;

            // Cache key = path + every definition. Keyed on the path alone, a second build of the
            // same kernel with a different definition set would silently reuse the first module.
            std::string cacheKey = fullPath;
            for (const auto &entry: defines) {
                cacheKey += '|';
                cacheKey += entry.first;
                cacheKey += '=';
                cacheKey += entry.second;
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                const auto hit = cache.find(cacheKey);
                if (hit != cache.end()) return hit->second;
            }

            std::ifstream f(fullPath, std::ios::binary);
            if (!f.is_open())
                throw std::runtime_error("ComputePipeline::Build: cannot open " + fullPath);
            std::ostringstream ss;
            ss << f.rdbuf();
            const std::string src = ss.str();

            const auto lastSlash = fullPath.rfind('/');
            const std::string dir = (lastSlash != std::string::npos) ? fullPath.substr(0, lastSlash) : ".";

            shaderc::Compiler compiler;
            shaderc::CompileOptions opts;
            opts.SetOptimizationLevel(shaderc_optimization_level_performance);
            std::vector<std::string> includeDirs = {dir};
            for (const std::string &root: shaderRoots()) includeDirs.push_back(root);
            opts.SetIncluder(std::make_unique<FilesystemIncluder>(includeDirs));
            for (const auto &entry: defines)
                opts.AddMacroDefinition(entry.first, entry.second);

            auto result = compiler.CompileGlslToSpv(src, shaderc_compute_shader, fullPath.c_str(), opts);
            if (result.GetCompilationStatus() != shaderc_compilation_status_success)
                throw std::runtime_error("ComputePipeline::Build: " + result.GetErrorMessage());

            std::vector<uint32_t> spv(result.cbegin(), result.cend());

            std::lock_guard<std::mutex> lock(mutex);
            cache.emplace(cacheKey, spv);
            return spv;
        }
    } // namespace

    ComputePipeline &ComputePipeline::Build(const std::string &filename) {
        destroyShaderResources();

        const std::string fullPath = resolveShaderPath(filename);
        const std::vector<uint32_t> spv = compileFileCached(fullPath, m_defines);

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size() * sizeof(uint32_t);
        smci.pCode = spv.data();
        if (vkCreateShaderModule(m_context.device, &smci, nullptr, &m_shaderModule) != VK_SUCCESS)
            throw std::runtime_error("ComputePipeline::Build: vkCreateShaderModule failed");

        reflectLocalSize(spv);
        return *this;
    }

    ComputePipeline &ComputePipeline::Build(const std::string &source, ShaderInput inputType) {
        destroyShaderResources();

        std::vector<uint32_t> spv = (inputType == ShaderInput::GlslSrc)
                                            ? compileGlslToSpv(source)
                                            : loadSPIRV(source);

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size() * sizeof(uint32_t);
        smci.pCode = spv.data();
        if (vkCreateShaderModule(m_context.device, &smci, nullptr, &m_shaderModule) != VK_SUCCESS)
            throw std::runtime_error("ComputePipeline::Build: vkCreateShaderModule failed");

        reflectLocalSize(spv);
        return *this;
    }

    void ComputePipeline::reflectLocalSize(const std::vector<uint32_t> &spv) {
        SpvReflectShaderModule module{};
        SpvReflectResult r = spvReflectCreateShaderModule(spv.size() * sizeof(uint32_t), spv.data(), &module);
        if (r != SPV_REFLECT_RESULT_SUCCESS) {
            fprintf(stderr, "[ComputePipeline] SPIRV-Reflect failed (code=%d), "
                            "local_size will be 0 — DispatchElements will throw\n",
                    static_cast<int>(r));
            return;
        }

        const SpvReflectEntryPoint *ep = spvReflectGetEntryPoint(&module, "main");
        if (ep) {
            m_localSize = {ep->local_size.x, ep->local_size.y, ep->local_size.z};
        } else {
            fprintf(stderr, "[ComputePipeline] SPIRV-Reflect: no 'main' entry point found\n");
        }

        spvReflectDestroyShaderModule(&module);
    }

    ComputePipeline::~ComputePipeline() {
        destroyShaderResources();
    }

    // Rebinding a slot REPLACES it wholesale, type included: the same binding number can carry a
    // buffer in one configuration of a kernel and a sampler in another, and leaving stale fields
    // behind would write a descriptor of the old type into the new layout.
    ComputePipeline &ComputePipeline::Bind(uint32_t binding, VkBuffer buffer, VkDeviceSize sizeBytes) {
        ResourceBinding entry{};
        entry.binding = binding;
        entry.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        entry.buffer = buffer;
        entry.size = sizeBytes;
        return bindResource(entry);
    }

    ComputePipeline &ComputePipeline::Bind(uint32_t binding, VkImageView view, VkSampler sampler,
                                           VkImageLayout layout) {
        if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE)
            throw std::runtime_error("ComputePipeline::Bind: binding " + std::to_string(binding) +
                                     " was given a null image view or sampler; an Image created "
                                     "with createView=false has no view to bind");

        ResourceBinding entry{};
        entry.binding = binding;
        entry.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        entry.view = view;
        entry.sampler = sampler;
        entry.layout = layout;
        return bindResource(entry);
    }

    ComputePipeline &ComputePipeline::bindResource(const ResourceBinding &entry) {
        for (auto &existing: m_bindings) {
            if (existing.binding == entry.binding) {
                // The descriptor set layout is built from the types, so a slot that changed type
                // needs a new layout, not just a new descriptor write.
                if (existing.type != entry.type) destroyShaderResources();
                existing = entry;
                m_dirty = true;
                return *this;
            }
        }
        m_bindings.push_back(entry);
        m_dirty = true;
        return *this;
    }

    void ComputePipeline::Dispatch(VkExtent3D grid) {
        Dispatch(grid.width, grid.height, grid.depth);
    }

    void ComputePipeline::Dispatch(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        ensurePipeline();
        if (m_dirty) updateDescriptors();
        submit(gridX, gridY, gridZ);
    }

    void ComputePipeline::DispatchElements(uint32_t numElements) {
        if (m_localSize.width == 0)
            throw std::runtime_error(
                    "ComputePipeline::DispatchElements: local_size.width=0 "
                    "(SPIR-V reflection failed — check Build() was called with valid shader)");
        uint32_t gridX = (numElements + m_localSize.width - 1) / m_localSize.width;
        Dispatch(gridX, 1, 1);
    }

    void ComputePipeline::Sync() {
        // Dispatch() already blocks via SubmitOneShot's vkQueueWaitIdle. Retained as a
        // no-op-equivalent call for API-shape compatibility with vkComputeBase.
        vkQueueWaitIdle(m_context.computeQueue);
    }

    void ComputePipeline::ensurePipeline() {
        if (m_pipeline != VK_NULL_HANDLE) return;

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (auto &b: m_bindings) {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = b.binding;
            lb.descriptorType = b.type;
            lb.descriptorCount = 1;
            lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(lb);
        }

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(m_context.device, &dlci, nullptr, &m_descLayout) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreateDescriptorSetLayout failed");

        // One pool size PER TYPE, as the spec requires: a pool offering only storage-buffer
        // descriptors cannot satisfy a set that contains a sampler.
        //
        // No test guards this. MoltenVK does not enforce the pool's type accounting and never
        // returns VK_ERROR_OUT_OF_POOL_MEMORY, so collapsing this back to a single pool size runs
        // fine on this machine and fails on a driver that checks. Verified by mutation.
        std::vector<VkDescriptorPoolSize> poolSizes;
        for (auto &b: m_bindings) {
            bool counted = false;
            for (auto &size: poolSizes) {
                if (size.type != b.type) continue;
                size.descriptorCount += 1;
                counted = true;
                break;
            }
            if (!counted) poolSizes.push_back({b.type, 1});
        }

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        dpci.pPoolSizes = poolSizes.data();
        if (vkCreateDescriptorPool(m_context.device, &dpci, nullptr, &m_descPool) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreateDescriptorPool failed");

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = m_descPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &m_descLayout;
        if (vkAllocateDescriptorSets(m_context.device, &dsai, &m_descSet) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkAllocateDescriptorSets failed");

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = m_pushData.empty() ? 0 : static_cast<uint32_t>(m_pushData.size());

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &m_descLayout;
        plci.pushConstantRangeCount = pcRange.size > 0 ? 1 : 0;
        plci.pPushConstantRanges = pcRange.size > 0 ? &pcRange : nullptr;
        if (vkCreatePipelineLayout(m_context.device, &plci, nullptr, &m_pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreatePipelineLayout failed");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = m_shaderModule;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = m_pipelineLayout;
        if (vkCreateComputePipelines(m_context.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline) != VK_SUCCESS)
            throw std::runtime_error("ensurePipeline: vkCreateComputePipelines failed");
    }

    void ComputePipeline::updateDescriptors() {
        std::vector<VkWriteDescriptorSet> writes;

        // Both info arrays are sized UP FRONT and never grown. Each write below stores a pointer
        // into them, so a push_back that reallocated would leave every earlier write pointing at
        // freed memory -- and vkUpdateDescriptorSets would read it.
        std::vector<VkDescriptorBufferInfo> bufferInfos(m_bindings.size());
        std::vector<VkDescriptorImageInfo> imageInfos(m_bindings.size());

        for (size_t i = 0; i < m_bindings.size(); i++) {
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = m_descSet;
            w.dstBinding = m_bindings[i].binding;
            w.dstArrayElement = 0;
            w.descriptorCount = 1;
            w.descriptorType = m_bindings[i].type;

            if (m_bindings[i].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                imageInfos[i].imageView = m_bindings[i].view;
                imageInfos[i].sampler = m_bindings[i].sampler;
                imageInfos[i].imageLayout = m_bindings[i].layout;
                w.pImageInfo = &imageInfos[i];
            } else {
                bufferInfos[i].buffer = m_bindings[i].buffer;
                bufferInfos[i].offset = 0;
                bufferInfos[i].range = m_bindings[i].size;
                w.pBufferInfo = &bufferInfos[i];
            }
            writes.push_back(w);
        }

        vkUpdateDescriptorSets(m_context.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        m_dirty = false;
    }

    void ComputePipeline::recordInto(VkCommandBuffer cmd, uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);
        if (!m_pushData.empty())
            vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               static_cast<uint32_t>(m_pushData.size()), m_pushData.data());
        vkCmdDispatch(cmd, gridX, gridY, gridZ);
    }

    void ComputePipeline::submit(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        SubmitOneShot(m_context, QueueRole::Compute, [&](VkCommandBuffer cmd) {
            recordInto(cmd, gridX, gridY, gridZ);
        });
    }

    void ComputePipeline::RecordDispatch(VkCommandBuffer cmd, uint32_t gridX, uint32_t gridY, uint32_t gridZ) {
        ensurePipeline();
        if (m_dirty) updateDescriptors();
        recordInto(cmd, gridX, gridY, gridZ);
    }

    std::vector<uint32_t> ComputePipeline::loadSPIRV(const std::string &path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("ComputePipeline: cannot open SPIR-V: " + path);

        size_t byteSize = static_cast<size_t>(file.tellg());
        if (byteSize % 4 != 0)
            throw std::runtime_error("ComputePipeline: SPIR-V size not 4-byte aligned");

        std::vector<uint32_t> code(byteSize / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char *>(code.data()), static_cast<std::streamsize>(byteSize));
        return code;
    }

} // namespace Engine::Core

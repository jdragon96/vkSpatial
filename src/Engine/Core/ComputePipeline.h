#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/Image.h"
#include "Engine/Core/OneShotCommands.h"
#include "Engine/Core/Sampler.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Core {

    enum class ShaderInput {
        SpvFile,
        GlslSrc,
    };

    class ComputePipeline {
    public:
        explicit ComputePipeline(Context &context);
        ~ComputePipeline();

        ComputePipeline(const ComputePipeline &) = delete;
        ComputePipeline &operator=(const ComputePipeline &) = delete;

        ComputePipeline &AddInclude(const std::string &name, const std::string &src);

        // Preprocessor definition applied to the next Build(path). Repeated calls accumulate;
        // re-defining a name replaces it. Definitions are part of the compile cache key, so the
        // same file built with different definitions yields different modules.
        ComputePipeline &Define(const std::string &name, const std::string &value = "1");

        ComputePipeline &Build(const std::string &source, ShaderInput inputType);
        ComputePipeline &Build(const std::string &path);

        ComputePipeline &Bind(uint32_t binding, VkBuffer buffer, VkDeviceSize sizeBytes);

        ComputePipeline &Bind(uint32_t binding, Buffer &buffer) {
            return Bind(binding, buffer.Handle(), static_cast<VkDeviceSize>(buffer.Size()));
        }

        // Combined image sampler -- a sampler2D / usampler2D / isampler2D in the shader.
        //
        // The image must already be in `layout` when the dispatch runs; this records no
        // transition, because the pipeline does not own the image and cannot know whether the
        // caller is about to write it from another pass in the same batch.
        ComputePipeline &Bind(uint32_t binding, VkImageView view, VkSampler sampler,
                              VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        ComputePipeline &Bind(uint32_t binding, Image &image, Sampler &sampler,
                              VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            return Bind(binding, image.View(), sampler.Handle(), layout);
        }

        template<typename T>
        ComputePipeline &Args(const T &value) {
            static_assert(sizeof(T) <= 256, "push constant must be <= 256 bytes");
            m_pushData.resize(sizeof(T));
            std::memcpy(m_pushData.data(), &value, sizeof(T));
            return *this;
        }

        void Dispatch(VkExtent3D grid);
        void Dispatch(uint32_t gridX, uint32_t gridY = 1, uint32_t gridZ = 1);
        void DispatchElements(uint32_t numElements);

        // Records pipeline bind + push constants + dispatch into a caller-owned command
        // buffer without submitting. Ensures the pipeline and descriptors are built first.
        void RecordDispatch(VkCommandBuffer cmd, uint32_t gridX, uint32_t gridY = 1, uint32_t gridZ = 1);

        // Retained for call-site compatibility with the vkComputeBase API this replaces.
        // Dispatch() already blocks via SubmitOneShot, so this is a no-op safety net, not
        // a required call.
        void Sync();

        VkExtent3D GetLocalSize() const { return m_localSize; }

    private:
        Context &m_context;

        VkShaderModule m_shaderModule = VK_NULL_HANDLE;
        VkDescriptorPool m_descPool = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_descLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_descSet = VK_NULL_HANDLE;
        VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_pipeline = VK_NULL_HANDLE;

        std::vector<uint8_t> m_pushData;

        // One entry per bound descriptor, whatever its type. A single list rather than one per
        // type because the descriptor set layout has to be built in binding order, and two lists
        // would have to be merged back together to do that.
        struct ResourceBinding {
            uint32_t binding = 0;
            VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

            VkBuffer buffer = VK_NULL_HANDLE; // STORAGE_BUFFER
            VkDeviceSize size = 0;

            VkImageView view = VK_NULL_HANDLE; // COMBINED_IMAGE_SAMPLER
            VkSampler sampler = VK_NULL_HANDLE;
            VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        };
        std::vector<ResourceBinding> m_bindings;
        bool m_dirty = true;

        VkExtent3D m_localSize{};

        std::unordered_map<std::string, std::string> m_includes;

        // std::map, not unordered: the cache key is built by walking this container, so the
        // iteration order must be deterministic across runs.
        std::map<std::string, std::string> m_defines;

        ComputePipeline &bindResource(const ResourceBinding &entry);
        void destroyShaderResources();
        void ensurePipeline();
        void updateDescriptors();
        void recordInto(VkCommandBuffer cmd, uint32_t gridX, uint32_t gridY, uint32_t gridZ);
        void submit(uint32_t gridX, uint32_t gridY, uint32_t gridZ);
        void reflectLocalSize(const std::vector<uint32_t> &spv);

        static std::vector<uint32_t> loadSPIRV(const std::string &path);
        std::vector<uint32_t> compileGlslToSpv(const std::string &src) const;
    };

} // namespace Engine::Core

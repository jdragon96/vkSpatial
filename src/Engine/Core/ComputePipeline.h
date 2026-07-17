#pragma once

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/OneShotCommands.h"

#include <cstdint>
#include <cstring>
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

        ComputePipeline &Build(const std::string &source, ShaderInput inputType);
        ComputePipeline &Build(const std::string &path);

        ComputePipeline &Bind(uint32_t binding, VkBuffer buffer, VkDeviceSize sizeBytes);

        ComputePipeline &Bind(uint32_t binding, Buffer &buffer) {
            return Bind(binding, buffer.Handle(), static_cast<VkDeviceSize>(buffer.Size()));
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

        struct BufferBinding {
            uint32_t binding;
            VkBuffer buffer;
            VkDeviceSize size;
        };
        std::vector<BufferBinding> m_bindings;
        bool m_dirty = true;

        VkExtent3D m_localSize{};

        std::unordered_map<std::string, std::string> m_includes;

        void destroyShaderResources();
        void ensurePipeline();
        void updateDescriptors();
        void submit(uint32_t gridX, uint32_t gridY, uint32_t gridZ);
        void reflectLocalSize(const std::vector<uint32_t> &spv);

        static std::vector<uint32_t> loadSPIRV(const std::string &path);
        std::vector<uint32_t> compileGlslToSpv(const std::string &src) const;
    };

} // namespace Engine::Core

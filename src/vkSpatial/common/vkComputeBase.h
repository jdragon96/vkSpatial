#pragma once

#include "vkSpatial/common/vkGPUMemory.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace vkCommon {

    enum class ShaderInput {
        SpvFile,
        GlslSrc,
    };

    class vkComputeBase {
    public:
        using UniquePtr = std::unique_ptr<vkComputeBase>;
        using SharedPtr = std::shared_ptr<vkComputeBase>;

        static UniquePtr MakeUnique(VkDevice device, VkPhysicalDevice physicalDevice,
                                    VkQueue computeQueue, VkCommandPool commandPool) {
            return std::make_unique<vkComputeBase>(device, physicalDevice, computeQueue, commandPool);
        }

        static SharedPtr MakeShared(VkDevice device, VkPhysicalDevice physicalDevice,
                                    VkQueue computeQueue, VkCommandPool commandPool) {
            return std::make_shared<vkComputeBase>(device, physicalDevice, computeQueue, commandPool);
        }

        vkComputeBase(VkDevice device,
                      VkPhysicalDevice physicalDevice,
                      VkQueue computeQueue,
                      VkCommandPool commandPool);

        virtual ~vkComputeBase();

        vkComputeBase(const vkComputeBase &) = delete;
        vkComputeBase &operator=(const vkComputeBase &) = delete;

        vkComputeBase &AddInclude(const std::string &name, const std::string &src);

        vkComputeBase &Build(const std::string &source, ShaderInput inputType);

        vkComputeBase &Build(const std::string &path);

        vkComputeBase &Bind(uint32_t binding, VkBuffer buffer, VkDeviceSize sizeBytes);

        vkComputeBase &Bind(uint32_t binding, vkGPUMemory &mem) {
            return Bind(binding, mem.GetBuffer(),
                        static_cast<VkDeviceSize>(mem.GetSize()));
        }

        vkComputeBase &Bind(uint32_t binding, vkGPUMemory::SharedPtr mem) {
            return Bind(binding, *mem);
        }

        template<typename T>
        vkComputeBase &Args(const T &value) {
            static_assert(sizeof(T) <= 256, "push constant must be <= 256 bytes");
            m_pushData.resize(sizeof(T));
            std::memcpy(m_pushData.data(), &value, sizeof(T));
            return *this;
        }

        void Dispatch(VkExtent3D grid);
        void Dispatch(uint32_t gridX, uint32_t gridY = 1, uint32_t gridZ = 1);
        void DispatchElements(uint32_t numElements);

        void Sync();

        VkExtent3D GetLocalSize() const { return m_localSize; }

    protected:
        VkDevice m_device = VK_NULL_HANDLE;
        VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
        VkQueue m_queue = VK_NULL_HANDLE;
        VkCommandPool m_cmdPool = VK_NULL_HANDLE;

    private:
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

        // AddInclude()로 등록된 인메모리 include 소스 (name → source)
        std::unordered_map<std::string, std::string> m_includes;

        void destroyShaderResources();
        void ensurePipeline();
        void updateDescriptors();
        void submit(uint32_t gridX, uint32_t gridY, uint32_t gridZ);
        void reflectLocalSize(const std::vector<uint32_t> &spv);

        static std::vector<uint32_t> loadSPIRV(const std::string &path);
        std::vector<uint32_t> compileGlslToSpv(const std::string &src) const;
    };

} // namespace vkCommon

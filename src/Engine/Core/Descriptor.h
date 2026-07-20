#pragma once

#include "Engine/Core/Context.h"

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

namespace Engine::Core {

    struct DescriptorBinding {
        uint32_t binding = 0;
        VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        uint32_t descriptorCount = 1;
        VkShaderStageFlags stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        static DescriptorBinding CombinedImageSampler(
                uint32_t binding,
                VkShaderStageFlags stageFlags,
                uint32_t descriptorCount = 1);

        VkDescriptorSetLayoutBinding Build() const;
    };

    class DescriptorSetLayout {
    public:
        explicit DescriptorSetLayout(Context &context);
        DescriptorSetLayout(Context &context, const std::vector<DescriptorBinding> &bindings);
        ~DescriptorSetLayout();

        DescriptorSetLayout(const DescriptorSetLayout &) = delete;
        DescriptorSetLayout &operator=(const DescriptorSetLayout &) = delete;
        DescriptorSetLayout(DescriptorSetLayout &&rhs) noexcept;
        DescriptorSetLayout &operator=(DescriptorSetLayout &&rhs) noexcept;

        DescriptorSetLayout &Create(const std::vector<DescriptorBinding> &bindings);
        void Destroy();

        bool Valid() const { return m_layout != VK_NULL_HANDLE; }
        VkDescriptorSetLayout Handle() const { return m_layout; }
        const std::vector<DescriptorBinding> &Bindings() const { return m_bindings; }

    private:
        Context *m_context = nullptr;
        VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
        std::vector<DescriptorBinding> m_bindings;

        void moveFrom(DescriptorSetLayout &&rhs) noexcept;
    };

    class DescriptorSet {
    public:
        DescriptorSet() = default;
        DescriptorSet(Context &context, VkDescriptorSet set);

        bool Valid() const { return m_set != VK_NULL_HANDLE; }
        VkDescriptorSet Handle() const { return m_set; }

        void Reset();
        void UpdateCombinedImageSampler(uint32_t binding,
                                        VkSampler sampler,
                                        VkImageView imageView,
                                        VkImageLayout imageLayout,
                                        uint32_t arrayElement = 0);
        void Bind(VkCommandBuffer commandBuffer,
                  VkPipelineLayout pipelineLayout,
                  uint32_t setIndex = 0,
                  VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS) const;

    private:
        Context *m_context = nullptr;
        VkDescriptorSet m_set = VK_NULL_HANDLE;
    };

    class DescriptorPool {
    public:
        explicit DescriptorPool(Context &context);
        DescriptorPool(Context &context,
                       const std::vector<VkDescriptorPoolSize> &poolSizes,
                       uint32_t maxSets);
        ~DescriptorPool();

        DescriptorPool(const DescriptorPool &) = delete;
        DescriptorPool &operator=(const DescriptorPool &) = delete;
        DescriptorPool(DescriptorPool &&rhs) noexcept;
        DescriptorPool &operator=(DescriptorPool &&rhs) noexcept;

        DescriptorPool &Create(const std::vector<VkDescriptorPoolSize> &poolSizes,
                               uint32_t maxSets);
        DescriptorSet Allocate(VkDescriptorSetLayout layout);
        void Destroy();

        bool Valid() const { return m_pool != VK_NULL_HANDLE; }
        VkDescriptorPool Handle() const { return m_pool; }
        uint32_t MaxSets() const { return m_maxSets; }
        const std::vector<VkDescriptorPoolSize> &PoolSizes() const { return m_poolSizes; }

    private:
        Context *m_context = nullptr;
        VkDescriptorPool m_pool = VK_NULL_HANDLE;
        std::vector<VkDescriptorPoolSize> m_poolSizes;
        uint32_t m_maxSets = 0;

        void moveFrom(DescriptorPool &&rhs) noexcept;
    };

} // namespace Engine::Core

#include "Engine/Core/Descriptor.h"

#include <stdexcept>
#include <utility>

namespace Engine::Core {

    DescriptorBinding DescriptorBinding::CombinedImageSampler(
            uint32_t binding,
            VkShaderStageFlags stageFlags,
            uint32_t descriptorCount) {
        DescriptorBinding descriptor{};
        descriptor.binding = binding;
        descriptor.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor.descriptorCount = descriptorCount;
        descriptor.stageFlags = stageFlags;
        return descriptor;
    }

    VkDescriptorSetLayoutBinding DescriptorBinding::Build() const {
        VkDescriptorSetLayoutBinding bindingInfo{};
        bindingInfo.binding = binding;
        bindingInfo.descriptorType = descriptorType;
        bindingInfo.descriptorCount = descriptorCount;
        bindingInfo.stageFlags = stageFlags;
        return bindingInfo;
    }

    DescriptorSetLayout::DescriptorSetLayout(Context &context) : m_context(&context) {}

    DescriptorSetLayout::DescriptorSetLayout(Context &context,
                                             const std::vector<DescriptorBinding> &bindings)
        : DescriptorSetLayout(context) {
        Create(bindings);
    }

    DescriptorSetLayout::~DescriptorSetLayout() {
        Destroy();
    }

    DescriptorSetLayout::DescriptorSetLayout(DescriptorSetLayout &&rhs) noexcept {
        moveFrom(std::move(rhs));
    }

    DescriptorSetLayout &DescriptorSetLayout::operator=(DescriptorSetLayout &&rhs) noexcept {
        if (this != &rhs) {
            Destroy();
            moveFrom(std::move(rhs));
        }
        return *this;
    }

    DescriptorSetLayout &DescriptorSetLayout::Create(const std::vector<DescriptorBinding> &bindings) {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            throw std::runtime_error("DescriptorSetLayout::Create requires a valid context");
        if (bindings.empty())
            throw std::runtime_error("DescriptorSetLayout::Create requires bindings");

        Destroy();
        m_bindings = bindings;

        std::vector<VkDescriptorSetLayoutBinding> bindingInfos;
        bindingInfos.reserve(m_bindings.size());
        for (const DescriptorBinding &binding: m_bindings)
            bindingInfos.push_back(binding.Build());

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindingInfos.size());
        layoutInfo.pBindings = bindingInfos.data();
        if (vkCreateDescriptorSetLayout(m_context->device, &layoutInfo,
                                        nullptr, &m_layout) != VK_SUCCESS)
            throw std::runtime_error("DescriptorSetLayout: failed to create layout");

        return *this;
    }

    void DescriptorSetLayout::Destroy() {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            return;

        if (m_layout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(m_context->device, m_layout, nullptr);
        m_layout = VK_NULL_HANDLE;
        m_bindings.clear();
    }

    void DescriptorSetLayout::moveFrom(DescriptorSetLayout &&rhs) noexcept {
        m_context = rhs.m_context;
        m_layout = rhs.m_layout;
        m_bindings = std::move(rhs.m_bindings);

        rhs.m_context = nullptr;
        rhs.m_layout = VK_NULL_HANDLE;
    }

    DescriptorSet::DescriptorSet(Context &context, VkDescriptorSet set)
        : m_context(&context), m_set(set) {}

    void DescriptorSet::Reset() {
        m_context = nullptr;
        m_set = VK_NULL_HANDLE;
    }

    void DescriptorSet::UpdateCombinedImageSampler(uint32_t binding,
                                                   VkSampler sampler,
                                                   VkImageView imageView,
                                                   VkImageLayout imageLayout,
                                                   uint32_t arrayElement) {
        if (!m_context || m_context->device == VK_NULL_HANDLE || m_set == VK_NULL_HANDLE)
            throw std::runtime_error("DescriptorSet::UpdateCombinedImageSampler requires a valid set");

        VkDescriptorImageInfo imageInfo{};
        imageInfo.sampler = sampler;
        imageInfo.imageView = imageView;
        imageInfo.imageLayout = imageLayout;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_set;
        write.dstBinding = binding;
        write.dstArrayElement = arrayElement;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(m_context->device, 1, &write, 0, nullptr);
    }

    void DescriptorSet::Bind(VkCommandBuffer commandBuffer,
                             VkPipelineLayout pipelineLayout,
                             uint32_t setIndex,
                             VkPipelineBindPoint bindPoint) const {
        if (m_set == VK_NULL_HANDLE)
            throw std::runtime_error("DescriptorSet::Bind requires a valid set");

        vkCmdBindDescriptorSets(commandBuffer,
                                bindPoint,
                                pipelineLayout,
                                setIndex,
                                1,
                                &m_set,
                                0,
                                nullptr);
    }

    DescriptorPool::DescriptorPool(Context &context) : m_context(&context) {}

    DescriptorPool::DescriptorPool(Context &context,
                                   const std::vector<VkDescriptorPoolSize> &poolSizes,
                                   uint32_t maxSets)
        : DescriptorPool(context) {
        Create(poolSizes, maxSets);
    }

    DescriptorPool::~DescriptorPool() {
        Destroy();
    }

    DescriptorPool::DescriptorPool(DescriptorPool &&rhs) noexcept {
        moveFrom(std::move(rhs));
    }

    DescriptorPool &DescriptorPool::operator=(DescriptorPool &&rhs) noexcept {
        if (this != &rhs) {
            Destroy();
            moveFrom(std::move(rhs));
        }
        return *this;
    }

    DescriptorPool &DescriptorPool::Create(const std::vector<VkDescriptorPoolSize> &poolSizes,
                                           uint32_t maxSets) {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            throw std::runtime_error("DescriptorPool::Create requires a valid context");
        if (poolSizes.empty() || maxSets == 0)
            throw std::runtime_error("DescriptorPool::Create requires pool sizes and maxSets");

        Destroy();
        m_poolSizes = poolSizes;
        m_maxSets = maxSets;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = m_maxSets;
        poolInfo.poolSizeCount = static_cast<uint32_t>(m_poolSizes.size());
        poolInfo.pPoolSizes = m_poolSizes.data();
        if (vkCreateDescriptorPool(m_context->device, &poolInfo,
                                   nullptr, &m_pool) != VK_SUCCESS)
            throw std::runtime_error("DescriptorPool: failed to create pool");

        return *this;
    }

    DescriptorSet DescriptorPool::Allocate(VkDescriptorSetLayout layout) {
        if (!m_context || m_context->device == VK_NULL_HANDLE || m_pool == VK_NULL_HANDLE)
            throw std::runtime_error("DescriptorPool::Allocate requires a valid pool");
        if (layout == VK_NULL_HANDLE)
            throw std::runtime_error("DescriptorPool::Allocate requires a valid layout");

        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_pool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &layout;
        if (vkAllocateDescriptorSets(m_context->device, &allocInfo, &set) != VK_SUCCESS)
            throw std::runtime_error("DescriptorPool: failed to allocate descriptor set");

        return DescriptorSet(*m_context, set);
    }

    void DescriptorPool::Destroy() {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            return;

        if (m_pool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(m_context->device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
        m_poolSizes.clear();
        m_maxSets = 0;
    }

    void DescriptorPool::moveFrom(DescriptorPool &&rhs) noexcept {
        m_context = rhs.m_context;
        m_pool = rhs.m_pool;
        m_poolSizes = std::move(rhs.m_poolSizes);
        m_maxSets = rhs.m_maxSets;

        rhs.m_context = nullptr;
        rhs.m_pool = VK_NULL_HANDLE;
        rhs.m_maxSets = 0;
    }

} // namespace Engine::Core

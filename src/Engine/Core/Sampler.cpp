#include "Engine/Core/Sampler.h"

#include <stdexcept>
#include <utility>

namespace Engine::Core {

    SamplerDescriptor SamplerDescriptor::ShadowMapManualPCF() {
        SamplerDescriptor descriptor{};
        descriptor.magFilter = VK_FILTER_NEAREST;
        descriptor.minFilter = VK_FILTER_NEAREST;
        descriptor.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        descriptor.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        descriptor.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        descriptor.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        descriptor.maxLod = 1.0f;
        descriptor.compareEnable = VK_FALSE;
        descriptor.compareOp = VK_COMPARE_OP_ALWAYS;
        return descriptor;
    }

    SamplerDescriptor SamplerDescriptor::ShadowMapCompare(VkFilter filter) {
        SamplerDescriptor descriptor = ShadowMapManualPCF();
        descriptor.magFilter = filter;
        descriptor.minFilter = filter;
        descriptor.mipmapMode = filter == VK_FILTER_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                           : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        descriptor.compareEnable = VK_TRUE;
        descriptor.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        return descriptor;
    }

    VkSamplerCreateInfo SamplerDescriptor::Build() const {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = magFilter;
        info.minFilter = minFilter;
        info.mipmapMode = mipmapMode;
        info.addressModeU = addressModeU;
        info.addressModeV = addressModeV;
        info.addressModeW = addressModeW;
        info.mipLodBias = mipLodBias;
        info.anisotropyEnable = anisotropyEnable;
        info.maxAnisotropy = maxAnisotropy;
        info.compareEnable = compareEnable;
        info.compareOp = compareOp;
        info.minLod = minLod;
        info.maxLod = maxLod;
        info.borderColor = borderColor;
        info.unnormalizedCoordinates = unnormalizedCoordinates;
        return info;
    }

    Sampler::Sampler(Context &context) : m_context(&context) {}

    Sampler::Sampler(Context &context, const SamplerDescriptor &descriptor) : Sampler(context) {
        Create(descriptor);
    }

    Sampler::~Sampler() {
        Destroy();
    }

    Sampler::Sampler(Sampler &&rhs) noexcept {
        moveFrom(std::move(rhs));
    }

    Sampler &Sampler::operator=(Sampler &&rhs) noexcept {
        if (this != &rhs) {
            Destroy();
            moveFrom(std::move(rhs));
        }
        return *this;
    }

    Sampler &Sampler::Create(const SamplerDescriptor &descriptor) {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            throw std::runtime_error("Sampler::Create requires a valid context");

        Destroy();
        m_descriptor = descriptor;

        const VkSamplerCreateInfo info = m_descriptor.Build();
        if (vkCreateSampler(m_context->device, &info, nullptr, &m_sampler) != VK_SUCCESS)
            throw std::runtime_error("Sampler: failed to create sampler");

        return *this;
    }

    void Sampler::Destroy() {
        if (!m_context || m_context->device == VK_NULL_HANDLE)
            return;

        if (m_sampler != VK_NULL_HANDLE)
            vkDestroySampler(m_context->device, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
        m_descriptor = {};
    }

    void Sampler::moveFrom(Sampler &&rhs) noexcept {
        m_context = rhs.m_context;
        m_sampler = rhs.m_sampler;
        m_descriptor = rhs.m_descriptor;

        rhs.m_context = nullptr;
        rhs.m_sampler = VK_NULL_HANDLE;
        rhs.m_descriptor = {};
    }

} // namespace Engine::Core

#pragma once

#include "Engine/Core/Context.h"

#include <vulkan/vulkan.h>

namespace Engine::Core {

    struct SamplerDescriptor {
        VkFilter magFilter = VK_FILTER_LINEAR;
        VkFilter minFilter = VK_FILTER_LINEAR;
        VkSamplerMipmapMode mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        VkSamplerAddressMode addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VkSamplerAddressMode addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VkSamplerAddressMode addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        float mipLodBias = 0.0f;
        VkBool32 anisotropyEnable = VK_FALSE;
        float maxAnisotropy = 1.0f;
        VkBool32 compareEnable = VK_FALSE;
        VkCompareOp compareOp = VK_COMPARE_OP_ALWAYS;
        float minLod = 0.0f;
        float maxLod = 0.0f;
        VkBorderColor borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        VkBool32 unnormalizedCoordinates = VK_FALSE;

        static SamplerDescriptor ShadowMapManualPCF();
        static SamplerDescriptor ShadowMapCompare(VkFilter filter = VK_FILTER_NEAREST);

        VkSamplerCreateInfo Build() const;
    };

    class Sampler {
    public:
        explicit Sampler(Context &context);
        Sampler(Context &context, const SamplerDescriptor &descriptor);
        ~Sampler();

        Sampler(const Sampler &) = delete;
        Sampler &operator=(const Sampler &) = delete;
        Sampler(Sampler &&rhs) noexcept;
        Sampler &operator=(Sampler &&rhs) noexcept;

        Sampler &Create(const SamplerDescriptor &descriptor);
        void Destroy();

        bool Valid() const { return m_sampler != VK_NULL_HANDLE; }
        VkSampler Handle() const { return m_sampler; }
        const SamplerDescriptor &Descriptor() const { return m_descriptor; }

    private:
        Context *m_context = nullptr;
        VkSampler m_sampler = VK_NULL_HANDLE;
        SamplerDescriptor m_descriptor{};

        void moveFrom(Sampler &&rhs) noexcept;
    };

} // namespace Engine::Core

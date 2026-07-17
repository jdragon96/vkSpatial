#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

namespace vkRender {

    template<typename Tag>
    class Handle {
    public:
        Handle() = default;
        explicit Handle(uint32_t id) : m_id(id) {}

        uint32_t Id() const { return m_id; }
        bool Valid() const { return m_id != 0; }
        explicit operator bool() const { return Valid(); }

        bool operator==(Handle rhs) const { return m_id == rhs.m_id; }
        bool operator!=(Handle rhs) const { return !(*this == rhs); }

    private:
        uint32_t m_id = 0;
    };

    struct MeshTag {};
    struct MaterialTag {};
    struct TextureTag {};

    using MeshHandle = Handle<MeshTag>;
    using MaterialHandle = Handle<MaterialTag>;
    using TextureHandle = Handle<TextureTag>;

    struct MeshDescriptor {
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        VkIndexType indexType = VK_INDEX_TYPE_UINT32;
    };

    struct MaterialDescriptor {
        float baseColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float metallic = 0.0f;
        float roughness = 0.8f;
        TextureHandle baseColorTexture;
        TextureHandle normalTexture;
    };

} // namespace vkRender

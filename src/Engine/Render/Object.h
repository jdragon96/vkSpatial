#pragma once

#include "Engine/Core/Buffer.h"
#include "utilities/Math.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Engine::Render {

    struct Vertex {
        float position[3] = {};
        float normal[3] = {};
        float color[3] = {};

        Vertex() = default;

        Vertex(const vkMath::Vec3 &p, const vkMath::Vec3 &n, const vkMath::Vec3 &c)
            : position{p.x(), p.y(), p.z()},
              normal{n.x(), n.y(), n.z()},
              color{c.x(), c.y(), c.z()} {}
    };

    template<typename VertexT = Vertex>
    class Object {
    public:
        using Vertex = VertexT;
        using Index = uint32_t;

        Object() = default;

        Object(std::vector<Vertex> vertices,
               std::vector<Index> indices,
               vkMath::Mat4 model = vkMath::Mat4::Identity())
            : m_vertices(std::move(vertices)),
              m_indices(std::move(indices)),
              m_model(model) {}

        Object(const Object &other)
            : m_vertices(other.m_vertices),
              m_indices(other.m_indices),
              m_model(other.m_model),
              m_firstIndex(other.m_firstIndex) {}

        Object &operator=(const Object &other) {
            if (this == &other)
                return *this;
            m_vertices = other.m_vertices;
            m_indices = other.m_indices;
            m_model = other.m_model;
            m_firstIndex = other.m_firstIndex;
            ResetUpload();
            return *this;
        }

        Object(Object &&) noexcept = default;
        Object &operator=(Object &&) noexcept = default;

        void SetGeometry(std::vector<Vertex> vertices, std::vector<Index> indices) {
            m_vertices = std::move(vertices);
            m_indices = std::move(indices);
            ResetUpload();
        }

        void SetVertices(std::vector<Vertex> vertices) {
            m_vertices = std::move(vertices);
            ResetUpload();
        }

        void SetIndices(std::vector<Index> indices) {
            m_indices = std::move(indices);
            ResetUpload();
        }

        Index AddVertex(const Vertex &vertex) {
            m_vertices.push_back(vertex);
            ResetUpload();
            return static_cast<Index>(m_vertices.size() - 1);
        }

        Index AddVertex(Vertex &&vertex) {
            m_vertices.push_back(std::move(vertex));
            ResetUpload();
            return static_cast<Index>(m_vertices.size() - 1);
        }

        void AddIndex(Index index) {
            m_indices.push_back(index);
            ResetUpload();
        }

        void AddIndices(std::initializer_list<Index> indices) {
            m_indices.insert(m_indices.end(), indices.begin(), indices.end());
            ResetUpload();
        }

        void AddTriangle(Index a, Index b, Index c) {
            AddIndices({a, b, c});
        }

        void AddQuad(Index a, Index b, Index c, Index d) {
            AddIndices({a, b, c, c, d, a});
        }

        void SetModel(const vkMath::Mat4 &model) {
            m_model = model;
        }

        void SetFirstIndex(uint32_t firstIndex) {
            m_firstIndex = firstIndex;
        }

        void Clear() {
            m_vertices.clear();
            m_indices.clear();
            m_firstIndex = 0;
            ResetUpload();
        }

        bool Empty() const { return m_vertices.empty() || m_indices.empty(); }
        bool Uploaded() const { return m_vertexBuffer && m_indexBuffer; }

        const std::vector<Vertex> &Vertices() const { return m_vertices; }
        const std::vector<Index> &Indices() const { return m_indices; }
        const vkMath::Mat4 &Model() const { return m_model; }

        const Vertex *VertexData() const { return m_vertices.data(); }
        const Index *IndexData() const { return m_indices.data(); }

        std::size_t VertexCount() const { return m_vertices.size(); }
        std::size_t IndexCount() const { return m_indices.size(); }
        std::size_t VertexByteSize() const { return m_vertices.size() * sizeof(Vertex); }
        std::size_t IndexByteSize() const { return m_indices.size() * sizeof(Index); }
        uint32_t FirstIndex() const { return m_firstIndex; }

        void Upload(Engine::Core::Context &context,
                    Engine::Core::QueueRole queueRole = Engine::Core::QueueRole::Graphics) {
            if (Empty())
                throw std::runtime_error("Object::Upload requires vertex and index data");

            m_vertexBuffer =
                    std::make_unique<Engine::Core::Buffer>(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            const uint32_t vertexBytes = static_cast<uint32_t>(VertexByteSize());
            m_vertexBuffer->Allocate(vertexBytes);
            m_vertexBuffer->Upload(VertexData(), vertexBytes, queueRole);

            m_indexBuffer =
                    std::make_unique<Engine::Core::Buffer>(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            const uint32_t indexBytes = static_cast<uint32_t>(IndexByteSize());
            m_indexBuffer->Allocate(indexBytes);
            m_indexBuffer->Upload(IndexData(), indexBytes, queueRole);
        }

        void Render(VkCommandBuffer commandBuffer) const {
            if (!Uploaded())
                throw std::runtime_error("Object::Render requires uploaded GPU buffers");

            VkBuffer vertexBuffer = m_vertexBuffer->Handle();
            const VkDeviceSize vertexOffset = 0;
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexOffset);
            vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer->Handle(), 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(IndexCount()), 1, 0, 0, 0);
        }

    private:
        std::vector<Vertex> m_vertices;
        std::vector<Index> m_indices;
        vkMath::Mat4 m_model = vkMath::Mat4::Identity();
        uint32_t m_firstIndex = 0;
        std::unique_ptr<Engine::Core::Buffer> m_vertexBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_indexBuffer;

        void ResetUpload() {
            m_vertexBuffer.reset();
            m_indexBuffer.reset();
        }
    };

    template<typename VertexT = Vertex>
    struct ObjectGeometry {
        using Vertex = VertexT;
        using Index = uint32_t;

        std::vector<Vertex> vertices;
        std::vector<Index> indices;

        const Vertex *VertexData() const { return vertices.data(); }
        const Index *IndexData() const { return indices.data(); }
        std::size_t VertexCount() const { return vertices.size(); }
        std::size_t IndexCount() const { return indices.size(); }
        std::size_t VertexByteSize() const { return vertices.size() * sizeof(Vertex); }
        std::size_t IndexByteSize() const { return indices.size() * sizeof(Index); }
        bool Empty() const { return vertices.empty() || indices.empty(); }
    };

    template<typename VertexT = Vertex>
    ObjectGeometry<VertexT> PackObjects(std::vector<Object<VertexT>> &objects) {
        ObjectGeometry<VertexT> geometry;

        std::size_t vertexCount = 0;
        std::size_t indexCount = 0;
        for (const Object<VertexT> &object: objects) {
            vertexCount += object.VertexCount();
            indexCount += object.IndexCount();
        }
        geometry.vertices.reserve(vertexCount);
        geometry.indices.reserve(indexCount);

        for (Object<VertexT> &object: objects) {
            const uint32_t baseVertex = static_cast<uint32_t>(geometry.vertices.size());
            object.SetFirstIndex(static_cast<uint32_t>(geometry.indices.size()));

            geometry.vertices.insert(
                    geometry.vertices.end(),
                    object.Vertices().begin(),
                    object.Vertices().end());
            for (uint32_t index: object.Indices())
                geometry.indices.push_back(baseVertex + index);
        }

        return geometry;
    }

} // namespace Engine::Render

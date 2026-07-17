#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace vkRender {

    using Entity = uint32_t;

    class Scene {
    public:
        using UniquePtr = std::unique_ptr<Scene>;

        Entity CreateEntity() {
            const Entity entity = m_nextEntity++;
            m_entities.push_back(entity);
            return entity;
        }

        void AddEntity(Entity entity) {
            if (!Contains(entity))
                m_entities.push_back(entity);
        }

        void Remove(Entity entity) {
            m_entities.erase(std::remove(m_entities.begin(), m_entities.end(), entity),
                             m_entities.end());
        }

        void Clear() { m_entities.clear(); }

        bool Contains(Entity entity) const {
            return std::find(m_entities.begin(), m_entities.end(), entity) != m_entities.end();
        }

        const std::vector<Entity> &Entities() const { return m_entities; }

    private:
        Entity m_nextEntity = 1;
        std::vector<Entity> m_entities;
    };

} // namespace vkRender

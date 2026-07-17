#include "Engine/Spatial/DirectionalHostStore.h"

#include <stdexcept>

namespace Engine::Spatial {

    bool DirectionalHostStore::Contains(const DirectionalGroupKey &key) const {
        return m_groups.find(key) != m_groups.end();
    }

    const DirectionalHostStore::Group &
    DirectionalHostStore::Get(const DirectionalGroupKey &key) const {
        auto it = m_groups.find(key);
        if (it == m_groups.end())
            throw std::runtime_error("DirectionalHostStore: group not found");
        return it->second;
    }

    DirectionalHostStore::Group &
    DirectionalHostStore::GetOrCreate(const DirectionalGroupKey &key) {
        // operator[] value-initializes: HostTsdfVoxel's member initializers zero every voxel.
        return m_groups[key];
    }

    void DirectionalHostStore::Put(const DirectionalGroupKey &key, const Group &data) {
        m_groups[key] = data;
    }

} // namespace Engine::Spatial

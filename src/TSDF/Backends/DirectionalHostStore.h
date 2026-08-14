#pragma once

#include "TSDF/Backends/DirectionalTSDFTypes.h"

#include <array>
#include <cstddef>
#include <unordered_map>

namespace Engine::Spatial {

    // Host-side authoritative sparse TSDF store (DirectionalTSDF design doc §11/§12).
    // Pure CPU data structure: the GPU active pool is a cache over this map.
    class DirectionalHostStore {
    public:
        using Group = std::array<HostTsdfVoxel, kVoxelsPerGroup>;

        bool Contains(const DirectionalGroupKey &key) const;
        // Throws std::runtime_error if the key is absent.
        const Group &Get(const DirectionalGroupKey &key) const;
        // Creates a zero-filled group on first touch.
        Group &GetOrCreate(const DirectionalGroupKey &key);
        void Put(const DirectionalGroupKey &key, const Group &data);
        size_t Size() const { return m_groups.size(); }

    private:
        std::unordered_map<DirectionalGroupKey, Group, DirectionalGroupKeyHash> m_groups;
    };

} // namespace Engine::Spatial

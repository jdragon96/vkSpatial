#include "TSDF/Volume.h"

#include "TSDF/ComposedVolume.h"
#include "TSDF/Memory/FlatStrategy.h"
#include "TSDF/Memory/SubmapStrategy.h"
#include "TSDF/Memory/TileStrategy.h"

#include <algorithm>

namespace TSDF {

    std::vector<std::string> VolumeRegistry::Names() const {
        std::vector<std::string> names;
        names.reserve(m_factories.size());
        for (const auto &entry: m_factories) names.push_back(entry.first);
        std::sort(names.begin(), names.end()); // unordered_map order is not reproducible
        return names;
    }

    VolumeRegistry VolumeRegistry::Default() {
        VolumeRegistry registry;
        registry.Register("flat", [] {
            return std::make_unique<ComposedVolume>(std::make_unique<FlatStrategy>());
        });
        registry.Register("tile", [] {
            return std::make_unique<ComposedVolume>(std::make_unique<TileStrategy>());
        });
        registry.Register("submap", [] {
            return std::make_unique<ComposedVolume>(std::make_unique<SubmapStrategy>());
        });
        return registry;
    }

} // namespace TSDF

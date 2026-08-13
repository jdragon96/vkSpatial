#include "TSDF/Volume.h"

#include <algorithm>

namespace TSDF {

    std::vector<std::string> VolumeRegistry::Names() const {
        std::vector<std::string> names;
        names.reserve(m_factories.size());
        for (const auto &entry: m_factories) names.push_back(entry.first);
        std::sort(names.begin(), names.end()); // unordered_map order is not reproducible
        return names;
    }

    // Task 6에서 flat/tile/submap이 여기에 등록된다.
    VolumeRegistry VolumeRegistry::Default() {
        VolumeRegistry registry;
        return registry;
    }

} // namespace TSDF

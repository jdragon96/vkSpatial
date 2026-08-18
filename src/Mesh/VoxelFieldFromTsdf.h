#pragma once

#include <vector>

#include "Mesh/VoxelField.h"

// Kept out of VoxelField.h on purpose: naming TSDF::AdvancedEntry opens `namespace TSDF`, which
// cannot coexist with the global `class TSDF` in TSDF/TSDF.h. Only include this from a translation
// unit that does not use that class.
namespace TSDF {
    struct AdvancedEntry;
}

namespace Mesh {

    // coord recovered from the entry's world-space centre, value = tsdf, gradient = normal.
    VoxelField FromAdvancedEntries(const std::vector<TSDF::AdvancedEntry> &entries, float cellSize);

} // namespace Mesh

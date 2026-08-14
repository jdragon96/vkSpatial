#pragma once

#include "TSDF/Backends/CompactDirectionalTSDF.h"
#include "TSDF/Backends/TiledDirectionalTSDF.h"

namespace TSDF {

    // TiledCompactDirectionalTSDF is the compact-backend instantiation of the generalized tiling
    // coordinator. See TiledDirectionalTSDF.h for geometry, ghost routing, and extraction.
    using TiledCompactDirectionalTSDF = TiledDirectionalTSDF<CompactDirectionalTSDF>;

} // namespace TSDF

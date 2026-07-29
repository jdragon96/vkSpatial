#pragma once

#include "Engine/Spatial/CompactDirectionalTSDF.h"
#include "Engine/Spatial/TiledDirectionalTSDF.h"

namespace Engine::Spatial {

    // TiledCompactDirectionalTSDF is the compact-backend instantiation of the generalized tiling
    // coordinator. See TiledDirectionalTSDF.h for geometry, ghost routing, and extraction.
    using TiledCompactDirectionalTSDF = TiledDirectionalTSDF<CompactDirectionalTSDF>;

} // namespace Engine::Spatial

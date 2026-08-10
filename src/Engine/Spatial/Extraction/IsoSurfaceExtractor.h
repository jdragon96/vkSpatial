#pragma once

#include "Engine/Spatial/Extraction/SurfaceMesh.h"
#include "Engine/Spatial/Extraction/VoxelField.h"

namespace Engine::Spatial::Extraction {

    struct ExtractParams {
        float isoLevel = 0.0f;
        float weldFraction = 0.25f;
        float featureAngleCosineThreshold = 0.9f;
    };

    class IsoSurfaceExtractor {
    public:
        virtual ~IsoSurfaceExtractor() = default;
        virtual const char *Name() const = 0;
        virtual SurfaceMesh Extract(const VoxelField &field, const ExtractParams &params) const = 0;
    };

} // namespace Engine::Spatial::Extraction

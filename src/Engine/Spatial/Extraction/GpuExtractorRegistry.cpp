#include "Engine/Spatial/Extraction/GpuExtractorRegistry.h"

#include <memory>

namespace Engine::Spatial::Extraction {

    // Factory functions for each registered GPU strategy, defined in their own translation units
    // (GpuMarchingCubesExtractor.cpp now; GpuMarchingCubes33Extractor.cpp / GpuMarchingTetrahedraExtractor.cpp
    // add their own forward declaration + Register() call here in Tasks 4-5) -- mirrors
    // ExtractorRegistry.cpp's CPU registry exactly.
    std::unique_ptr<IsoSurfaceExtractor> CreateGpuMarchingCubesExtractor(Engine::Core::Context &context);

    GpuExtractorRegistry GpuExtractorRegistry::Default() {
        GpuExtractorRegistry reg;
        reg.Register("mc-gpu", CreateGpuMarchingCubesExtractor);
        return reg;
    }

} // namespace Engine::Spatial::Extraction

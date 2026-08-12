#include "Engine/Spatial/Extraction/GpuExtractorRegistry.h"

#include <memory>

namespace Engine::Spatial::Extraction {

    // Factory functions for each registered GPU strategy, defined in their own translation units
    // (GpuMarchingCubesExtractor.cpp, GpuMarchingCubes33Extractor.cpp now; GpuMarchingTetrahedraExtractor.cpp
    // adds its own forward declaration + Register() call here in Task 5) -- mirrors
    // ExtractorRegistry.cpp's CPU registry exactly.
    std::unique_ptr<IsoSurfaceExtractor> CreateGpuMarchingCubesExtractor(Engine::Core::Context &context);
    std::unique_ptr<IsoSurfaceExtractor> CreateGpuMarchingCubes33Extractor(Engine::Core::Context &context);

    GpuExtractorRegistry GpuExtractorRegistry::Default() {
        GpuExtractorRegistry reg;
        reg.Register("mc-gpu", CreateGpuMarchingCubesExtractor);
        reg.Register("mc33-gpu", CreateGpuMarchingCubes33Extractor);
        return reg;
    }

} // namespace Engine::Spatial::Extraction

#include "Engine/Spatial/Extraction/ExtractorRegistry.h"

#include <memory>

namespace Engine::Spatial::Extraction {

    // Factory functions for each registered strategy, defined in their own translation units
    // (e.g. MarchingCubesExtractor.cpp). Later tasks add one forward declaration + one
    // Register() call each to Default() below.
    std::unique_ptr<IsoSurfaceExtractor> CreateMarchingCubesExtractor();

    ExtractorRegistry ExtractorRegistry::Default() {
        ExtractorRegistry reg;
        reg.Register("mc", CreateMarchingCubesExtractor);
        return reg;
    }

} // namespace Engine::Spatial::Extraction

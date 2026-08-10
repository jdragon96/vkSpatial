#include "Engine/Spatial/Extraction/ExtractorRegistry.h"

#include <memory>

namespace Engine::Spatial::Extraction {

    // Factory functions for each registered strategy, defined in their own translation units
    // (e.g. MarchingCubesExtractor.cpp). Later tasks add one forward declaration + one
    // Register() call each to Default() below.
    std::unique_ptr<IsoSurfaceExtractor> CreateMarchingCubesExtractor();
    std::unique_ptr<IsoSurfaceExtractor> CreateMarchingCubes33Extractor();
    std::unique_ptr<IsoSurfaceExtractor> CreateMarchingTetrahedraExtractor();
    std::unique_ptr<IsoSurfaceExtractor> CreateExtendedMarchingCubesExtractor();

    ExtractorRegistry ExtractorRegistry::Default() {
        ExtractorRegistry reg;
        reg.Register("mc", CreateMarchingCubesExtractor);
        reg.Register("mc33", CreateMarchingCubes33Extractor);
        reg.Register("mtet", CreateMarchingTetrahedraExtractor);
        reg.Register("emc", CreateExtendedMarchingCubesExtractor);
        return reg;
    }

} // namespace Engine::Spatial::Extraction

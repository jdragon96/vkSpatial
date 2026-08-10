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
    std::unique_ptr<IsoSurfaceExtractor> CreateDualContouringExtractor();
    std::unique_ptr<IsoSurfaceExtractor> CreateDualMarchingCubesExtractor();

    ExtractorRegistry ExtractorRegistry::Default() {
        ExtractorRegistry reg;
        reg.Register("mc", CreateMarchingCubesExtractor);
        reg.Register("mc33", CreateMarchingCubes33Extractor);
        reg.Register("mtet", CreateMarchingTetrahedraExtractor);
        reg.Register("emc", CreateExtendedMarchingCubesExtractor);
        reg.Register("dc", CreateDualContouringExtractor);
        reg.Register("dmc", CreateDualMarchingCubesExtractor);
        return reg;
    }

} // namespace Engine::Spatial::Extraction

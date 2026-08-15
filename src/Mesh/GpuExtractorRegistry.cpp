#include "Mesh/GpuExtractorRegistry.h"

#include <memory>

namespace Mesh {

    // Factory functions for each registered GPU strategy, defined in their own translation units
    // (GpuMarchingCubesExtractor.cpp, GpuMarchingCubes33Extractor.cpp, GpuMarchingTetrahedraExtractor.cpp)
    // -- mirrors ExtractorRegistry.cpp's CPU registry exactly.
    std::unique_ptr<IsoSurfaceExtractor> CreateGpuMarchingCubesExtractor(Engine::Core::Context &context);
    std::unique_ptr<IsoSurfaceExtractor> CreateGpuMarchingCubes33Extractor(Engine::Core::Context &context);
    std::unique_ptr<IsoSurfaceExtractor> CreateGpuMarchingTetrahedraExtractor(Engine::Core::Context &context);

    GpuExtractorRegistry GpuExtractorRegistry::Default() {
        GpuExtractorRegistry reg;
        reg.Register("mc-gpu", CreateGpuMarchingCubesExtractor);
        reg.Register("mc33-gpu", CreateGpuMarchingCubes33Extractor);
        reg.Register("mtet-gpu", CreateGpuMarchingTetrahedraExtractor);
        return reg;
    }

} // namespace Mesh

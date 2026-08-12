#pragma once

// GPU-compute twin of the "mc33" CPU strategy (MarchingCubes33Extractor.cpp): the same
// case/subcase resolution (asymptotic-decider face test + interior test, transcribed tables from
// MarchingCubes33Tables.h) ported to extract_mc33.comp, sharing GpuIsoSurfaceExtractorCommon's
// upload/readback verbatim with GpuMarchingCubesExtractor -- and the same core::
// WeldAndComputeNormals weld, so Extract() produces a SurfaceMesh matching the CPU "mc33"
// extractor to floating-point tolerance (see GpuExtractors.Mc33GpuMatchesCpuOnSphere in
// test/test_gpu_extractors.cpp). Registered as "mc33-gpu" by GpuExtractorRegistry::Default().

#include "Engine/Core/Context.h"
#include "Engine/Spatial/Extraction/IsoSurfaceExtractor.h"

namespace Engine::Spatial::Extraction {

    class GpuMarchingCubes33Extractor : public IsoSurfaceExtractor {
    public:
        explicit GpuMarchingCubes33Extractor(Engine::Core::Context &context) : m_context(&context) {}

        const char *Name() const override { return "mc33-gpu"; }

        SurfaceMesh Extract(const VoxelField &field, const ExtractParams &params) const override;

    private:
        Engine::Core::Context *m_context;
    };

} // namespace Engine::Spatial::Extraction

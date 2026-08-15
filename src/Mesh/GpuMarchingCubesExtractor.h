#pragma once

// GPU-compute twin of the "mc" CPU strategy (MarchingCubesExtractor.cpp): the same edgeTable/
// triTable walk and winding swap (extract_mc.comp, sharing voxel_common.glsl's tables with
// voxel_tsdf_mc.comp), the same core::WeldAndComputeNormals weld -- so Extract() produces a
// SurfaceMesh matching the CPU "mc" extractor to floating-point tolerance (see
// GpuExtractors.McGpuMatchesCpuOnSphere in test/test_gpu_extractors.cpp). Registered as "mc-gpu" by
// GpuExtractorRegistry::Default().

#include "Engine/Core/Context.h"
#include "Mesh/IsoSurfaceExtractor.h"

namespace Mesh {

    class GpuMarchingCubesExtractor : public IsoSurfaceExtractor {
    public:
        explicit GpuMarchingCubesExtractor(Engine::Core::Context &context) : m_context(&context) {}

        const char *Name() const override { return "mc-gpu"; }

        SurfaceMesh Extract(const VoxelField &field, const ExtractParams &params) const override;

    private:
        Engine::Core::Context *m_context;
    };

} // namespace Mesh

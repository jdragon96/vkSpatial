#pragma once

// GPU-compute twin of the "mtet" CPU strategy (MarchingTetrahedraExtractor.cpp): each candidate
// cube is split into the 6 tetrahedra sharing the cube's 0-6 main diagonal (extract_mtet.comp),
// and each tetrahedron is triangulated independently from its own 4 corner signs -- no trilinear
// ambiguity by construction (see MarchingTetrahedraExtractor.cpp's file header for the full
// argument), so unlike "mc33" this needs no asymptotic-decider/interior test. Shares
// GpuIsoSurfaceExtractorCommon's upload/readback verbatim with GpuMarchingCubesExtractor /
// GpuMarchingCubes33Extractor. CPU-welds with the SAME reduced weld distance the CPU "mtet"
// extractor uses (params.weldFraction * field.CellSize() / kWeldDistanceDivisor, with
// kWeldDistanceDivisor == 32 -- see GpuMarchingTetrahedraExtractor.cpp and
// MarchingTetrahedraExtractor.cpp's own extensive comment on why "mtet" needs a tighter weld
// tolerance than "mc"/"mc33"), so Extract() produces a SurfaceMesh matching the CPU "mtet"
// extractor to floating-point tolerance (see GpuExtractors.MtetGpuMatchesCpuOnSphere in
// test/test_gpu_extractors.cpp). Registered as "mtet-gpu" by GpuExtractorRegistry::Default().

#include "Engine/Core/Context.h"
#include "Mesh/IsoSurfaceExtractor.h"

namespace Mesh {

    class GpuMarchingTetrahedraExtractor : public IsoSurfaceExtractor {
    public:
        explicit GpuMarchingTetrahedraExtractor(Engine::Core::Context &context) : m_context(&context) {}

        const char *Name() const override { return "mtet-gpu"; }

        SurfaceMesh Extract(const VoxelField &field, const ExtractParams &params) const override;

    private:
        Engine::Core::Context *m_context;
    };

} // namespace Mesh

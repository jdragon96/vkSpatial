// The "mc" strategy: single-resolution CPU Marching Cubes over a VoxelField, built entirely
// from the shared core (MarchingCubesCore.h). No dedicated header -- this extractor is only
// ever reached through ExtractorRegistry::Default() via the CreateMarchingCubesExtractor()
// factory function declared (and used) in ExtractorRegistry.cpp, matching how later extractor
// strategies register into the same Default().

#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Spatial/Extraction/IsoSurfaceExtractor.h"
#include "Engine/Spatial/Extraction/MarchingCubesCore.h"

#include <memory>

namespace Engine::Spatial::Extraction {

    namespace {

        class MarchingCubesExtractor : public IsoSurfaceExtractor {
        public:
            const char *Name() const override { return "mc"; }

            SurfaceMesh Extract(const VoxelField &field, const ExtractParams &params) const override {
                const auto bases = core::CandidateBases(field.OccupiedCoords());

                // Shift sampled values by isoLevel so the shared core's hard-coded
                // zero-crossing test (sdf[c] < 0) extracts the params.isoLevel isosurface.
                const auto sampler = [&](const std::array<int, 3> &coord, float &out) {
                    if (!field.Sample(coord, out)) return false;
                    out -= params.isoLevel;
                    return true;
                };

                std::vector<core::RawTriangle> raw;
                core::GenerateRawTriangles(bases, sampler, field.CellSize(), raw);

                return core::WeldAndComputeNormals(raw, params.weldFraction * field.CellSize());
            }
        };

    } // namespace

    std::unique_ptr<IsoSurfaceExtractor> CreateMarchingCubesExtractor() {
        return std::make_unique<MarchingCubesExtractor>();
    }

} // namespace Engine::Spatial::Extraction

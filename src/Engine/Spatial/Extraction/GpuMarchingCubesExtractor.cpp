#include "Engine/Spatial/Extraction/GpuMarchingCubesExtractor.h"

#include "Engine/Core/ComputePipeline.h"
#include "Engine/Spatial/Extraction/GpuIsoSurfaceExtractorCommon.h"
#include "Engine/Spatial/Extraction/MarchingCubesCore.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Spatial::Extraction {

    namespace {

        // Mirrors extract_mc.comp's PushConstants block field-for-field: SCALAR fields only (no
        // vec3/ivec3 -- a push-constant vec3/ivec3 pads to 16 bytes in GLSL and would desync from
        // this tightly-packed C++ struct; see icp_iterate.comp.glsl's doc comment for the same
        // reasoning applied to GpuPointToPlaneIcp).
        struct ExtractMarchingCubesPushConstants {
            int32_t originX, originY, originZ;
            int32_t dimsX, dimsY, dimsZ;
            float cellSize;
            float isoLevel;
            uint32_t candidateCount;
            uint32_t maxTriangles;
        };

        // A cube emits at most 5 triangles (the longest triTable row), so this bound is tight and
        // exact -- unlike voxel_tsdf_mc.comp's fixed MC_MAX_TRIS budget, it is never clamped away
        // real geometry.
        constexpr uint32_t kMaxTrianglesPerCube = 5u;

        // MUST match extract_mc.comp's order-key stride EXACTLY -- that shader has no
        // ORDER_KEY_STRIDE #define (unlike mc33/mtet's macro-shared emit path); its
        // processCube() hardcodes `candidateIndex * 8u + uint(i / 3)` inline. See
        // GpuIsoSurfaceExtractorCommon.h's "Order key" doc for the full contract: the stride must
        // strictly exceed a cube's maximum triangle count, or ReadbackRawTriangles' order-key sort
        // silently reconstructs the WRONG per-cube emission order instead of failing loudly. The
        // static_assert below enforces that contract at compile time.
        constexpr uint32_t kOrderKeyStride = 8u;
        static_assert(kMaxTrianglesPerCube < kOrderKeyStride,
                      "extract_mc.comp's order-key stride must strictly exceed kMaxTrianglesPerCube, "
                      "or ReadbackRawTriangles' order-key sort silently reconstructs the wrong "
                      "per-cube emission order");

    } // namespace

    SurfaceMesh GpuMarchingCubesExtractor::Extract(const VoxelField &field, const ExtractParams &params) const {
        const float weldDistance = params.weldFraction * field.CellSize();

        // (a) Upload the candidate cube bases + a dense coord->value lookup.
        const GpuVoxelFieldUpload upload = UploadField(*m_context, field);
        if (upload.candidateCount == 0) return core::WeldAndComputeNormals({}, weldDistance);

        // A cube emits at most kMaxTrianglesPerCube triangles (the longest triTable row), so this
        // bound is tight and exact -- unlike voxel_tsdf_mc.comp's fixed MC_MAX_TRIS budget, it is
        // never clamped away real geometry.
        const uint32_t maxTriangles = upload.candidateCount * kMaxTrianglesPerCube;
        GpuTriangleOutput output = AllocateTriangleOutput(*m_context, maxTriangles);

        // (b) Dispatch extract_mc.comp: one invocation per candidate cube.
        Engine::Core::ComputePipeline kernel(*m_context);
        kernel.Build("extract_mc.comp");
        kernel.Bind(0, *output.counter)
                .Bind(1, *output.vertices)
                .Bind(2, *upload.candidateBases)
                .Bind(3, *upload.valueGrid);

        ExtractMarchingCubesPushConstants pushConstants{};
        pushConstants.originX = upload.origin[0];
        pushConstants.originY = upload.origin[1];
        pushConstants.originZ = upload.origin[2];
        pushConstants.dimsX = upload.dims[0];
        pushConstants.dimsY = upload.dims[1];
        pushConstants.dimsZ = upload.dims[2];
        pushConstants.cellSize = field.CellSize();
        pushConstants.isoLevel = params.isoLevel;
        pushConstants.candidateCount = upload.candidateCount;
        pushConstants.maxTriangles = maxTriangles;
        kernel.Args(pushConstants);

        kernel.DispatchElements(upload.candidateCount);

        // (c) Read back the raw triangles and weld exactly like the CPU "mc" extractor.
        const std::vector<core::RawTriangle> raw = ReadbackRawTriangles(output);
        return core::WeldAndComputeNormals(raw, weldDistance);
    }

    std::unique_ptr<IsoSurfaceExtractor> CreateGpuMarchingCubesExtractor(Engine::Core::Context &context) {
        return std::make_unique<GpuMarchingCubesExtractor>(context);
    }

} // namespace Engine::Spatial::Extraction

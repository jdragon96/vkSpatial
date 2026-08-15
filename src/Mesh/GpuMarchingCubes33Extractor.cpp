#include "Mesh/GpuMarchingCubes33Extractor.h"

#include "Engine/Core/ComputePipeline.h"
#include "Mesh/GpuIsoSurfaceExtractorCommon.h"
#include "Mesh/MarchingCubesCore.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Mesh {

    namespace {

        // Mirrors extract_mc33.comp's PushConstants block field-for-field: SCALAR fields only (no
        // vec3/ivec3 -- a push-constant vec3/ivec3 pads to 16 bytes in GLSL and would desync from
        // this tightly-packed C++ struct; see extract_mc.comp / GpuMarchingCubesExtractor.cpp's
        // identical reasoning).
        struct ExtractMarchingCubes33PushConstants {
            int32_t originX, originY, originZ;
            int32_t dimsX, dimsY, dimsZ;
            float cellSize;
            float isoLevel;
            uint32_t candidateCount;
            uint32_t maxTriangles;
        };

        // Marching Cubes 33's case/subcase tables emit up to 12 triangles for a single cube --
        // case 13.4 (tiling13_4, 4 configurations per (case13-face-test) configuration index,
        // each row 36 ints == 12 triangles), the maximum across every case/subcase in
        // MarchingCubes33Tables.h (audited by hand against every `emit(table, N)` call in
        // MarchingCubes33Extractor.cpp's EmitCubeTriangles -- the next-largest are the 10-triangle
        // cases 13.3/13.5.2). This is markedly higher than "mc"'s 5-triangle bound (the longest
        // mc::triTable row), which is WHY extract_mc33.comp's order-key stride must be 16, not
        // mc's 8 -- see that file's ORDER_KEY_STRIDE comment for the full "why".
        constexpr uint32_t kMaxTrianglesPerCube33 = 12u;

        // MUST match extract_mc33.comp's #define ORDER_KEY_STRIDE (16u) EXACTLY -- see that file's
        // own ORDER_KEY_STRIDE comment for the full contract. See
        // GpuIsoSurfaceExtractorCommon.h's "Order key" doc for why: the stride must strictly
        // exceed a cube's maximum triangle count, or ReadbackRawTriangles' order-key sort silently
        // reconstructs the WRONG per-cube emission order instead of failing loudly. The
        // static_assert below enforces that contract at compile time.
        constexpr uint32_t kOrderKeyStride33 = 16u;
        static_assert(kMaxTrianglesPerCube33 < kOrderKeyStride33,
                      "extract_mc33.comp's ORDER_KEY_STRIDE must strictly exceed "
                      "kMaxTrianglesPerCube33, or ReadbackRawTriangles' order-key sort silently "
                      "reconstructs the wrong per-cube emission order");

    } // namespace

    SurfaceMesh GpuMarchingCubes33Extractor::Extract(const VoxelField &field, const ExtractParams &params) const {
        const float weldDistance = params.weldFraction * field.CellSize();

        // (a) Upload the candidate cube bases + a dense coord->value lookup (identical to "mc-gpu").
        const GpuVoxelFieldUpload upload = UploadField(*m_context, field);
        if (upload.candidateCount == 0) return core::WeldAndComputeNormals({}, weldDistance);

        const uint32_t maxTriangles = upload.candidateCount * kMaxTrianglesPerCube33;
        GpuTriangleOutput output = AllocateTriangleOutput(*m_context, maxTriangles);

        // (b) Dispatch extract_mc33.comp: one invocation per candidate cube.
        Engine::Core::ComputePipeline kernel(*m_context);
        kernel.Build("Mesh/kernel_extract_mc33.comp.glsl");
        kernel.Bind(0, *output.counter)
                .Bind(1, *output.vertices)
                .Bind(2, *upload.candidateBases)
                .Bind(3, *upload.valueGrid);

        ExtractMarchingCubes33PushConstants pushConstants{};
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

        // (c) Read back the raw triangles and weld exactly like the CPU "mc33" extractor.
        const std::vector<core::RawTriangle> raw = ReadbackRawTriangles(output);
        return core::WeldAndComputeNormals(raw, weldDistance);
    }

    std::unique_ptr<IsoSurfaceExtractor> CreateGpuMarchingCubes33Extractor(Engine::Core::Context &context) {
        return std::make_unique<GpuMarchingCubes33Extractor>(context);
    }

} // namespace Mesh

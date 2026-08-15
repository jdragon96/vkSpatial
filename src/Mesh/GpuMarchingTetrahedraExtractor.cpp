#include "Mesh/GpuMarchingTetrahedraExtractor.h"

#include "Engine/Core/ComputePipeline.h"
#include "Mesh/GpuIsoSurfaceExtractorCommon.h"
#include "Mesh/MarchingCubesCore.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Mesh {

    namespace {

        // Mirrors extract_mtet.comp's PushConstants block field-for-field: SCALAR fields only (no
        // vec3/ivec3 -- a push-constant vec3/ivec3 pads to 16 bytes in GLSL and would desync from
        // this tightly-packed C++ struct; see GpuMarchingCubesExtractor.cpp's identical reasoning).
        struct ExtractMarchingTetrahedraPushConstants {
            int32_t originX, originY, originZ;
            int32_t dimsX, dimsY, dimsZ;
            float cellSize;
            float isoLevel;
            uint32_t candidateCount;
            uint32_t maxTriangles;
        };

        // A cube's 6 tetrahedra (kCubeTetrahedra, extract_mtet.comp) each emit at most 2 triangles
        // (TriangulateTetrahedron's negativeCount==2 quadrilateral case; see
        // MarchingTetrahedraExtractor.cpp), so 6*2 = 12 triangles/cube is the exact bound -- the SAME
        // maximum "mc33" has, so extract_mtet.comp reuses that file's ORDER_KEY_STRIDE=16 (see
        // extract_mtet.comp's own "ORDER-KEY STRIDE" header section for the "why strictly exceed the
        // max" contract).
        constexpr uint32_t kMaxTrianglesPerCubeMtet = 12u;

        // MUST match extract_mtet.comp's #define ORDER_KEY_STRIDE (16u, reused from mc33's) EXACTLY
        // -- see that file's own "ORDER-KEY STRIDE" header section for the full contract. See
        // GpuIsoSurfaceExtractorCommon.h's "Order key" doc for why: the stride must strictly exceed
        // a cube's maximum triangle count, or ReadbackRawTriangles' order-key sort silently
        // reconstructs the WRONG per-cube emission order instead of failing loudly. The
        // static_assert below enforces that contract at compile time.
        constexpr uint32_t kOrderKeyStrideMtet = 16u;
        static_assert(kMaxTrianglesPerCubeMtet < kOrderKeyStrideMtet,
                      "extract_mtet.comp's ORDER_KEY_STRIDE must strictly exceed "
                      "kMaxTrianglesPerCubeMtet, or ReadbackRawTriangles' order-key sort silently "
                      "reconstructs the wrong per-cube emission order");

        // MUST match MarchingTetrahedraExtractor.cpp's kWeldDistanceDivisor EXACTLY (currently 32 --
        // see that file's extensive comment on why "mtet"'s denser diagonal-cut vertices need a
        // reduced weld distance vs "mc"/"mc33"'s plain weldFraction*cellSize). A mismatch here would
        // silently weld the GPU mesh to a DIFFERENT vertex set than the CPU one, failing
        // GpuExtractors.MtetGpuMatchesCpuOnSphere.
        constexpr float kWeldDistanceDivisor = 32.0f;

    } // namespace

    SurfaceMesh GpuMarchingTetrahedraExtractor::Extract(const VoxelField &field, const ExtractParams &params) const {
        const float weldDistance = params.weldFraction * field.CellSize() / kWeldDistanceDivisor;

        // (a) Upload the candidate cube bases + a dense coord->value lookup (identical to
        // "mc-gpu"/"mc33-gpu").
        const GpuVoxelFieldUpload upload = UploadField(*m_context, field);
        if (upload.candidateCount == 0) return core::WeldAndComputeNormals({}, weldDistance);

        const uint32_t maxTriangles = upload.candidateCount * kMaxTrianglesPerCubeMtet;
        GpuTriangleOutput output = AllocateTriangleOutput(*m_context, maxTriangles);

        // (b) Dispatch extract_mtet.comp: one invocation per candidate cube.
        Engine::Core::ComputePipeline kernel(*m_context);
        kernel.Build("Mesh/kernel_extract_mtet.comp.glsl");
        kernel.Bind(0, *output.counter)
                .Bind(1, *output.vertices)
                .Bind(2, *upload.candidateBases)
                .Bind(3, *upload.valueGrid);

        ExtractMarchingTetrahedraPushConstants pushConstants{};
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

        // (c) Read back the raw triangles and weld exactly like the CPU "mtet" extractor (SAME
        // reduced weld distance -- see kWeldDistanceDivisor above).
        const std::vector<core::RawTriangle> raw = ReadbackRawTriangles(output);
        return core::WeldAndComputeNormals(raw, weldDistance);
    }

    std::unique_ptr<IsoSurfaceExtractor> CreateGpuMarchingTetrahedraExtractor(Engine::Core::Context &context) {
        return std::make_unique<GpuMarchingTetrahedraExtractor>(context);
    }

} // namespace Mesh

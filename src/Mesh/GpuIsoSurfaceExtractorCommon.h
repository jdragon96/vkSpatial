#pragma once

// Shared GPU upload/dispatch/readback plumbing for every Gpu*Extractor (GpuMarchingCubesExtractor
// now; GpuMarchingCubes33Extractor / GpuMarchingTetrahedraExtractor reuse this verbatim in later
// tasks). Mirrors GpuPointToPlaneIcp.{h,cpp}'s compute pattern: Engine::Core::Buffer Allocate/Upload
// for inputs, AllocateHostVisibleReadback + MakeVisibleToGPU/MakeVisibleToCPU for a buffer BOTH the
// host and the GPU touch (zero-copy, no staging), Engine::Core::ComputePipeline Bind/Args/
// DispatchElements for the dispatch itself (built by each concrete extractor, since the compute
// kernel differs per strategy: extract_mc.comp, extract_mc33.comp, extract_mtet.comp).
//
// Field representation (documented here per the plan's request): the SIMPLEST first cut is a DENSE
// value grid over the candidate cube bases' integer bounding box, uploaded as one flat float buffer
// + origin/dims. This trades memory for simplicity -- a real streamed/tiled TSDF covering a large
// scene would want a sparse GPU hash instead (mirroring voxel_tsdf_mc.comp's wangHash probe), but
// the VoxelField inputs this task targets (synthetic test fixtures, single-tile extraction) are
// small enough that a dense grid is fine, and it turns extract_mc.comp's corner lookup into a
// single flat-array index instead of a hash probe loop. UploadField() guards against the dense
// grid exceeding a fixed cell budget so a much larger field fails loudly (std::runtime_error)
// instead of silently exhausting GPU memory -- see GpuIsoSurfaceExtractorCommon.cpp.
//
// Emit contract: every Gpu*Extractor kernel reserves output vertex slots with an int atomicAdd on
// a SINGLE counter (MoltenVK has no float atomics, so nothing is ever atomically accumulated --
// only this one integer slot-reservation), then writes 3 world-space vertices per triangle into
// the reserved slots. AllocateTriangleOutput()/ReadbackRawTriangles() below own that buffer pair.
// AllocateTriangleOutput() guards its vertex-buffer size against a fixed triangle budget the same
// way UploadField() guards the dense grid above -- see GpuIsoSurfaceExtractorCommon.cpp.

#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"
#include "Mesh/VoxelField.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace Mesh {

    // Sentinel written into every dense-grid cell the VoxelField never observed. Every Gpu*Extractor
    // kernel sharing this contract (extract_mc.comp now) must treat any sampled value >= this as
    // "absent", exactly mirroring VoxelField::Sample()'s bool return on the CPU. Chosen comfortably
    // below FLT_MAX so no legitimate SDF/TSDF value (bounded by the field's truncation distance)
    // can ever collide with it. KEEP IN SYNC with UNRESOLVED_FIELD_VALUE in every .comp that
    // includes this contract.
    inline constexpr float kGpuUnresolvedFieldValue = 1e37f;

    // GPU buffers produced by UploadField(): the candidate cube bases (one ivec4 per candidate cube,
    // xyz used) to dispatch one compute invocation per, and the dense value grid + its origin/dims
    // those invocations sample. origin/dims describe the grid in the FIELD's own integer coordinate
    // space (see extract_mc.comp's sampleFieldValue()). Binding-agnostic: the caller Bind()s these
    // wherever its own kernel's layout expects them.
    struct GpuVoxelFieldUpload {
        std::array<int, 3> origin{0, 0, 0};
        std::array<int, 3> dims{1, 1, 1};
        uint32_t candidateCount = 0;

        std::unique_ptr<Engine::Core::Buffer> candidateBases; // unallocated (Size()==0) when
        std::unique_ptr<Engine::Core::Buffer> valueGrid;       // candidateCount==0 -- caller must
                                                                // skip dispatch entirely in that case
    };

    // Builds core::CandidateBases(field.OccupiedCoords()) and a dense value grid spanning their
    // bounding box (+1 cell so every candidate's 8 corners stay in-bounds), then uploads both to
    // GPU buffers via `context`. candidateCount == 0 for an empty field (nothing to extract);
    // callers must not Bind()/dispatch in that case, since the buffers are left unallocated.
    // Throws std::runtime_error if the resulting dense grid would exceed the dense-first-cut cell
    // budget (see the file header comment above).
    GpuVoxelFieldUpload UploadField(Engine::Core::Context &context, const VoxelField &field);

    // GPU buffers produced by AllocateTriangleOutput(): the int-atomic triangle counter (bind at
    // binding 0 of the emit contract) and the vertex-slot buffer it indexes into (binding 1).
    // Every Gpu*Extractor kernel must Bind() these at binding 0/1 respectively -- see
    // extract_mc.comp's header comment for the full contract.
    struct GpuTriangleOutput {
        std::unique_ptr<Engine::Core::Buffer> counter;  // binding 0: single `int`, zero-initialised
        std::unique_ptr<Engine::Core::Buffer> vertices; // binding 1: vec4[3 * maxTriangles]
        uint32_t maxTriangles = 0;
    };

    // Allocates (host-visible, zero-copy readback) a zeroed triangle counter + a vertex-slot buffer
    // sized for up to `maxTriangles` triangles (3 world-space vec4 vertices each). Ready to Bind()
    // at binding 0/1 immediately; the counter is already zeroed and flushed to the GPU.
    // Throws std::runtime_error if maxTriangles exceeds a sane output-buffer budget (see
    // GpuIsoSurfaceExtractorCommon.cpp's kMaxTrianglesPerExtraction comment) instead of silently
    // wrapping the vertex-buffer byte-size computation and under-allocating.
    GpuTriangleOutput AllocateTriangleOutput(Engine::Core::Context &context, uint32_t maxTriangles);

    // Reads back whatever a compute dispatch wrote under the AllocateTriangleOutput() contract:
    // invalidates + reads the counter, clamps it to maxTriangles (defensive -- the shader itself
    // already refuses to write past maxTriangles, see extract_mc.comp), and copies that many raw
    // (unwelded) triangles out for the caller's core::WeldAndComputeNormals() pass.
    //
    // core::WeldAndComputeNormals's first-hit tie-break makes its output depend on RAW TRIANGLE
    // ORDER, not just the raw triangle set -- but GPU invocations race on the atomic slot counter,
    // so the buffer fills in whatever order threads happen to finish, NOT candidate order. Every
    // kernel sharing this contract therefore also bit-casts a deterministic emission-order key
    // into every vertex's .w (see extract_mc.comp's "Order key" doc); this function sorts by that
    // key before returning, which reconstructs the CPU extractor's exact sequential emission order
    // so the two sides weld to the SAME vertex positions (not merely the same triangle count).
    std::vector<core::RawTriangle> ReadbackRawTriangles(GpuTriangleOutput &output);

} // namespace Mesh

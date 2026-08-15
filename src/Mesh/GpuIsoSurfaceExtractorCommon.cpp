#include "Mesh/GpuIsoSurfaceExtractorCommon.h"

#include "Mesh/MarchingCubesCore.h"

#include <Eigen/Core>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace Mesh {

    namespace {

        // Above this many dense-grid cells, UploadField() refuses (see GpuIsoSurfaceExtractorCommon.h's
        // file-header doc for why a dense grid is this task's deliberate "simplest first cut" and when
        // a sparse GPU hash would be needed instead). 128M float cells = 512 MB, comfortably above
        // anything the synthetic test fixtures or a single TiledAdvancedTSDF tile produce.
        constexpr uint64_t kMaxDenseValueGridCells = 128ull * 1024 * 1024;

        // Above this many triangles, AllocateTriangleOutput() refuses. The vertex-slot buffer it
        // sizes is 3 vertices/triangle * 16 bytes/vertex (std430 vec4) = 48 bytes/triangle, and
        // Engine::Core::Buffer's Allocate*() calls take that byte size as a uint32_t -- so a
        // maxTriangles beyond UINT32_MAX/48 (~89.5M) would wrap the byte-size computation and
        // silently under-allocate. 16M triangles = 768 MB, comfortably below that wraparound point
        // (~5.3x headroom) and comfortably above anything a synthetic test fixture or single
        // candidate-cube dispatch (candidateCount * kMaxTrianglesPerCube, see each
        // Gpu*Extractor.cpp) produces today -- same "dense-first-cut" budget philosophy as
        // kMaxDenseValueGridCells above.
        constexpr uint64_t kMaxTrianglesPerExtraction = 16ull * 1024 * 1024;

        // std430 array stride for ivec4 is 16 bytes; padding the 4th component keeps the CPU-side
        // upload byte-identical to what extract_mc.comp's `ivec4 g_candidateBases[]` expects.
        struct PaddedCoordinate {
            int32_t x, y, z, padding;
        };

    } // namespace

    GpuVoxelFieldUpload UploadField(Engine::Core::Context &context, const VoxelField &field) {
        GpuVoxelFieldUpload upload;
        upload.candidateBases = std::make_unique<Engine::Core::Buffer>(context);
        upload.valueGrid = std::make_unique<Engine::Core::Buffer>(context);

        const auto candidateBaseSet = core::CandidateBases(field.OccupiedCoords());
        upload.candidateCount = static_cast<uint32_t>(candidateBaseSet.size());
        if (candidateBaseSet.empty()) return upload; // empty field: leave buffers unallocated

        // 1. Flatten to a vector (iteration order is whatever the unordered_set gives -- fine, see
        // the header doc: WeldAndComputeNormals()'s result doesn't depend on triangle order except
        // at weld-distance tie-breaks, which the RMSE-based consistency gate tolerates) and its
        // integer bounding box.
        const std::vector<std::array<int, 3>> candidateBaseList(candidateBaseSet.begin(), candidateBaseSet.end());
        std::array<int, 3> baseMinimum = candidateBaseList.front();
        std::array<int, 3> baseMaximum = candidateBaseList.front();
        for (const auto &base: candidateBaseList)
            for (int axis = 0; axis < 3; ++axis) {
                baseMinimum[axis] = std::min(baseMinimum[axis], base[axis]);
                baseMaximum[axis] = std::max(baseMaximum[axis], base[axis]);
            }

        upload.origin = baseMinimum;
        for (int axis = 0; axis < 3; ++axis)
            upload.dims[axis] = (baseMaximum[axis] - baseMinimum[axis]) + 2; // +1 corner reach, inclusive count

        const uint64_t cellCount =
                uint64_t(upload.dims[0]) * uint64_t(upload.dims[1]) * uint64_t(upload.dims[2]);
        if (cellCount > kMaxDenseValueGridCells)
            throw std::runtime_error(
                    "GpuIsoSurfaceExtractorCommon::UploadField: dense value grid of " + std::to_string(cellCount) +
                    " cells exceeds the dense-first-cut budget of " + std::to_string(kMaxDenseValueGridCells) +
                    " (see GpuIsoSurfaceExtractorCommon.h's field-representation doc comment)");

        // 2. Dense value grid: sentinel-filled, then sparsely overwritten from the field's own
        // occupied coordinates. Every occupied coordinate is guaranteed inside [origin, origin+dims)
        // because candidate bases are occupied +/- {0,-1} per axis, so occupied coordinates never
        // leave [baseMinimum, baseMaximum+1] -- exactly this grid's range.
        std::vector<float> denseValues(size_t(cellCount), kGpuUnresolvedFieldValue);
        for (const auto &coordinate: field.OccupiedCoords()) {
            float value = 0.0f;
            if (!field.Sample(coordinate, value)) continue; // defensive; OccupiedCoords entries always Sample()
            const std::array<int, 3> local{coordinate[0] - upload.origin[0], coordinate[1] - upload.origin[1],
                                            coordinate[2] - upload.origin[2]};
            const size_t flatIndex = (size_t(local[2]) * size_t(upload.dims[1]) + size_t(local[1])) *
                                              size_t(upload.dims[0]) +
                                      size_t(local[0]);
            denseValues[flatIndex] = value;
        }

        // 3. Upload both buffers.
        std::vector<PaddedCoordinate> paddedBases(candidateBaseList.size());
        for (size_t i = 0; i < candidateBaseList.size(); ++i)
            paddedBases[i] = {int32_t(candidateBaseList[i][0]), int32_t(candidateBaseList[i][1]),
                              int32_t(candidateBaseList[i][2]), 0};
        const auto paddedBasesBytes = uint32_t(paddedBases.size() * sizeof(PaddedCoordinate));
        upload.candidateBases->Allocate(paddedBasesBytes);
        upload.candidateBases->Upload(paddedBases.data(), paddedBasesBytes);

        const auto denseValuesBytes = uint32_t(denseValues.size() * sizeof(float));
        upload.valueGrid->Allocate(denseValuesBytes);
        upload.valueGrid->Upload(denseValues.data(), denseValuesBytes);

        return upload;
    }

    GpuTriangleOutput AllocateTriangleOutput(Engine::Core::Context &context, uint32_t maxTriangles) {
        if (uint64_t(maxTriangles) > kMaxTrianglesPerExtraction)
            throw std::runtime_error(
                    "GpuIsoSurfaceExtractorCommon::AllocateTriangleOutput: maxTriangles of " +
                    std::to_string(maxTriangles) + " exceeds the dense-first-cut triangle budget of " +
                    std::to_string(kMaxTrianglesPerExtraction) +
                    " (see GpuIsoSurfaceExtractorCommon.cpp's kMaxTrianglesPerExtraction comment)");

        GpuTriangleOutput output;
        output.maxTriangles = maxTriangles;
        output.counter = std::make_unique<Engine::Core::Buffer>(context);
        output.vertices = std::make_unique<Engine::Core::Buffer>(context);

        // Zero-copy: write the initial zero straight into the persistently-mapped counter, flush to
        // the GPU, and later ReadbackRawTriangles() invalidates + reads the SAME mapping after the
        // shader's atomicAdd's have run -- mirrors AdvancedTSDF::Reset()/DownloadEntries()'s
        // MappedPtr()+MakeVisibleToGPU()/MakeVisibleToCPU() idiom for a GPU-accumulated counter.
        output.counter->AllocateHostVisibleReadback(sizeof(int32_t));
        *static_cast<int32_t *>(output.counter->MappedPtr()) = 0;
        output.counter->MakeVisibleToGPU(sizeof(int32_t));

        // Computed in uint64_t: maxTriangles * 3 vertices/triangle * 16 bytes/vertex can exceed
        // UINT32_MAX for maxTriangles beyond ~89.5M (see kMaxTrianglesPerExtraction's comment above).
        // Narrowing this multiplication to uint32_t BEFORE the budget check above would silently
        // wrap and under-allocate `vertices`, and extract_*.comp's atomicAdd-reserved slots would
        // then write past the end of the (too-small) buffer. The narrowing static_cast below is
        // safe only because the throw above already bounds maxTriangles well under that wraparound
        // point.
        constexpr uint64_t kBytesPerVertex = 4ull * sizeof(float); // vec4 per vertex, std430
        const uint64_t vertexBytes = std::max<uint64_t>(1, maxTriangles) * 3ull * kBytesPerVertex;
        output.vertices->AllocateHostVisibleReadback(static_cast<uint32_t>(vertexBytes));

        return output;
    }

    std::vector<core::RawTriangle> ReadbackRawTriangles(GpuTriangleOutput &output) {
        output.counter->MakeVisibleToCPU(sizeof(int32_t));
        int32_t rawTriangleCount = 0;
        std::memcpy(&rawTriangleCount, output.counter->MappedPtr(), sizeof(int32_t));
        const uint32_t triangleCount =
                std::min<uint32_t>(uint32_t(std::max<int32_t>(rawTriangleCount, 0)), output.maxTriangles);

        if (triangleCount == 0) return {};

        constexpr uint32_t kBytesPerVertex = 4u * sizeof(float);
        const uint32_t vertexBytes = triangleCount * 3u * kBytesPerVertex;
        output.vertices->MakeVisibleToCPU(vertexBytes);
        const auto *vertexFloats = static_cast<const float *>(output.vertices->MappedPtr());

        // Pair each triangle with the emission-order key its shader stashed in every vertex's .w
        // (bit-cast, not a float value -- read back via memcpy so no rounding is possible), then
        // sort ascending: see the header doc for why this is required to match the CPU weld.
        std::vector<std::pair<uint32_t, core::RawTriangle>> orderedTriangles(triangleCount);
        for (uint32_t i = 0; i < triangleCount; ++i) {
            const float *a = vertexFloats + (size_t(i) * 3 + 0) * 4;
            const float *b = vertexFloats + (size_t(i) * 3 + 1) * 4;
            const float *c = vertexFloats + (size_t(i) * 3 + 2) * 4;
            uint32_t orderKey = 0;
            std::memcpy(&orderKey, &a[3], sizeof(uint32_t));
            orderedTriangles[i] = {orderKey,
                                   core::RawTriangle{Eigen::Vector3f(a[0], a[1], a[2]), Eigen::Vector3f(b[0], b[1], b[2]),
                                                      Eigen::Vector3f(c[0], c[1], c[2])}};
        }
        std::sort(orderedTriangles.begin(), orderedTriangles.end(),
                  [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

        std::vector<core::RawTriangle> raw(triangleCount);
        for (uint32_t i = 0; i < triangleCount; ++i) raw[i] = orderedTriangles[i].second;
        return raw;
    }

} // namespace Mesh

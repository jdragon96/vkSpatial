#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Compute/StagingBuffer.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Pipeline/Reconstruction/DepthCameraFrameSource.h" // CameraIntrinsics

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <string>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

struct ValidationMaskProperty {
    std::uint32_t valid = 0;   // [H2] the pixel carries a depth measurement
    std::uint32_t emitted = 0; // [H3] the pixel produced a trustworthy point + normal
};

static_assert(alignof(ValidationMaskProperty) == 4,
              "ValidationMaskProperty gained a member wider than 4 bytes; std430 will pad it and "
              "the GPU stride will no longer match sizeof()");
static_assert(sizeof(ValidationMaskProperty) % 4 == 0,
              "ValidationMaskProperty must stay a pack of 4-byte scalars");
// Mirrors ValidationMaskCounters in ValidationMask.common.glsl, and Pipeline::DepthFilterStats
// field for field so a GPU run and a CPU run report the same numbers under the same names.
struct ValidationMaskCounters {
    std::uint32_t emittedPoints = 0;
    std::uint32_t rejectedByRange = 0;
    std::uint32_t rejectedByNeighbourSupport = 0;
    std::uint32_t rejectedByIncidence = 0;
};

static_assert(sizeof(ValidationMaskCounters) == 16, "the GLSL mirror is four 4-byte scalars");
static_assert(offsetof(ValidationMaskCounters, rejectedByRange) == 4, "field order drifted");
static_assert(offsetof(ValidationMaskCounters, rejectedByNeighbourSupport) == 8, "field order drifted");
static_assert(offsetof(ValidationMaskCounters, rejectedByIncidence) == 12, "field order drifted");

static_assert(offsetof(ValidationMaskProperty, emitted) == 4,
              "the GLSL struct lists emitted second; the two orders must not drift");
static_assert(offsetof(ValidationMaskProperty, valid) == 0,
              "valid must stay the first member: the clear kernel writes it by name, but every "
              "other pass indexes the struct by its GLSL offsets");

class ValidationMask {
public:
    ValidationMask(Engine::Core::Context &context, int width, int height)
        : m_width(width), m_height(height) {
        const std::size_t pixels = std::size_t(width) * std::size_t(height);
        const uint32_t vectorBytes = uint32_t(pixels * 4u * sizeof(float));
        auto device = [&context](uint32_t bytes) {
            auto buffer = std::make_unique<Engine::Core::Buffer>(context);
            buffer->Allocate(bytes);
            return buffer;
        };
        auto readback = [&context](uint32_t bytes) {
            auto buffer = std::make_unique<Engine::Core::Buffer>(context);
            buffer->AllocateHostVisibleReadback(bytes);
            return buffer;
        };
        m_filteredDepth = device(uint32_t(pixels * sizeof(float)));
        m_vertices = device(vectorBytes);
        m_normals = device(vectorBytes);
        m_counters = readback(uint32_t(sizeof(ValidationMaskCounters)));
        m_rowOffset = readback(uint32_t((std::size_t(height) + 1u) * sizeof(std::uint32_t)));
        m_points = readback(vectorBytes);
        m_compactNormals = readback(vectorBytes);

        m_propertyBuffer = std::make_unique<Engine::Core::Buffer>(context);
        m_propertyBuffer->Allocate(uint32_t(std::size_t(width) * std::size_t(height) * sizeof(ValidationMaskProperty)));

        kernel_clearProperty = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_clearProperty->Build("Pipeline/Reconstruction/Algorithm/kernel_ClearValidMask.comp.glsl");
        kernel_windowAveraging = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_windowAveraging->Build("Pipeline/Reconstruction/Algorithm/kernel_WindowAveraging.comp.glsl");
        kernel_buildVertexGrid = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_buildVertexGrid->Build("Pipeline/Reconstruction/Algorithm/kernel_BuildVertexGrid.comp.glsl");
        kernel_estimateNormal = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_estimateNormal->Build("Pipeline/Reconstruction/Algorithm/kernel_EstimateNormal.comp.glsl");
        kernel_countEmittedPerRow = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_countEmittedPerRow->Build("Pipeline/Reconstruction/Algorithm/kernel_CountEmittedPerRow.comp.glsl");
        kernel_scanRows = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_scanRows->Build("Pipeline/Reconstruction/Algorithm/kernel_ScanRows.comp.glsl");
        kernel_scatterPoints = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_scatterPoints->Build("Pipeline/Reconstruction/Algorithm/kernel_ScatterPoints.comp.glsl");
    }

    // The whole front end, recorded into one batch. The caller submits, then reads PointCount(),
    // DownloadPoints(), DownloadNormals(), DownloadCounters().
    //
    // Clear runs first even though BuildVertexGrid writes `valid` and EstimateNormal writes
    // `emitted` for every pixel, so nothing is stale. It costs one 8-byte store per pixel, and it
    // is what keeps a future pass that writes only SOME pixels from inheriting the previous
    // frame's verdict.
    void Execute(Engine::Compute::CommandBatch &batch, Engine::Core::Buffer &sourceDepth,
                 const Pipeline::CameraIntrinsics &intrinsics,
                 const Pipeline::DepthFilterOptions &filter) {
        if (intrinsics.width != m_width || intrinsics.height != m_height)
            throw std::runtime_error("ValidationMask::Execute: intrinsics are " +
                                     std::to_string(intrinsics.width) + "x" +
                                     std::to_string(intrinsics.height) + " but the buffers were "
                                     "sized for " + std::to_string(m_width) + "x" +
                                     std::to_string(m_height));

        // Per frame, not cumulative. Pipeline::DepthFilterStats is owned by its caller and
        // accumulates; these are scoped to the pass, and mixing the conventions would inflate
        // every report.
        batch.FillBuffer(m_counters->Handle(), 0, sizeof(ValidationMaskCounters), 0u);
        batch.Barrier();

        RecordClear(batch);
        batch.Barrier();
        RecordWindowAveraging(batch, sourceDepth, *m_filteredDepth, m_width, m_height,
                              filter.prefilterWindow, filter.relativeDepthJump,
                              filter.minimumDepthJump);
        batch.Barrier();
        RecordBuildVertexGrid(batch, *m_filteredDepth, *m_vertices, *m_propertyBuffer, *m_counters,
                              intrinsics, filter.minimumDepthMeters, filter.maximumDepthMeters);
        batch.Barrier();
        RecordEstimateNormal(batch, *m_vertices, *m_propertyBuffer, *m_normals, *m_counters,
                             m_width, m_height, filter);
        batch.Barrier();
        RecordCompactPoints(batch, *m_propertyBuffer, *m_vertices, *m_normals, *m_rowOffset,
                            *m_points, *m_compactNormals, m_width, m_height);
    }

    // Valid only after the batch Execute() was recorded into has been submitted.
    std::uint32_t PointCount() const {
        const uint32_t bytes = uint32_t((std::size_t(m_height) + 1u) * sizeof(std::uint32_t));
        m_rowOffset->MakeVisibleToCPU(bytes);
        std::uint32_t total = 0;
        std::memcpy(&total, static_cast<const std::uint32_t *>(m_rowOffset->MappedPtr()) + m_height,
                    sizeof total);
        return total;
    }

    ValidationMaskCounters DownloadCounters() const {
        m_counters->MakeVisibleToCPU(uint32_t(sizeof(ValidationMaskCounters)));
        ValidationMaskCounters counters;
        std::memcpy(&counters, m_counters->MappedPtr(), sizeof counters);
        return counters;
    }

    std::vector<Eigen::Vector3f> DownloadPoints() const { return downloadVectors(*m_points); }
    std::vector<Eigen::Vector3f> DownloadNormals() const {
        return downloadVectors(*m_compactNormals);
    }

    int Width() const { return m_width; }
    int Height() const { return m_height; }
    Engine::Core::Buffer &PropertyBuffer() { return *m_propertyBuffer; }

    void RecordClear(Engine::Compute::CommandBatch &batch) {
        RecordClear(batch, *m_propertyBuffer, m_width, m_height);
    }

    void RecordClear(Engine::Compute::CommandBatch &batch,
                     Engine::Core::Buffer &target,
                     int width,
                     int height) {
        ClearPushConstants pushConstants{width, height};
        kernel_clearProperty->Bind(0, target);
        kernel_clearProperty->Args(pushConstants);
        dispatchOverImage(batch, *kernel_clearProperty, width, height);
    }

    void RecordWindowAveraging(Engine::Compute::CommandBatch &batch,
                              Engine::Core::Buffer &source,
                              Engine::Core::Buffer &filtered,
                              int width,
                              int height,
                              int window,
                              float relativeDepthJump,
                              float minimumDepthJump) {
        PrefilterPushConstants pushConstants{width,
                                             height,
                                             window,
                                             relativeDepthJump,
                                             minimumDepthJump};
        kernel_windowAveraging->Bind(0, source).Bind(1, filtered);
        kernel_windowAveraging->Args(pushConstants);
        dispatchOverImage(batch, *kernel_windowAveraging, width, height);
    }

    // [H2] back-project the depth image into camera-frame vertices and mark which pixels the
    // range gate admits. `rejectionCounter` holds one uint the kernel atomically increments; the
    // caller zeroes it per frame. A range rejection is counted, a sensor dropout is not -- the two
    // want opposite corrections.
    void RecordBuildVertexGrid(Engine::Compute::CommandBatch &batch,
                               Engine::Core::Buffer &depth,
                               Engine::Core::Buffer &vertices,
                               Engine::Core::Buffer &mask,
                               Engine::Core::Buffer &rejectionCounter,
                               const Pipeline::CameraIntrinsics &intrinsics,
                               float minimumDepthMeters,
                               float maximumDepthMeters) {
        VertexGridPushConstants pushConstants{intrinsics.width,
                                              intrinsics.height,
                                              intrinsics.fx,
                                              intrinsics.fy,
                                              intrinsics.cx,
                                              intrinsics.cy,
                                              minimumDepthMeters,
                                              maximumDepthMeters};
        kernel_buildVertexGrid->Bind(0, depth).Bind(1, vertices).Bind(2, mask).Bind(3, rejectionCounter);
        kernel_buildVertexGrid->Args(pushConstants);
        dispatchOverImage(batch, *kernel_buildVertexGrid, intrinsics.width, intrinsics.height);
    }

    // [H3] forward jump guard + normal estimation. Writes `emitted`, never `valid`: a pixel the
    // guard refuses still has a measurement, and the neighbour counts in [H4]/[H5] read `valid`.
    void RecordEstimateNormal(Engine::Compute::CommandBatch &batch, Engine::Core::Buffer &vertices,
                              Engine::Core::Buffer &mask, Engine::Core::Buffer &normals,
                              Engine::Core::Buffer &counters, int width, int height,
                              const Pipeline::DepthFilterOptions &filter) {
        // cos() once here rather than per pixel: the gate is one comparison and the value never
        // changes across the image.
        const float minimumIncidenceCosine =
                filter.maximumIncidenceDegrees > 0.0f
                        ? std::cos(filter.maximumIncidenceDegrees * float(M_PI) / 180.0f)
                        : 0.0f;
        NormalPushConstants pushConstants{width,
                                          height,
                                          filter.relativeDepthJump,
                                          filter.minimumDepthJump,
                                          filter.symmetricDepthJumpGuard ? 1u : 0u,
                                          filter.minimumValidNeighbours,
                                          minimumIncidenceCosine};
        kernel_estimateNormal->Bind(0, vertices).Bind(1, mask).Bind(2, normals).Bind(3, counters);
        kernel_estimateNormal->Args(pushConstants);
        dispatchOverImage(batch, *kernel_estimateNormal, width, height);
    }

    // Gathers the `emitted` pixels into compact point + normal arrays, in the SAME row-major order
    // BackprojectDepth emits in.
    //
    // Three dispatches rather than one atomicAdd append, on purpose. This cloud is the ICP source,
    // and its centroid is a float sum, so the compacted ORDER reaches the solve -- an
    // atomicAdd order makes a replay diverge, which this repo has measured (four runs of one
    // command: 1.47, 6.45, 7.84 and 136.76 metres of trajectory).
    //
    // `rowOffset` must hold height + 1 uints; slot [height] comes back as the point count.
    void RecordCompactPoints(Engine::Compute::CommandBatch &batch, Engine::Core::Buffer &mask,
                             Engine::Core::Buffer &vertices, Engine::Core::Buffer &normals,
                             Engine::Core::Buffer &rowOffset, Engine::Core::Buffer &points,
                             Engine::Core::Buffer &compactNormals, int width, int height) {
        ImagePushConstants imageSize{width, height};
        kernel_countEmittedPerRow->Bind(0, mask).Bind(1, rowOffset);
        kernel_countEmittedPerRow->Args(imageSize);
        batch.DispatchElements(*kernel_countEmittedPerRow, uint32_t(height));
        batch.Barrier();

        RowScanPushConstants rowCount{height};
        kernel_scanRows->Bind(0, rowOffset);
        kernel_scanRows->Args(rowCount);
        batch.Dispatch(*kernel_scanRows, 1, 1, 1);
        batch.Barrier();

        kernel_scatterPoints->Bind(0, mask).Bind(1, vertices).Bind(2, normals).Bind(3, rowOffset).Bind(4, points).Bind(5, compactNormals);
        kernel_scatterPoints->Args(imageSize);
        batch.DispatchElements(*kernel_scatterPoints, uint32_t(height));
    }

private:
    // Must match the push_constant block in kernel_ClearValidMask.comp.glsl.
    struct ClearPushConstants {
        std::int32_t width;
        std::int32_t height;
    };

    // Must match the push_constant block in kernel_WindowAveraging.comp.glsl.
    struct PrefilterPushConstants {
        std::int32_t width;
        std::int32_t height;
        std::int32_t window;
        float relativeDepthJump;
        float minimumDepthJump;
    };

    // Must match the push_constant block in kernel_BuildVertexGrid.comp.glsl.
    struct VertexGridPushConstants {
        std::int32_t width;
        std::int32_t height;
        float fx;
        float fy;
        float cx;
        float cy;
        float minimumDepthMeters;
        float maximumDepthMeters;
    };

    // Must match the push_constant block in kernel_EstimateNormal.comp.glsl.
    struct NormalPushConstants {
        std::int32_t width;
        std::int32_t height;
        float relativeDepthJump;
        float minimumDepthJump;
        std::uint32_t symmetricDepthJumpGuard;
        std::int32_t minimumValidNeighbours;
        float minimumIncidenceCosine;
    };

    // Must match the push_constant blocks in kernel_CountEmittedPerRow / kernel_ScatterPoints.
    struct ImagePushConstants {
        std::int32_t width;
        std::int32_t height;
    };

    // Must match the push_constant block in kernel_ScanRows.comp.glsl.
    struct RowScanPushConstants {
        std::int32_t height;
    };

    // The GPU stores vec4 for std430's sake; the CPU-side Frame wants vec3, so the w is dropped
    // here rather than leaving every caller to know about the padding.
    std::vector<Eigen::Vector3f> downloadVectors(Engine::Core::Buffer &buffer) const {
        const std::uint32_t count = PointCount();
        std::vector<Eigen::Vector3f> out(count);
        if (count == 0) return out;
        const uint32_t bytes = count * 4u * uint32_t(sizeof(float));
        buffer.MakeVisibleToCPU(bytes);
        const float *words = static_cast<const float *>(buffer.MappedPtr());
        for (std::uint32_t i = 0; i < count; ++i)
            out[i] = Eigen::Vector3f(words[4 * i + 0], words[4 * i + 1], words[4 * i + 2]);
        return out;
    }

    static void dispatchOverImage(Engine::Compute::CommandBatch &batch,
                                  Engine::Core::ComputePipeline &pipeline,
                                  int width,
                                  int height) {
        if (width <= 0 || height <= 0) return;
        const VkExtent3D localSize = pipeline.GetLocalSize();
        if (localSize.width == 0 || localSize.height == 0) {
            throw std::runtime_error("ValidationMask::dispatchOverImage: kernel has a zero local size");
        }
        batch.Dispatch(pipeline,
                       (uint32_t(width) + localSize.width - 1) / localSize.width,
                       (uint32_t(height) + localSize.height - 1) / localSize.height,
                       1);
    }

    int m_width = 0;
    int m_height = 0;
    std::unique_ptr<Engine::Core::Buffer> m_propertyBuffer;
    std::unique_ptr<Engine::Core::Buffer> m_filteredDepth;
    std::unique_ptr<Engine::Core::Buffer> m_vertices;
    std::unique_ptr<Engine::Core::Buffer> m_normals;
    std::unique_ptr<Engine::Core::Buffer> m_counters;
    std::unique_ptr<Engine::Core::Buffer> m_rowOffset;
    std::unique_ptr<Engine::Core::Buffer> m_points;
    std::unique_ptr<Engine::Core::Buffer> m_compactNormals;
    std::unique_ptr<Engine::Core::ComputePipeline> kernel_clearProperty;
    std::unique_ptr<Engine::Core::ComputePipeline> kernel_windowAveraging;
    std::unique_ptr<Engine::Core::ComputePipeline> kernel_buildVertexGrid;
    // [H3]
    std::unique_ptr<Engine::Core::ComputePipeline> kernel_estimateNormal;
    // Compaction: count per row, scan the rows, scatter into a compact array.
    std::unique_ptr<Engine::Core::ComputePipeline> kernel_countEmittedPerRow;
    std::unique_ptr<Engine::Core::ComputePipeline> kernel_scanRows;
    std::unique_ptr<Engine::Core::ComputePipeline> kernel_scatterPoints;
};

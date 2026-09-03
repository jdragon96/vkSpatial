#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Compute/StagingBuffer.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Pipeline/Reconstruction/DepthCameraFrameSource.h" // CameraIntrinsics

#include <Eigen/Dense>
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
// Mirrors ValidationMaskCounters in kernel_ValidationMaskCommmon.glsl, and Pipeline::DepthFilterStats
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
        m_propertyBuffer = std::make_unique<Engine::Core::Buffer>(context);
        m_propertyBuffer->Allocate(uint32_t(std::size_t(width) * std::size_t(height) * sizeof(ValidationMaskProperty)));

        kernel_clearProperty = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_clearProperty->Build("Pipeline/Reconstruction/Algorithm/kernel_ClearValidMask.comp.glsl");
        kernel_PrefilterDepth = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_PrefilterDepth->Build("Pipeline/Reconstruction/Algorithm/kernel_PrefilterDepth.comp.glsl");
        kernel_buildVertexGrid = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_buildVertexGrid->Build("Pipeline/Reconstruction/Algorithm/kernel_BuildVertexGrid.comp.glsl");
        kernel_estimateNormal = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_estimateNormal->Build("Pipeline/Reconstruction/Algorithm/kernel_EstimateNormal.comp.glsl");
    }

    void Execute() {
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

    void RecordPrefilterDepth(Engine::Compute::CommandBatch &batch,
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
        kernel_PrefilterDepth->Bind(0, source).Bind(1, filtered);
        kernel_PrefilterDepth->Args(pushConstants);
        dispatchOverImage(batch, *kernel_PrefilterDepth, width, height);
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
                              Engine::Core::Buffer &mask, Engine::Core::Buffer &normals, int width,
                              int height, float relativeDepthJump, float minimumDepthJump) {
        NormalPushConstants pushConstants{width, height, relativeDepthJump, minimumDepthJump};
        kernel_estimateNormal->Bind(0, vertices).Bind(1, mask).Bind(2, normals);
        kernel_estimateNormal->Args(pushConstants);
        dispatchOverImage(batch, *kernel_estimateNormal, width, height);
    }

private:
    // Must match the push_constant block in kernel_ClearValidMask.comp.glsl.
    struct ClearPushConstants {
        std::int32_t width;
        std::int32_t height;
    };

    // Must match the push_constant block in kernel_PrefilterDepth.comp.glsl.
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
    };

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
    std::unique_ptr<Engine::Core::ComputePipeline> kernel_clearProperty;
    std::unique_ptr<Engine::Core::ComputePipeline> kernel_PrefilterDepth;
    std::unique_ptr<Engine::Core::ComputePipeline> kernel_buildVertexGrid;
    // [H3]
    std::unique_ptr<Engine::Core::ComputePipeline> kernel_estimateNormal;
};

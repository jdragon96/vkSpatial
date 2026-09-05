#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include "Realsense/Algorithm/DownSample.h"
#include "Realsense/Algorithm/NormalEstimation.h"
#include "Realsense/Algorithm/ValidationMask.h"
#include "Realsense/RealSenseTypes.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace Realsense {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // The D400 depth front end: raw Z16 in, a compacted cloud of points with confidences and
    // normals out.
    //
    //   upload + score -> threshold + back-project -> normals -> downsample -> count -> scan -> scatter
    //
    // Why this is its own class rather than more methods on ValidationMask: the stages are
    // separable algorithms with their own tests -- the score is a confidence model, the normal is
    // an estimator with three interchangeable strategies -- but the BUFFERS between them and the
    // ORDER they run in belong to neither. Folding the sequence into whichever stage happened to
    // be written first makes that stage own six buffers it never reads, and makes "what runs after
    // what" a fact you can only learn by reading a 60-line method on a class named after one pass.
    //
    // The split follows this repository's usual shape: a facade that holds lifetime, routing and
    // aggregation, and implementations that hold algorithms and do not know about each other.
    //
    // Two entry points rather than one, because they cost different things. Scoring is the
    // expensive half (four terms and a 3x3 gather per pixel); the threshold is one compare. A tool
    // that sweeps the bar re-runs only Extract, which is the whole reason the bar lives in its own
    // kernel instead of inside the score.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class RealSensePipeline {
    public:
        RealSensePipeline(Engine::Core::Context &context, int width, int height)
            : m_context(context), m_width(width), m_height(height), m_mask(context, width, height),
              m_normalEstimation(context), m_downSample(context, width, height) {
            if (width <= 0 || height <= 0)
                throw std::runtime_error("Realsense::RealSensePipeline::RealSensePipeline: the "
                                         "frame is " +
                                         std::to_string(width) + "x" + std::to_string(height));

            const std::size_t pixels = std::size_t(width) * std::size_t(height);

            // Dense per-pixel grids. Device-local: they exist only to be read by the next stage,
            // and the compaction below is what the readback is for.
            m_vertices = std::make_unique<Engine::Core::Buffer>(context);
            m_vertices->Allocate(std::uint32_t(pixels * 4u * sizeof(float)));
            m_normals = std::make_unique<Engine::Core::Buffer>(context);
            m_normals->Allocate(std::uint32_t(pixels * 4u * sizeof(float)));

            auto readback = [&context](std::uint32_t bytes) {
                auto buffer = std::make_unique<Engine::Core::Buffer>(context);
                buffer->AllocateHostVisibleReadback(bytes);
                return buffer;
            };

            // height + 1: the scan parks the grand total one past the last row, which is where
            // ValidPointCount reads it from.
            m_rowOffset = readback(std::uint32_t((std::size_t(height) + 1u) * sizeof(std::uint32_t)));
            // Sized for the worst case, every pixel surviving: the count is only known after the
            // scan has already run on the device.
            m_points = readback(std::uint32_t(pixels * 4u * sizeof(float)));
            m_compactNormals = readback(std::uint32_t(pixels * 4u * sizeof(float)));
            m_compactScores = readback(std::uint32_t(pixels * sizeof(float)));
            m_normalCounters = readback(std::uint32_t(sizeof(NormalEstimationCounters)));

            const auto build = [this](const char *path) {
                auto pipeline = std::make_unique<Engine::Core::ComputePipeline>(m_context);
                pipeline->Build(path);
                return pipeline;
            };
            kernel_RemainValidDepth = build("Realsense/Algorithm/ValidationMask.RemainValidDepth.glsl");
            kernel_CountEmittedPerRow = build("Realsense/Algorithm/ValidationMask.CountEmittedPerRow.glsl");
            kernel_ScanRows = build("Realsense/Algorithm/ValidationMask.ScanRows.glsl");
            kernel_ScatterValidPoints = build("Realsense/Algorithm/ValidationMask.ScatterValidPoints.glsl");
        }

        // The whole front end, one call. Records into `batch`; nothing is submitted here, so a
        // caller can put a frame's worth of other work in the same submission.
        void Execute(Engine::Compute::CommandBatch &batch,
                     const std::uint16_t *depthZ16,
                     const ValidationScoreOptions &options,
                     const PinholeIntrinsics &intrinsics,
                     float scoreThreshold,
                     const NormalEstimationOptions &normalOptions = {},
                     const DownSampleOptions &downSampleOptions = {},
                     const std::uint8_t *infraredY8 = nullptr) {
            RecordScore(batch, depthZ16, options, infraredY8);
            batch.Barrier();
            RecordExtract(batch, options, intrinsics, scoreThreshold, normalOptions,
                          downSampleOptions);
        }

        // Stage 1 alone: upload and score. Separated so a caller can re-run Extract at several
        // thresholds against ONE score, which has to give the same answer as scoring again.
        void RecordScore(Engine::Compute::CommandBatch &batch,
                         const std::uint16_t *depthZ16,
                         const ValidationScoreOptions &options,
                         const std::uint8_t *infraredY8 = nullptr) {
            m_mask.Execute(batch, depthZ16, options, infraredY8);
        }

        // Stages 2..5: threshold and back-project, estimate normals, then compact.
        //
        // The order is not arbitrary. Normals run BEFORE the count because that pass cancels
        // emitted pixels and the row counts have to see the final verdict; reversing them hands a
        // slot to a pixel that was then dropped.
        void RecordExtract(Engine::Compute::CommandBatch &batch,
                           const ValidationScoreOptions &options,
                           const PinholeIntrinsics &intrinsics,
                           float scoreThreshold,
                           const NormalEstimationOptions &normalOptions = {},
                           const DownSampleOptions &downSampleOptions = {}) {
            ValidationMask::ValidateOptions(options);
            DownSample::ValidateOptions(downSampleOptions);
            if (!(intrinsics.fx > 0.0f) || !(intrinsics.fy > 0.0f))
                throw std::runtime_error("Realsense::RealSensePipeline::RecordExtract: fx and fy "
                                         "must be positive; back-projection divides by them");

            const RemainPushConstants remain{m_width, m_height, intrinsics.fx,
                                             intrinsics.fy, intrinsics.cx, intrinsics.cy,
                                             options.depthScale, scoreThreshold};
            kernel_RemainValidDepth->Bind(0, m_mask.DepthImage(), m_mask.ImageSampler())
                    .Bind(1, m_mask.Properties())
                    .Bind(2, *m_vertices);
            kernel_RemainValidDepth->Args(remain);
            dispatchOverImage(batch, *kernel_RemainValidDepth);
            batch.Barrier();

            // Skippable so a caller that wants only the compacted coordinates does not pay the
            // estimator's border, which costs a ring of pixels a difference stencil cannot reach.
            if (normalOptions.enabled) {
                batch.FillBuffer(m_normalCounters->Handle(), 0, sizeof(NormalEstimationCounters), 0u);
                batch.Barrier();
                m_normalEstimation.RecordEstimate(batch, *m_vertices, m_mask.Properties(), *m_normals,
                                                  *m_normalCounters, m_width, m_height, options,
                                                  normalOptions);
                batch.Barrier();
            }

            // After the normal pass so only pixels that survived every gate are thinned, and
            // before the count so the compaction sees the final verdict -- the same two-sided
            // ordering constraint the normal pass has.
            if (downSampleOptions.enabled) {
                m_downSample.RecordDownSample(batch, *m_vertices, m_mask.Properties(), m_width,
                                              m_height, downSampleOptions);
                batch.Barrier();
            }

            const ImagePushConstants imageSize{m_width, m_height};
            kernel_CountEmittedPerRow->Bind(0, m_mask.Properties()).Bind(1, *m_rowOffset);
            kernel_CountEmittedPerRow->Args(imageSize);
            batch.DispatchElements(*kernel_CountEmittedPerRow, std::uint32_t(m_height));
            batch.Barrier();

            const RowScanPushConstants rowCount{m_height};
            kernel_ScanRows->Bind(0, *m_rowOffset);
            kernel_ScanRows->Args(rowCount);
            batch.Dispatch(*kernel_ScanRows, 1, 1, 1);
            batch.Barrier();

            kernel_ScatterValidPoints->Bind(0, m_mask.Properties())
                    .Bind(1, *m_vertices)
                    .Bind(2, *m_rowOffset)
                    .Bind(3, *m_points)
                    .Bind(4, *m_compactScores)
                    .Bind(5, *m_normals)
                    .Bind(6, *m_compactNormals);
            kernel_ScatterValidPoints->Args(imageSize);
            // One workgroup PER ROW, not DispatchElements over rows: gl_WorkGroupID.x IS the row in
            // that kernel, and its shared-memory scan needs the whole row's workgroup intact.
            batch.Dispatch(*kernel_ScatterValidPoints, std::uint32_t(m_height), 1, 1);
        }

        // The grand total the scan parked one past the last row.
        std::uint32_t ValidPointCount() const {
            const std::uint32_t bytes =
                    std::uint32_t((std::size_t(m_height) + 1u) * sizeof(std::uint32_t));
            m_rowOffset->MakeVisibleToCPU(bytes);
            std::uint32_t total = 0;
            std::memcpy(&total,
                        static_cast<const std::uint32_t *>(m_rowOffset->MappedPtr()) + m_height,
                        sizeof total);
            return total;
        }

        // The three compacted arrays share one index: point i, its confidence and its normal all
        // describe the same pixel. Splitting them across calls rather than into one struct keeps
        // the device layout (three separate std430 arrays) visible in the API.
        std::vector<Eigen::Vector3f> DownloadValidPoints() const {
            return downloadVectors(*m_points);
        }
        std::vector<Eigen::Vector3f> DownloadValidNormals() const {
            return downloadVectors(*m_compactNormals);
        }
        std::vector<float> DownloadValidScores() const {
            const std::uint32_t count = ValidPointCount();
            std::vector<float> out(count);
            if (count == 0) return out;
            const std::uint32_t bytes = count * std::uint32_t(sizeof(float));
            m_compactScores->MakeVisibleToCPU(bytes);
            std::memcpy(out.data(), m_compactScores->MappedPtr(), bytes);
            return out;
        }

        DownSampleCounters DownloadDownSampleCounters() const {
            return m_downSample.DownloadCounters();
        }

        NormalEstimationCounters DownloadNormalCounters() const {
            m_normalCounters->MakeVisibleToCPU(std::uint32_t(sizeof(NormalEstimationCounters)));
            NormalEstimationCounters out;
            std::memcpy(&out, m_normalCounters->MappedPtr(), sizeof out);
            return out;
        }

        // The dense score image and the score stage's own counters, forwarded rather than
        // re-implemented -- a caller holding a pipeline should not have to reach past it.
        std::vector<float> DownloadScores() const { return m_mask.DownloadScores(); }
        ValidationScoreCounters DownloadCounters() const { return m_mask.DownloadCounters(); }

        int Width() const { return m_width; }
        int Height() const { return m_height; }
        ValidationMask &Mask() { return m_mask; }

    private:
        // The GPU stores vec4 for std430's sake; the w is dropped here rather than leaving every
        // caller to know about the padding.
        std::vector<Eigen::Vector3f> downloadVectors(Engine::Core::Buffer &buffer) const {
            const std::uint32_t count = ValidPointCount();
            std::vector<Eigen::Vector3f> out(count);
            if (count == 0) return out;
            buffer.MakeVisibleToCPU(count * 4u * std::uint32_t(sizeof(float)));
            const float *words = static_cast<const float *>(buffer.MappedPtr());
            for (std::uint32_t i = 0; i < count; ++i)
                out[i] = Eigen::Vector3f(words[4 * i + 0], words[4 * i + 1], words[4 * i + 2]);
            return out;
        }

        void dispatchOverImage(Engine::Compute::CommandBatch &batch,
                               Engine::Core::ComputePipeline &kernel) const {
            const VkExtent3D localSize = kernel.GetLocalSize();
            if (localSize.width == 0 || localSize.height == 0)
                throw std::runtime_error("Realsense::RealSensePipeline: the kernel reported a zero "
                                         "local size");
            batch.Dispatch(kernel,
                           (std::uint32_t(m_width) + localSize.width - 1) / localSize.width,
                           (std::uint32_t(m_height) + localSize.height - 1) / localSize.height, 1);
        }

        Engine::Core::Context &m_context;
        int m_width = 0;
        int m_height = 0;

        ValidationMask m_mask;
        NormalEstimation m_normalEstimation;
        DownSample m_downSample;

        std::unique_ptr<Engine::Core::Buffer> m_vertices;
        std::unique_ptr<Engine::Core::Buffer> m_normals;
        std::unique_ptr<Engine::Core::Buffer> m_rowOffset;
        std::unique_ptr<Engine::Core::Buffer> m_points;
        std::unique_ptr<Engine::Core::Buffer> m_compactScores;
        std::unique_ptr<Engine::Core::Buffer> m_compactNormals;
        std::unique_ptr<Engine::Core::Buffer> m_normalCounters;

        std::unique_ptr<Engine::Core::ComputePipeline> kernel_RemainValidDepth;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_CountEmittedPerRow;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_ScanRows;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_ScatterValidPoints;
    };

} // namespace Realsense

#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include "Realsense/RealSenseTypes.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace Realsense {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // One point per detail voxel.
    //
    // Two points closer together than one voxel are redundant to everything downstream: a weighted
    // TSDF fusion averages them into the same cell and an ICP correspondence search pays for both.
    // Removing one here removes it from the scatter's output, from the readback, and from every
    // consumer after that.
    //
    // Like the normal pass, this stage may only CANCEL an emitted pixel, never elect one.
    //
    // Measured on capture/ (8 frames, 1.95M points, scene median 0.82 m, f = 383): a 5 mm detail
    // voxel leaves 28.4% of the points, 10 mm leaves 8.4%, and 1.25 mm leaves 99.1%. The lateral
    // sample spacing is z/f, so there is nothing to remove unless z < detailVoxel * f -- which is
    // why the option is off by default rather than merely tunable.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class DownSample {
    public:
        DownSample(Engine::Core::Context &context, int width, int height) : m_context(context) {
            const std::size_t pixels = std::size_t(width) * std::size_t(height);

            // Two slots per pixel. The occupancy this table has to hold is the number of distinct
            // voxels, which at a fine voxel approaches one per point -- 99.1% was measured -- so
            // sizing by pixels rather than by an expected reduction is what keeps the probe budget
            // from being the thing that decides the result.
            m_slotCount = std::uint32_t(pixels * 2u);

            m_slots = std::make_unique<Engine::Core::Buffer>(context);
            m_slots->Allocate(std::uint32_t(std::size_t(m_slotCount) * 2u * sizeof(std::uint32_t)));

            m_counters = std::make_unique<Engine::Core::Buffer>(context);
            m_counters->AllocateHostVisibleReadback(std::uint32_t(sizeof(DownSampleCounters)));

            m_claim = build("DOWNSAMPLE_PASS_CLAIM");
            m_cancel = build(nullptr);
        }

        // Records the clear, both passes and the barrier between them. The barrier is not optional:
        // every claim has to be settled before any pixel asks whether it won.
        void RecordDownSample(Engine::Compute::CommandBatch &batch,
                              Engine::Core::Buffer &vertices,
                              Engine::Core::Buffer &properties,
                              int width,
                              int height,
                              const DownSampleOptions &options) {
            ValidateOptions(options);

            // 0xFFFFFFFF is both EMPTY_KEY and "no winner yet", which is what lets one fill reset
            // the whole table rather than needing a clear kernel.
            batch.FillBuffer(m_slots->Handle(), 0, VkDeviceSize(m_slots->Size()), 0xFFFFFFFFu);
            batch.FillBuffer(m_counters->Handle(), 0, sizeof(DownSampleCounters), 0u);
            batch.Barrier();

            const DownSamplePushConstants pushConstants{width, height, options.detailVoxelMeters,
                                                        m_slotCount};

            dispatch(batch, *m_claim, vertices, properties, pushConstants, width, height);
            batch.Barrier();
            dispatch(batch, *m_cancel, vertices, properties, pushConstants, width, height);
        }

        DownSampleCounters DownloadCounters() const {
            m_counters->MakeVisibleToCPU(std::uint32_t(sizeof(DownSampleCounters)));
            DownSampleCounters out;
            std::memcpy(&out, m_counters->MappedPtr(), sizeof out);
            return out;
        }

        std::uint32_t SlotCount() const { return m_slotCount; }

        static void ValidateOptions(const DownSampleOptions &options) {
            // Silently doing nothing would be worse than throwing: the caller asked for a reduction
            // and would get the full cloud back with no way to tell why.
            if (options.enabled && !(options.detailVoxelMeters > 0.0f))
                throw std::runtime_error("Realsense::DownSample: enabled with detailVoxelMeters " +
                                         std::to_string(options.detailVoxelMeters) +
                                         "; it must be positive");
        }

    private:
        std::unique_ptr<Engine::Core::ComputePipeline> build(const char *macroName) {
            auto pipeline = std::make_unique<Engine::Core::ComputePipeline>(m_context);
            if (macroName) pipeline->Define(macroName);
            pipeline->Build("Realsense/Algorithm/DownSample.ToDetailVoxel.glsl");
            return pipeline;
        }

        void dispatch(Engine::Compute::CommandBatch &batch,
                      Engine::Core::ComputePipeline &kernel,
                      Engine::Core::Buffer &vertices,
                      Engine::Core::Buffer &properties,
                      const DownSamplePushConstants &pushConstants,
                      int width,
                      int height) const {
            kernel.Bind(0, vertices).Bind(1, properties).Bind(2, *m_slots).Bind(3, *m_counters);
            kernel.Args(pushConstants);

            const VkExtent3D localSize = kernel.GetLocalSize();
            if (localSize.width == 0 || localSize.height == 0)
                throw std::runtime_error("Realsense::DownSample: the kernel reported a zero local "
                                         "size");
            batch.Dispatch(kernel,
                           (std::uint32_t(width) + localSize.width - 1) / localSize.width,
                           (std::uint32_t(height) + localSize.height - 1) / localSize.height, 1);
        }

        Engine::Core::Context &m_context;
        std::uint32_t m_slotCount = 0;
        std::unique_ptr<Engine::Core::Buffer> m_slots;
        std::unique_ptr<Engine::Core::Buffer> m_counters;
        std::unique_ptr<Engine::Core::ComputePipeline> m_claim;
        std::unique_ptr<Engine::Core::ComputePipeline> m_cancel;
    };

} // namespace Realsense

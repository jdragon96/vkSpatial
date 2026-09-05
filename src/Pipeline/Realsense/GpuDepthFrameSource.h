#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"

#include "Pipeline/Acquisition/DepthCameraFrameSource.h" // IDepthProvider, CameraIntrinsics
#include "Pipeline/Acquisition/ReconstructionSource.h"   // IFrameSource

#include "Realsense/RealSensePipeline.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Pipeline {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Acquisition through the GPU depth front end.
    //
    // The sibling DepthCameraFrameSource does the same job entirely on the CPU: PrefilterDepth,
    // then a per-pixel back-projection and normal estimation in BackprojectDepth. This one hands
    // the frame to Realsense::RealSensePipeline instead, so scoring, thresholding, back-projection,
    // normals, downsampling and compaction all happen on the device and only the surviving points
    // are read back.
    //
    // OPT-IN, not a replacement. Both sources satisfy IFrameSource, so a caller picks one in its
    // AcquisitionConfig::makeSource and the two can be run against the same recording.
    //
    // Two things follow from that placement and are worth knowing before using it:
    //
    //  - Leave AcquisitionConfig::downsampleVoxel at 0. That reduce runs on the CPU AFTER the frame
    //    has been read back; DownSampleOptions here thins on the device BEFORE, so the transfer
    //    shrinks too. Running both would thin twice.
    //  - The confidence the front end computes is spent HERE, as the score threshold, and does not
    //    cross into Pipeline::Frame -- which carries points and normals and no per-point weight.
    //    Carrying it through to fusion is a separate piece of work.
    //
    // Two constructors, for the two situations a caller is in. A tool that already has a device --
    // a viewer with a renderer -- passes its Context by reference and gets one device for
    // everything. A headless tool that has none says so and the source builds its own, which is
    // what GpuIcpTracker and IntegrationThread already do for their own GPU work.
    //
    // Either way the Context stays out of AcquisitionConfig: the source is built inside a
    // caller-supplied makeSource lambda, the same way DepthCameraFrameSource receives its device.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class GpuDepthFrameSource : public IFrameSource {
    public:
        GpuDepthFrameSource(Engine::Core::Context &context,
                            std::unique_ptr<IDepthProvider> device,
                            Realsense::ValidationScoreOptions scoreOptions,
                            Realsense::NormalEstimationOptions normalOptions = {},
                            Realsense::DownSampleOptions downSampleOptions = {},
                            float scoreThreshold = 0.9f)
            : m_context(context), m_device(std::move(device)), m_scoreOptions(scoreOptions),
              m_normalOptions(normalOptions), m_downSampleOptions(downSampleOptions),
              m_scoreThreshold(scoreThreshold) {
            if (!m_device)
                throw std::runtime_error("Pipeline::GpuDepthFrameSource: the depth provider is null");
        }

        // Builds and owns a Context. For a headless caller with no device of its own; the Context
        // is created here rather than lazily because a reference to it has to be handed to the
        // pipeline below, and a member that moves would invalidate it.
        GpuDepthFrameSource(std::unique_ptr<IDepthProvider> device,
                            Realsense::ValidationScoreOptions scoreOptions,
                            Realsense::NormalEstimationOptions normalOptions = {},
                            Realsense::DownSampleOptions downSampleOptions = {},
                            float scoreThreshold = 0.9f)
            : m_ownedContext(std::make_unique<Engine::Core::Context>()),
              m_context(*m_ownedContext), m_device(std::move(device)), m_scoreOptions(scoreOptions),
              m_normalOptions(normalOptions), m_downSampleOptions(downSampleOptions),
              m_scoreThreshold(scoreThreshold) {
            if (!m_device)
                throw std::runtime_error("Pipeline::GpuDepthFrameSource: the depth provider is null");
        }

        EAcquisitionType Type() const override { return EAcquisitionType::DepthCamera; }
        const char *Name() const override { return "depth-camera-gpu"; }

        // No Open() override: IDepthProvider has no open step -- a provider is live from
        // construction and only Close() is part of its contract.

        bool Next(Frame &out) override {
            DepthFrame depthFrame;
            if (!m_device || !m_device->Grab(depthFrame)) return false;

            const CameraIntrinsics &intrinsics = m_device->Intrinsics();
            ensurePipeline(intrinsics);

            requantiseToZ16(depthFrame.depth);

            const Realsense::PinholeIntrinsics pinhole{intrinsics.fx, intrinsics.fy, intrinsics.cx,
                                                       intrinsics.cy};
            {
                Engine::Compute::CommandBatch batch(m_context);
                m_pipeline->Execute(batch, m_depthZ16.data(), m_scoreOptions, pinhole,
                                    m_scoreThreshold, m_normalOptions, m_downSampleOptions);
                batch.Submit();
            }

            out.pts = m_pipeline->DownloadValidPoints();
            out.nrm = m_pipeline->DownloadValidNormals();
            // The sensor is the origin of the frame it produced; the tracker places it in the world.
            out.cam = Eigen::Vector3f::Zero();
            return true;
        }

        void Close() override {
            if (m_device) m_device->Close();
        }

        // For a caller that wants the front end's own diagnostics rather than DepthFilterStats,
        // which describes the CPU path's gates and has no counterpart here. Null before the first
        // frame, because the pipeline is sized from intrinsics the provider may not know until it
        // has opened.
        const Realsense::RealSensePipeline *FrontEnd() const { return m_pipeline.get(); }

    private:
        void ensurePipeline(const CameraIntrinsics &intrinsics) {
            if (m_pipeline) return;
            if (intrinsics.width <= 0 || intrinsics.height <= 0)
                throw std::runtime_error("Pipeline::GpuDepthFrameSource: the provider reports a " +
                                         std::to_string(intrinsics.width) + "x" +
                                         std::to_string(intrinsics.height) + " frame");
            // Built on the first frame, not in the constructor: the width and height come from the
            // provider, and a device does not know them until it has opened.
            m_pipeline = std::make_unique<Realsense::RealSensePipeline>(m_context, intrinsics.width,
                                                                       intrinsics.height);
        }

        // IDepthProvider hands out float metres; the front end samples R16_UINT and scales on the
        // device. This is the one lossy step the GPU path adds, and it is bounded by the quantum:
        // for a RealSense the floats came FROM Z16 at this same scale, so the round trip is exact.
        void requantiseToZ16(const std::vector<float> &metres) {
            m_depthZ16.assign(metres.size(), 0);
            const float inverseScale = 1.0f / m_scoreOptions.depthScale;
            for (std::size_t i = 0; i < metres.size(); ++i) {
                if (!(metres[i] > 0.0f)) continue;
                const float units = std::round(metres[i] * inverseScale);
                // Past 65535 the sensor could not have reported it either. Clamping would invent a
                // surface at the far limit, so the pixel becomes "no measurement" instead.
                m_depthZ16[i] = units <= 65535.0f ? std::uint16_t(units) : std::uint16_t(0);
            }
        }

        // Declared before m_context so the reference is bound to a live object and outlives it.
        std::unique_ptr<Engine::Core::Context> m_ownedContext;
        Engine::Core::Context &m_context;
        std::unique_ptr<IDepthProvider> m_device;
        std::unique_ptr<Realsense::RealSensePipeline> m_pipeline;

        Realsense::ValidationScoreOptions m_scoreOptions;
        Realsense::NormalEstimationOptions m_normalOptions;
        Realsense::DownSampleOptions m_downSampleOptions;
        float m_scoreThreshold = 0.9f;

        std::vector<std::uint16_t> m_depthZ16;
    };

} // namespace Pipeline

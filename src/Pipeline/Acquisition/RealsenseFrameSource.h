#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"

#include "Pipeline/Acquisition/AcquisitionThread.h"      // IFrameSource
#include "Pipeline/Acquisition/DepthRecording.h"         // RecordedDepthProvider
#include "Realsense/RealSenseD435.h"
#include "Realsense/RealSensePipeline.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Pipeline {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Frames for the ICP thread, produced by the GPU depth front end.
    //
    // Two inputs, one output. A live D400 hands over Z16 UNTOUCHED -- Realsense::RealSenseD435 does
    // not unpack it -- so the frame goes to the device exactly as the sensor produced it: half the
    // upload of a float image and no host pass over it at all. A recording holds the float metres an
    // older front end unpacked it into, so it is re-quantised on the way back; that round trip is
    // exact, because the floats came FROM Z16 at this same scale.
    //
    // Everything after the upload is identical either way: score, threshold, back-project, normals,
    // downsample, compact, and only the survivors are read back.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class RealsenseFrameSource : public IFrameSource {
    public:
        // Live device.
        RealsenseFrameSource(Realsense::D435StreamOptions stream,
                             Realsense::ValidationScoreOptions score,
                             Realsense::NormalEstimationOptions normal,
                             Realsense::DownSampleOptions downSample,
                             float scoreThreshold)
            : m_score(score), m_normal(normal), m_downSample(downSample),
              m_scoreThreshold(scoreThreshold), m_stream(std::move(stream)),
              m_camera(std::make_unique<Realsense::RealSenseD435>()) {}

        // Anything that hands out float metres: a recording, or a scripted provider in a test.
        // Those get re-quantised on the way in, which is exact for a recording and is why the live
        // constructor above exists to skip it.
        RealsenseFrameSource(std::unique_ptr<IDepthProvider> device,
                             Realsense::ValidationScoreOptions score,
                             Realsense::NormalEstimationOptions normal = {},
                             Realsense::DownSampleOptions downSample = {},
                             float scoreThreshold = 0.9f)
            : m_score(score), m_normal(normal), m_downSample(downSample),
              m_scoreThreshold(scoreThreshold), m_device(std::move(device)) {
            if (!m_device)
                throw std::runtime_error("Pipeline::RealsenseFrameSource: the depth provider is null");
        }

        const char *Name() const override { return m_camera ? "realsense" : "realsense-depth"; }

        // Opened here rather than in the constructor: the source is built on the acquisition
        // thread, and a camera that throws should do it where the stage can report it.
        void Open() override {
            if (m_camera && !m_camera->IsOpen()) m_camera->Open(m_stream);
            configure();
        }

        bool Next(Frame &out) override {
            const std::uint16_t *depthZ16 = nullptr;

            if (m_camera) {
                Realsense::D435Frame frame;
                if (!m_camera->Grab(frame) || !frame.depthZ16) return false;
                depthZ16 = frame.depthZ16; // the driver's own buffer, valid until the next Grab
            } else {
                DepthFrame frame;
                if (!m_device || !m_device->Grab(frame)) return false;
                requantise(frame.depth);
                depthZ16 = m_requantised.data();
            }

            {
                Engine::Compute::CommandBatch batch(m_context);
                m_pipeline->Execute(batch, depthZ16, m_score, m_intrinsics, m_scoreThreshold,
                                    m_normal, m_downSample);
                batch.Submit();
            }

            out.pts = m_pipeline->DownloadValidPoints();
            out.nrm = m_pipeline->DownloadValidNormals();
            // The sensor is the origin of the frame it produced; the tracker places it in the world.
            out.cam = Eigen::Vector3f::Zero();
            return true;
        }

        void Close() override {
            if (m_camera) m_camera->Close();
            if (m_device) m_device->Close();
        }

    private:
        // Sized and filled once the input can answer for itself. focalLengthPixels defaults to 0
        // precisely so a frame cannot be scored without it; depthScale and the baseline carry D4xx
        // defaults that an explicit config still overrides.
        void configure() {
            if (m_pipeline) return;

            int width = 0, height = 0;
            if (m_camera) {
                const Realsense::D435Calibration &calibration = m_camera->Calibration();
                width = calibration.width;
                height = calibration.height;
                m_intrinsics = m_camera->MakeIntrinsics();
                const Realsense::ValidationScoreOptions device = m_camera->MakeScoreOptions();
                if (!(m_score.focalLengthPixels > 0.0f))
                    m_score.focalLengthPixels = device.focalLengthPixels;
                if (device.depthScale > 0.0f) m_score.depthScale = device.depthScale;
                if (device.baselineMeters > 0.0f) m_score.baselineMeters = device.baselineMeters;
            } else {
                const CameraIntrinsics &k = m_device->Intrinsics();
                width = k.width;
                height = k.height;
                m_intrinsics = Realsense::PinholeIntrinsics{k.fx, k.fy, k.cx, k.cy};
                // A recording carries intrinsics but no baseline -- it predates the module that
                // needs one -- so ValidationScoreOptions' D435 default stands. Every comparison
                // over a recording is relative, so a baseline off by a few percent shifts the whole
                // run rather than its shape.
                if (!(m_score.focalLengthPixels > 0.0f)) m_score.focalLengthPixels = k.fx;
            }

            if (width <= 0 || height <= 0)
                throw std::runtime_error("Pipeline::RealsenseFrameSource: the source reports a " +
                                         std::to_string(width) + "x" + std::to_string(height) +
                                         " frame");
            m_pipeline = std::make_unique<Realsense::RealSensePipeline>(m_context, width, height);
        }

        void requantise(const std::vector<float> &metres) {
            m_requantised.assign(metres.size(), 0);
            const float inverseScale = 1.0f / m_score.depthScale;
            for (std::size_t i = 0; i < metres.size(); ++i) {
                if (!(metres[i] > 0.0f)) continue;
                const float units = std::round(metres[i] * inverseScale);
                // Past 65535 the sensor could not have reported it either. Clamping would invent a
                // surface at the far limit, so the pixel becomes "no measurement" instead.
                m_requantised[i] = units <= 65535.0f ? std::uint16_t(units) : std::uint16_t(0);
            }
        }

        Realsense::ValidationScoreOptions m_score;
        Realsense::NormalEstimationOptions m_normal;
        Realsense::DownSampleOptions m_downSample;
        float m_scoreThreshold = 0.9f;
        Realsense::PinholeIntrinsics m_intrinsics;

        Realsense::D435StreamOptions m_stream;
        std::unique_ptr<Realsense::RealSenseD435> m_camera; // set for the live source
        std::unique_ptr<IDepthProvider> m_device;           // set for every other source
        std::vector<std::uint16_t> m_requantised;           // m_device path only

        Engine::Core::Context m_context;
        std::unique_ptr<Realsense::RealSensePipeline> m_pipeline;
    };

} // namespace Pipeline

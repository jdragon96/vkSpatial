#pragma once

#include "Pipeline/PipelineStage.h"
#include "Pipeline/Types.h"

#include "Realsense/RealSenseD435.h"
#include "Realsense/RealSenseTypes.h"

#include "utilities/RunningMean.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Pipeline {

    // The acquisition axis' vocabulary is owned by Realsense (which implements it) and pulled in
    // here rather than spelled out at every use. Pipeline already depends on Realsense, so this
    // needs no third home -- it had one, and the extra namespace bought nothing.
    using Realsense::CameraIntrinsics;
    using Realsense::DepthFrame;
    using Realsense::IDepthProvider;

    struct GpuFrontEndStats {
        std::atomic<std::uint64_t> emittedPoints{0};
        std::atomic<std::uint32_t> lastFramePoints{0};
        std::atomic<std::uint64_t> normalOutOfDomain{0};
        std::atomic<std::uint64_t> normalNoSupport{0};
        std::atomic<std::uint64_t> downSampleInsertFailures{0};
        std::atomic<std::uint64_t> downSampleOutOfRange{0};
    };

    // One axis, one kind of thing on it: a source of depth images. Both entries go through the
    // same GPU front end, so a fix to that front end is exercised by the recording too.
    enum class EAcquisitionSource {
        Realsense,     // a live D400
        RealsenseFile, // a depth recording replayed in its place
    };

    struct AcquisitionConfig {
        EAcquisitionSource source = EAcquisitionSource::Realsense;

        // --- RealsenseFile -----------------------------------------------------------------
        std::string recordingDirectory;

        Realsense::ValidationScoreOptions score;
        Realsense::NormalEstimationOptions normal;
        Realsense::DownSampleOptions downSample;
        float scoreThreshold = 0.9f;
        Realsense::D435StreamOptions stream;

        std::shared_ptr<GpuFrontEndStats> gpuStats;

        float downsampleVoxel = 0.0f;

        bool realTime = true;

        std::function<std::unique_ptr<IDepthProvider>()> makeProvider;
    };

    std::unique_ptr<IDepthProvider> MakeDepthProvider(const AcquisitionConfig &config);

    class DepthFrontEnd;

    class AcquisitionThread : public PipelineStage {
    public:
        AcquisitionThread(CommunicationModule &comm, AcquisitionConfig config);
        ~AcquisitionThread() override;

        EAcquisitionSource Source() const { return m_config.source; }

        std::string VisualPresetRefusal() const;

        void SetPaused(bool paused);
        bool IsPaused() const { return m_paused.load(); }

        double AcquireMsAvg() const { return m_acquireMs.Mean(); }
        float DownsampleVoxel() const { return m_config.downsampleVoxel; }
        std::uint64_t AcquiredFrames() const { return m_acquireMs.Count(); }

    protected:
        void Interrupt() override;
        void Run() override;

    private:
        bool waitWhilePaused();
        void reduceFrame(Frame &frame) const;

        void open();
        bool nextDepthFrame(Frame &out);

        AcquisitionConfig m_config;
        std::unique_ptr<IDepthProvider> m_provider;
        std::unique_ptr<DepthFrontEnd> m_frontEnd;
        bool m_closed = false;
        std::atomic<bool> m_paused{false};
        std::mutex m_pauseMutex;
        std::condition_variable m_pauseCv;
        util::RunningMean m_acquireMs;
    };

} // namespace Pipeline

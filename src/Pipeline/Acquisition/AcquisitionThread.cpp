#include "Pipeline/Acquisition/AcquisitionThread.h"

#include "Pipeline/CommunicationModule.h"

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Registration/Features/Downsample.h"
#include "Realsense/RealSenseD435.h"
#include "Realsense/RealSenseD435Recorder.h"
#include "Realsense/RealSensePipeline.h"
#include "Common/PointCloud.h"

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace Pipeline {
    class DepthFrontEnd {
    public:
        DepthFrontEnd(Realsense::ValidationScoreOptions score,
                      Realsense::NormalEstimationOptions normal,
                      Realsense::DownSampleOptions downSample,
                      float scoreThreshold,
                      std::shared_ptr<GpuFrontEndStats> stats)
            : m_score(score), m_normal(normal), m_downSample(downSample),
              m_scoreThreshold(scoreThreshold), m_stats(std::move(stats)) {}

        void Configure(const CameraIntrinsics &intrinsics) {
            if (intrinsics.width <= 0 || intrinsics.height <= 0)
                throw std::runtime_error("Pipeline::AcquisitionThread: the device reports a " +
                                         std::to_string(intrinsics.width) + "x" +
                                         std::to_string(intrinsics.height) + " frame");

            m_intrinsics = Realsense::PinholeIntrinsics{intrinsics.fx,
                                                        intrinsics.fy,
                                                        intrinsics.cx,
                                                        intrinsics.cy};
            if (!(m_score.focalLengthPixels > 0.0f)) m_score.focalLengthPixels = intrinsics.fx;
            if (intrinsics.depthScale > 0.0f) m_score.depthScale = intrinsics.depthScale;
            if (intrinsics.stereoBaselineMeters > 0.0f)
                m_score.baselineMeters = intrinsics.stereoBaselineMeters;

            m_pipeline = std::make_unique<Realsense::RealSensePipeline>(m_context,
                                                                        intrinsics.width,
                                                                        intrinsics.height);
        }

        // Swapped from the caller's thread while Run() executes on the acquisition thread. Guarded
        // rather than swapped during a pause: SetPaused only parks the worker at the TOP of its
        // loop, so a pause request says nothing about whether it is currently inside Run(). One
        // mutex next to a GPU dispatch costs nothing.
        void SetOptions(Realsense::ValidationScoreOptions score,
                        Realsense::NormalEstimationOptions normal,
                        Realsense::DownSampleOptions downSample,
                        float scoreThreshold) {
            std::lock_guard<std::mutex> lock(m_optionMutex);
            // focalLengthPixels / depthScale / baselineMeters were filled from the device in
            // Configure and the caller does not know them; keep what the device said.
            const Realsense::ValidationScoreOptions device = m_score;
            m_score = score;
            m_score.focalLengthPixels = device.focalLengthPixels;
            m_score.depthScale = device.depthScale;
            m_score.baselineMeters = device.baselineMeters;
            m_normal = normal;
            m_downSample = downSample;
            m_scoreThreshold = scoreThreshold;
        }

        void Run(const DepthFrame &depth, Frame &out) {
            const std::uint16_t *depthZ16 = depth.rawZ16;
            if (!depthZ16)
                throw std::runtime_error("Pipeline::AcquisitionThread: the provider returned a "
                                         "frame with no Z16 image");
            Realsense::ValidationScoreOptions score;
            Realsense::NormalEstimationOptions normal;
            Realsense::DownSampleOptions downSample;
            float scoreThreshold;
            {
                std::lock_guard<std::mutex> lock(m_optionMutex);
                score = m_score;
                normal = m_normal;
                downSample = m_downSample;
                scoreThreshold = m_scoreThreshold;
            }
            {
                Engine::Compute::CommandBatch batch(m_context);
                m_pipeline->Execute(batch, depthZ16, score, m_intrinsics, scoreThreshold, normal,
                                    downSample);
                batch.Submit();
            }

            out.pts = m_pipeline->DownloadValidPoints();
            out.nrm = m_pipeline->DownloadValidNormals();
            out.cam = Eigen::Vector3f::Zero();
            publishStats(std::uint32_t(out.pts.size()));
        }

    private:
        void publishStats(std::uint32_t emitted) {
            if (!m_stats) return;
            m_stats->lastFramePoints.store(emitted, std::memory_order_relaxed);
            m_stats->emittedPoints.fetch_add(emitted, std::memory_order_relaxed);

            std::lock_guard<std::mutex> lock(m_optionMutex);
            if (m_normal.enabled) {
                const Realsense::NormalEstimationCounters normal = m_pipeline->DownloadNormalCounters();
                m_stats->normalOutOfDomain.fetch_add(normal.outOfDomain, std::memory_order_relaxed);
                m_stats->normalNoSupport.fetch_add(normal.noSupport, std::memory_order_relaxed);
            }
            if (m_downSample.enabled) {
                const Realsense::DownSampleCounters down = m_pipeline->DownloadDownSampleCounters();
                m_stats->downSampleInsertFailures.fetch_add(down.insertFailures, std::memory_order_relaxed);
                m_stats->downSampleOutOfRange.fetch_add(down.outOfPackableRange, std::memory_order_relaxed);
            }
        }

        mutable std::mutex m_optionMutex; // guards the four option members below
        Realsense::ValidationScoreOptions m_score;
        Realsense::NormalEstimationOptions m_normal;
        Realsense::DownSampleOptions m_downSample;
        float m_scoreThreshold = 0.9f;
        std::shared_ptr<GpuFrontEndStats> m_stats;
        Realsense::PinholeIntrinsics m_intrinsics;

        Engine::Core::Context m_context;
        std::unique_ptr<Realsense::RealSensePipeline> m_pipeline;
    };

    std::unique_ptr<IDepthProvider> MakeDepthProvider(const AcquisitionConfig &config) {
        if (config.makeProvider) {
            std::unique_ptr<IDepthProvider> provider = config.makeProvider();
            if (!provider)
                throw std::invalid_argument(
                        "MakeDepthProvider: the injected factory returned no provider");
            return provider;
        }

        switch (config.source) {
            case EAcquisitionSource::Realsense: {
                auto camera = std::make_unique<Realsense::RealSenseD435>();
                camera->Open(config.stream);
                return camera;
            }

            case EAcquisitionSource::RealsenseFile:
                if (config.recordingDirectory.empty())
                    throw std::invalid_argument("MakeDepthProvider: RealsenseFile needs "
                                                "AcquisitionConfig::recordingDirectory");
                return std::make_unique<Realsense::RealSenseD435Recorder>(
                        config.recordingDirectory);
        }
        throw std::invalid_argument("MakeDepthProvider: unknown source");
    }

    AcquisitionThread::AcquisitionThread(CommunicationModule &comm, AcquisitionConfig config)
        : PipelineStage(comm), m_config(std::move(config)) {
        m_provider = MakeDepthProvider(m_config);
    }

    AcquisitionThread::~AcquisitionThread() { Stop(); }

    std::string AcquisitionThread::VisualPresetRefusal() const {
        const auto *camera = dynamic_cast<const Realsense::RealSenseD435 *>(m_provider.get());
        return camera ? camera->VisualPresetRefusal() : std::string();
    }

    void AcquisitionThread::SetOptions(const AcquisitionConfig &config) {
        m_config.score = config.score;
        m_config.normal = config.normal;
        m_config.downSample = config.downSample;
        m_config.scoreThreshold = config.scoreThreshold;
        m_config.downsampleVoxel = config.downsampleVoxel;
        // Null until Run() has reached open(); the values above are then picked up there instead.
        if (m_frontEnd)
            m_frontEnd->SetOptions(config.score, config.normal, config.downSample,
                                   config.scoreThreshold);
    }

    void AcquisitionThread::SetPaused(bool paused) {
        {
            std::lock_guard<std::mutex> lock(m_pauseMutex);
            m_paused.store(paused);
        }
        m_pauseCv.notify_all();
    }

    void AcquisitionThread::Interrupt() {
        m_paused.store(false);
        {
            std::lock_guard<std::mutex> lock(m_pauseMutex);
            m_closed = true;
        }
        m_pauseCv.notify_all();
        if (m_provider) m_provider->Close();
    }

    void AcquisitionThread::reduceFrame(Frame &frame) const {
        Common::PointCloud cloud;
        cloud.points = std::move(frame.pts);
        cloud.normals = std::move(frame.nrm);
        Common::PointCloud reduced =
                Features::DownsampleVoxel(cloud, m_config.downsampleVoxel);
        frame.pts = std::move(reduced.points);
        frame.nrm = std::move(reduced.normals);
    }

    bool AcquisitionThread::waitWhilePaused() {
        std::unique_lock<std::mutex> lock(m_pauseMutex);
        if (m_paused.load() && !StopRequested()) {
            m_parked = true;
            m_parkedCv.notify_all();
        }
        m_pauseCv.wait(lock, [this] { return !m_paused.load() || StopRequested(); });
        m_parked = false;
        return !StopRequested();
    }

    bool AcquisitionThread::WaitUntilPaused(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(m_pauseMutex);
        return m_parkedCv.wait_for(lock, timeout, [this] { return m_parked || StopRequested(); });
    }

    void AcquisitionThread::open() {
        {
            std::lock_guard<std::mutex> lock(m_pauseMutex);
            m_closed = false;
        }

        m_frontEnd = std::make_unique<DepthFrontEnd>(m_config.score,
                                                     m_config.normal,
                                                     m_config.downSample,
                                                     m_config.scoreThreshold,
                                                     m_config.gpuStats);
        m_frontEnd->Configure(m_provider->Intrinsics());
    }

    bool AcquisitionThread::nextDepthFrame(Frame &out) {
        DepthFrame depth;
        if (!m_provider->Grab(depth)) return false;

        m_frontEnd->Run(depth, out);
        return true;
    }

    void AcquisitionThread::Run() {
        open();
        Frame f;
        while (!StopRequested()) {
            if (!waitWhilePaused()) break;
            bool ok;
            {
                util::ScopedMean t(m_acquireMs);
                ok = nextDepthFrame(f);
                if (ok && m_config.downsampleVoxel > 0.0f) reduceFrame(f);
            }
            if (!ok) break;
            if (!m_comm.capturedFrames.Push(std::move(f))) break;
        }
        m_provider->Close();
        m_comm.capturedFrames.Close();
    }
}

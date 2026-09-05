#include "Pipeline/Acquisition/AcquisitionThread.h"

#include "Pipeline/Acquisition/D435DepthProvider.h"
#include "Pipeline/Acquisition/DepthRecording.h" // RecordedDepthProvider
#include "Pipeline/Acquisition/FrameLoader.h"    // EstimateCameraHint
#include "Pipeline/CommunicationModule.h"

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Features/Downsample.h"
#include "Realsense/RealSensePipeline.h"

#include "utilities/PointCloudIO.h" // util::LoadPly

#include <cmath>
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

        void Configure(const CameraIntrinsics &intrinsics, float depthScale,
                       const Realsense::D435Calibration *calibration) {
            if (intrinsics.width <= 0 || intrinsics.height <= 0)
                throw std::runtime_error("Pipeline::AcquisitionThread: the device reports a " +
                                         std::to_string(intrinsics.width) + "x" +
                                         std::to_string(intrinsics.height) + " frame");

            m_intrinsics = Realsense::PinholeIntrinsics{intrinsics.fx,
                                                        intrinsics.fy,
                                                        intrinsics.cx,
                                                        intrinsics.cy};
            if (!(m_score.focalLengthPixels > 0.0f)) m_score.focalLengthPixels = intrinsics.fx;
            if (depthScale > 0.0f) m_score.depthScale = depthScale;
            if (calibration && calibration->baselineMeters > 0.0f)
                m_score.baselineMeters = calibration->baselineMeters;

            m_pipeline = std::make_unique<Realsense::RealSensePipeline>(m_context,
                                                                        intrinsics.width,
                                                                        intrinsics.height);
            m_pixels = std::size_t(intrinsics.width) * std::size_t(intrinsics.height);
        }

        void Run(const DepthFrame &depth, Frame &out) {
            const std::uint16_t *depthZ16 = depth.rawZ16;
            if (!depthZ16) {
                requantise(depth.depth);
                depthZ16 = m_requantised.data();
            }

            {
                Engine::Compute::CommandBatch batch(m_context);
                m_pipeline->Execute(batch,
                                    depthZ16,
                                    m_score,
                                    m_intrinsics,
                                    m_scoreThreshold,
                                    m_normal,
                                    m_downSample);
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

        void requantise(const std::vector<float> &metres) {
            m_requantised.assign(m_pixels, 0);
            const float inverseScale = 1.0f / m_score.depthScale;
            const std::size_t count = std::min(m_pixels, metres.size());
            for (std::size_t i = 0; i < count; ++i) {
                if (!(metres[i] > 0.0f)) continue;
                const float units = std::round(metres[i] * inverseScale);
                m_requantised[i] = units <= 65535.0f ? std::uint16_t(units) : std::uint16_t(0);
            }
        }

        Realsense::ValidationScoreOptions m_score;
        Realsense::NormalEstimationOptions m_normal;
        Realsense::DownSampleOptions m_downSample;
        float m_scoreThreshold = 0.9f;
        std::shared_ptr<GpuFrontEndStats> m_stats; // optional
        Realsense::PinholeIntrinsics m_intrinsics;

        std::size_t m_pixels = 0;
        std::vector<std::uint16_t> m_requantised; // only when the provider has no device buffer

        Engine::Core::Context m_context;
        std::unique_ptr<Realsense::RealSensePipeline> m_pipeline;
    };

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // The device
    ///////////////////////////////////////////////////////////////////////////////////////////////

    std::unique_ptr<IDepthProvider> MakeDepthProvider(const AcquisitionConfig &config) {
        if (config.makeProvider) {
            std::unique_ptr<IDepthProvider> provider = config.makeProvider();
            if (!provider)
                throw std::invalid_argument(
                        "MakeDepthProvider: the injected factory returned no provider");
            return provider;
        }

        switch (config.source) {
            case EAcquisitionSource::PlyFolder:
                return nullptr; // no device: the thread reads the files itself

            case EAcquisitionSource::Realsense:
                return std::make_unique<D435DepthProvider>(config.stream);

            case EAcquisitionSource::RealsenseFile:
                if (config.recordingDirectory.empty())
                    throw std::invalid_argument("MakeDepthProvider: RealsenseFile needs "
                                                "AcquisitionConfig::recordingDirectory");
                return std::make_unique<RecordedDepthProvider>(config.recordingDirectory);
        }
        throw std::invalid_argument("MakeDepthProvider: unknown source");
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // The stage
    ///////////////////////////////////////////////////////////////////////////////////////////////
    AcquisitionThread::AcquisitionThread(CommunicationModule &comm, AcquisitionConfig config)
        : PipelineStage(comm), m_config(std::move(config)) {
        m_provider = MakeDepthProvider(m_config);
    }

    AcquisitionThread::~AcquisitionThread() { Stop(); }

    std::string AcquisitionThread::VisualPresetRefusal() const {
        const auto *camera = dynamic_cast<const D435DepthProvider *>(m_provider.get());
        return camera ? camera->VisualPresetRefusal() : std::string();
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
        if (m_provider) m_provider->Close(); // wake a Grab() blocked on a device
    }

    void AcquisitionThread::reduceFrame(Frame &frame) const {
        Registration::PointCloud cloud;
        cloud.points = std::move(frame.pts);
        cloud.normals = std::move(frame.nrm);
        Registration::PointCloud reduced =
                Features::DownsampleVoxel(cloud, m_config.downsampleVoxel);
        frame.pts = std::move(reduced.points);
        frame.nrm = std::move(reduced.normals);
    }

    bool AcquisitionThread::waitWhilePaused() {
        std::unique_lock<std::mutex> lock(m_pauseMutex);
        m_pauseCv.wait(lock, [this] { return !m_paused.load() || StopRequested(); });
        return !StopRequested();
    }

    void AcquisitionThread::open() {
        m_plyCursor = 0;
        m_hasLastPlyEmit = false;
        {
            std::lock_guard<std::mutex> lock(m_pauseMutex);
            m_closed = false;
        }
        if (!m_provider) return;

        m_frontEnd = std::make_unique<DepthFrontEnd>(m_config.score, m_config.normal,
                                                     m_config.downSample, m_config.scoreThreshold,
                                                     m_config.gpuStats);

        const auto *camera = dynamic_cast<const D435DepthProvider *>(m_provider.get());
        m_frontEnd->Configure(m_provider->Intrinsics(),
                              camera ? camera->Calibration().depthScale : 0.0f,
                              camera ? &camera->Calibration() : nullptr);
    }

    bool AcquisitionThread::next(Frame &out) {
        return m_provider ? nextDepthFrame(out) : nextPlyFrame(out);
    }

    bool AcquisitionThread::nextDepthFrame(Frame &out) {
        DepthFrame depth;
        if (!m_provider->Grab(depth)) return false;
        m_frontEnd->Run(depth, out);
        return true;
    }

    bool AcquisitionThread::nextPlyFrame(Frame &out) {
        for (;;) {
            if (m_plyCursor >= m_config.framePaths.size()) {
                if (!m_config.loop || m_config.framePaths.empty()) return false;
                m_plyCursor = 0;
            }
            if (!waitForNextPlySlot()) return false;

            Frame frame;
            const std::string &filePath = m_config.framePaths[m_plyCursor++];
            if (!util::LoadPly(filePath, frame.pts, frame.nrm) ||
                frame.nrm.size() != frame.pts.size())
                continue;
            frame.cam = EstimateCameraHint(frame.pts, frame.nrm);
            m_lastPlyEmit = std::chrono::steady_clock::now();
            m_hasLastPlyEmit = true;
            out = std::move(frame);
            return true;
        }
    }

    bool AcquisitionThread::waitForNextPlySlot() {
        std::unique_lock<std::mutex> lock(m_pauseMutex);
        if (m_closed) return false;
        if (m_config.intervalMs <= 0.0 || !m_hasLastPlyEmit) return true;
        const auto due = m_lastPlyEmit +
                         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                 std::chrono::duration<double, std::milli>(m_config.intervalMs));
        m_pauseCv.wait_until(lock, due, [&] { return m_closed; }); // early wake on Interrupt()
        return !m_closed;
    }

    void AcquisitionThread::Run() {
        open();
        Frame f;
        while (!StopRequested()) {
            if (!waitWhilePaused()) break;
            bool ok;
            {
                util::ScopedMean t(m_acquireMs);
                ok = next(f);
                if (ok && m_config.downsampleVoxel > 0.0f) reduceFrame(f);
            }
            if (!ok) break;
            if (!m_comm.capturedFrames.Push(std::move(f))) break;
        }
        if (m_provider) m_provider->Close();
        m_comm.capturedFrames.Close(); // exhausted -> let ICP drain and exit
    }

} // namespace Pipeline

#include "Pipeline/Acquisition/AcquisitionThread.h"

#include "Pipeline/Acquisition/DepthRecording.h" // RecordedDepthProvider
#include "Pipeline/Acquisition/FileFrameSource.h"
#include "Pipeline/CommunicationModule.h"
#include "Pipeline/Acquisition/RealsenseFrameSource.h"

#include "Features/Downsample.h"

#include <stdexcept>
#include <utility>

namespace Pipeline {

    std::unique_ptr<IFrameSource> MakeAcquisitionSource(const AcquisitionConfig &config) {
        if (config.makeSource) {
            std::unique_ptr<IFrameSource> source = config.makeSource();
            if (!source)
                throw std::invalid_argument(
                        "MakeAcquisitionSource: the injected factory returned no source");
            return source;
        }

        switch (config.source) {
            case EAcquisitionSource::PlyFolder: {
                FileSourceConfig files;
                files.files = config.framePaths;
                files.intervalMs = config.intervalMs;
                files.loop = config.loop;
                return std::make_unique<FileFrameSource>(std::move(files));
            }

            case EAcquisitionSource::Realsense:
                return std::make_unique<RealsenseFrameSource>(config.stream, config.score,
                                                              config.normal, config.downSample,
                                                              config.scoreThreshold);

            case EAcquisitionSource::RealsenseFile:
                if (config.recordingDirectory.empty())
                    throw std::invalid_argument("MakeAcquisitionSource: RealsenseFile needs "
                                                "AcquisitionConfig::recordingDirectory");
                return std::make_unique<RealsenseFrameSource>(
                        std::make_unique<RecordedDepthProvider>(config.recordingDirectory),
                        config.score, config.normal, config.downSample, config.scoreThreshold);
        }
        throw std::invalid_argument("MakeAcquisitionSource: unknown source");
    }

    AcquisitionThread::AcquisitionThread(CommunicationModule &comm, AcquisitionConfig config)
        : PipelineStage(comm), m_config(std::move(config)),
          m_source(MakeAcquisitionSource(m_config)) {}

    AcquisitionThread::~AcquisitionThread() { Stop(); }

    void AcquisitionThread::SetPaused(bool paused) {
        {
            std::lock_guard<std::mutex> lock(m_pauseMutex);
            m_paused.store(paused);
        }
        m_pauseCv.notify_all();
    }

    void AcquisitionThread::Interrupt() {
        m_paused.store(false);
        m_pauseCv.notify_all();
        if (m_source) m_source->Close(); // wake a Next() blocked on a device
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

    void AcquisitionThread::Run() {
        if (m_source) m_source->Open();
        Frame f;
        while (!StopRequested() && m_source) {
            if (!waitWhilePaused()) break;
            bool ok;
            {
                util::ScopedMean t(m_acquireMs);
                ok = m_source->Next(f);
                // Timed with the acquire, because from every later stage's point of view this IS
                // what acquisition produced.
                if (ok && m_config.downsampleVoxel > 0.0f) reduceFrame(f);
            }
            if (!ok) break;
            if (!m_comm.capturedFrames.Push(std::move(f))) break;
        }
        if (m_source) m_source->Close();
        m_comm.capturedFrames.Close(); // source exhausted -> let ICP drain and exit
    }

} // namespace Pipeline

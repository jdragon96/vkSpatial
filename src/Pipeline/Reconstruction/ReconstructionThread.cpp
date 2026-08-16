#include "Pipeline/Reconstruction/ReconstructionThread.h"

#include "Pipeline/CommunicationModule.h"
#include "Engine/Features/Downsample.h"
#include "Pipeline/Reconstruction/FileFrameSource.h"

#include <stdexcept>
#include <utility>

namespace Pipeline {

    std::unique_ptr<IFrameSource> MakeAcquisitionSource(const AcquisitionConfig &sourceType) {
        // An injected factory wins for every type: it is the only way to describe a device, and on
        // a File source it lets a caller substitute a decorated or synthetic source without
        // inventing a config field for it.
        if (sourceType.makeSource) {
            std::unique_ptr<IFrameSource> source = sourceType.makeSource();
            if (!source)
                throw std::invalid_argument(
                        "MakeAcquisitionSource: the injected factory returned no source");
            return source;
        }

        switch (sourceType.type) {
            case EAcquisitionType::File: {
                FileSourceConfig fc;
                fc.files = sourceType.framePaths;
                fc.intervalMs = sourceType.intervalMs;
                fc.loop = sourceType.loop;
                return std::make_unique<FileFrameSource>(std::move(fc));
            }
            case EAcquisitionType::DepthCamera:
            case EAcquisitionType::StructuredLight:
                throw std::invalid_argument("MakeAcquisitionSource: this source type has no "
                                            "by-value description; set AcquisitionConfig::makeSource");
        }
        throw std::invalid_argument("MakeAcquisitionSource: unknown source type");
    }

    ReconstructionThread::ReconstructionThread(CommunicationModule &comm,
                                               AcquisitionConfig sourceType)
        : PipelineStage(comm), m_source(MakeAcquisitionSource(sourceType)),
          m_downsampleVoxel(sourceType.downsampleVoxel) {}

    ReconstructionThread::~ReconstructionThread() { Stop(); }

    EAcquisitionType ReconstructionThread::Type() const {
        return m_source ? m_source->Type() : EAcquisitionType::File;
    }

    void ReconstructionThread::SetPaused(bool paused) {
        m_paused.store(paused);
        m_pauseCv.notify_all();
    }

    void ReconstructionThread::Interrupt() {
        if (m_source) m_source->Close();
        m_pauseCv.notify_all();
    }


    // Engine::Features::DownsampleVoxel takes the cell centroid and the renormalized mean normal,
    // so this is not merely a decimation: averaging inside a cell also cancels part of the stereo
    // sensor's per-pixel depth noise. Vectors move both ways, so no point is copied.
    void ReconstructionThread::reduceFrame(Frame &frame) const {
        Engine::Registration::PointCloud cloud;
        cloud.points = std::move(frame.pts);
        cloud.normals = std::move(frame.nrm);
        Engine::Registration::PointCloud reduced =
                Engine::Features::DownsampleVoxel(cloud, m_downsampleVoxel);
        frame.pts = std::move(reduced.points);
        frame.nrm = std::move(reduced.normals);
    }

    bool ReconstructionThread::waitWhilePaused() {
        std::unique_lock<std::mutex> lock(m_pauseMutex);
        m_pauseCv.wait(lock, [this] { return !m_paused.load() || StopRequested(); });
        return !StopRequested();
    }

    void ReconstructionThread::Run() {
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
                if (ok && m_downsampleVoxel > 0.0f) reduceFrame(f);
            }
            if (!ok) break;
            if (!m_comm.capturedFrames.Push(std::move(f))) break;
        }
        if (m_source) m_source->Close();
        m_comm.capturedFrames.Close(); // source exhausted -> let ICP drain and exit
    }

} // namespace Pipeline

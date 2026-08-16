#include "Pipeline/Reconstruction/ReconstructionThread.h"

#include "Pipeline/CommunicationModule.h"
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
        : PipelineStage(comm), m_source(MakeAcquisitionSource(sourceType)) {}

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
            }
            if (!ok) break;
            if (!m_comm.capturedFrames.Push(std::move(f))) break;
        }
        if (m_source) m_source->Close();
        m_comm.capturedFrames.Close(); // source exhausted -> let ICP drain and exit
    }

} // namespace Pipeline

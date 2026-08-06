#pragma once

#include "Engine/Pipeline/PipelineStage.h"
#include "Engine/Pipeline/Reconstruction/ReconstructionSource.h" // IFrameSource, AcquisitionConfig

#include "utilities/RunningMean.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

namespace Engine::Pipeline {

    std::unique_ptr<IFrameSource> MakeAcquisitionSource(const AcquisitionConfig &sourceType);

    class ReconstructionThread : public PipelineStage {
    public:
        ReconstructionThread(CommunicationModule &comm, AcquisitionConfig sourceType);
        ~ReconstructionThread() override; // join before m_source dies (Run uses it)

        EAcquisitionType Type() const;

        void SetPaused(bool paused);
        bool IsPaused() const { return m_paused.load(); }

        double AcquireMsAvg() const { return m_acquireMs.Mean(); }
        std::uint64_t AcquiredFrames() const { return m_acquireMs.Count(); }

    protected:
        void Interrupt() override;
        void Run() override;

    private:
        bool waitWhilePaused();

        std::unique_ptr<IFrameSource> m_source;
        std::atomic<bool> m_paused{false};
        std::mutex m_pauseMutex;
        std::condition_variable m_pauseCv;
        util::RunningMean m_acquireMs; // per-frame acquire time (thread-safe)
    };

} // namespace Engine::Pipeline

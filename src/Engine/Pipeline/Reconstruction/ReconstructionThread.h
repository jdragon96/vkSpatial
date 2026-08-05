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

        // Play / pause the acquisition. While paused, Run() blocks at the frame gate (no frames are
        // pulled or pushed) until resumed or stopped. Thread-safe; callable from the render thread.
        void SetPaused(bool paused);
        bool IsPaused() const { return m_paused.load(); }

        // Liveness/timing: average time to acquire one frame + how many frames pulled so far.
        double AcquireMsAvg() const { return m_acquireMs.Mean(); }
        std::uint64_t AcquiredFrames() const { return m_acquireMs.Count(); }

    protected:
        void Interrupt() override;
        void Run() override;

    private:
        bool waitWhilePaused(); // block while paused; false if stopped while waiting

        std::unique_ptr<IFrameSource> m_source;
        std::atomic<bool> m_paused{false};
        std::mutex m_pauseMutex;
        std::condition_variable m_pauseCv;
        util::RunningMean m_acquireMs; // per-frame acquire time (thread-safe)
    };

} // namespace Engine::Pipeline

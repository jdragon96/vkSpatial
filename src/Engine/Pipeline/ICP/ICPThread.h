#pragma once

#include "Engine/Pipeline/PipelineStage.h"

#include "utilities/RunningMean.h"

#include <cstdint>
#include <memory>

namespace Engine::Pipeline {

    class AlignmentCommand; // Alignment.h (injected strategy — forward decl)

    // Tracking stage — resolve each frame's pose via a swappable AlignmentCommand (frame-to-model),
    // then hand the posed frame downstream. The model it aligns against is the latest published
    // snapshot (Mailbox), so tracking and mapping run concurrently.
    class ICPThread : public PipelineStage {
    public:
        ICPThread(CommunicationModule &comm, std::unique_ptr<AlignmentCommand> align);
        ~ICPThread() override; // join before m_align dies (Run uses it)

        // Liveness/timing: average time to align one frame + how many frames aligned so far.
        double AlignMsAvg() const { return m_alignMs.Mean(); }
        std::uint64_t AlignedFrames() const { return m_alignMs.Count(); }

    protected:
        void Interrupt() override;
        void Run() override;

    private:
        std::unique_ptr<AlignmentCommand> m_align;
        util::RunningMean m_alignMs; // per-frame alignment time (thread-safe)
    };

} // namespace Engine::Pipeline

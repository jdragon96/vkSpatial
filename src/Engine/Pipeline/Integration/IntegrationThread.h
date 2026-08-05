#pragma once

#include "Engine/Pipeline/PipelineStage.h"
#include "Engine/Pipeline/Types.h" // MapConfig, Frame

#include "utilities/RunningMean.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Pipeline {

    // Mapping stage — owns its OWN Vulkan device + SubmapAdvancedTSDF (no device is shared across
    // threads). Integrates posed frames off the tracking thread and publishes an immutable
    // ModelSnapshot per frame. The density set is precomputed once from `densityFrames` (optional).
    class IntegrationThread : public PipelineStage {
    public:
        IntegrationThread(CommunicationModule &comm, MapConfig cfg,
                          std::shared_ptr<const std::vector<Frame>> densityFrames);
        ~IntegrationThread() override;

        int ProcessedFrame() const { return m_processed.load(); }

        // Liveness/timing: average time to integrate one frame (integrate + download + tracker) +
        // how many frames integrated so far.
        double IntegrateMsAvg() const { return m_integrateMs.Mean(); }
        std::uint64_t IntegratedFrames() const { return m_integrateMs.Count(); }

    protected:
        void Interrupt() override;
        void Run() override;

    private:
        MapConfig m_cfg;
        std::shared_ptr<const std::vector<Frame>> m_densityFrames;
        std::atomic<int> m_processed{-1};
        util::RunningMean m_integrateMs; // per-frame integrate+download+tracker time (thread-safe)
    };

} // namespace Engine::Pipeline

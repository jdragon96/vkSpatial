#pragma once

#include "Engine/Pipeline/PipelineStage.h"
#include "Engine/Pipeline/Types.h" // MapConfig, Frame

#include "utilities/RunningMean.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Pipeline {

    class IntegrationThread : public PipelineStage {
    public:
        IntegrationThread(CommunicationModule &comm, MapConfig cfg);
        ~IntegrationThread() override;

        int ProcessedFrame() const { return m_processed.load(); }

        double IntegrateMsAvg() const { return m_integrateMs.Mean(); }

        std::uint64_t IntegratedFrames() const { return m_integrateMs.Count(); }

    protected:
        void Interrupt() override;
        void Run() override;

    private:
        MapConfig m_cfg;
        std::atomic<int> m_processed{-1};
        util::RunningMean m_integrateMs; // per-frame integrate+download+tracker time (thread-safe)
    };

} // namespace Engine::Pipeline

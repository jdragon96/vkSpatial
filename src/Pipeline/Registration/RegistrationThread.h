#pragma once

#include "Pipeline/PipelineStage.h"

#include "utilities/RunningMean.h"

#include <cstdint>
#include <memory>

namespace Pipeline {

    class Tracker;

    class RegistrationThread : public PipelineStage {
    public:
        RegistrationThread(CommunicationModule &comm, std::unique_ptr<Tracker> align);
        ~RegistrationThread() override;

        double AlignMsAvg() const { return m_trackerMs.Mean(); }
        std::uint64_t AlignedFrames() const { return m_trackerMs.Count(); }
        double TrackerRmseAvg() const { return m_trackerRmse.Mean(); }

    protected:
        void Interrupt() override;
        void Run() override;

    private:
        std::unique_ptr<Tracker> m_tracker;
        util::RunningMean m_trackerMs;
        util::RunningMean m_trackerRmse;
    };

} // namespace Pipeline

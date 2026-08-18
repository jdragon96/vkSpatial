#pragma once

#include "Pipeline/PipelineStage.h"

#include "utilities/RunningMean.h"

#include <atomic>

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
        std::uint64_t Rejected() const { return m_rejected.load(); }
        std::uint64_t RejectedNoModel() const { return m_rejectedNoModel.load(); }
        std::uint64_t RejectedNoLocalTarget() const { return m_rejectedNoLocalTarget.load(); }
        std::uint64_t RejectedTooFewInliers() const { return m_rejectedTooFewInliers.load(); }
        std::uint64_t RejectedLowOverlap() const { return m_rejectedLowOverlap.load(); }
        std::uint64_t RejectedImplausibleMotion() const { return m_rejectedImplausibleMotion.load(); }
        // Frames handed downstream marked "do not fuse" -- a rejected track whose pose would have
        // corrupted the map. Observable because a silently discarded frame looks like a frame that
        // was never captured. See ShouldFuse() in Types.h for which causes are skipped and why.
        std::uint64_t SkippedFusions() const { return m_skippedFusions.load(); }
        double PoseDeltaMetersAvg() const { return m_poseDeltaMeters.Mean(); }
        double PoseDeltaMetersMax() const { return m_poseDeltaMetersMax.load(); }
        double PoseDeltaDegreesMax() const { return m_poseDeltaDegreesMax.load(); }
        double TrajectoryLengthMeters() const { return m_trajectoryLengthMeters.load(); }

    protected:
        void Interrupt() override;
        void Run() override;

    private:
        std::unique_ptr<Tracker> m_tracker;
        util::RunningMean m_trackerMs;
        util::RunningMean m_trackerRmse;
        std::atomic<std::uint64_t> m_rejected{0};
        std::atomic<std::uint64_t> m_rejectedNoModel{0}, m_rejectedNoLocalTarget{0};
        std::atomic<std::uint64_t> m_rejectedTooFewInliers{0}, m_rejectedLowOverlap{0};
        std::atomic<std::uint64_t> m_rejectedImplausibleMotion{0};
        std::atomic<std::uint64_t> m_skippedFusions{0};
        util::RunningMean m_poseDeltaMeters;
        std::atomic<double> m_poseDeltaMetersMax{0.0};
        std::atomic<double> m_poseDeltaDegreesMax{0.0};
        std::atomic<double> m_trajectoryLengthMeters{0.0};
    };

} // namespace Pipeline

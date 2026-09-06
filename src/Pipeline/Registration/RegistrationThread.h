#pragma once

#include "Pipeline/Integration/FusionGate.h" // Pipeline::FusionGate
#include "Pipeline/PipelineStage.h"
#include "Pipeline/Types.h" // TrackerStats, FusionGateConfig

#include "utilities/RunningMean.h"

#include <atomic>

#include <cstdint>
#include <memory>

namespace Pipeline {

    class Tracker;

    class RegistrationThread : public PipelineStage {
    public:
        RegistrationThread(CommunicationModule &comm, std::unique_ptr<Tracker> align, FusionGateConfig fusion = {});
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
        // The FusionGate's refusals, split by cause (see FusionGate.h). All zero unless the caller
        // configured the gate, which is off by default.
        std::uint64_t BootstrapHeldFrames() const { return m_bootstrapHeldFrames.load(); }
        std::uint64_t FusionRejectedByFitness() const { return m_fusionRejectedByFitness.load(); }
        std::uint64_t FusionRejectedByRmse() const { return m_fusionRejectedByRmse.load(); }
        bool FusionArmed() const { return m_fusionArmed.load(); }
        double PoseDeltaMetersAvg() const { return m_poseDeltaMeters.Mean(); }
        double PoseDeltaMetersMax() const { return m_poseDeltaMetersMax.load(); }
        double PoseDeltaDegreesMax() const { return m_poseDeltaDegreesMax.load(); }
        double TrajectoryLengthMeters() const { return m_trajectoryLengthMeters.load(); }
        // The tracker's own counters (relocalization attempts/successes); zeros for trackers
        // without the mechanism. In the .cpp because Tracker is only forward-declared here.
        TrackerStats TrackerCounters() const;

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
        // Touched only by Run(); mirrored into atomics so the caller can read them mid-stream.
        FusionGate m_fusionGate;
        std::atomic<std::uint64_t> m_bootstrapHeldFrames{0};
        std::atomic<std::uint64_t> m_fusionRejectedByFitness{0}, m_fusionRejectedByRmse{0};
        std::atomic<bool> m_fusionArmed{true};
        util::RunningMean m_poseDeltaMeters;
        std::atomic<double> m_poseDeltaMetersMax{0.0};
        std::atomic<double> m_poseDeltaDegreesMax{0.0};
        std::atomic<double> m_trajectoryLengthMeters{0.0};
    };

} // namespace Pipeline

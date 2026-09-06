#include "Pipeline/Registration/RegistrationThread.h"
#include "Pipeline/CommunicationModule.h"
#include "Pipeline/Registration/Tracker.h"

#include <Eigen/Geometry>

#include <memory>
#include <utility>

namespace Pipeline {

    RegistrationThread::RegistrationThread(CommunicationModule &comm, std::unique_ptr<Tracker> tracker, FusionGateConfig fusion)
        : PipelineStage(comm), m_tracker(std::move(tracker)), m_fusionGate(fusion) {
        m_fusionArmed.store(m_fusionGate.IsArmed());
    }

    RegistrationThread::~RegistrationThread() { Stop(); }

    TrackerStats RegistrationThread::TrackerCounters() const {
        return m_tracker ? m_tracker->Stats() : TrackerStats{};
    }

    void RegistrationThread::Interrupt() {
        m_comm.capturedFrames.Close();
        m_comm.handshake.Close();
    }

    void RegistrationThread::Run() {
        Eigen::Isometry3f previousPose = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f previousPreviousPose = Eigen::Isometry3f::Identity();
        int consecutiveAdoptions = 0;
        Frame f;

        constexpr float kVelocityPriorMinimumStepMeters = 0.02f;

        while (!StopRequested() && m_comm.capturedFrames.Pop(f)) {
            const std::shared_ptr<const ModelSnapshot> model = m_comm.model.Latest();
            const Eigen::Isometry3f lastDelta = previousPreviousPose.inverse() * previousPose;
            const bool velocityWorthUsing =
                    consecutiveAdoptions >= 2 &&
                    lastDelta.translation().norm() > kVelocityPriorMinimumStepMeters;
            const Eigen::Isometry3f prior =
                    velocityWorthUsing ? previousPose * lastDelta : previousPose;
            TrackingResult a;
            {
                util::ScopedMean t(m_trackerMs);
                a = m_tracker->Track(f, model.get(), prior);
            }
            const Eigen::Isometry3f pose = a.valid ? a.pose : previousPose;
            if (a.valid) {
                const Eigen::Isometry3f step = previousPose.inverse() * a.pose;
                const double stepMeters = double(step.translation().norm());
                const double stepDegrees =
                        double(Eigen::AngleAxisf(step.rotation()).angle()) * 180.0 / M_PI;
                m_poseDeltaMeters.Add(stepMeters);
                if (stepMeters > m_poseDeltaMetersMax.load()) m_poseDeltaMetersMax.store(stepMeters);
                if (stepDegrees > m_poseDeltaDegreesMax.load()) m_poseDeltaDegreesMax.store(stepDegrees);
                m_trajectoryLengthMeters.store(m_trajectoryLengthMeters.load() + stepMeters);

                previousPreviousPose = previousPose;
                previousPose = a.pose;
                if (consecutiveAdoptions < 2) ++consecutiveAdoptions;
                m_trackerRmse.Add(a.rmse);
            } else {
                consecutiveAdoptions = 0;
                m_rejected.fetch_add(1);
                switch (a.failure) {
                    case ETrackFailure::NoModel:
                        m_rejectedNoModel.fetch_add(1);
                        break;
                    case ETrackFailure::NoLocalTarget:
                        m_rejectedNoLocalTarget.fetch_add(1);
                        break;
                    case ETrackFailure::TooFewInliers:
                        m_rejectedTooFewInliers.fetch_add(1);
                        break;
                    case ETrackFailure::LowOverlap:
                        m_rejectedLowOverlap.fetch_add(1);
                        break;
                    case ETrackFailure::ImplausibleMotion:
                        m_rejectedImplausibleMotion.fetch_add(1);
                        break;
                    case ETrackFailure::None:
                        break;
                }
            }

            TrackedFrame tf;
            // sensor camera -> world
            tf.cameraWorld = pose * f.cam;
            tf.pose = pose;
            tf.failure = a.failure;
            tf.fuse = ShouldFuse(a.valid, a.failure);
            if (tf.fuse) {
                tf.fuse = m_fusionGate.Admit(a.valid, a.failure, a.fitness, a.rmse);
            } else {
                m_skippedFusions.fetch_add(1);
            }

            // Update statistic and TrackedFrame
            m_bootstrapHeldFrames.store(m_fusionGate.BootstrapHeldFrames());
            m_fusionRejectedByFitness.store(m_fusionGate.FusionRejectedByFitness());
            m_fusionRejectedByRmse.store(m_fusionGate.FusionRejectedByRmse());
            m_fusionArmed.store(m_fusionGate.IsArmed());
            tf.frame = std::move(f);
            m_comm.handshake.NotePushed();
            m_comm.trackedFrames.Push(std::move(tf));
            if (!m_comm.handshake.WaitForDrain()) break;
        }
        m_comm.trackedFrames.Close(); // upstream done -> let Integration drain and exit
    }

} // namespace Pipeline

#include "Pipeline/Registration/RegistrationThread.h"
#include "Pipeline/CommunicationModule.h"
#include "Pipeline/Registration/Tracker.h"

#include <Eigen/Geometry>

#include <memory>
#include <utility>

namespace Pipeline {

    RegistrationThread::RegistrationThread(CommunicationModule &comm, std::unique_ptr<Tracker> tracker)
        : PipelineStage(comm), m_tracker(std::move(tracker)) {}

    RegistrationThread::~RegistrationThread() { Stop(); }

    void RegistrationThread::Interrupt() {
        m_comm.capturedFrames.Close(); // wake a blocked Pop
        m_comm.handshake.Close();      // and a blocked WaitForDrain, or Stop() never returns
    }

    void RegistrationThread::Run() {
        Eigen::Isometry3f previousPose = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f previousPreviousPose = Eigen::Isometry3f::Identity();
        // The constant-velocity prior may extrapolate only from two CONSECUTIVELY adopted poses.
        // A plain "have two poses" flag re-arms it one frame after any rejection, when the last two
        // adopted poses are two frame intervals apart -- re-applying that delta as if it were one
        // interval overshoots by a full frame of motion. Measured on the 477-frame capture/
        // recording, that overshoot pushed every second solve out of its convergence basin: a
        // self-sustaining period-2 oscillation (adopt fitness ~0.9 / reject ~0.12, perfectly
        // alternating) that rejected 229 of 477 frames. Counting consecutive adoptions (saturated
        // at 2) makes the delta always span exactly one interval.
        int consecutiveAdoptions = 0;
        Frame f;

        // The velocity prior's prediction error is the pose-estimate noise in the last delta,
        // re-applied -- so it only helps when the motion it predicts is large against that noise.
        // Below this threshold the previous pose is already inside the solve's convergence basin
        // and extrapolation adds noise and nothing else: measured on capture/ (~5 mm/frame), the
        // correctly re-armed one-interval extrapolation STILL pushed every third solve out of its
        // basin (153/477 frames gated as implausible, period-3), while the plain previous-pose
        // prior tracked all 476. The straight-line fixture (0.1 m/frame), where the model is worth
        // a full frame of motion, sits far above the threshold and keeps it.
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
                // Measured against the pose actually adopted last frame, so a rejected track does
                // not show up as a jump on the frame after it.
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
                    case ETrackFailure::NoModel: m_rejectedNoModel.fetch_add(1); break;
                    case ETrackFailure::NoLocalTarget: m_rejectedNoLocalTarget.fetch_add(1); break;
                    case ETrackFailure::TooFewInliers: m_rejectedTooFewInliers.fetch_add(1); break;
                    case ETrackFailure::LowOverlap: m_rejectedLowOverlap.fetch_add(1); break;
                    case ETrackFailure::ImplausibleMotion:
                        m_rejectedImplausibleMotion.fetch_add(1);
                        break;
                    case ETrackFailure::None: break;
                }
            }

            TrackedFrame tf;
            tf.cameraWorld = pose * f.cam; // sensor camera -> world
            tf.pose = pose;
            tf.failure = a.failure;
            tf.fuse = ShouldFuse(a.valid, a.failure);
            if (!tf.fuse) m_skippedFusions.fetch_add(1);
            tf.frame = std::move(f);
            m_comm.handshake.NotePushed();
            m_comm.trackedFrames.Push(std::move(tf));
            // Offline only (no-op on a live sensor): wait until integration has finished this frame,
            // so the next Track() sees a map holding every earlier frame rather than whichever
            // version the scheduler happened to publish. Without it the same recording produces a
            // different trajectory every run -- see FrameHandshake in CommunicationModule.h.
            if (!m_comm.handshake.WaitForDrain()) break;
        }
        m_comm.trackedFrames.Close(); // upstream done -> let Integration drain and exit
    }

} // namespace Pipeline

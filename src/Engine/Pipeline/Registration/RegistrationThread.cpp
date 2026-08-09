#include "Engine/Pipeline/Registration/RegistrationThread.h"
#include "Engine/Pipeline/CommunicationModule.h"
#include "Engine/Pipeline/Registration/Tracker.h"

#include <Eigen/Geometry>

#include <memory>
#include <utility>

namespace Engine::Pipeline {

    RegistrationThread::RegistrationThread(CommunicationModule &comm, std::unique_ptr<Tracker> tracker)
        : PipelineStage(comm), m_tracker(std::move(tracker)) {}

    RegistrationThread::~RegistrationThread() { Stop(); }

    void RegistrationThread::Interrupt() { m_comm.capturedFrames.Close(); } // wake a blocked Pop

    void RegistrationThread::Run() {
        // Constant-velocity motion model: predict the NEXT prior as the previous pose advanced by the
        // same delta that got from previousPreviousPose to previousPose, instead of just re-handing the
        // tracker the previous pose unchanged. Requires two valid poses in a row (haveTwoPoses); after
        // any invalid track, reset so a stale/wrong velocity estimate isn't carried into future frames.
        Eigen::Isometry3f previousPose = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f previousPreviousPose = Eigen::Isometry3f::Identity();
        bool haveTwoPoses = false;
        Frame f;
        while (!StopRequested() && m_comm.capturedFrames.Pop(f)) {
            const std::shared_ptr<const ModelSnapshot> model = m_comm.model.Latest();
            const Eigen::Isometry3f prior =
                    haveTwoPoses ? previousPose * (previousPreviousPose.inverse() * previousPose)
                                 : previousPose;
            TrackingResult a;
            {
                util::ScopedMean t(m_trackerMs);
                a = m_tracker->Track(f, model.get(), prior);
            }
            const Eigen::Isometry3f pose = a.valid ? a.pose : previousPose;
            if (a.valid) {
                previousPreviousPose = previousPose;
                previousPose = a.pose;
                haveTwoPoses = true;
                m_trackerRmse.Add(a.rmse);
            } else {
                haveTwoPoses = false; // stale velocity after a dropped track
            }

            TrackedFrame tf;
            tf.cameraWorld = pose * f.cam; // sensor camera -> world
            tf.pose = pose;
            tf.frame = std::move(f);
            m_comm.trackedFrames.Push(std::move(tf));
        }
        m_comm.trackedFrames.Close(); // upstream done -> let Integration drain and exit
    }

} // namespace Engine::Pipeline

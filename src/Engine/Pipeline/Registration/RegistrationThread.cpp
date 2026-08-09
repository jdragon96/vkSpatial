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
        Eigen::Isometry3f prev = Eigen::Isometry3f::Identity();
        Frame f;
        while (!StopRequested() && m_comm.capturedFrames.Pop(f)) {
            const std::shared_ptr<const ModelSnapshot> model = m_comm.model.Latest();
            TrackingResult a;
            {
                util::ScopedMean t(m_trackerMs);
                a = m_tracker->Track(f, model.get(), prev);
            }
            const Eigen::Isometry3f pose = a.valid ? a.pose : prev;
            if (a.valid) {
                prev = a.pose;
                m_trackerRmse.Add(a.rmse);
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

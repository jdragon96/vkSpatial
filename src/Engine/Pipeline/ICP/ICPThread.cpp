#include "Engine/Pipeline/ICP/ICPThread.h"

#include "Engine/Pipeline/ICP/Alignment.h"
#include "Engine/Pipeline/CommunicationModule.h"

#include <Eigen/Geometry>

#include <memory>
#include <utility>

namespace Engine::Pipeline {

    ICPThread::ICPThread(CommunicationModule &comm, std::unique_ptr<AlignmentCommand> align)
        : PipelineStage(comm), m_align(std::move(align)) {}

    ICPThread::~ICPThread() { Stop(); }

    void ICPThread::Interrupt() { m_comm.capturedFrames.Close(); } // wake a blocked Pop

    void ICPThread::Run() {
        Eigen::Isometry3f prev = Eigen::Isometry3f::Identity();
        Frame f;
        while (!StopRequested() && m_comm.capturedFrames.Pop(f)) {
            const std::shared_ptr<const ModelSnapshot> model = m_comm.model.Latest();
            AlignmentResult a;
            {
                util::ScopedMean t(m_alignMs);
                a = m_align->Execute(f, model.get(), prev);
            }
            const Eigen::Isometry3f pose = a.valid ? a.pose : prev;
            if (a.valid) prev = a.pose;

            TrackedFrame tf;
            tf.cameraWorld = pose * f.cam; // sensor camera -> world
            tf.pose = pose;
            tf.frame = std::move(f);
            m_comm.trackedFrames.Push(std::move(tf));
        }
        m_comm.trackedFrames.Close(); // upstream done -> let Integration drain and exit
    }

} // namespace Engine::Pipeline

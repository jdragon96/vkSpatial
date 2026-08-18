#pragma once

#include "Pipeline/Types.h" // Frame, TrackedFrame, ModelSnapshot

#include "utilities/Channel.h"
#include "utilities/Mailbox.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace Pipeline {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Lock-step handshake between registration and integration, for reproducible offline runs.
    //
    // Blocking channels make a replay LOSSLESS -- every frame is processed. They do NOT make it
    // DETERMINISTIC. The map reaches the tracker through a latest-wins Mailbox, and registration may
    // run ahead of integration by the whole trackedFrames capacity, so which map version frame N
    // aligns against depends on thread scheduling. Since that map is the alignment target, the pose
    // changes, so the next map changes: the run diverges chaotically. Measured on the 477-frame
    // capture/ recording, four runs of ONE command reported trajectory lengths of 1.47, 6.45, 7.84
    // and 136.76 metres, with rejection causes ranging from 227 TooFewInliers/0 LowOverlap to
    // 44/170. Any A/B taken through the pipeline on a recording is measuring this, not the setting.
    //
    // When enabled, registration waits after each push until integration has finished that frame, so
    // frame N always aligns against a map holding exactly frames 0..N-1.
    class FrameHandshake {
    public:
        void SetEnabled(bool on) { m_enabled = on; }
        bool Enabled() const { return m_enabled; }

        void NotePushed() {
            if (!m_enabled) return;
            std::lock_guard<std::mutex> lock(m_mutex);
            ++m_pushed;
        }

        // Called once per frame POPPED by integration, fused or not -- a skipped fusion still has to
        // release the waiter, or the pipeline deadlocks on the first frame integration declines.
        void NoteCompleted() {
            if (!m_enabled) return;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                ++m_completed;
            }
            m_cv.notify_all();
        }

        // Returns false if the pipeline is shutting down (so the caller stops rather than blocking).
        bool WaitForDrain() {
            if (!m_enabled) return true;
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] { return m_closed || m_completed >= m_pushed; });
            return !m_closed;
        }

        // Idempotent; every stop path calls it, so a waiter is never left blocked on a dead stage.
        void Close() {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_closed = true;
            }
            m_cv.notify_all();
        }

    private:
        std::mutex m_mutex;
        std::condition_variable m_cv;
        std::uint64_t m_pushed = 0, m_completed = 0;
        bool m_closed = false;
        bool m_enabled = false;
    };

    struct CommunicationModule {
        // dropWhenBehind picks the overflow policy for BOTH inter-stage links.
        //
        //   true  -- a live sensor. Frames keep arriving whether or not the map can keep up, so a
        //            full queue drops its oldest entry: latency stays bounded and the map tracks
        //            the present rather than falling further behind forever.
        //   false -- a recording. There is nothing to fall behind: the source waits. Blocking makes
        //            the run LOSSLESS, so every frame is processed and two configurations can
        //            actually be compared -- with dropping, a slower setting silently processes
        //            fewer frames and every measurement taken across settings is confounded.
        // A recording (dropWhenBehind == false) also runs the registration/integration handshake, so
        // the run is reproducible as well as lossless -- see FrameHandshake above. A live sensor
        // must not: waiting for the map would add the whole integrate time to the capture latency.
        explicit CommunicationModule(bool dropWhenBehind = true)
            : capturedFrames(8, dropWhenBehind), trackedFrames(4, dropWhenBehind) {
            handshake.SetEnabled(!dropWhenBehind);
        }

        util::Channel<Frame> capturedFrames;          // ReconstructionThread -> RegistrationThread
        util::Channel<TrackedFrame> trackedFrames;    // RegistrationThread -> IntegrationThread
        util::Mailbox<ModelSnapshot> model;           // IntegrationThread -> caller (latest only)
        FrameHandshake handshake;                     // lock-step for reproducible offline runs
    };

} // namespace Pipeline

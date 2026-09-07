#pragma once

#include "Pipeline/Types.h" // Frame, TrackedFrame, ModelSnapshot

#include "utilities/Channel.h"
#include "utilities/Mailbox.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace Pipeline {

    class FrameHandshake {
    public:
        void SetEnabled(bool on) { m_enabled = on; }
        bool Enabled() const { return m_enabled; }

        void NotePushed() {
            if (!m_enabled) return;
            std::lock_guard<std::mutex> lock(m_mutex);
            ++m_pushed;
        }

        void NoteCompleted() {
            if (!m_enabled) return;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                ++m_completed;
            }
            m_cv.notify_all();
        }

        bool WaitForDrain() {
            if (!m_enabled) return true;
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] { return m_closed || m_completed >= m_pushed; });
            return !m_closed;
        }

        // Idempotent; every stop path calls it, so a waiter is never left blocked on a dead stage.
        // Back to a fresh run: counters zeroed and reopened. The counters are cumulative, so a
        // rebuilt registration stage that starts pushing against a stale m_completed would either
        // never wait or wait forever depending on which side was ahead.
        void Reset(bool enabled) {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_pushed = 0;
                m_completed = 0;
                m_closed = false;
                m_enabled = enabled;
            }
            m_cv.notify_all();
        }

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
        explicit CommunicationModule(bool dropWhenBehind = true)
            : capturedFrames(8, dropWhenBehind), trackedFrames(4, dropWhenBehind) {
            handshake.SetEnabled(!dropWhenBehind);
        }

        // Put every link back to a fresh run WITHOUT replacing this object. PipelineStage holds a
        // CommunicationModule REFERENCE, so a stage that survives a reconfigure (the acquisition
        // stage keeps the open camera) would be left pointing at a destroyed one.
        void Reset(bool dropWhenBehind) {
            capturedFrames.Reopen();
            trackedFrames.Reopen();
            model.Clear(); // the old map must not be the new run's first alignment target
            handshake.Reset(!dropWhenBehind);
        }

        util::Channel<Frame> capturedFrames;       // AcquisitionThread -> RegistrationThread (ICP)
        util::Channel<TrackedFrame> trackedFrames; // RegistrationThread -> IntegrationThread
        util::Mailbox<ModelSnapshot> model;        // IntegrationThread -> caller (latest only)
        FrameHandshake handshake;                  // lock-step for reproducible offline runs
    };

} // namespace Pipeline

#pragma once

#include <atomic>
#include <exception>
#include <thread>

namespace Pipeline {

    struct CommunicationModule; // CommunicationModule.h (reference member — forward decl suffices)

    class PipelineStage {
    public:
        PipelineStage(const PipelineStage &) = delete;
        PipelineStage &operator=(const PipelineStage &) = delete;
        virtual ~PipelineStage();

        void Start();
        void Stop();

        // Rethrow a worker-thread exception on the caller (nullptr if the stage ran clean).
        std::exception_ptr Error() const { return m_error; }

    protected:
        explicit PipelineStage(CommunicationModule &comm);

        virtual void Run() = 0;     // the stage loop; return once StopRequested() is true
        virtual void Interrupt() {} // unblock whatever Run() waits on so Stop() joins promptly
        bool StopRequested() const { return m_stop.load(); }

        CommunicationModule &m_comm;

    private:
        void RunGuarded();

        std::thread m_thread;
        std::atomic<bool> m_stop{false};
        std::exception_ptr m_error;
    };

} // namespace Pipeline

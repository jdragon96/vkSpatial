#pragma once

#include <atomic>
#include <exception>
#include <thread>

// Base for one pipeline stage: a worker thread with a Start/Stop lifecycle, cooperative stop flag,
// and worker-exception capture. Composition over inheritance for the thread — the base owns the
// std::thread; derived classes only implement Run() (the stage loop) and Interrupt() (unblock
// whatever Run() waits on). Because Run() and Interrupt() touch derived members, EVERY derived
// class MUST call Stop() from its own destructor before its members die (the base destructor's
// Stop() would call the base Interrupt(), too late for the derived state).
namespace Engine::Pipeline {

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

} // namespace Engine::Pipeline

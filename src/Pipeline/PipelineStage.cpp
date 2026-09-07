#include "Pipeline/PipelineStage.h"

namespace Pipeline {

    PipelineStage::PipelineStage(CommunicationModule &comm) : m_comm(comm) {}

    PipelineStage::~PipelineStage() { Stop(); }

    void PipelineStage::Start() {
        if (m_thread.joinable()) return; // already running
        m_stop = false;
        m_running = true;
        m_thread = std::thread([this] { RunGuarded(); });
    }

    void PipelineStage::Stop() {
        m_stop = true;
        Interrupt(); // wake any external blocking wait (e.g. a paced source / channel Pop)
        if (m_thread.joinable()) m_thread.join();
    }

    void PipelineStage::RunGuarded() {
        try {
            Run();
        } catch (...) {
            m_error = std::current_exception();
        }
        m_running = false;
    }

} // namespace Pipeline

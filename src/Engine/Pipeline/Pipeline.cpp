#include "Engine/Pipeline/Pipeline.h"

#include "Engine/Pipeline/CommunicationModule.h"
#include "Engine/Pipeline/ICP/Alignment.h"
#include "Engine/Pipeline/ICP/ICPThread.h"
#include "Engine/Pipeline/Integration/IntegrationThread.h"
#include "Engine/Pipeline/Reconstruction/ReconstructionThread.h"

#include <utility>

namespace Engine::Pipeline {

    Pipeline::Pipeline(Config cfg, std::unique_ptr<AlignmentCommand> align) {
        buildStages(std::move(cfg), std::move(align));
    }

    Pipeline::~Pipeline() { Stop(); }

    // (Re)build the comm module + the three worker stages from `cfg`. Shared by the constructor and
    // Reconfigure; does not Start (callers decide when to run).
    void Pipeline::buildStages(Config cfg, std::unique_ptr<AlignmentCommand> align) {
        // Destroy any existing stages BEFORE replacing the comm they reference: each stage's destructor
        // runs Stop() -> Interrupt() -> m_comm.*.Close(), so the old comm must still be alive when the
        // old stages die. Replacing m_comm first (then the stages) would touch a freed comm on
        // Reconfigure (the constructor path has no old stages, so these resets are no-ops there).
        m_integration.reset();
        m_icp.reset();
        m_reconstruction.reset();
        m_comm = std::make_unique<CommunicationModule>();
        m_reconstruction = std::make_unique<ReconstructionThread>(*m_comm, std::move(cfg.source));
        m_icp = std::make_unique<ICPThread>(*m_comm, std::move(align));
        m_integration =
                std::make_unique<IntegrationThread>(*m_comm, cfg.map, std::move(cfg.densityFrames));
    }

    void Pipeline::Reconfigure(Config cfg, std::unique_ptr<AlignmentCommand> align) {
        Stop();                                          // join the current workers, close channels
        buildStages(std::move(cfg), std::move(align));   // fresh comm + stages with the new config
        Start();                                         // replay from frame 0 with the new settings
    }

    // Consumers first, producer last, so no frame is dropped for a not-yet-ready stage.
    void Pipeline::Start() {
        m_integration->Start();
        m_icp->Start();
        m_reconstruction->Start();
    }

    // Close channels first (wakes blocked Pop/Next), then join producer-first. Idempotent.
    void Pipeline::Stop() {
        m_comm->capturedFrames.Close();
        m_comm->trackedFrames.Close();
        m_reconstruction->Stop();
        m_icp->Stop();
        m_integration->Stop();
    }

    void Pipeline::SetPaused(bool paused) { m_reconstruction->SetPaused(paused); }
    bool Pipeline::IsPaused() const { return m_reconstruction->IsPaused(); }

    std::shared_ptr<const ModelSnapshot> Pipeline::LatestModel() const {
        return m_comm->model.Latest();
    }
    int Pipeline::ProcessedFrame() const { return m_integration->ProcessedFrame(); }
    EAcquisitionType Pipeline::Type() const { return m_reconstruction->Type(); }

    PipelineStats Pipeline::GetStats() const {
        PipelineStats s;
        s.processedFrame = m_integration->ProcessedFrame();
        s.captureDepth = m_comm->capturedFrames.Size();
        s.trackDepth = m_comm->trackedFrames.Size();
        s.trackDropped = m_comm->trackedFrames.Dropped();
        // Per-thread liveness + timing (avg ms + processed count).
        s.acquireMsAvg = m_reconstruction->AcquireMsAvg();
        s.acquiredFrames = m_reconstruction->AcquiredFrames();
        s.alignMsAvg = m_icp->AlignMsAvg();
        s.alignedFrames = m_icp->AlignedFrames();
        s.integrateMsAvg = m_integration->IntegrateMsAvg();
        s.integratedFrames = m_integration->IntegratedFrames();
        return s;
    }

    void Pipeline::CheckErrors() const {
        for (const PipelineStage *s: {static_cast<const PipelineStage *>(m_reconstruction.get()),
                                      static_cast<const PipelineStage *>(m_icp.get()),
                                      static_cast<const PipelineStage *>(m_integration.get())})
            if (const std::exception_ptr e = s->Error()) std::rethrow_exception(e);
    }

} // namespace Engine::Pipeline

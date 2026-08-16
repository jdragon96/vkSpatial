#include "Pipeline/Pipeline.h"

#include "Pipeline/CommunicationModule.h"
#include "Pipeline/Integration/IntegrationThread.h"
#include "Pipeline/Reconstruction/ReconstructionThread.h"
#include "Pipeline/Registration/RegistrationThread.h"
#include "Pipeline/Registration/Tracker.h"

#include <utility>

namespace Pipeline {

    Pipeline::Pipeline(Config cfg, std::unique_ptr<Tracker> align) {
        buildStages(std::move(cfg), std::move(align));
    }

    Pipeline::~Pipeline() { Stop(); }

    void Pipeline::buildStages(Config cfg, std::unique_ptr<Tracker> align) {
        m_integration.reset();
        m_registration.reset();
        m_reconstruction.reset();
        m_comm = std::make_unique<CommunicationModule>(cfg.acquisition.realTime);
        // The finest level the map can represent. Anything below it is detail the map cannot hold,
        // so carrying it through tracking and the queues is pure cost. Left alone if the caller
        // set a value, and disabled by setting it negative.
        if (cfg.acquisition.downsampleVoxel == 0.0f)
            cfg.acquisition.downsampleVoxel =
                    cfg.map.submap ? cfg.map.baseVoxel * 0.5f : cfg.map.baseVoxel;
        m_reconstruction = std::make_unique<ReconstructionThread>(*m_comm, std::move(cfg.acquisition));
        m_registration = std::make_unique<RegistrationThread>(*m_comm, std::move(align));
        m_integration = std::make_unique<IntegrationThread>(*m_comm, cfg.map);
    }

    void Pipeline::Reconfigure(Config cfg, std::unique_ptr<Tracker> align) {
        Stop();
        buildStages(std::move(cfg), std::move(align));
        Start();
    }

    void Pipeline::Start() {
        m_integration->Start();
        m_registration->Start();
        m_reconstruction->Start();
    }

    void Pipeline::Stop() {
        m_comm->capturedFrames.Close();
        m_comm->trackedFrames.Close();
        m_reconstruction->Stop();
        m_registration->Stop();
        m_integration->Stop();
    }

    float Pipeline::DownsampleVoxel() const { return m_reconstruction->DownsampleVoxel(); }

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
        s.acquireMsAvg = m_reconstruction->AcquireMsAvg();
        s.acquiredFrames = m_reconstruction->AcquiredFrames();
        s.alignMsAvg = m_registration->AlignMsAvg();
        s.alignedFrames = m_registration->AlignedFrames();
        s.trackerRmseAvg = m_registration->TrackerRmseAvg();
        s.integrateMsAvg = m_integration->IntegrateMsAvg();
        s.integratedFrames = m_integration->IntegratedFrames();
        return s;
    }

    void Pipeline::CheckErrors() const {
        for (const PipelineStage *s: {static_cast<const PipelineStage *>(m_reconstruction.get()),
                                      static_cast<const PipelineStage *>(m_registration.get()),
                                      static_cast<const PipelineStage *>(m_integration.get())})
            if (const std::exception_ptr e = s->Error()) std::rethrow_exception(e);
    }

} // namespace Pipeline

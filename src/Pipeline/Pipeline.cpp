#include "Pipeline/Pipeline.h"

#include "Pipeline/CommunicationModule.h"
#include "Pipeline/Integration/IntegrationThread.h"
#include "Pipeline/Acquisition/AcquisitionThread.h"
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
        m_acquisition.reset();
        m_comm = std::make_unique<CommunicationModule>(cfg.acquisition.realTime);
        m_acquisition = std::make_unique<AcquisitionThread>(*m_comm, std::move(cfg.acquisition));
        m_registration = std::make_unique<RegistrationThread>(*m_comm, std::move(align), cfg.fusion);
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
        m_acquisition->Start();
    }

    void Pipeline::Stop() {
        m_comm->capturedFrames.Close();
        m_comm->trackedFrames.Close();
        m_comm->handshake.Close();
        m_acquisition->Stop();
        m_registration->Stop();
        m_integration->Stop();
    }

    float Pipeline::DownsampleVoxel() const { return m_acquisition->DownsampleVoxel(); }

    std::string Pipeline::VisualPresetRefusal() const { return m_acquisition->VisualPresetRefusal(); }

    void Pipeline::SetPaused(bool paused) { m_acquisition->SetPaused(paused); }
    bool Pipeline::IsPaused() const { return m_acquisition->IsPaused(); }

    std::shared_ptr<const ModelSnapshot> Pipeline::LatestModel() const {
        return m_comm->model.Latest();
    }
    int Pipeline::ProcessedFrame() const { return m_integration->ProcessedFrame(); }
    EAcquisitionSource Pipeline::Source() const { return m_acquisition->Source(); }

    PipelineStats Pipeline::GetStats() const {
        PipelineStats s;
        s.processedFrame = m_integration->ProcessedFrame();
        s.captureDepth = m_comm->capturedFrames.Size();
        s.trackDepth = m_comm->trackedFrames.Size();
        s.trackDropped = m_comm->trackedFrames.Dropped();
        s.acquireMsAvg = m_acquisition->AcquireMsAvg();
        s.acquiredFrames = m_acquisition->AcquiredFrames();
        s.alignMsAvg = m_registration->AlignMsAvg();
        s.alignedFrames = m_registration->AlignedFrames();
        s.trackerRmseAvg = m_registration->TrackerRmseAvg();
        s.trackRejected = m_registration->Rejected();
        s.rejectedNoModel = m_registration->RejectedNoModel();
        s.rejectedNoLocalTarget = m_registration->RejectedNoLocalTarget();
        s.rejectedTooFewInliers = m_registration->RejectedTooFewInliers();
        s.rejectedLowOverlap = m_registration->RejectedLowOverlap();
        s.rejectedImplausibleMotion = m_registration->RejectedImplausibleMotion();
        s.skippedFusions = m_registration->SkippedFusions();
        s.bootstrapHeldFrames = m_registration->BootstrapHeldFrames();
        s.fusionRejectedByFitness = m_registration->FusionRejectedByFitness();
        s.fusionRejectedByRmse = m_registration->FusionRejectedByRmse();
        s.fusionArmed = m_registration->FusionArmed();
        const TrackerStats trackerStats = m_registration->TrackerCounters();
        s.relocalizationAttempts = trackerStats.relocalizationAttempts;
        s.relocalizationSuccesses = trackerStats.relocalizationSuccesses;
        s.poseDeltaMetersAvg = m_registration->PoseDeltaMetersAvg();
        s.poseDeltaMetersMax = m_registration->PoseDeltaMetersMax();
        s.poseDeltaDegreesMax = m_registration->PoseDeltaDegreesMax();
        s.trajectoryLengthMeters = m_registration->TrajectoryLengthMeters();
        s.integrateMsAvg = m_integration->IntegrateMsAvg();
        s.integratedFrames = m_integration->IntegratedFrames();
        return s;
    }

    void Pipeline::CheckErrors() const {
        for (const PipelineStage *s: {static_cast<const PipelineStage *>(m_acquisition.get()),
                                      static_cast<const PipelineStage *>(m_registration.get()),
                                      static_cast<const PipelineStage *>(m_integration.get())})
            if (const std::exception_ptr e = s->Error()) std::rethrow_exception(e);
    }

} // namespace Pipeline

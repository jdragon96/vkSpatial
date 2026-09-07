#include "Pipeline/Pipeline.h"

#include <chrono>

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
        m_lastSource = cfg.acquisition.source;
        m_comm = std::make_unique<CommunicationModule>(cfg.acquisition.realTime);
        m_acquisition = std::make_unique<AcquisitionThread>(*m_comm, std::move(cfg.acquisition));
        m_registration = std::make_unique<RegistrationThread>(*m_comm, std::move(align), cfg.fusion);
        m_integration = std::make_unique<IntegrationThread>(*m_comm, cfg.map);
    }

    void Pipeline::Reconfigure(Config cfg, std::unique_ptr<Tracker> align) {
        // A live camera keeps its acquisition stage. Nothing the caller can retune here is baked
        // into the device or the GPU front end -- RealSensePipeline is built from the frame size
        // alone, and the score / normal / downsample options are Execute arguments -- so closing
        // the camera to change a map voxel buys nothing and costs the one operation that actually
        // fails: reopening a D400 immediately after Close.
        //
        // A recording does NOT qualify. There "restart" means rewinding to frame 0, and rebuilding
        // the recorder IS the rewind. It has no device to contend for, so rebuilding is free.
        // An injected factory does not disqualify reuse -- the opposite. realsense_scan --record
        // injects a recorder wrapping the device, and rebuilding that THROWS: the recorder refuses
        // a directory that already holds a recording. Keeping the provider is the only thing that
        // works there, and it is what the live case wants anyway.
        const bool reuseAcquisition = m_acquisition && m_comm &&
                                      m_lastSource == EAcquisitionSource::Realsense &&
                                      cfg.acquisition.source == EAcquisitionSource::Realsense;
        if (!reuseAcquisition) {
            Stop();
            buildStages(std::move(cfg), std::move(align));
            Start();
            return;
        }

        // Park acquisition FIRST. Stopping the downstream stages closes capturedFrames, and a
        // worker that is mid-Push reads that as end-of-stream and leaves its loop permanently --
        // the stage object would survive and quietly stop producing frames.
        const bool wasPaused = m_acquisition->IsPaused();
        m_acquisition->SetPaused(true);
        m_acquisition->WaitUntilPaused(std::chrono::milliseconds(2000));

        // Only the two downstream stages restart. The acquisition stage keeps running against the
        // same CommunicationModule -- which is why that object is RESET rather than replaced: every
        // PipelineStage holds a reference to it.
        m_registration->Stop();
        m_integration->Stop();
        m_integration.reset();
        m_registration.reset();

        m_comm->Reset(cfg.acquisition.realTime);
        m_acquisition->SetOptions(cfg.acquisition);

        m_registration = std::make_unique<RegistrationThread>(*m_comm, std::move(align), cfg.fusion);
        m_integration = std::make_unique<IntegrationThread>(*m_comm, cfg.map);
        m_integration->Start();
        m_registration->Start();
        m_acquisition->SetPaused(wasPaused);
    }

    void Pipeline::Start() {
        if (!m_acquisition || !m_registration || !m_integration) return;
        m_integration->Start();
        m_registration->Start();
        m_acquisition->Start();
    }

    void Pipeline::Stop() {
        if (m_comm) {
            m_comm->capturedFrames.Close();
            m_comm->trackedFrames.Close();
            m_comm->handshake.Close();
        }
        if (m_acquisition) m_acquisition->Stop();
        if (m_registration) m_registration->Stop();
        if (m_integration) m_integration->Stop();
    }

    float Pipeline::DownsampleVoxel() const { return m_acquisition ? m_acquisition->DownsampleVoxel() : 0.0f; }

    std::string Pipeline::VisualPresetRefusal() const { return m_acquisition ? m_acquisition->VisualPresetRefusal() : std::string(); }

    void Pipeline::SetPaused(bool paused) {
        if (m_acquisition) m_acquisition->SetPaused(paused);
    }
    bool Pipeline::IsPaused() const { return m_acquisition && m_acquisition->IsPaused(); }

    std::shared_ptr<const ModelSnapshot> Pipeline::LatestModel() const {
        return m_comm ? m_comm->model.Latest() : nullptr;
    }
    int Pipeline::ProcessedFrame() const { return m_integration ? m_integration->ProcessedFrame() : 0; }
    EAcquisitionSource Pipeline::Source() const {
        return m_acquisition ? m_acquisition->Source() : m_lastSource;
    }

    // Zeros rather than a crash when a rebuild failed: the caller is a viewer drawing a panel,
    // and a half-built pipeline is exactly when it most needs to keep drawing to show the error.
    PipelineStats Pipeline::GetStats() const {
        PipelineStats s;
        if (!m_comm || !m_acquisition || !m_registration || !m_integration) return s;
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
                                      static_cast<const PipelineStage *>(m_integration.get())}) {
            if (!s) continue; // a rebuild that threw left this stage unbuilt
            if (const std::exception_ptr e = s->Error()) std::rethrow_exception(e);
        }
    }

} // namespace Pipeline

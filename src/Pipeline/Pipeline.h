#pragma once

#include "Pipeline/Acquisition/AcquisitionThread.h" // AcquisitionConfig, EAcquisitionSource
#include "Pipeline/Types.h"

#include <memory>
#include <vector>

namespace Pipeline {

    struct CommunicationModule; // CommunicationModule.h
    class AcquisitionThread;
    class RegistrationThread;
    class IntegrationThread;
    class Tracker; // Tracker.h

    class Pipeline {
    public:
        struct Config {
            MapConfig map;                 // TSDF/submap parameters
            AcquisitionConfig acquisition; // which source, and its parameters
            FusionGateConfig fusion;       // extra fusion conditions layered on ShouldFuse; off by default
        };

        Pipeline(Config cfg, std::unique_ptr<Tracker> align);
        ~Pipeline();

        void Start();
        void Stop();

        // Rebuild every worker stage with a new config + alignment and restart, IN PLACE -- the same
        // Pipeline object, so a held reference (e.g. RenderThread's) stays valid. Stops the current run
        // first; an accumulating map cannot be reconfigured mid-stream, so this replays from frame 0.
        // Used for runtime option toggles (submap / point-to-plane / ...) in a debug viewer.
        void Reconfigure(Config cfg, std::unique_ptr<Tracker> align);

        // Play / pause the acquisition (the mapping + render keep running). Safe from the render thread.
        void SetPaused(bool paused);
        bool IsPaused() const;

        std::shared_ptr<const ModelSnapshot> LatestModel() const;
        int ProcessedFrame() const;
        EAcquisitionSource Source() const;
        float DownsampleVoxel() const; // acquisition-stage voxel reduction actually in effect
        std::string VisualPresetRefusal() const; // empty when the requested D400 preset took
        PipelineStats GetStats() const;
        void CheckErrors() const; // rethrow the first worker-stage exception, if any

    private:
        void buildStages(Config cfg, std::unique_ptr<Tracker> align); // ctor + Reconfigure

        std::unique_ptr<CommunicationModule> m_comm;
        std::unique_ptr<AcquisitionThread> m_acquisition;
        std::unique_ptr<RegistrationThread> m_registration;
        std::unique_ptr<IntegrationThread> m_integration;
    };

} // namespace Pipeline

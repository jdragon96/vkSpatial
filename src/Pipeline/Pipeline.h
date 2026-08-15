#pragma once

#include "Pipeline/Reconstruction/ReconstructionSource.h" // AcquisitionConfig, EAcquisitionType
#include "Pipeline/Types.h"

#include <memory>
#include <vector>

namespace Pipeline {

    struct CommunicationModule; // CommunicationModule.h
    class ReconstructionThread;
    class RegistrationThread;
    class IntegrationThread;
    class Tracker; // Tracker.h

    class Pipeline {
    public:
        struct Config {
            MapConfig map;                 // TSDF/submap parameters
            AcquisitionConfig acquisition; // which acquisition strategy + its params
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
        EAcquisitionType Type() const;
        PipelineStats GetStats() const;
        void CheckErrors() const; // rethrow the first worker-stage exception, if any

    private:
        void buildStages(Config cfg, std::unique_ptr<Tracker> align); // ctor + Reconfigure

        std::unique_ptr<CommunicationModule> m_comm;
        std::unique_ptr<ReconstructionThread> m_reconstruction;
        std::unique_ptr<RegistrationThread> m_registration;
        std::unique_ptr<IntegrationThread> m_integration;
    };

} // namespace Pipeline

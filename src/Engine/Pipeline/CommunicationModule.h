#pragma once

#include "Engine/Pipeline/Types.h" // Frame, TrackedFrame, ModelSnapshot

#include "utilities/Channel.h"
#include "utilities/Mailbox.h"

namespace Engine::Pipeline {

    struct CommunicationModule {
        // ReconstructionThread -> ICPThread (drop-oldest)
        util::Channel<Frame> capturedFrames{8};
        // ICPThread -> IntegrationThread (drop-oldest)
        util::Channel<TrackedFrame> trackedFrames{4};
        // IntegrationThread -> caller (latest)
        util::Mailbox<ModelSnapshot> model;
    };

} // namespace Engine::Pipeline

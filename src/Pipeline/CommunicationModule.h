#pragma once

#include "Pipeline/Types.h" // Frame, TrackedFrame, ModelSnapshot

#include "utilities/Channel.h"
#include "utilities/Mailbox.h"

namespace Pipeline {

    struct CommunicationModule {
        // dropWhenBehind picks the overflow policy for BOTH inter-stage links.
        //
        //   true  -- a live sensor. Frames keep arriving whether or not the map can keep up, so a
        //            full queue drops its oldest entry: latency stays bounded and the map tracks
        //            the present rather than falling further behind forever.
        //   false -- a recording. There is nothing to fall behind: the source waits. Blocking makes
        //            the run LOSSLESS, so every frame is processed and two configurations can
        //            actually be compared -- with dropping, a slower setting silently processes
        //            fewer frames and every measurement taken across settings is confounded.
        explicit CommunicationModule(bool dropWhenBehind = true)
            : capturedFrames(8, dropWhenBehind), trackedFrames(4, dropWhenBehind) {}

        util::Channel<Frame> capturedFrames;          // ReconstructionThread -> RegistrationThread
        util::Channel<TrackedFrame> trackedFrames;    // RegistrationThread -> IntegrationThread
        util::Mailbox<ModelSnapshot> model;           // IntegrationThread -> caller (latest only)
    };

} // namespace Pipeline

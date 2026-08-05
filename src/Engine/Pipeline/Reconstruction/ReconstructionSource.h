#pragma once

#include "Engine/Pipeline/Types.h" // Engine::Pipeline::Frame

#include <string>
#include <vector>

namespace Engine::Pipeline {

    // The acquisition strategy the pipeline pulls frames from. One Frame per Next(); Close() also
    // wakes a blocked Next() so the pipeline can stop promptly.
    class IFrameSource {
    public:
        virtual ~IFrameSource() = default;

        virtual EAcquisitionType Type() const = 0;
        virtual const char *Name() const = 0;

        virtual void Open() {}             // acquire the device / open the dataset (optional)
        virtual bool Next(Frame &out) = 0; // fill the next frame; false when exhausted/stopped
        virtual void Close() {}            // release the device / wake a blocked Next() (optional)
    };

    // Config-driven description of a source. File is fully described here (a path list + pacing);
    // device types need an injected provider/decoder.
    struct AcquisitionConfig {
        EAcquisitionType type = EAcquisitionType::File;
        std::vector<std::string> framePaths;
        double intervalMs = 0.0;
        bool loop = false;
    };

} // namespace Engine::Pipeline

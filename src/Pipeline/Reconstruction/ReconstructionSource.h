#pragma once

#include "Pipeline/Types.h" // Pipeline::Frame

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Pipeline {

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

        // How a device source gets built. A camera is a handle, not a path list, so it cannot be
        // described by value the way File is -- and putting device parameters in this struct would
        // drag every sensor SDK the pipeline might ever support into its config type.
        //
        // Called once per stage build, which includes every Pipeline::Reconfigure. Stages are torn
        // down before the new ones are constructed, so a live device is released before this is
        // asked to open it again; a factory that captures an already-open handle instead of
        // creating one would hand the second pipeline a source the first has closed.
        std::function<std::unique_ptr<IFrameSource>()> makeSource;
    };

} // namespace Pipeline

#pragma once

#include "Pipeline/PipelineStage.h"
#include "Pipeline/Types.h" // Pipeline::Frame

#include "Realsense/RealSenseD435.h" // D435StreamOptions
#include "Realsense/RealSenseTypes.h"

#include "utilities/RunningMean.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Pipeline {

    // Where frames come from. One Frame per Next(); Close() also wakes a blocked Next() so the
    // pipeline can stop promptly.
    class IFrameSource {
    public:
        virtual ~IFrameSource() = default;
        virtual const char *Name() const = 0;

        virtual void Open() {}             // acquire the device / open the dataset (optional)
        virtual bool Next(Frame &out) = 0; // fill the next frame; false when exhausted/stopped
        virtual void Close() {}            // release the device / wake a blocked Next() (optional)
    };

    enum class EAcquisitionSource {
        Realsense,     // a live D400 through the GPU depth front end
        RealsenseFile, // a depth recording replayed through the same front end
        PlyFolder,     // frame_*.ply on disk
    };

    // Everything a source needs, by value.
    //
    // Devices used to be undescribable here, so every caller that wanted one had to write a factory
    // lambda -- which meant each tool re-derived how to build a RealSense source, and got it subtly
    // different. The source now says which one it is and the thread builds it.
    struct AcquisitionConfig {
        EAcquisitionSource source = EAcquisitionSource::PlyFolder;

        // --- PlyFolder ---------------------------------------------------------------------
        std::vector<std::string> framePaths;
        double intervalMs = 0.0; // pace the replay; 0 = as fast as the pipeline drains
        bool loop = false;

        // --- RealsenseFile -----------------------------------------------------------------
        std::string recordingDirectory;

        // --- Realsense and RealsenseFile ---------------------------------------------------
        // The GPU front end's knobs. focalLengthPixels and depthScale are filled from the device
        // (or the recording) when they are left at zero, because a frame scored with the wrong
        // sigma_z looks entirely plausible.
        Realsense::ValidationScoreOptions score;
        Realsense::NormalEstimationOptions normal;
        Realsense::DownSampleOptions downSample;
        float scoreThreshold = 0.9f;
        Realsense::D435StreamOptions stream; // live device only

        // Voxel-grid reduce each frame as it is acquired; 0 (default) disables it.
        //
        // Reducing HERE rather than at integration is what makes it worth anything: a 640x480 depth
        // frame is 307k points, and carrying all of them costs ICP, the frame queues, and every
        // copy in between. On a 477-frame D435 capture it took ICP from 133 ms to 7 ms per frame.
        //
        // Leave it at 0 for the two Realsense sources: their DownSampleOptions thins on the device
        // BEFORE the readback, so this would only thin twice.
        float downsampleVoxel = 0.0f;

        // Drop old frames to keep latency bounded, instead of blocking. TRUE only for a live
        // camera. For a recording or a dataset it must be false, which makes the run lossless --
        // the stages block rather than drop, so every frame is processed and two configurations
        // can be compared. A recording left on `true` silently processes fewer frames on the
        // slower setting and every A/B taken over it is worthless. See CommunicationModule.
        bool realTime = true;

        // Escape hatch, and the only reason it survives: tests inject synthetic sources that no
        // enum can describe. When set it wins over `source`. Production code should not need it --
        // if a real source cannot be described above, add it above.
        std::function<std::unique_ptr<IFrameSource>()> makeSource;
    };

    std::unique_ptr<IFrameSource> MakeAcquisitionSource(const AcquisitionConfig &config);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // The acquisition stage: pulls frames from one source and hands them to the ICP thread.
    //
    // Every source ends in the same place -- comm.capturedFrames -- so which one is running changes
    // nothing downstream. A live camera and a replayed recording are two configurations of this one
    // stage, not two pipelines.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class AcquisitionThread : public PipelineStage {
    public:
        AcquisitionThread(CommunicationModule &comm, AcquisitionConfig config);
        ~AcquisitionThread() override; // join before m_source dies (Run uses it)

        EAcquisitionSource Source() const { return m_config.source; }

        void SetPaused(bool paused);
        bool IsPaused() const { return m_paused.load(); }

        double AcquireMsAvg() const { return m_acquireMs.Mean(); }
        float DownsampleVoxel() const { return m_config.downsampleVoxel; }
        std::uint64_t AcquiredFrames() const { return m_acquireMs.Count(); }

    protected:
        void Interrupt() override;
        void Run() override;

    private:
        bool waitWhilePaused();
        void reduceFrame(Frame &frame) const;

        AcquisitionConfig m_config;
        std::unique_ptr<IFrameSource> m_source;
        std::atomic<bool> m_paused{false};
        std::mutex m_pauseMutex;
        std::condition_variable m_pauseCv;
        util::RunningMean m_acquireMs; // per-frame acquire time (thread-safe)
    };

} // namespace Pipeline

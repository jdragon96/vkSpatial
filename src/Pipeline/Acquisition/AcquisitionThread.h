#pragma once

#include "Pipeline/Acquisition/DepthProvider.h"
#include "Pipeline/PipelineStage.h"
#include "Pipeline/Types.h" // Pipeline::Frame

#include "Realsense/RealSenseD435.h" // D435StreamOptions
#include "Realsense/RealSenseTypes.h"

#include "utilities/RunningMean.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Pipeline {

    // What the GPU depth front end discarded, published once per frame by RealsenseFrameSource.
    //
    // Without this the panel over a live scan reads zero while points flow, and the two ceilings
    // below fail OPEN -- a full probe table or a voxel outside the packable range silently drops
    // the point. Atomic because the reader is the render thread and the writer is acquisition.
    //
    // The score kernel's own rejection breakdown is NOT here: it costs an atomic per rejected
    // pixel and most of a depth image is usually invalid, so it stays behind
    // ValidationScoreOptions::countRejections for the lab tools that want it.
    struct GpuFrontEndStats {
        std::atomic<std::uint64_t> emittedPoints{0}; // cumulative, over the whole run
        std::atomic<std::uint32_t> lastFramePoints{0};
        // The estimator's stencil hung off the edge of the image: its domain, not the scene.
        std::atomic<std::uint64_t> normalOutOfDomain{0};
        // The stencil fitted and still found too few same-surface samples. The scene refusing the
        // pixel, which is the number worth watching.
        std::atomic<std::uint64_t> normalNoSupport{0};
        std::atomic<std::uint64_t> downSampleInsertFailures{0};
        std::atomic<std::uint64_t> downSampleOutOfRange{0};
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

        // Optional. Owned by the caller because the source is built on the acquisition thread and
        // never handed back, so this is the only way to read what the front end discarded.
        std::shared_ptr<GpuFrontEndStats> gpuStats;

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

        // Escape hatch on the DEVICE, not on the frame: a test injects a scripted depth image and
        // gets the real front end over it. When set it wins over `source`, and the frames it
        // produces go through exactly the path a camera's do.
        std::function<std::unique_ptr<IDepthProvider>()> makeProvider;
    };

    // Built from `source`, or from makeProvider when that is set. PlyFolder has no device and
    // returns null -- the thread reads the files itself.
    std::unique_ptr<IDepthProvider> MakeDepthProvider(const AcquisitionConfig &config);

    // The depth front end, hidden here because it owns a Vulkan context and every consumer of
    // Pipeline.h would otherwise pay for it. Defined in AcquisitionThread.cpp.
    class DepthFrontEnd;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // The acquisition stage: makes frames and hands them to the ICP thread.
    //
    // It MAKES them rather than pulling them from a source object. There are only two ways a frame
    // comes into being -- a depth image through the GPU front end, or a PLY off disk -- and an
    // interface over two cases bought nothing but a second axis to configure wrongly.
    //
    // Every path ends in the same place, comm.capturedFrames, so which one is running changes
    // nothing downstream. A live camera and a replayed recording are two configurations of this one
    // stage, not two pipelines.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class AcquisitionThread : public PipelineStage {
    public:
        AcquisitionThread(CommunicationModule &comm, AcquisitionConfig config);
        ~AcquisitionThread() override; // joins before the device and front end die (Run uses them)

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

        void open();
        bool next(Frame &out);
        bool nextDepthFrame(Frame &out);
        bool nextPlyFrame(Frame &out);
        bool waitForNextPlySlot();

        AcquisitionConfig m_config;
        std::unique_ptr<IDepthProvider> m_provider;   // null for PlyFolder
        std::unique_ptr<DepthFrontEnd> m_frontEnd;    // built on Open, once the device can be asked
        std::size_t m_plyCursor = 0;
        bool m_hasLastPlyEmit = false;
        std::chrono::steady_clock::time_point m_lastPlyEmit{};
        bool m_closed = false;
        std::atomic<bool> m_paused{false};
        std::mutex m_pauseMutex;
        std::condition_variable m_pauseCv;
        util::RunningMean m_acquireMs; // per-frame acquire time (thread-safe)
    };

} // namespace Pipeline

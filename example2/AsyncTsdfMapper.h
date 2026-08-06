#pragma once

#include "VoxelFillDebug.h" // voxdbg::FillTracker

#include "Engine/Core/Context.h"
#include "Engine/Spatial/AdvancedTSDF.h" // AdvancedEntry
#include "Engine/Spatial/SubmapAdvancedTSDF.h"
#include "utilities/Mailbox.h"
#include "utilities/StageProfiler.h"

#include <Eigen/Core>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

// Background TSDF mapping thread: a worker owns its OWN Vulkan device + SubmapAdvancedTSDF and
// integrates frames off the render thread, publishing an immutable MapSnapshot the render thread
// polls. Because the worker's device is separate from the render device, there is no shared Vulkan
// state — the only handoff is the snapshot. Scrub/play is a single coalesced target frame: the
// worker always steps toward the newest requested frame (reset+replay when scrubbing backward).
namespace asyncmap {

    using Engine::Spatial::AdvancedEntry;

    // One captured frame the mapper integrates. Owned by the caller (shared, read-only).
    struct MapperFrame {
        std::vector<Eigen::Vector3f> pts, nrm;
        Eigen::Vector3f cam = Eigen::Vector3f::Zero();
    };

    using Box = std::pair<Eigen::Vector3f, Eigen::Vector3f>; // world AABB (min, max)

    // Immutable per-frame result handed to the render thread.
    struct MapSnapshot {
        std::vector<AdvancedEntry> entries; // occupied voxels (precedence-deduped base+detail)
        std::vector<char> isNew;            // parallel to entries: first filled this frame
        std::vector<int> firstFrame;        // parallel to entries: frame that first filled it
        int processedFrame = -1;
        uint32_t baseTiles = 0, detailTiles = 0, denseBlocks = 0;
        Eigen::Vector3f allocMin = Eigen::Vector3f::Zero(), allocMax = Eigen::Vector3f::Zero();
        bool hasAlloc = false;
        double integrateMs = 0, downloadMs = 0, trackerMs = 0;
        std::vector<Box> baseCoreBoxes;   // coarse 512^3 tile windows
        std::vector<Box> denseBlockBoxes; // submap (detail) regions
    };

    struct Config {
        float baseVoxel = 0.5f;
        float truncation = 1.5f;
        int blockVoxels = 32;
        float detailK = 4.0f;
        uint32_t tileHash = 1u << 20;
        uint32_t maxPoints = 1u << 15;
        Engine::Spatial::IntegrationQuality quality{3, 4, true};
        bool pointToPlane = true;
        float confidence = 0.5f;
        bool hermite = false;
    };

    // The single step the worker takes toward `target` from the currently-shown frame. Pure (unit
    // tested): backward target -> Reset+replay from 0; caught up -> Wait; else integrate shown+1.
    enum class StepKind { Wait,
                          Reset,
                          Forward };
    struct Step {
        StepKind kind;
        int frame; // valid only for Forward
    };
    inline Step planStep(int shown, int target) {
        if (target < 0 || target == shown) return {StepKind::Wait, -1};
        if (target < shown) return {StepKind::Reset, -1};
        return {StepKind::Forward, shown + 1};
    }

    class AsyncTsdfMapper {
    public:
        ~AsyncTsdfMapper() { joinWorker(); } // never rethrows (no throwing during destruction)

        // Launches the worker (builds its device + submap + density precompute, then idles until the
        // first RequestFrame). `frames` is shared read-only with the worker.
        void Start(const Config &cfg, std::shared_ptr<const std::vector<MapperFrame>> frames) {
            m_cfg = cfg;
            m_frames = std::move(frames);
            m_stop = false;
            m_target = -1;
            m_requested = -1;
            m_processed = -1;
            m_thread = std::thread([this] { workerMain(); });
        }

        void Stop() {
            joinWorker();
            rethrowIfFailed(); // surface a worker failure to an explicit Stop() caller
        }

        // Ask the worker to integrate up to `target` (coalesced: only the newest request matters).
        void RequestFrame(int target) {
            const int n = m_frames ? int(m_frames->size()) : 0;
            if (n == 0) return;
            const int t = std::max(0, std::min(target, n - 1));
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_target = t;
            }
            m_requested = t;
            m_cv.notify_one();
        }

        std::shared_ptr<const MapSnapshot> Latest() {
            rethrowIfFailed();
            return m_mailbox.Latest();
        }

        int RequestedFrame() const { return m_requested.load(); }
        int ProcessedFrame() const { return m_processed.load(); }

    private:
        void joinWorker() {
            if (m_thread.joinable()) {
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_stop = true;
                }
                m_cv.notify_one();
                m_thread.join();
            }
        }

        void rethrowIfFailed() {
            std::exception_ptr e;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                e = m_exception;
            }
            if (e) std::rethrow_exception(e);
        }

        void workerMain() {
            try {
                Engine::Core::Context ctx;
                Engine::Spatial::SubmapAdvancedTSDF submap;
                submap.Build(ctx, m_cfg.baseVoxel, m_cfg.truncation, m_cfg.blockVoxels, m_cfg.detailK,
                             m_cfg.tileHash, m_cfg.maxPoints);
                submap.SetIntegrationQuality(m_cfg.quality);
                submap.SetPointToPlane(m_cfg.pointToPlane);
                submap.SetConfidenceWeight(m_cfg.confidence);
                submap.SetHermitePosition(m_cfg.hermite);
                // Density learned online during the integrate loop below (no pre-scan).

                voxdbg::FillTracker tracker(m_cfg.baseVoxel * 0.5f); // detail voxel -> unique keys
                util::StageProfiler prof;
                int shown = -1;

                while (true) {
                    int target;
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_cv.wait(lock, [&] { return m_stop || m_target != shown; });
                        if (m_stop) break;
                        target = m_target;
                    }

                    const Step step = planStep(shown, target);
                    if (step.kind == StepKind::Wait) continue;
                    if (step.kind == StepKind::Reset) {
                        submap.Reset();
                        tracker.reset();
                        shown = -1;
                        m_processed = -1;
                        continue;
                    }

                    const MapperFrame &fr = (*m_frames)[step.frame];
                    auto snap = std::make_shared<MapSnapshot>();
                    {
                        util::ScopedStageTimer t(prof, "integrate");
                        submap.Integrate(fr.pts, fr.nrm, fr.cam);
                    }
                    snap->integrateMs = prof.LastMs("integrate");
                    {
                        util::ScopedStageTimer t(prof, "download");
                        snap->entries = submap.DownloadEntries();
                    }
                    snap->downloadMs = prof.LastMs("download");
                    {
                        util::ScopedStageTimer t(prof, "tracker");
                        snap->isNew = tracker.update(snap->entries, step.frame);
                        snap->firstFrame.resize(snap->entries.size());
                        const float keyVoxel = m_cfg.baseVoxel * 0.5f;
                        for (std::size_t i = 0; i < snap->entries.size(); ++i)
                            snap->firstFrame[i] =
                                    tracker.firstFrame(voxdbg::keyOf(snap->entries[i], keyVoxel));
                    }
                    snap->trackerMs = prof.LastMs("tracker");

                    snap->processedFrame = step.frame;
                    snap->baseTiles = submap.BaseTileCount();
                    snap->detailTiles = submap.DetailTileCount();
                    snap->denseBlocks = submap.DenseBlockCount();
                    snap->baseCoreBoxes = submap.BaseCoreBoxes();
                    snap->denseBlockBoxes = submap.DenseBlockBoxes();
                    fillAlloc(*snap);

                    m_mailbox.Publish(snap);
                    shown = step.frame;
                    m_processed = step.frame;
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_exception = std::current_exception();
            }
        }

        void fillAlloc(MapSnapshot &snap) const {
            if (snap.entries.empty()) return;
            Eigen::Vector3f mn = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
            Eigen::Vector3f mx = Eigen::Vector3f::Constant(-std::numeric_limits<float>::max());
            for (const AdvancedEntry &e: snap.entries) {
                mn = mn.cwiseMin(e.center);
                mx = mx.cwiseMax(e.center);
            }
            const Eigen::Vector3f h = Eigen::Vector3f::Constant(0.5f * m_cfg.baseVoxel);
            snap.allocMin = mn - h;
            snap.allocMax = mx + h;
            snap.hasAlloc = true;
        }

        Config m_cfg;
        std::shared_ptr<const std::vector<MapperFrame>> m_frames;

        std::thread m_thread;
        mutable std::mutex m_mutex;
        std::condition_variable m_cv;
        bool m_stop = false;
        int m_target = -1; // requested frame (guarded by m_mutex, for the worker's condvar)
        std::exception_ptr m_exception;

        std::atomic<int> m_requested{-1};
        std::atomic<int> m_processed{-1};
        util::Mailbox<MapSnapshot> m_mailbox;
    };

} // namespace asyncmap

#pragma once

#include "Alignment.h"
#include "PipelineTypes.h"
#include "VoxelFillDebug.h" // voxdbg::FillTracker

#include "Engine/Core/Context.h"
#include "TSDF/Backends/SubmapAdvancedTSDF.h"
#include "utilities/Channel.h"
#include "utilities/Mailbox.h"
#include "utilities/StageProfiler.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace pipeline {

    struct PipelineStats {
        int requestedFrame = -1;
        int processedFrame = -1;
        std::size_t captureDepth = 0;
        std::size_t trackDepth = 0;
        std::size_t trackDropped = 0;
    };

    class ReconstructionPipeline {
    public:
        struct Config {
            asyncmap::Config map;
            std::size_t captureQueue = 8; // Frame channel capacity
            std::size_t trackQueue = 4;   // TrackedFrame channel capacity (drop-oldest)
            int downloadEveryN = 1;       // Map download cadence (raise to decouple slow download)
        };

        ~ReconstructionPipeline() { joinAll(); }

        void Start(const Config &cfg, std::unique_ptr<AlignmentCommand> align) {
            m_cfg = cfg;
            m_align = std::move(align);
            m_capture = std::make_unique<util::Channel<Frame>>(cfg.captureQueue, /*dropOldest=*/true);
            m_tracked = std::make_unique<util::Channel<TrackedFrame>>(cfg.trackQueue, /*dropOldest=*/true);
            m_resetGen = 0;
            m_pushed = -1;
            m_processed = -1;
            m_mapThread = std::thread([this] { mapMain(); });
            m_trackThread = std::thread([this] { trackMain(); });
        }

        void Stop() {
            joinAll();
            rethrowIfFailed();
        }

        // Backward-scrub reset: drop pending work + bump generation so both workers reset their state
        // (Track prior pose, Map submap + fill tracker) before consuming re-pushed frames. Keeps the
        // dense set (SubmapAdvancedTSDF::Reset). Caller re-pushes frames 0..target afterwards.
        void Reset() {
            if (m_capture) m_capture->Clear();
            if (m_tracked) m_tracked->Clear();
            m_pushed = -1;
            m_processed = -1;
            m_resetGen.fetch_add(1);
        }

        bool PushFrame(Frame frame) {
            if (!m_capture || m_capture->Closed()) return false;
            m_pushed.fetch_add(1);
            return m_capture->Push(std::move(frame));
        }

        std::shared_ptr<const ModelSnapshot> LatestModel() {
            rethrowIfFailed();
            return m_model.Latest();
        }

        int RequestedFrame() const { return m_pushed.load(); }
        int ProcessedFrame() const { return m_processed.load(); }
        PipelineStats GetStats() const {
            PipelineStats s;
            s.requestedFrame = m_pushed.load();
            s.processedFrame = m_processed.load();
            if (m_capture) s.captureDepth = m_capture->Size();
            if (m_tracked) {
                s.trackDepth = m_tracked->Size();
                s.trackDropped = m_tracked->Dropped();
            }
            return s;
        }

    private:
        /// Track thread: pop a frame, resolve its pose via the alignment command (seeded by the
        /// previous pose + the latest model), push a posed frame downstream.
        void trackMain() {
            try {
                Eigen::Isometry3f prev = Eigen::Isometry3f::Identity();
                int myGen = m_resetGen.load();
                Frame f;
                while (m_capture->Pop(f)) {
                    const int gen = m_resetGen.load();
                    if (gen != myGen) { // reset: forget the accumulated prior
                        prev = Eigen::Isometry3f::Identity();
                        myGen = gen;
                    }
                    const std::shared_ptr<const ModelSnapshot> model = m_model.Latest();
                    const AlignmentResult a = m_align->Execute(f, model.get(), prev);
                    const Eigen::Isometry3f pose = a.valid ? a.pose : prev;
                    if (a.valid) prev = a.pose;

                    TrackedFrame tf;
                    tf.cameraWorld = pose * f.cam; // sensor camera -> world
                    tf.pose = pose;
                    tf.gen = gen;
                    tf.frame = std::move(f);
                    m_tracked->Push(std::move(tf));
                }
                m_tracked->Close(); // capture closed+drained -> signal Map to finish
            } catch (...) {
                captureException();
                m_tracked->Close();
            }
        }

        /// Map thread: owns the device + TSDF; integrate posed frames, throttle-download a snapshot.
        void mapMain() {
            try {
                Engine::Core::Context ctx;
                TSDF::SubmapAdvancedTSDF submap;
                submap.Build(
                        ctx,
                        m_cfg.map.baseVoxel,
                        m_cfg.map.truncation,
                        m_cfg.map.blockVoxels,
                        m_cfg.map.detailK,
                        m_cfg.map.tileHash,
                        m_cfg.map.maxPoints);
                submap.SetIntegrationQuality(m_cfg.map.quality);
                submap.SetPointToPlane(m_cfg.map.pointToPlane);
                submap.SetConfidenceWeight(m_cfg.map.confidence);
                submap.SetHermitePosition(m_cfg.map.hermite);
                // Density learned online during the integrate loop below (no pre-scan).

                voxdbg::FillTracker tracker(m_cfg.map.baseVoxel * 0.5f);
                util::StageProfiler prof;
                int myGen = m_resetGen.load();
                int processed = -1;

                TrackedFrame tf;
                while (m_tracked->Pop(tf)) {
                    const int gen = m_resetGen.load();
                    if (gen != myGen) { // reset: rebuild from scratch (keep dense set)
                        submap.Reset();
                        tracker.reset();
                        myGen = gen;
                        processed = -1;
                        m_processed = -1;
                    }
                    if (tf.gen != gen) continue; // stale (pre-reset) frame

                    integrateWorld(submap, tf, prof);
                    ++processed;

                    if ((processed % std::max(1, m_cfg.downloadEveryN)) == 0)
                        publishSnapshot(submap, tracker, prof, processed);
                }
            } catch (...) {
                captureException();
            }
        }

        // Integrate a posed frame in world coords (no-op transform for an identity pose).
        void integrateWorld(TSDF::SubmapAdvancedTSDF &submap, const TrackedFrame &tf,
                            util::StageProfiler &prof) {
            util::ScopedStageTimer t(prof, "integrate");
            if (tf.pose.matrix().isApprox(Eigen::Matrix4f::Identity())) {
                submap.Integrate(tf.frame.pts, tf.frame.nrm, tf.cameraWorld);
                return;
            }
            const std::size_t n = std::min(tf.frame.pts.size(), tf.frame.nrm.size());
            std::vector<Eigen::Vector3f> wp(n), wn(n);
            const Eigen::Matrix3f R = tf.pose.rotation();
            for (std::size_t i = 0; i < n; ++i) {
                wp[i] = tf.pose * tf.frame.pts[i];
                wn[i] = R * tf.frame.nrm[i];
            }
            submap.Integrate(wp, wn, tf.cameraWorld);
        }

        void publishSnapshot(const TSDF::SubmapAdvancedTSDF &submap,
                             voxdbg::FillTracker &tracker, util::StageProfiler &prof, int processed) {
            auto snap = std::make_shared<ModelSnapshot>();
            {
                util::ScopedStageTimer t(prof, "download");
                snap->entries = submap.DownloadEntries();
            }
            snap->downloadMs = prof.LastMs("download");
            snap->integrateMs = prof.LastMs("integrate");
            {
                util::ScopedStageTimer t(prof, "tracker");
                snap->isNew = tracker.update(snap->entries, processed);
                snap->firstFrame.resize(snap->entries.size());
                const float keyVoxel = m_cfg.map.baseVoxel * 0.5f;
                for (std::size_t i = 0; i < snap->entries.size(); ++i)
                    snap->firstFrame[i] =
                            tracker.firstFrame(voxdbg::keyOf(snap->entries[i], keyVoxel));
            }
            snap->trackerMs = prof.LastMs("tracker");
            snap->processedFrame = processed;
            snap->baseTiles = submap.BaseTileCount();
            snap->detailTiles = submap.DetailTileCount();
            snap->denseBlocks = submap.DenseBlockCount();
            snap->baseCoreBoxes = submap.BaseCoreBoxes();
            snap->denseBlockBoxes = submap.DenseBlockBoxes();
            fillAlloc(*snap);
            m_model.Publish(snap);
            m_processed = processed; // now reflected in a published snapshot
        }

        void fillAlloc(ModelSnapshot &snap) const {
            if (snap.entries.empty()) return;
            Eigen::Vector3f mn = Eigen::Vector3f::Constant(1e30f);
            Eigen::Vector3f mx = Eigen::Vector3f::Constant(-1e30f);
            for (const TSDF::AdvancedEntry &e: snap.entries) {
                mn = mn.cwiseMin(e.center);
                mx = mx.cwiseMax(e.center);
            }
            const Eigen::Vector3f h = Eigen::Vector3f::Constant(0.5f * m_cfg.map.baseVoxel);
            snap.allocMin = mn - h;
            snap.allocMax = mx + h;
            snap.hasAlloc = true;
        }

        void joinAll() { // close + join without rethrowing (safe from the destructor)
            if (m_capture) m_capture->Close();
            if (m_tracked) m_tracked->Close();
            if (m_trackThread.joinable()) m_trackThread.join();
            if (m_mapThread.joinable()) m_mapThread.join();
        }

        void captureException() {
            std::lock_guard<std::mutex> lock(m_errMutex);
            if (!m_exception) m_exception = std::current_exception();
        }
        void rethrowIfFailed() {
            std::exception_ptr e;
            {
                std::lock_guard<std::mutex> lock(m_errMutex);
                e = m_exception;
            }
            if (e) std::rethrow_exception(e);
        }

        Config m_cfg;
        std::unique_ptr<AlignmentCommand> m_align;
        std::unique_ptr<util::Channel<Frame>> m_capture;
        std::unique_ptr<util::Channel<TrackedFrame>> m_tracked;
        util::Mailbox<ModelSnapshot> m_model;

        std::thread m_trackThread, m_mapThread;
        std::atomic<int> m_resetGen{0};
        std::atomic<int> m_pushed{-1};
        std::atomic<int> m_processed{-1};

        std::mutex m_errMutex;
        std::exception_ptr m_exception;
    };

} // namespace pipeline

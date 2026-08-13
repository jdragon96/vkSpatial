#include "Engine/Pipeline/Integration/IntegrationThread.h"

#include "Engine/Pipeline/CommunicationModule.h"

#include "Engine/Core/Context.h"
#include "Engine/Spatial/SubmapAdvancedTSDF.h"
#include "utilities/StageProfiler.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <utility>

namespace Engine::Pipeline {

    namespace {

        void integrateWorld(Engine::Spatial::SubmapAdvancedTSDF &submap,
                            const TrackedFrame &tf,
                            util::StageProfiler &prof) {
            util::ScopedStageTimer t(prof, "integrate");

            // step 1: frame RT가 identity일 때
            if (tf.pose.matrix().isApprox(Eigen::Matrix4f::Identity())) {
                submap.IntegrateGPU(tf.frame.pts, tf.frame.nrm, tf.cameraWorld);
                return;
            }

            // step 2: frame RT가 있을 때
            const std::size_t n = std::min(tf.frame.pts.size(), tf.frame.nrm.size());
            std::vector<Eigen::Vector3f> vecWorldPosition(n), vecWorldNormal(n);
            const Eigen::Matrix3f R = tf.pose.rotation();
            for (std::size_t i = 0; i < n; ++i) {
                vecWorldPosition[i] = tf.pose * tf.frame.pts[i];
                vecWorldNormal[i] = R * tf.frame.nrm[i];
            }
            submap.IntegrateGPU(vecWorldPosition, vecWorldNormal, tf.cameraWorld);
        }

        void buildSnapshot(ModelSnapshot &snap,
                           const Engine::Spatial::SubmapAdvancedTSDF &submap,
                           util::StageProfiler &prof,
                           int processed,
                           float baseVoxel,
                           float truncationDistance) {
            {
                util::ScopedStageTimer t(prof, "download");
                submap.DownloadEntries(snap.entries);
            }
            snap.downloadMs = prof.LastMs("download");
            snap.integrateMs = prof.LastMs("integrate");
            snap.trackerMs = 0.0;
            snap.processedFrame = processed;
            snap.voxel = baseVoxel;                       // so the tracker can scale its correspondence distance to the map
            snap.truncationDistance = truncationDistance; // so the tracker can recover a sub-voxel target
                                                          // point via center - tsdf*truncationDistance*normal
            snap.baseTiles = submap.BaseTileCount();
            snap.detailTiles = submap.DetailTileCount();
            snap.denseBlocks = submap.DenseBlockCount();
            snap.baseCoreBoxes = submap.BaseCoreBoxes();
            snap.denseBlockBoxes = submap.DenseBlockBoxes();
            snap.hasAlloc = false;
            if (!snap.entries.empty()) {
                float mnx = 1e30f, mny = 1e30f, mnz = 1e30f;
                float mxx = -1e30f, mxy = -1e30f, mxz = -1e30f;
                for (const Engine::Spatial::AdvancedEntry &e: snap.entries) {
                    const float x = e.center.x(), y = e.center.y(), z = e.center.z();
                    mnx = std::min(mnx, x);
                    mny = std::min(mny, y);
                    mnz = std::min(mnz, z);
                    mxx = std::max(mxx, x);
                    mxy = std::max(mxy, y);
                    mxz = std::max(mxz, z);
                }
                const float h = 0.5f * baseVoxel;
                snap.allocMin = Eigen::Vector3f(mnx - h, mny - h, mnz - h);
                snap.allocMax = Eigen::Vector3f(mxx + h, mxy + h, mxz + h);
                snap.hasAlloc = true;
            }
        }

    } // namespace

    IntegrationThread::IntegrationThread(CommunicationModule &comm, MapConfig cfg)
        : PipelineStage(comm), m_cfg(cfg) {}

    IntegrationThread::~IntegrationThread() { Stop(); }

    void IntegrationThread::Interrupt() { m_comm.trackedFrames.Close(); } // wake a blocked Pop

    void IntegrationThread::Run() {
        Engine::Core::Context ctx;
        Engine::Spatial::SubmapAdvancedTSDF submap;
        submap.Build(ctx,
                     m_cfg.baseVoxel,
                     m_cfg.truncation,
                     m_cfg.blockVoxels,
                     m_cfg.detailK,
                     m_cfg.tileHash,
                     m_cfg.maxPoints,
                     m_cfg.detailTruncVoxels);
        submap.SetIntegrationQuality(m_cfg.quality);
        submap.SetPointToPlane(m_cfg.pointToPlane);
        submap.SetConfidenceWeight(m_cfg.confidence);
        submap.SetHermitePosition(m_cfg.hermite);
        submap.SetDownsample(m_cfg.downsample);
        submap.PreWarm();

        std::vector<std::shared_ptr<ModelSnapshot>> pool;
        auto acquireSnapshot = [&pool]() -> std::shared_ptr<ModelSnapshot> {
            for (const auto &s: pool)
                if (s.use_count() == 1) return s; // only the pool holds it -> free to reuse
            pool.push_back(std::make_shared<ModelSnapshot>());
            return pool.back();
        };

        util::StageProfiler prof;
        int processed = -1;
        TrackedFrame tf;
        while (!StopRequested() && m_comm.trackedFrames.Pop(tf)) {
            {
                util::ScopedMean t(m_integrateMs); // integrate + download + snapshot for this frame
                ++processed;
                submap.SetCurrentFrame(processed); // GPU stamps newly-filled voxels with this frame
                integrateWorld(submap, tf, prof);
                std::shared_ptr<ModelSnapshot> snap = acquireSnapshot();
                buildSnapshot(*snap, submap, prof, processed, m_cfg.baseVoxel, m_cfg.truncation);
                m_comm.model.Publish(snap);
            }
            m_processed = processed; // reflected in a published snapshot
        }
    }

} // namespace Engine::Pipeline

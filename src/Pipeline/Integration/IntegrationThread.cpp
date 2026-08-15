#include "Pipeline/Integration/IntegrationThread.h"

#include "Pipeline/CommunicationModule.h"

#include "Engine/Core/Context.h"
#include "TSDF/TSDF.h"
#include "utilities/StageProfiler.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <utility>

namespace Pipeline {

    namespace {

        TSDFConfiguration ConfigureMap(const MapConfig &cfg) {
            TSDFConfiguration config;
            config.backend = "advanced";
            config.splitter = cfg.submap ? "dense" : "none";
            config.useSubmap = cfg.submap;
            config.windowVoxels = 512;
            config.backendConfig.voxelSize = cfg.baseVoxel;
            config.backendConfig.truncation = cfg.truncation;
            config.backendConfig.hashCapacity = cfg.tileHash;
            config.backendConfig.maxPointPerFrame = cfg.maxPoints;
            config.backendConfig.pointToPlane = cfg.pointToPlane;
            config.backendConfig.confidenceWeight = cfg.confidence;
            config.backendConfig.hermitePosition = cfg.hermite;
            config.backendConfig.maxDirections = cfg.maxDirections;
            config.backendConfig.directionExponent = cfg.directionExponent;
            config.backendConfig.viewAngleWeight = cfg.viewAngleWeight;
            config.backendConfig.probeStats = cfg.probeStats;
            config.splitterConfig.baseResolution = cfg.baseVoxel;
            config.splitterConfig.blockVoxels = cfg.blockVoxels;
            config.splitterConfig.maxPointPerFrame = int(cfg.maxPoints);
            return config;
        }

        void IntegrateWorld(TSDF &tsdf, const TrackedFrame &tf, util::StageProfiler &prof) {
            util::ScopedStageTimer timer(prof, "integrate");

            if (tf.pose.matrix().isApprox(Eigen::Matrix4f::Identity())) {
                tsdf.Integrate(tf.frame.pts, tf.frame.nrm, tf.cameraWorld);
                return;
            }

            const std::size_t n = std::min(tf.frame.pts.size(), tf.frame.nrm.size());
            std::vector<Eigen::Vector3f> worldPosition(n), worldNormal(n);
            const Eigen::Matrix3f rotation = tf.pose.rotation();
            for (std::size_t i = 0; i < n; ++i) {
                worldPosition[i] = tf.pose * tf.frame.pts[i];
                worldNormal[i] = rotation * tf.frame.nrm[i];
            }
            tsdf.Integrate(worldPosition, worldNormal, tf.cameraWorld);
        }

        void BuildSnapshot(ModelSnapshot &snap,
                           const TSDF &tsdf,
                           util::StageProfiler &prof,
                           int processed,
                           float baseVoxel,
                           float truncationDistance) {
            {
                util::ScopedStageTimer timer(prof, "download");
                tsdf.Download(snap.entries);
            }
            snap.downloadMs = prof.LastMs("download");
            snap.integrateMs = prof.LastMs("integrate");
            snap.trackerMs = 0.0;
            snap.processedFrame = processed;
            snap.voxel = baseVoxel;                       // lets the tracker scale its correspondence distance
            snap.truncationDistance = truncationDistance; // lets it recover a sub-voxel target point

            snap.baseTiles = uint32_t(tsdf.BaseWindowCount());
            snap.detailTiles = uint32_t(tsdf.DetailWindowCount());
            snap.denseBlocks = tsdf.LastDivision().denseBlockCount;
            snap.map = tsdf.Stats();
            snap.windowLimitRefusals = tsdf.WindowLimitRefusalCount();
            snap.baseCoreBoxes = tsdf.BaseWindowBoxes();
            // Detail WINDOWS, not the dense blocks the old submap map drew -- the new map places the
            // detail level per window, so this is the coarser outline of the same thing.
            snap.denseBlockBoxes = tsdf.DetailWindowBoxes();

            snap.hasAlloc = false;
            if (snap.entries.empty()) return;

            Eigen::Vector3f minimum = Eigen::Vector3f::Constant(1e30f);
            Eigen::Vector3f maximum = Eigen::Vector3f::Constant(-1e30f);
            for (const TSDFVoxel &voxel: snap.entries) {
                minimum = minimum.cwiseMin(voxel.center);
                maximum = maximum.cwiseMax(voxel.center);
            }
            const Eigen::Vector3f half = Eigen::Vector3f::Constant(0.5f * baseVoxel);
            snap.allocMin = minimum - half;
            snap.allocMax = maximum + half;
            snap.hasAlloc = true;
        }

    } // namespace

    IntegrationThread::IntegrationThread(CommunicationModule &comm, MapConfig cfg)
        : PipelineStage(comm), m_cfg(cfg) {}

    IntegrationThread::~IntegrationThread() { Stop(); }

    void IntegrationThread::Interrupt() { m_comm.trackedFrames.Close(); } // wake a blocked Pop

    void IntegrationThread::Run() {
        Engine::Core::Context ctx;
        TSDF tsdf;
        tsdf.Build(ctx, ConfigureMap(m_cfg));

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
                IntegrateWorld(tsdf, tf, prof);
                std::shared_ptr<ModelSnapshot> snap = acquireSnapshot();
                BuildSnapshot(*snap, tsdf, prof, processed, m_cfg.baseVoxel, m_cfg.truncation);
                m_comm.model.Publish(snap);
            }
            m_processed = processed; // reflected in a published snapshot
        }
    }

} // namespace Pipeline

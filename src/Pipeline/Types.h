#pragma once

#include "TSDF/Backends/TSDFBackend.h" // TSDFVoxel

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace Pipeline {

    enum class EAcquisitionType {
        File,
        DepthCamera,
        StructuredLight,
    };

    struct Frame {
        std::vector<Eigen::Vector3f> pts, nrm;
        Eigen::Vector3f cam = Eigen::Vector3f::Zero();
    };

    // Why a track was not adopted. Lives here, not with the Tracker, because TrackedFrame carries it
    // across the registration -> integration boundary: the fusion decision is per-cause, and one
    // "rejected" bit cannot express it. "The map has not been built yet" and "the solve latched onto
    // strays" call for opposite responses.
    enum class ETrackFailure {
        None,              // the track was adopted
        NoModel,           // no map yet, or an empty frame -- expected on the first frames
        NoLocalTarget,     // the frame's neighbourhood holds almost no map: it is somewhere new
        TooFewInliers,     // the solve found fewer correspondences than minInliers
        LowOverlap,        // enough inliers, but too small a share of the frame -- the fitness gate
        ImplausibleMotion, // the solve moved further from the prior than a hand-held camera can in
                           // one frame -- a mis-convergence by physics, whatever its fitness says
    };

    // Should a frame with this tracking outcome be fused into the map?
    //
    // Not "was the track valid". A frame whose solve failed its gates carries a knowingly-wrong pose,
    // and the map it corrupts is the NEXT frame's alignment target -- on a 477-frame D435 capture
    // roughly half the frames took that path. But refusing every rejected frame does not work either:
    // GpuIcpTracker reports NoModel while the map is still empty, so declining to fuse frame 0 means
    // the map never bootstraps and every later frame is NoModel too -- an empty reconstruction. And
    // NoLocalTarget means the frame is somewhere the map does not reach yet, so declining those
    // freezes the map at whatever the first frames happened to see.
    //
    // So: fuse when there is nothing to corrupt, skip when a real local map was there and the solve
    // still failed against it.
    inline bool ShouldFuse(bool trackValid, ETrackFailure failure) {
        if (trackValid) return true;
        return failure == ETrackFailure::NoModel || failure == ETrackFailure::NoLocalTarget;
    }

    struct TrackedFrame {
        Frame frame;
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity(); // sensor -> world
        Eigen::Vector3f cameraWorld = Eigen::Vector3f::Zero();  // world camera position (view weight)
        ETrackFailure failure = ETrackFailure::None;
        bool fuse = true; // ShouldFuse(...) for this frame; false -> integration skips it
    };

    using Box = std::pair<Eigen::Vector3f, Eigen::Vector3f>; // world AABB (min, max)

    // Immutable per-frame model handed to the caller / render thread.
    struct ModelSnapshot {
        std::vector<TSDFVoxel> entries; // occupied voxels (precedence-deduped base+detail)
        std::vector<char> isNew;            // parallel to entries: first filled this frame
        std::vector<int> firstFrame;        // parallel to entries: frame that first filled it
        int processedFrame = -1;
        float voxel = 0.0f; // map base voxel size -> lets trackers scale the ICP correspondence distance
                            // to the map resolution (a fixed maxCorrDist mismatched to a coarse voxel both
                            // finds too few correspondences AND blows up the GPU LocalGrid cell count).
        float truncationDistance = 0.0f; // TSDF truncation band -> lets a tracker recover a sub-voxel
                                         // target point via center - tsdf*truncationDistance*normal.
        uint32_t baseTiles = 0, detailTiles = 0, denseBlocks = 0; // windows/level + dense blocks

        // Memory, hash occupancy and probe cost for the whole map. probe* is zero unless
        // MapConfig::probeStats was on.
        TSDFBackendStats map;
        uint32_t windowLimitRefusals = 0;
        Eigen::Vector3f allocMin = Eigen::Vector3f::Zero(), allocMax = Eigen::Vector3f::Zero();
        bool hasAlloc = false;
        double integrateMs = 0, downloadMs = 0, trackerMs = 0;
        std::vector<Box> baseCoreBoxes;   // coarse 512^3 tile windows
        std::vector<Box> denseBlockBoxes; // submap (detail) regions
    };

    // Map parameters — the integration stage builds a TSDF from these.
    struct MapConfig {
        float baseVoxel = 0.5f;
        float truncation = 1.5f;
        int blockVoxels = 32;
        float detailK = 4.0f;
        float detailTruncVoxels = 3.0f; // detail-level band radius (in detail voxels); quality<->speed
        uint32_t tileHash = 1u << 19;   // per-tile hash slots; smaller = cheaper compaction + tile alloc
        uint32_t maxPoints = 1u << 15;
        uint32_t maxDirections = 3;
        uint32_t directionExponent = 4;
        bool viewAngleWeight = true;

        // Count hash probes. Development switch: recompiles the integrate kernel and adds three
        // atomics per lookup.
        bool probeStats = false;

        // Integration-stage feature toggles (each independently on/off).
        bool submap = true;         // false -> base-only map: skip density, no detail (voxel/2) level
        bool pointToPlane = true;   // false -> projective ray distance instead of point-to-plane
        float confidence = 0.5f;    // A1 surface-proximity weight in [0,1]; 0 -> off (uniform)
        bool hermite = false;       // A2 cubic-Hermite zero-crossing position; false -> linear
        bool downsample = false;    // voxel-grid downsample integrate input to baseVoxel/2; opt-in --
                                    //   a no-op O(N) pass on already-sparse scans, helps only over-sampled ones
    };

    // Live per-thread readout for the caller (e.g. a debug HUD): queue depths, plus each worker
    // stage's average per-frame time and processed count. A count that keeps rising = that thread is
    // alive and advancing; the avg ms is that stage's typical cost.
    struct PipelineStats {
        int processedFrame = -1;
        std::size_t captureDepth = 0, trackDepth = 0, trackDropped = 0;

        // ReconstructionThread: acquire one frame (File source: dominated by the --interval pacing).
        double acquireMsAvg = 0.0;
        std::uint64_t acquiredFrames = 0;
        // ICPThread: align one frame to the model (0 for identity; real cost for icp/global).
        double alignMsAvg = 0.0;
        std::uint64_t alignedFrames = 0;
        // ICPThread: running mean of the tracker's residual rmse over valid tracks (0 if none yet).
        double trackerRmseAvg = 0.0;
        // Per-frame pose change, and the total path walked. A hand-held camera at 30 fps moves
        // well under 0.05 m per frame; anything far above that is not motion the sensor could have
        // made, so these separate a tracker that spread the frames correctly from one that
        // diverged -- which the residual rmse cannot do, since it only scores the correspondences
        // the tracker itself chose.
        std::uint64_t trackRejected = 0; // tracker returned invalid -> previous pose reused
        // Rejections split by cause. One total cannot be acted on: a run rejecting frames because
        // the map is not built yet needs nothing done, while one rejecting them for low overlap has
        // a tracking problem.
        std::uint64_t rejectedNoModel = 0, rejectedNoLocalTarget = 0;
        std::uint64_t rejectedTooFewInliers = 0, rejectedLowOverlap = 0;
        std::uint64_t rejectedImplausibleMotion = 0;
        // Rejected frames that were NOT fused, so their wrong pose never entered the map. Fewer than
        // trackRejected: NoModel/NoLocalTarget frames are still fused (ShouldFuse, above).
        std::uint64_t skippedFusions = 0;
        double poseDeltaMetersAvg = 0.0, poseDeltaMetersMax = 0.0;
        double poseDeltaDegreesMax = 0.0;
        double trajectoryLengthMeters = 0.0;

        // IntegrationThread: integrate + download + first-seen tracking for one frame.
        double integrateMsAvg = 0.0;
        std::uint64_t integratedFrames = 0;
    };
} // namespace Pipeline

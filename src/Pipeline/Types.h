#pragma once

#include "TSDF/Backends/TSDFBackend.h" // TSDFVoxel

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Pipeline {

    struct Frame {
        std::vector<Eigen::Vector3f> pts, nrm;
        Eigen::Vector3f cam = Eigen::Vector3f::Zero();
    };

    enum class ETrackFailure {
        None,              // the track was adopted
        NoModel,           // no map yet, or an empty frame -- expected on the first frames
        NoLocalTarget,     // the frame's neighbourhood holds almost no map: it is somewhere new
        TooFewInliers,     // the solve found fewer correspondences than minInliers
        LowOverlap,        // enough inliers, but too small a share of the frame -- the fitness gate
        ImplausibleMotion, // the solve moved further from the prior than a hand-held camera can in
                           // one frame -- a mis-convergence by physics, whatever its fitness says
    };

    // 이 프레임의 포즈로 융합되어도 괜찮은가?
    // 1. ICP 성공 시 허용
    // 2. 최초 프레임의 경우 허용
    inline bool ShouldFuse(bool trackValid, ETrackFailure failure) {
        if (trackValid) return true;
        return failure == ETrackFailure::NoModel || failure == ETrackFailure::NoLocalTarget;
    }

    // Extra, opt-in conditions layered on top of ShouldFuse (see FusionGate). ShouldFuse answers
    // "would this pose corrupt the map"; this answers "is the solve good enough to be worth
    // fusing at all", which is a judgement the caller tunes per sensor rather than a fixed rule.
    //
    // Every field is off by default, so a pipeline that does not set one behaves exactly as before.
    // That default matters more than it looks: the identity tracker never computes a fitness and
    // reports 0.0 for every frame, so a gate that shipped enabled would refuse the entire
    // scan_out / scanData corpus and reconstruct nothing.
    struct FusionGateConfig {
        // Hold fusion after the map's seed frame until this many CONSECUTIVE frames clear
        // bootstrapMinFitness -- proof that tracking actually locked onto the seed before the map
        // is allowed to grow on top of it. 0 = off. The run restarts on any frame that misses.
        int bootstrapConsecutiveFrames = 0;
        // Fitness a frame must reach to count towards the bootstrap run. Fitness is the share of
        // the frame's points that found a correspondence, so a depth frame's border, occluded and
        // not-yet-mapped points put a ceiling well below 1.0 -- 0.85 against a grown map on a D435
        // indoor capture, and lower still here, because while the run is held the map is only the
        // single seed frame. Measured on capture_live (450 frames, voxel 0.03): the gate arms in
        // exactly 5 frames at 0.76 and never arms at 0.78. A threshold above that cliff holds the
        // map at its seed forever, which starves tracking into LowOverlap and never recovers, so
        // the default sits a clear margin below it rather than at the edge.
        float bootstrapMinFitness = 0.70f;
        // Steady-state per-frame gates, applied once armed. 0 = off for both.
        float minimumFusionFitness = 0.0f; // fuse only when fitness >= this
        float maximumFusionRmse = 0.0f;    // fuse only when rmse <= this
    };

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
        std::vector<char> isNew;        // parallel to entries: first filled this frame
        std::vector<int> firstFrame;    // parallel to entries: frame that first filled it
        int processedFrame = -1;
        float voxel = 0.0f; // map base voxel size -> lets trackers scale the ICP correspondence distance
                            // to the map resolution (a fixed maxCorrDist mismatched to a coarse voxel both
                            // finds too few correspondences AND blows up the GPU LocalGrid cell count).
        float truncationDistance = 0.0f;
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

        // Spatial bucket index over `entries` (BuildSnapshotEntryIndex, built once per snapshot on
        // the integration thread). Trackers crop the snapshot to the frame's AABB every frame;
        // without an index that crop is a full O(entries) scan on the REGISTRATION thread, and it
        // grows with the whole map while the frame's neighbourhood does not. entryBucketSize == 0
        // means "no index" (hand-built snapshots, older producers): ForEachEntryInBox then falls
        // back to the linear scan, so the index is never load-bearing for correctness.
        float entryBucketSize = 0.0f;
        std::unordered_map<std::int64_t, std::vector<std::uint32_t>> entryBuckets;
    };

    // Bucket side in base voxels. Trades bucket count against crop sharpness: at the capture/
    // scale (voxel 0.05) a bucket is 0.8 m, so a room-scale map stays in hundreds of buckets while
    // a frame's crop box touches only a handful.
    inline constexpr int kEntryIndexBucketVoxels = 16;

    // Same 21-bit-per-axis packing as the ICP NN grid: coordinates past +-2^20 buckets alias,
    // which only adds false candidates -- every query re-checks the exact box below, so results
    // stay correct regardless.
    inline std::int64_t EntryBucketKey(int x, int y, int z) {
        return (std::int64_t(x) & 0x1FFFFF) | ((std::int64_t(y) & 0x1FFFFF) << 21) |
               ((std::int64_t(z) & 0x1FFFFF) << 42);
    }

    inline void BuildSnapshotEntryIndex(ModelSnapshot &snap) {
        snap.entryBuckets.clear(); // snapshots are pooled -- a stale index must never survive reuse
        snap.entryBucketSize = snap.voxel > 0.0f ? float(kEntryIndexBucketVoxels) * snap.voxel : 0.0f;
        if (snap.entryBucketSize <= 0.0f) return;
        for (std::uint32_t entryIndex = 0; entryIndex < std::uint32_t(snap.entries.size()); ++entryIndex) {
            const Eigen::Vector3f &center = snap.entries[entryIndex].center;
            snap.entryBuckets[EntryBucketKey(int(std::floor(center.x() / snap.entryBucketSize)),
                                             int(std::floor(center.y() / snap.entryBucketSize)),
                                             int(std::floor(center.z() / snap.entryBucketSize)))]
                    .push_back(entryIndex);
        }
    }

    // Visit every entry whose center lies inside [minimum, maximum], via the bucket index when the
    // snapshot carries one and a linear scan when it does not.
    template<typename EntryVisitor>
    inline void ForEachEntryInBox(const ModelSnapshot &snap,
                                  const Eigen::Vector3f &minimum,
                                  const Eigen::Vector3f &maximum,
                                  EntryVisitor &&visitEntry) {
        const auto centerInsideBox = [&](const TSDFVoxel &entry) {
            return (entry.center.array() >= minimum.array()).all() &&
                   (entry.center.array() <= maximum.array()).all();
        };
        if (snap.entryBucketSize <= 0.0f) {
            for (const TSDFVoxel &entry: snap.entries)
                if (centerInsideBox(entry)) visitEntry(entry);
            return;
        }
        const float bucketSize = snap.entryBucketSize;
        const int firstBucketX = int(std::floor(minimum.x() / bucketSize));
        const int lastBucketX = int(std::floor(maximum.x() / bucketSize));
        const int firstBucketY = int(std::floor(minimum.y() / bucketSize));
        const int lastBucketY = int(std::floor(maximum.y() / bucketSize));
        const int firstBucketZ = int(std::floor(minimum.z() / bucketSize));
        const int lastBucketZ = int(std::floor(maximum.z() / bucketSize));
        // A box spanning more buckets than the map has (or one so large the count overflows int
        // arithmetic) is cheaper to answer by walking the occupied buckets themselves.
        const double boxBucketCount =
                (double(lastBucketX) - double(firstBucketX) + 1.0) *
                (double(lastBucketY) - double(firstBucketY) + 1.0) *
                (double(lastBucketZ) - double(firstBucketZ) + 1.0);
        if (boxBucketCount > double(snap.entryBuckets.size())) {
            for (const auto &[bucketKey, entryIndices]: snap.entryBuckets) {
                (void) bucketKey;
                for (const std::uint32_t entryIndex: entryIndices)
                    if (centerInsideBox(snap.entries[entryIndex])) visitEntry(snap.entries[entryIndex]);
            }
            return;
        }
        for (int z = firstBucketZ; z <= lastBucketZ; ++z)
            for (int y = firstBucketY; y <= lastBucketY; ++y)
                for (int x = firstBucketX; x <= lastBucketX; ++x) {
                    const auto found = snap.entryBuckets.find(EntryBucketKey(x, y, z));
                    if (found == snap.entryBuckets.end()) continue;
                    for (const std::uint32_t entryIndex: found->second)
                        if (centerInsideBox(snap.entries[entryIndex])) visitEntry(snap.entries[entryIndex]);
                }
    }

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
        // Range-adaptive truncation band; bandSigmaMultiplier 0 (default) = the fixed band.
        // Axial depth noise grows quadratically with range, so one truncation is too wide up
        // close and barely adequate far away. See TSDFAdaptiveBand / AdvancedTSDF.h.
        float bandSigmaMultiplier = 0.0f;
        float bandMinimumVoxels = 2.0f;
        // Discount only the occluded side of the truncation band instead of both sides equally.
        // false (default) = the symmetric `confidence` profile below.
        bool behindSurfaceDropoff = false;

        // Count hash probes. Development switch: recompiles the integrate kernel and adds three
        // atomics per lookup.
        bool probeStats = false;

        // Integration-stage feature toggles (each independently on/off).
        bool submap = true;       // false -> base-only map: skip density, no detail (voxel/2) level
        bool pointToPlane = true; // false -> projective ray distance instead of point-to-plane
        float confidence = 0.5f;  // A1 surface-proximity weight in [0,1]; 0 -> off (uniform)
        bool hermite = false;     // A2 cubic-Hermite zero-crossing position; false -> linear
        bool downsample = false;  // voxel-grid downsample integrate input to baseVoxel/2; opt-in --
                                  //   a no-op O(N) pass on already-sparse scans, helps only over-sampled ones
    };

    // Tracker-side counters surfaced through Pipeline::GetStats(). A boundary type like
    // ModelSnapshot: the base Tracker returns zeros, and a tracker that owns an expensive internal
    // mechanism (the relocalizing composite's global fallback) reports how often it ran -- silent
    // success/failure inside the registration thread would otherwise be undiagnosable from a live
    // run.
    struct TrackerStats {
        std::uint64_t relocalizationAttempts = 0;  // global fallback solves started
        std::uint64_t relocalizationSuccesses = 0; // ...whose refined pose passed the local gates
    };

    // Live per-thread readout for the caller (e.g. a debug HUD): queue depths, plus each worker
    // stage's average per-frame time and processed count. A count that keeps rising = that thread is
    // alive and advancing; the avg ms is that stage's typical cost.
    struct PipelineStats {
        int processedFrame = -1;
        std::size_t captureDepth = 0, trackDepth = 0, trackDropped = 0;

        // AcquisitionThread: grab one depth image and run the GPU front end over it.
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
        // FusionGate refusals, a breakdown of the share of skippedFusions the gate caused (see
        // FusionGate.h). Zero unless Config::fusion was set, since the gate ships off.
        std::uint64_t bootstrapHeldFrames = 0;
        std::uint64_t fusionRejectedByFitness = 0;
        std::uint64_t fusionRejectedByRmse = 0;
        bool fusionArmed = true;
        double poseDeltaMetersAvg = 0.0, poseDeltaMetersMax = 0.0;
        double poseDeltaDegreesMax = 0.0;
        double trajectoryLengthMeters = 0.0;
        // Tracker-internal counters (TrackerStats above); zero for trackers without the mechanism.
        std::uint64_t relocalizationAttempts = 0, relocalizationSuccesses = 0;

        // IntegrationThread: integrate + download + first-seen tracking for one frame.
        double integrateMsAvg = 0.0;
        std::uint64_t integratedFrames = 0;
    };
} // namespace Pipeline

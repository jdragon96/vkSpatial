#pragma once

#include "Engine/Spatial/AdvancedTSDF.h"                  // Engine::Spatial::AdvancedEntry
#include "Engine/Spatial/DirectionalIntegrationQuality.h" // Engine::Spatial::IntegrationQuality

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace Engine::Pipeline {

    using Engine::Spatial::AdvancedEntry;

    enum class EAcquisitionType {
        File,
        DepthCamera,
        StructuredLight,
    };

    struct Frame {
        std::vector<Eigen::Vector3f> pts, nrm;
        Eigen::Vector3f cam = Eigen::Vector3f::Zero();
    };

    struct TrackedFrame {
        Frame frame;
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity(); // sensor -> world
        Eigen::Vector3f cameraWorld = Eigen::Vector3f::Zero();  // world camera position (view weight)
        int gen = 0;                                            // reset generation (stale-frame guard)
    };

    using Box = std::pair<Eigen::Vector3f, Eigen::Vector3f>; // world AABB (min, max)

    // Immutable per-frame model handed to the caller / render thread.
    struct ModelSnapshot {
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

    // TSDF / submap parameters — the integration stage builds a SubmapAdvancedTSDF from these.
    struct MapConfig {
        float baseVoxel = 0.5f;
        float truncation = 1.5f;
        int blockVoxels = 32;
        float detailK = 4.0f;
        float detailTruncVoxels = 3.0f; // detail-level band radius (in detail voxels); quality<->speed
        uint32_t tileHash = 1u << 19;   // per-tile hash slots; smaller = cheaper compaction + tile alloc
        uint32_t maxPoints = 1u << 15;
        Engine::Spatial::IntegrationQuality quality{3, 4, true};

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
        // IntegrationThread: integrate + download + first-seen tracking for one frame.
        double integrateMsAvg = 0.0;
        std::uint64_t integratedFrames = 0;
    };
} // namespace Engine::Pipeline

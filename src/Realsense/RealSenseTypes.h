#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace Realsense {
    struct ValidationMaskProperty {
        std::uint32_t valid = 0;
        std::uint32_t emitted = 0;
        float score = 0.f;
    };

    static_assert(sizeof(ValidationMaskProperty) == 12, "the GLSL mirror is three 4-byte scalars");
    static_assert(offsetof(ValidationMaskProperty, emitted) == 4,
                  "the GLSL struct lists emitted second; the two orders must not drift");
    // ValidationMask::DownloadScores gathers at this offset, and the scatter kernel reads it by
    // name -- a reordered struct would hand the CPU `emitted` bits reinterpreted as a float.
    static_assert(offsetof(ValidationMaskProperty, score) == 8, "field order drifted");

    struct ValidationScoreCounters {
        std::uint32_t scoredPixels = 0;
        std::uint32_t zeroedByNoMeasurement = 0;
        std::uint32_t zeroedByRange = 0;
        std::uint32_t zeroedByInfrared = 0;
        std::uint32_t zeroedByNeighbourSupport = 0;
    };

    static_assert(sizeof(ValidationScoreCounters) == 20, "the GLSL mirror is five 4-byte scalars");
    static_assert(offsetof(ValidationScoreCounters, zeroedByNoMeasurement) == 4, "field order drifted");
    static_assert(offsetof(ValidationScoreCounters, zeroedByRange) == 8, "field order drifted");
    static_assert(offsetof(ValidationScoreCounters, zeroedByInfrared) == 12, "field order drifted");
    static_assert(offsetof(ValidationScoreCounters, zeroedByNeighbourSupport) == 16, "field order drifted");

    // Kept apart from ValidationScoreCounters on purpose: that one partitions the image and must
    // keep summing to width*height, while these two count pixels a LATER pass refused.
    struct NormalEstimationCounters {
        // The stencil hung off the edge of the image. Every estimator loses a frame this way and a
        // wider one loses more, so this tracks the estimator's domain, not the scene.
        std::uint32_t outOfDomain = 0;
        // The stencil fitted and still found too few same-surface samples. This one is the scene
        // refusing the pixel, which is the number worth watching.
        std::uint32_t noSupport = 0;
    };

    static_assert(sizeof(NormalEstimationCounters) == 8, "the GLSL mirror is two 4-byte scalars");
    static_assert(offsetof(NormalEstimationCounters, noSupport) == 4, "field order drifted");

    struct ScorePushConstants {
        std::int32_t width;
        std::int32_t height;
        float depthScale;
        float subpixelRms;
        float focalLengthPixels;
        float baselineMeters;
        float sameSurfaceSigmaMultiplier;
        float nearFadeStart;
        float nearFadeEnd;
        float farFadeStart;
        float farFadeEnd;
        float infraredFloor;
        float infraredReference;
        float infraredSaturation;
    };

    // Must match the push_constant block in ValidationMask.RemainValidDepth.glsl.
    struct RemainPushConstants {
        std::int32_t width;
        std::int32_t height;
        float fx;
        float fy;
        float cx;
        float cy;
        float depthScale;
        float scoreThreshold;
    };

    // Must match ValidationMask.CountEmittedPerRow / ValidationMask.ScatterValidPoints.
    struct ImagePushConstants {
        std::int32_t width;
        std::int32_t height;
    };

    // Must match ValidationMask.ScanRows.glsl.
    struct RowScanPushConstants {
        std::int32_t height;
    };

    // Must match the push_constant block in NormalEstimation.EstimateNormal.glsl.
    struct NormalPushConstants {
        std::int32_t width;
        std::int32_t height;
        float subpixelRms;
        float focalLengthPixels;
        float baselineMeters;
        float sameSurfaceSigmaMultiplier;
        std::int32_t planeFitRadius;
        std::int32_t minimumPlaneFitSamples;
    };

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Options.
    //
    // Both option structs live here rather than beside the class that consumes them because
    // NormalEstimation needs ValidationScoreOptions (the sensor constants that set the
    // same-surface tolerance) while ValidationMask owns a NormalEstimation -- defining either one
    // in its own class header would make the two headers include each other.
    ///////////////////////////////////////////////////////////////////////////////////////////////

    // The four numbers back-projection needs, as one argument. Not D435Calibration: that carries
    // depthScale and the baseline too, and a replayed recording has intrinsics without a device.
    struct PinholeIntrinsics {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
    };

    struct ValidationScoreOptions {
        // metres per Z16 unit, from rs2::depth_sensor::get_depth_scale
        float depthScale = 0.001f;
        float subpixelRms = 0.08f;
        // [px] -- 0.08 where the projector finds texture, 0.25 on blank wall
        float focalLengthPixels = 0.0f;
        // D435 0.05, D455 0.095
        float baselineMeters = 0.05f;
        // 3 sigma covers 99.7% of the axial noise
        float sameSurfaceSigmaMultiplier = 3.0f;
        float nearFadeStart = 0.3f;
        float nearFadeEnd = 0.5f;
        float farFadeStart = 2.5f;
        float farFadeEnd = 3.5f;

        float infraredFloor = 20.0f;       // [Y8 count * m^2]
        float infraredReference = 120.0f;  // [Y8 count * m^2]
        float infraredSaturation = 250.0f; // [Y8 count]

        bool useInfrared = false;
        bool countRejections = false;
    };

    // Two ceilings, both fail OPEN: a pixel that could not be placed in the table keeps its
    // emitted flag. Less downsampling costs throughput; dropping it would punch a hole in the
    // surface, and a hole is the failure that shows no symptom until the mesh is wrong.
    struct DownSampleCounters {
        std::uint32_t insertFailures = 0;      // the probe budget ran out
        std::uint32_t outOfPackableRange = 0;  // the voxel sits outside the 11/11/10-bit key
    };

    static_assert(sizeof(DownSampleCounters) == 8, "the GLSL mirror is two 4-byte scalars");
    static_assert(offsetof(DownSampleCounters, outOfPackableRange) == 4, "field order drifted");

    // Must match the push_constant block in DownSample.ToDetailVoxel.glsl.
    struct DownSamplePushConstants {
        std::int32_t width;
        std::int32_t height;
        float detailVoxelMeters;
        std::uint32_t slotCount;
    };

    struct DownSampleOptions {
        // Off by default: whether there is anything to remove depends entirely on the voxel size
        // against the sample spacing z/f, and at a fine voxel the pass costs a dispatch to delete
        // nothing. See docs/superpowers/specs/2026-09-05-realsense-downsample-design.md.
        bool enabled = false;
        // One TSDF detail voxel. The pipeline does not know what a TSDF is -- this is just the
        // distance below which two points are redundant to whatever consumes the cloud.
        float detailVoxelMeters = 0.0f;
    };

    struct NormalEstimationOptions {
        // "planefit" rather than Pipeline's "forward": measured at 12.5 degrees against the ground
        // truth where "forward" reads 50.6, and on real frames it needs no depth prefilter, so the
        // point that reaches the TSDF is not smoothed to fix the normal.
        std::string estimator = "planefit";
        // The fit spans planeFitRadius pixels either side of the centre; 2 is a 5x5 window, which
        // matches Pipeline's planeFitWindow = 5. Ignored by the two difference estimators.
        int planeFitRadius = 2;
        int minimumPlaneFitSamples = 8;
        // Skips the pass outright. For callers that want the compacted points and nothing else --
        // the normal stencil costs a border of pixels, so a test pinning an exact point count has
        // to be able to ask for the chain without it.
        bool enabled = true;
    };
} // namespace Realsense
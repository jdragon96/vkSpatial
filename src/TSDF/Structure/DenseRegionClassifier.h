#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <vector>

namespace TSDF {

    // When a block earns a detail level. All four conditions are ANDed: any one of them alone
    // admits blocks whose detail level would be waste, and the detail level is a second full tile
    // hierarchy -- the measured dominant memory cost.
    struct DensityCriteria {
        // occupiedFine / occupiedCoarse over a surface equals min(4, (v/s)^2), so 3.2 means
        // "sample spacing s <= v / sqrt(3.2) ~= 0.56 v" -- the fine grid is actually resolved.
        float occupancyRatio = 3.2f;
        // |sum(normal)| / pointCount over ONE frame: ~1 on a plane, lower on curvature and edges. A
        // densely scanned flat face has nothing to refine, and this is the only condition that sees
        // that.
        //
        // MUST BE CALIBRATED AGAINST THE CALLER'S NORMAL ESTIMATOR. This condition is exactly as
        // good as the normals it is handed, and normal noise is indistinguishable from curvature
        // here. For unit normals with per-axis RMS angular error sigma, |sum(n)|/N is about
        // 1 - sigma^2, so the 0.9 default trips at sigma ~= 18 degrees -- inside depth-sensor normal
        // error at grazing incidence or on low albedo. A FAILED estimate is worse than a noisy one:
        // a zero normal adds 0 to the sum and 1 to the count, so a perfectly flat wall with 10%
        // failed normals reads exactly 0.90 and refines. The verdict is a latch, so one bad frame
        // is permanent. A caller whose estimator drops normals should either filter them out before
        // Record() or lower this threshold to match its measured failure rate.
        float normalCoherence = 0.9f;
        // Samples per fine cell after refinement. Below this the refined voxels are noise.
        float samplesPerFineCell = 3.0f;
        // Absolute floor so a block glimpsed by a handful of points cannot latch to dense.
        uint32_t minimumFineOccupied = 64u;
    };

    // Per-block statistics. Mirrors the GPU-side layout exactly: 10 tightly packed 4-byte scalars.
    // The GPU side has ONE definition, in DenseRegionClassifier.common.glsl, shared by all three
    // passes; this struct is its only other copy, so a field added there must be added here in the
    // same position and the static_assert below is what catches a size drift.
    //
    // The "Frame" fields hold THIS frame only and are zeroed by the clear pass every frame. The
    // normal sum in particular is per-frame ON PURPOSE: summed cumulatively it overflows int32 on
    // exactly the dense flat wall its veto exists for, and averaging normals across viewpoints
    // mixes differently-misregistered estimates rather than sharpening one. The remaining bound is
    // one frame, one block: 2^31 / 10000 = 214,748 points before int32 wraps -- about 3.5x the
    // worst real scan density measured for a 0.32 m block, so it carries no counter.
    struct BlockRecord {
        uint32_t blockKey;          // packed block coordinate; 0xFFFFFFFF = empty slot
        uint32_t pointCount;        // cumulative
        uint32_t pointCountFrame;   // THIS frame -- denominator of the normal coherence
        uint32_t coarseOccupied;    // cumulative sum of per-frame counts
        uint32_t fineOccupied;      // cumulative sum of per-frame counts
        uint32_t fineOccupiedFrame; // THIS frame -- the absolute floor, and the sizing input
        uint32_t fineOccupiedMax;   // max over frames -- the detail table sizing input
        int32_t sumNormalFrameX;    // THIS frame, fixed point x10000
        int32_t sumNormalFrameY;
        int32_t sumNormalFrameZ;
    };
    static_assert(sizeof(BlockRecord) == 40, "BlockRecord must be 10 packed 4-byte scalars");

    // Decides which blocks earn a half-voxel detail level, and splits a frame's points into the
    // two levels. Owns only its own buffers; it knows nothing about any memory strategy, so it can
    // be built and tested on its own.
    //
    // PER-FRAME CONTRACT. Record(), Classify() and Partition() are ONE frame's sequence, and the
    // three must go into one CommandBatch which is submitted before the next frame's triple is
    // recorded. This is not a style preference: CommandBatch forbids dispatching one pipeline twice
    // per batch, and this class relies on that to reset m_totals and m_partitionCount from the host
    // without racing a pending dispatch. A caller that batches two frames together gets doubled
    // totals and partition counts, partition writes past the two index lists, and -- if the second
    // frame grows the buffers -- the first frame's already-recorded dispatch reading destroyed ones.
    //
    // COST. Build() ends in Reset(), and Reset() submits its own CommandBatch and blocks on a full
    // queue wait (the dense latch is device-local, so there is no host pointer to memset). Neither
    // is a per-frame path; both are per-scene.
    class DenseRegionClassifier {
    public:
        // `blockVoxels` must be in [1, 32] -- wider aliases distinct cells onto one packed key --
        // and throws otherwise. `maxPointPerFrame` is a hint, not a cap; Record() grows past it.
        // Safe to call again on a built classifier, in either size direction: it reallocates
        // unconditionally and Reset()s.
        void Build(Engine::Core::Context &context, float baseVoxel, int blockVoxels,
                   uint32_t maxPointPerFrame, const DensityCriteria &criteria = {});

        // Empties the block records, the dense latch, and every counter the accessors serve. Cell
        // hashes are per-frame and cleared by Record itself. Costs a queue submit and a full wait.
        void Reset();

        // Records the accumulate pass into `batch` without submitting. Grows the point buffers
        // when a frame exceeds the current capacity -- never clamps.
        //
        // `normals` must be the same length as `points` (throws otherwise -- a mismatch has no
        // correct interpretation and truncating to the shorter one would drop the caller's points
        // silently), and every normal must be UNIT LENGTH. The whole curvature test is
        // |sum(normal)| / pointCount against a threshold near 1, so a non-unit normal does not just
        // add noise, it rescales the measurement: short normals read as curvature that is not there
        // and refine a flat wall. A zero normal is the degenerate case of that; see
        // DensityCriteria::normalCoherence.
        void Record(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    Engine::Compute::CommandBatch &batch);

        // Applies the criteria to every block record. Latched: a dense block stays dense.
        void Classify(Engine::Compute::CommandBatch &batch);

        // Records the partition pass into `batch` without submitting. Splits the most recent
        // Record()'d frame's points into a base-level index list and a detail-level index list,
        // using the verdicts Classify() wrote to m_denseFlags. A point whose block record could not
        // be created (HASH_INSERT_FAILED, i.e. EMPTY_KEY, in m_blockIndex) still goes to the base
        // level -- every point must land somewhere, so the two output counts always total the
        // input count.
        void Partition(Engine::Compute::CommandBatch &batch);

        // Reads back the two index lists Partition() produced, sized to exactly the counts the GPU
        // wrote -- never the buffers' full capacity.
        void ReadPartition(std::vector<uint32_t> &base, std::vector<uint32_t> &detail) const;

        uint32_t DenseBlockCount() const;

        // Occupied fine cells summed over every dense block, from per-frame maximum occupancy
        // rather than the cumulative count, which re-counts a cell once per frame.
        //
        // THIS IS THE OCCUPANCY, NOT THE TABLE SIZE. Spec section 4 defines the detail table's
        // capacity as this figure times a load-factor margin, and picking that margin is the
        // caller's job -- this class has no idea what hash the caller will put the slots in. Size a
        // detail table at exactly this number and it runs at load factor 1.0. The recorded failure
        // mode for that in this repository is not "slow", it is HOLES: SubmapAdvancedTSDF's detail
        // hash overflows and the surface disappears where it overflowed.
        //
        // It is also a lower bound in a second way. fineOccupiedMax is a per-frame MAXIMUM, while
        // the detail table has to hold the UNION over frames. Spec section 5 argues max ~= union,
        // but that only holds when each frame covers the whole block -- a scanner sweeping ACROSS a
        // block sees a different third of it each frame, and the union is then several times the max.
        uint32_t DetailSlotEstimate() const;

        uint32_t BlockCount() const;

        // Points whose block record could not be created: the block table's probing gave up, or the
        // block sits outside the packed key's +/-512-block coordinate range (which is
        // 512 x baseVoxel x blockVoxels -- only +/-1.638 m at a 100 um base voxel). Those points are
        // NOT dropped; Partition() routes them to the base level. What is lost is their contribution
        // to a block verdict, so such a block can never earn a detail level. Zero in a healthy run.
        // Mirrors how the TSDF backends (e.g. AdvancedTSDF::InsertFailureCount) surface their own
        // insert failures: a counter the caller can observe rather than a silent drop.
        uint32_t BlockInsertFailureCount() const;

        // Fine or coarse cell claims whose probing gave up before finding a slot, so that cell's
        // occupancy went uncounted for that frame. Zero in a healthy run -- the cell hashes are
        // sized at twice the frame's point count, so load factor is bounded at 0.5 by construction
        // -- but linear probing clusters, and "unlikely" is not the same as observable. A non-zero
        // value means the occupancy ratio, and therefore the verdicts and DetailSlotEstimate(), are
        // under-reported.
        uint32_t CellInsertFailureCount() const;

        // Occupied block records, for tests and diagnostics. Not a per-frame path.
        std::vector<BlockRecord> ReadBlocks() const;

    private:
        void growPointBuffers(uint32_t pointCount);

        Engine::Core::Context *m_context = nullptr;
        float m_baseVoxel = 0.01f;
        int m_blockVoxels = 32;
        uint32_t m_maxPointPerFrame = 0;
        uint32_t m_recordedPointCount = 0; // points passed to the most recent Record() call
        DensityCriteria m_criteria;

        std::unique_ptr<Engine::Core::Buffer> m_pointBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_normalBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_blockRecords;
        std::unique_ptr<Engine::Core::Buffer> m_blockCount;
        std::unique_ptr<Engine::Core::Buffer> m_blockInsertFailureCount;
        std::unique_ptr<Engine::Core::Buffer> m_cellInsertFailureCount;
        std::unique_ptr<Engine::Core::Buffer> m_blockIndex;   // per point -> record index
        std::unique_ptr<Engine::Core::Buffer> m_fineCells;    // per-frame key-only hash
        std::unique_ptr<Engine::Core::Buffer> m_coarseCells;  // per-frame key-only hash
        std::unique_ptr<Engine::Core::Buffer> m_denseFlags;   // per-record latch, parallel to m_blockRecords
        std::unique_ptr<Engine::Core::Buffer> m_totals;       // [0] denseBlockCount, [1] detailSlots
        std::unique_ptr<Engine::Core::Buffer> m_baseIndex;      // per frame -- points routed to the base level
        std::unique_ptr<Engine::Core::Buffer> m_detailIndex;    // per frame -- points routed to the detail level
        std::unique_ptr<Engine::Core::Buffer> m_partitionCount; // [0] baseCount, [1] detailCount

        std::unique_ptr<Engine::Core::ComputePipeline> kernel_clearCells;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_accumulate;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_classify;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_partition;
    };

} // namespace TSDF

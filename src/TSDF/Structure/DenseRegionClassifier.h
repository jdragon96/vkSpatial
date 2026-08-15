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
        // |sum(normal)| / pointCount: ~1 on a plane, lower on curvature and edges. A densely
        // scanned flat face has nothing to refine, and this is the only condition that sees that.
        float normalCoherence = 0.9f;
        // Samples per fine cell after refinement. Below this the refined voxels are noise.
        float samplesPerFineCell = 3.0f;
        // Absolute floor so a block glimpsed by a handful of points cannot latch to dense.
        uint32_t minimumFineOccupied = 64u;
    };

    // Per-block statistics, accumulated across frames. Mirrors the GPU-side layout exactly:
    // 9 tightly packed 4-byte scalars.
    struct BlockRecord {
        uint32_t blockKey;        // packed block coordinate; 0xFFFFFFFF = empty slot
        uint32_t pointCount;      // cumulative
        uint32_t coarseOccupied;  // cumulative sum of per-frame counts
        uint32_t fineOccupied;      // cumulative sum of per-frame counts
        uint32_t fineOccupiedFrame; // THIS frame's count; zeroed by the clear pass each frame
        uint32_t fineOccupiedMax;   // max over frames -- the detail table sizing input
        int32_t sumNormalX;         // fixed point, x10000
        int32_t sumNormalY;
        int32_t sumNormalZ;
    };
    static_assert(sizeof(BlockRecord) == 36, "BlockRecord must be 9 packed 4-byte scalars");

    // Decides which blocks earn a half-voxel detail level, and splits a frame's points into the
    // two levels. Owns only its own buffers; it knows nothing about any memory strategy, so it can
    // be built and tested on its own.
    class DenseRegionClassifier {
    public:
        void Build(Engine::Core::Context &context, float baseVoxel, int blockVoxels,
                   uint32_t maxPointPerFrame, const DensityCriteria &criteria = {});

        // Empties the block records. Cell hashes are per-frame and cleared by Record itself.
        void Reset();

        // Records the accumulate pass into `batch` without submitting. Grows the point buffers
        // when a frame exceeds the current capacity -- never clamps.
        void Record(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    Engine::Compute::CommandBatch &batch);

        // Applies the criteria to every block record. Latched: a dense block stays dense.
        void Classify(Engine::Compute::CommandBatch &batch);

        uint32_t DenseBlockCount() const;

        // Slots the detail table needs across every dense block, from per-frame maximum occupancy
        // rather than the cumulative count, which re-counts a cell once per frame.
        uint32_t DetailSlotEstimate() const;

        uint32_t BlockCount() const;

        // Points dropped because the block table's probing gave up before finding a slot. Zero in a
        // healthy run; non-zero means kBlockCapacity is too small for the scene. Mirrors how the TSDF
        // backends (e.g. AdvancedTSDF::InsertFailureCount) surface their own insert failures: a
        // counter the caller can observe rather than a silent drop.
        uint32_t BlockInsertFailureCount() const;

        // Occupied block records, for tests and diagnostics. Not a per-frame path.
        std::vector<BlockRecord> ReadBlocks() const;

    private:
        void growPointBuffers(uint32_t pointCount);

        Engine::Core::Context *m_context = nullptr;
        float m_baseVoxel = 0.01f;
        int m_blockVoxels = 32;
        uint32_t m_maxPointPerFrame = 0;
        DensityCriteria m_criteria;

        std::unique_ptr<Engine::Core::Buffer> m_pointBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_normalBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_blockRecords;
        std::unique_ptr<Engine::Core::Buffer> m_blockCount;
        std::unique_ptr<Engine::Core::Buffer> m_blockInsertFailureCount;
        std::unique_ptr<Engine::Core::Buffer> m_blockIndex;   // per point -> record index
        std::unique_ptr<Engine::Core::Buffer> m_fineCells;    // per-frame key-only hash
        std::unique_ptr<Engine::Core::Buffer> m_coarseCells;  // per-frame key-only hash
        std::unique_ptr<Engine::Core::Buffer> m_denseFlags;   // per-record latch, parallel to m_blockRecords
        std::unique_ptr<Engine::Core::Buffer> m_totals;       // [0] denseBlockCount, [1] detailSlots

        std::unique_ptr<Engine::Core::ComputePipeline> kernel_clearCells;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_accumulate;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_classify;
    };

} // namespace TSDF

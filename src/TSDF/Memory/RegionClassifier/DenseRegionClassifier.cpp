#include "TSDF/Memory/RegionClassifier/DenseRegionClassifier.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace TSDF {

    namespace {
        // Blocks are coarse (32 base voxels a side), so a frame touches thousands, not millions.
        // Kept well below 1<<14 because the cell key packs the block's record index in 14 bits.
        constexpr uint32_t kBlockCapacity = 1u << 13;
        // The fine cell key is (blockSlot << 18) | 18 bits of local cell, so the slot has exactly 14
        // bits. Slot 16383 with fine cell (63,63,63) packs to 0xFFFFFFFF, which IS EMPTY_KEY -- the
        // hash would read that cell as permanently empty and re-count it for every point in it. So
        // the last representable slot is unusable and the true ceiling is (1<<14) - 1, not 1<<14.
        // Whoever grows the block table next has to stop there, not at the round number.
        static_assert(kBlockCapacity <= (1u << 14) - 1u,
                      "block slot 16383 packs a fine cell key equal to EMPTY_KEY");
        // The fine cell key packs 6 bits per axis and the coarse key 5, so a block wider than 32
        // base voxels aliases distinct cells onto one key -- and for the fine key it aliases ACROSS
        // blocks, because the overflow runs into the slot field.
        constexpr int kMaxBlockVoxels = 32;
        // One entry per distinct fine cell in a frame. Sized against maxPointPerFrame because a
        // frame can never occupy more cells than it has points.
        constexpr float kCellCapacityFactor = 2.0f;

        struct AccumulatePC {
            uint32_t numPoints;
            uint32_t blockCapacity;
            uint32_t cellCapacity;
            float voxelSize;
            int32_t blockVoxels;
        };

        struct ClearPC {
            uint32_t cellCapacity;
            uint32_t blockCapacity;
        };

        struct ClassifyPC {
            uint32_t blockCapacity;
            float occupancyRatio;
            float normalCoherence;
            float samplesPerFineCell;
            uint32_t minimumFineOccupied;
        };

        struct PartitionPC {
            uint32_t numPoints;
        };
    } // namespace

    void DenseRegionClassifier::Build(Engine::Core::Context &context, float baseVoxel,
                                      int blockVoxels, uint32_t maxPointPerFrame,
                                      const DensityCriteria &criteria) {
        // Rejected rather than clamped: a wider block silently aliases distinct cells onto one key,
        // so the occupancy under-reports and the verdicts change with no symptom at all. Nothing
        // needs the flexibility today. A throw, not an assert, because the project builds with
        // NDEBUG -- an assert here would be a no-op in exactly the configuration that ships.
        if (blockVoxels < 1 || blockVoxels > kMaxBlockVoxels)
            throw std::runtime_error("DenseRegionClassifier::Build: blockVoxels must be in [1, 32]");

        m_context = &context;
        m_baseVoxel = baseVoxel;
        m_blockVoxels = blockVoxels;
        m_criteria = criteria;

        // Dropped BEFORE m_maxPointPerFrame is assigned, so growPointBuffers() below cannot take its
        // early return. Without this, a second Build() with a LARGER hint set the member to the new
        // size and then early-returned out of the grow (the member already satisfied the test),
        // leaving the four point-sized buffers at the OLD capacity while the member claimed the new
        // one -- every later Record() then memcpy'd past a mapped allocation and partition wrote its
        // index lists past the same ceiling. Measured: a 4096-point frame after Build(1024) then
        // Build(32768) reported 2,223,218,252 accumulated points instead of 4096.
        m_pointBuffer.reset();
        m_maxPointPerFrame = maxPointPerFrame;

        m_blockRecords = std::make_unique<Engine::Core::Buffer>(context);
        m_blockCount = std::make_unique<Engine::Core::Buffer>(context);
        m_blockInsertFailureCount = std::make_unique<Engine::Core::Buffer>(context);
        m_cellInsertFailureCount = std::make_unique<Engine::Core::Buffer>(context);

        m_blockRecords->AllocateHostVisibleReadback(kBlockCapacity * sizeof(BlockRecord));
        m_blockCount->AllocateHostVisibleReadback(sizeof(uint32_t));
        m_blockInsertFailureCount->AllocateHostVisibleReadback(sizeof(uint32_t));
        m_cellInsertFailureCount->AllocateHostVisibleReadback(sizeof(uint32_t));

        // m_blockIndex and the two cell hashes are NOT allocated here. growPointBuffers owns them --
        // they are all sized off the point count, and it is the one place that count changes -- so
        // allocating them here too would only mean three allocate/destroy round-trips per Build.
        growPointBuffers(maxPointPerFrame);

        kernel_clearCells = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_clearCells->Build("TSDF/Memory/RegionClassifier/kernel_DenseRegionClassifier.clear.comp.glsl")
                .Bind(0, *m_fineCells)
                .Bind(1, *m_coarseCells)
                .Bind(2, *m_blockRecords);

        kernel_accumulate = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_accumulate->Build("TSDF/Memory/RegionClassifier/kernel_DenseRegionClassifier.accumulate.comp.glsl")
                .Bind(2, *m_blockRecords)
                .Bind(3, *m_blockCount)
                .Bind(4, *m_blockIndex)
                .Bind(5, *m_fineCells)
                .Bind(6, *m_coarseCells)
                .Bind(7, *m_blockInsertFailureCount)
                .Bind(8, *m_cellInsertFailureCount);

        m_denseFlags = std::make_unique<Engine::Core::Buffer>(context);
        m_totals = std::make_unique<Engine::Core::Buffer>(context);
        m_denseFlags->Allocate(kBlockCapacity * sizeof(uint32_t));
        m_totals->AllocateHostVisibleReadback(2u * sizeof(uint32_t));

        kernel_classify = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_classify->Build("TSDF/Memory/RegionClassifier/kernel_DenseRegionClassifier.classify.comp.glsl")
                .Bind(0, *m_blockRecords)
                .Bind(1, *m_denseFlags)
                .Bind(2, *m_totals);

        m_partitionCount = std::make_unique<Engine::Core::Buffer>(context);
        m_partitionCount->AllocateHostVisibleReadback(2u * sizeof(uint32_t));

        // m_baseIndex/m_detailIndex were already (re)created by the growPointBuffers() call above,
        // so they and m_blockIndex are all valid to bind here.
        kernel_partition = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_partition->Build("TSDF/Memory/RegionClassifier/kernel_DenseRegionClassifier.partition.comp.glsl")
                .Bind(0, *m_blockIndex)
                .Bind(1, *m_denseFlags)
                .Bind(2, *m_baseIndex)
                .Bind(3, *m_detailIndex)
                .Bind(4, *m_partitionCount);

        Reset();
    }

    void DenseRegionClassifier::growPointBuffers(uint32_t pointCount) {
        if (m_pointBuffer && pointCount <= m_maxPointPerFrame) return;
        // 1.5x slack so a stream of similar frames reallocates once, not every call. Growing
        // rather than clamping is deliberate: a clamp drops points with no symptom.
        const uint32_t grown = pointCount + pointCount / 2u;
        m_pointBuffer = std::make_unique<Engine::Core::Buffer>(*m_context);
        m_normalBuffer = std::make_unique<Engine::Core::Buffer>(*m_context);
        m_pointBuffer->AllocateHostVisible(grown * 3u * sizeof(float));
        m_normalBuffer->AllocateHostVisible(grown * 3u * sizeof(float));
        m_blockIndex = std::make_unique<Engine::Core::Buffer>(*m_context);
        m_blockIndex->Allocate(grown * sizeof(uint32_t));

        // The partition output lists are sized 1:1 with the point buffers -- a frame can never
        // route more points into base+detail combined than it has points -- so, like m_blockIndex,
        // they must be recreated (not just resized) here and rebound below whenever this function
        // actually grows. Host-visible readback because ReadPartition() copies them back to the CPU.
        m_baseIndex = std::make_unique<Engine::Core::Buffer>(*m_context);
        m_detailIndex = std::make_unique<Engine::Core::Buffer>(*m_context);
        m_baseIndex->AllocateHostVisibleReadback(grown * sizeof(uint32_t));
        m_detailIndex->AllocateHostVisibleReadback(grown * sizeof(uint32_t));

        // The cell hashes are sized off maxPointPerFrame too (a frame can never occupy more cells
        // than it has points), so they must grow in lockstep with it, here, in the one place
        // m_maxPointPerFrame itself changes. Previously they were sized once in Build() from the
        // raw, pre-grow parameter, while Record() kept recomputing cellCapacity from the (now
        // inflated) member -- the two could silently drift apart the moment this function's own
        // 1.5x growth ran, handing the kernels a cellCapacity past the buffers' true allocation.
        const uint32_t cellCapacity = uint32_t(float(grown) * kCellCapacityFactor);
        m_fineCells = std::make_unique<Engine::Core::Buffer>(*m_context);
        m_coarseCells = std::make_unique<Engine::Core::Buffer>(*m_context);
        m_fineCells->Allocate(cellCapacity * sizeof(uint32_t));
        m_coarseCells->Allocate(cellCapacity * sizeof(uint32_t));

        m_maxPointPerFrame = grown;
        if (kernel_accumulate)
            kernel_accumulate->Bind(4, *m_blockIndex).Bind(5, *m_fineCells).Bind(6, *m_coarseCells);
        if (kernel_clearCells) kernel_clearCells->Bind(0, *m_fineCells).Bind(1, *m_coarseCells);
        // m_blockIndex was just recreated above too, so kernel_partition's binding to it (as well
        // as to the two output lists) needs refreshing along with everything else.
        if (kernel_partition)
            kernel_partition->Bind(0, *m_blockIndex).Bind(2, *m_baseIndex).Bind(3, *m_detailIndex);
    }

    void DenseRegionClassifier::Reset() {
        // Guarded like every other public method: Reset() is reachable on a default-constructed
        // classifier (`DenseRegionClassifier c; c.Reset();`), and every dereference below would
        // otherwise be a null one.
        if (!m_context || !m_blockRecords || !m_totals || !m_partitionCount) return;

        // The most-recently-recorded-frame state must not survive a scene reset either, for the
        // same reason Record() invalidates it up front: a stale count would have Partition()
        // replay whatever frame was last recorded before this Reset().
        m_recordedPointCount = 0;

        // Empty every block slot on the host: the record buffer is host-visible and this runs once
        // per scene, not per frame.
        auto *records = static_cast<BlockRecord *>(m_blockRecords->MappedPtr());
        std::memset(records, 0, kBlockCapacity * sizeof(BlockRecord));
        for (uint32_t i = 0; i < kBlockCapacity; ++i) records[i].blockKey = 0xFFFFFFFFu;
        m_blockRecords->MakeVisibleToGPU(kBlockCapacity * sizeof(BlockRecord));
        *static_cast<uint32_t *>(m_blockCount->MappedPtr()) = 0;
        m_blockCount->MakeVisibleToGPU(sizeof(uint32_t));
        *static_cast<uint32_t *>(m_blockInsertFailureCount->MappedPtr()) = 0;
        m_blockInsertFailureCount->MakeVisibleToGPU(sizeof(uint32_t));
        *static_cast<uint32_t *>(m_cellInsertFailureCount->MappedPtr()) = 0;
        m_cellInsertFailureCount->MakeVisibleToGPU(sizeof(uint32_t));

        // Every readback the accessors serve has to be reset too, not just the records they are
        // derived from. These two buffers are the accessors' ONLY storage -- DenseBlockCount(),
        // DetailSlotEstimate() and ReadPartition() read them straight back with no recomputation --
        // so leaving them resident would have `Reset(); DenseBlockCount();` answer with the
        // pre-Reset count and ReadPartition() hand back the pre-Reset frame's index lists.
        auto *totals = static_cast<uint32_t *>(m_totals->MappedPtr());
        totals[0] = 0;
        totals[1] = 0;
        m_totals->MakeVisibleToGPU(2u * sizeof(uint32_t));
        auto *partitionCounts = static_cast<uint32_t *>(m_partitionCount->MappedPtr());
        partitionCounts[0] = 0;
        partitionCounts[1] = 0;
        m_partitionCount->MakeVisibleToGPU(2u * sizeof(uint32_t));

        // The dense latch must not survive across scenes. m_denseFlags is device-local (Allocate,
        // not AllocateHostVisible*), so it has no mapped pointer to memset from the host -- a
        // one-shot GPU-side fill is the only way to zero it here. This is also why Reset(), and
        // therefore Build(), costs a queue submit and a full wait; see the header.
        Engine::Compute::CommandBatch batch(*m_context);
        batch.FillBuffer(m_denseFlags->Handle(), 0, VK_WHOLE_SIZE, 0u);
        batch.Submit();
    }

    void DenseRegionClassifier::Record(const std::vector<Eigen::Vector3f> &points,
                                       const std::vector<Eigen::Vector3f> &normals,
                                       Engine::Compute::CommandBatch &batch) {
        // Set first, above the throw and the early return below: Record/Classify/Partition is a
        // per-frame sequence and this count is that sequence's state, so an empty or malformed frame
        // must invalidate it right here, where the frame is decided. Leaving the previous frame's
        // count resident would have Partition() dispatch over -- and ReadPartition() hand back --
        // the previous frame's still-resident m_blockIndex, silently duplicating it into the
        // caller's integration.
        m_recordedPointCount = 0;
        // A mismatch used to fall back to min(points, normals), which is a silent truncation of the
        // caller's frame -- points past the shorter array vanish from the reconstruction with no
        // counter and no error. There is no correct count to pick, so this is the caller's bug to
        // hear about. Same wording, and same choice, as DirectionalTSDF::Integrate.
        if (points.size() != normals.size())
            throw std::runtime_error("DenseRegionClassifier::Record: points/normals size mismatch");
        if (!m_context || points.empty()) return;
        const uint32_t n = uint32_t(points.size());
        growPointBuffers(n);
        // Partition() has no points parameter of its own -- it repartitions THIS frame, so it reads
        // the count back from here rather than the caller passing it twice.
        m_recordedPointCount = n;

        std::memcpy(m_pointBuffer->MappedPtr(), points.data(), n * 3u * sizeof(float));
        std::memcpy(m_normalBuffer->MappedPtr(), normals.data(), n * 3u * sizeof(float));
        m_pointBuffer->MakeVisibleToGPU(n * 3u * sizeof(float));
        m_normalBuffer->MakeVisibleToGPU(n * 3u * sizeof(float));

        const uint32_t cellCapacity =
                uint32_t(float(m_maxPointPerFrame) * kCellCapacityFactor);

        // The cell hashes hold one frame's occupancy, so they are emptied at the head of every
        // frame rather than at Reset.
        kernel_clearCells->Args(ClearPC{cellCapacity, kBlockCapacity});
        batch.DispatchElements(*kernel_clearCells, std::max(cellCapacity, kBlockCapacity));

        // Orders the accumulate dispatch after the clear dispatch's writes to the cell hashes and
        // fineOccupiedFrame: Vulkan gives no ordering guarantee between two dispatches recorded back
        // to back, so without this the accumulate kernel could read a stale or partially-cleared
        // hash from a previous frame.
        batch.Barrier();

        kernel_accumulate->Bind(0, *m_pointBuffer).Bind(1, *m_normalBuffer);
        kernel_accumulate->Args(AccumulatePC{n, kBlockCapacity, cellCapacity, m_baseVoxel,
                                             int32_t(m_blockVoxels)});
        batch.DispatchElements(*kernel_accumulate, n);
    }

    void DenseRegionClassifier::Classify(Engine::Compute::CommandBatch &batch) {
        if (!m_context) return;
        // The two totals are recomputed from scratch every call, not accumulated across calls. Safe
        // to reset unconditionally on the host: CommandBatch already forbids dispatching the same
        // ComputePipeline twice in one batch, so Classify() can never run twice against a
        // not-yet-submitted dispatch of its own kernel_classify.
        auto *totals = static_cast<uint32_t *>(m_totals->MappedPtr());
        totals[0] = 0;
        totals[1] = 0;
        m_totals->MakeVisibleToGPU(2u * sizeof(uint32_t));

        // Orders the classify dispatch after Record()'s accumulate dispatch: classify reads the
        // block records accumulate just wrote, and Vulkan gives no ordering guarantee between two
        // dispatches recorded back to back on the same command buffer.
        batch.Barrier();

        kernel_classify->Args(ClassifyPC{kBlockCapacity, m_criteria.occupancyRatio,
                                         m_criteria.normalCoherence, m_criteria.samplesPerFineCell,
                                         m_criteria.minimumFineOccupied});
        batch.DispatchElements(*kernel_classify, kBlockCapacity);
    }

    void DenseRegionClassifier::Partition(Engine::Compute::CommandBatch &batch) {
        if (!m_context) return;
        // The two counts are this frame's partition sizes, recomputed from scratch every call --
        // not accumulated across calls -- matching how Classify() resets m_totals. Safe to reset
        // unconditionally on the host: CommandBatch already forbids dispatching the same
        // ComputePipeline twice in one batch, so Partition() can never run twice against a
        // not-yet-submitted dispatch of its own kernel_partition.
        auto *counts = static_cast<uint32_t *>(m_partitionCount->MappedPtr());
        counts[0] = 0;
        counts[1] = 0;
        m_partitionCount->MakeVisibleToGPU(2u * sizeof(uint32_t));

        // Orders the partition dispatch after Classify()'s writes to m_denseFlags (and, further
        // back in the same command buffer, Record()'s writes to m_blockIndex): Vulkan gives no
        // ordering guarantee between two dispatches recorded back to back.
        batch.Barrier();

        kernel_partition->Args(PartitionPC{m_recordedPointCount});
        batch.DispatchElements(*kernel_partition, m_recordedPointCount);
    }

    void DenseRegionClassifier::ReadPartition(std::vector<uint32_t> &base,
                                              std::vector<uint32_t> &detail) const {
        base.clear();
        detail.clear();
        if (!m_partitionCount) return;

        m_partitionCount->MakeVisibleToCPU(2u * sizeof(uint32_t));
        const auto *counts = static_cast<const uint32_t *>(m_partitionCount->MappedPtr());
        const uint32_t baseCount = counts[0];
        const uint32_t detailCount = counts[1];

        // Copy back only the counts the GPU actually wrote, never the buffers' full capacity.
        m_baseIndex->MakeVisibleToCPU(baseCount * sizeof(uint32_t));
        m_detailIndex->MakeVisibleToCPU(detailCount * sizeof(uint32_t));
        const auto *baseData = static_cast<const uint32_t *>(m_baseIndex->MappedPtr());
        const auto *detailData = static_cast<const uint32_t *>(m_detailIndex->MappedPtr());
        base.assign(baseData, baseData + baseCount);
        detail.assign(detailData, detailData + detailCount);
    }

    uint32_t DenseRegionClassifier::DenseBlockCount() const {
        if (!m_totals) return 0;
        m_totals->MakeVisibleToCPU(2u * sizeof(uint32_t));
        return static_cast<const uint32_t *>(m_totals->MappedPtr())[0];
    }

    uint32_t DenseRegionClassifier::DetailSlotEstimate() const {
        if (!m_totals) return 0;
        m_totals->MakeVisibleToCPU(2u * sizeof(uint32_t));
        return static_cast<const uint32_t *>(m_totals->MappedPtr())[1];
    }

    uint32_t DenseRegionClassifier::BlockCount() const {
        if (!m_blockCount) return 0;
        m_blockCount->MakeVisibleToCPU(sizeof(uint32_t));
        return *static_cast<const uint32_t *>(m_blockCount->MappedPtr());
    }

    uint32_t DenseRegionClassifier::BlockInsertFailureCount() const {
        if (!m_blockInsertFailureCount) return 0;
        m_blockInsertFailureCount->MakeVisibleToCPU(sizeof(uint32_t));
        return *static_cast<const uint32_t *>(m_blockInsertFailureCount->MappedPtr());
    }

    uint32_t DenseRegionClassifier::CellInsertFailureCount() const {
        if (!m_cellInsertFailureCount) return 0;
        m_cellInsertFailureCount->MakeVisibleToCPU(sizeof(uint32_t));
        return *static_cast<const uint32_t *>(m_cellInsertFailureCount->MappedPtr());
    }

    std::vector<BlockRecord> DenseRegionClassifier::ReadBlocks() const {
        std::vector<BlockRecord> out;
        if (!m_blockRecords) return out;
        m_blockRecords->MakeVisibleToCPU(kBlockCapacity * sizeof(BlockRecord));
        const auto *records = static_cast<const BlockRecord *>(m_blockRecords->MappedPtr());
        for (uint32_t i = 0; i < kBlockCapacity; ++i)
            if (records[i].blockKey != 0xFFFFFFFFu) out.push_back(records[i]);
        return out;
    }

} // namespace TSDF

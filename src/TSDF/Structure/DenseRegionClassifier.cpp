#include "TSDF/Structure/DenseRegionClassifier.h"

#include <algorithm>
#include <cstring>

namespace TSDF {

    namespace {
        // Blocks are coarse (32 base voxels a side), so a frame touches thousands, not millions.
        // Kept well below 1<<14 because the cell key packs the block's record index in 14 bits.
        constexpr uint32_t kBlockCapacity = 1u << 13;
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
    } // namespace

    void DenseRegionClassifier::Build(Engine::Core::Context &context, float baseVoxel,
                                      int blockVoxels, uint32_t maxPointPerFrame,
                                      const DensityCriteria &criteria) {
        m_context = &context;
        m_baseVoxel = baseVoxel;
        m_blockVoxels = blockVoxels;
        m_maxPointPerFrame = maxPointPerFrame;
        m_criteria = criteria;

        const uint32_t cellCapacity = uint32_t(float(maxPointPerFrame) * kCellCapacityFactor);

        m_blockRecords = std::make_unique<Engine::Core::Buffer>(context);
        m_blockCount = std::make_unique<Engine::Core::Buffer>(context);
        m_blockInsertFailureCount = std::make_unique<Engine::Core::Buffer>(context);
        m_fineCells = std::make_unique<Engine::Core::Buffer>(context);
        m_coarseCells = std::make_unique<Engine::Core::Buffer>(context);
        m_blockIndex = std::make_unique<Engine::Core::Buffer>(context);

        m_blockRecords->AllocateHostVisibleReadback(kBlockCapacity * sizeof(BlockRecord));
        m_blockCount->AllocateHostVisibleReadback(sizeof(uint32_t));
        m_blockInsertFailureCount->AllocateHostVisibleReadback(sizeof(uint32_t));
        m_fineCells->Allocate(cellCapacity * sizeof(uint32_t));
        m_coarseCells->Allocate(cellCapacity * sizeof(uint32_t));
        m_blockIndex->Allocate(maxPointPerFrame * sizeof(uint32_t));

        growPointBuffers(maxPointPerFrame);

        kernel_clearCells = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_clearCells->Build("TSDF/Structure/DenseRegionClassifier.clear.comp.glsl")
                .Bind(0, *m_fineCells)
                .Bind(1, *m_coarseCells)
                .Bind(2, *m_blockRecords);

        kernel_accumulate = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_accumulate->Build("TSDF/Structure/DenseRegionClassifier.accumulate.comp.glsl")
                .Bind(2, *m_blockRecords)
                .Bind(3, *m_blockCount)
                .Bind(4, *m_blockIndex)
                .Bind(5, *m_fineCells)
                .Bind(6, *m_coarseCells)
                .Bind(7, *m_blockInsertFailureCount);

        m_denseFlags = std::make_unique<Engine::Core::Buffer>(context);
        m_totals = std::make_unique<Engine::Core::Buffer>(context);
        m_denseFlags->Allocate(kBlockCapacity * sizeof(uint32_t));
        m_totals->AllocateHostVisibleReadback(2u * sizeof(uint32_t));

        kernel_classify = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_classify->Build("TSDF/Structure/DenseRegionClassifier.classify.comp.glsl")
                .Bind(0, *m_blockRecords)
                .Bind(1, *m_denseFlags)
                .Bind(2, *m_totals);

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
    }

    void DenseRegionClassifier::Reset() {
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

        // The dense latch must not survive across scenes. m_denseFlags is device-local (Allocate,
        // not AllocateHostVisible*), so it has no mapped pointer to memset from the host -- a
        // one-shot GPU-side fill is the only way to zero it here.
        Engine::Compute::CommandBatch batch(*m_context);
        batch.FillBuffer(m_denseFlags->Handle(), 0, VK_WHOLE_SIZE, 0u);
        batch.Submit();
    }

    void DenseRegionClassifier::Record(const std::vector<Eigen::Vector3f> &points,
                                       const std::vector<Eigen::Vector3f> &normals,
                                       Engine::Compute::CommandBatch &batch) {
        if (!m_context || points.empty()) return;
        const uint32_t n =
                std::min(uint32_t(points.size()), uint32_t(normals.size()));
        if (n == 0) return;
        growPointBuffers(n);

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

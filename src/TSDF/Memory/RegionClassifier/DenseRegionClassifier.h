#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <vector>

#include "TSDF/Memory/DataSplitter.h"

namespace TSDF {
    struct DensityCriteria {
        float occupancyRatio = 3.2f;
        float normalCoherence = 0.9f;
        float samplesPerFineCell = 3.0f;
        uint32_t minimumFineOccupied = 64u;
    };


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

    class DenseRegionClassifier {
    public:
        void Build(Engine::Core::Context &context,
                   float baseVoxel,
                   int blockVoxels,
                   uint32_t maxPointPerFrame,
                   const DensityCriteria &criteria = {});

        void Reset();

        void Record(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    Engine::Compute::CommandBatch &batch);

        void Classify(Engine::Compute::CommandBatch &batch);

        void Partition(Engine::Compute::CommandBatch &batch);

        void ReadPartition(
                std::vector<uint32_t> &base,
                std::vector<uint32_t> &detail) const;

        uint32_t DenseBlockCount() const;

        uint32_t DetailSlotEstimate() const;

        uint32_t BlockCount() const;

        uint32_t BlockInsertFailureCount() const;

        uint32_t CellInsertFailureCount() const;

        std::vector<BlockRecord> ReadBlocks() const;

    private:
        void growPointBuffers(uint32_t pointCount);

        Engine::Core::Context *m_context = nullptr;
        float m_baseVoxel = 0.01f;
        int m_blockVoxels = 32;
        uint32_t m_maxPointPerFrame = 0;
        uint32_t m_recordedPointCount = 0;
        DensityCriteria m_criteria;

        std::unique_ptr<Engine::Core::Buffer> m_pointBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_normalBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_blockRecords;
        std::unique_ptr<Engine::Core::Buffer> m_blockCount;
        std::unique_ptr<Engine::Core::Buffer> m_blockInsertFailureCount;
        std::unique_ptr<Engine::Core::Buffer> m_cellInsertFailureCount;
        std::unique_ptr<Engine::Core::Buffer> m_blockIndex;     // per point -> record index
        std::unique_ptr<Engine::Core::Buffer> m_fineCells;      // per-frame key-only hash
        std::unique_ptr<Engine::Core::Buffer> m_coarseCells;    // per-frame key-only hash
        std::unique_ptr<Engine::Core::Buffer> m_denseFlags;     // per-record latch, parallel to m_blockRecords
        std::unique_ptr<Engine::Core::Buffer> m_totals;         // [0] denseBlockCount, [1] detailSlots
        std::unique_ptr<Engine::Core::Buffer> m_baseIndex;      // per frame -- points routed to the base level
        std::unique_ptr<Engine::Core::Buffer> m_detailIndex;    // per frame -- points routed to the detail level
        std::unique_ptr<Engine::Core::Buffer> m_partitionCount; // [0] baseCount, [1] detailCount

        std::unique_ptr<Engine::Core::ComputePipeline> kernel_clearCells;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_accumulate;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_classify;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_partition;
    };

    class DenseRegionStrategy final : public DataSplitter {
    public:
        void Build(Engine::Core::Context &context, const DataSplitterConfig &config) override {
            m_context = &context;
            m_classifier.Build(context, config.baseResolution, config.blockVoxels,
                               uint32_t(config.maxPointPerFrame));
        }

        void Reset() override { m_classifier.Reset(); }

        void DividePoint(const std::vector<Eigen::Vector3f> &points,
                         const std::vector<Eigen::Vector3f> &normals,
                         DividePointOutput &output) override {
            output.Clear();
            if (!m_context || points.empty()) return;
            if (points.size() != normals.size())
                throw std::runtime_error("DataSplitter::DividePoint: points/normals size mismatch");

            // The three passes are one frame and must share one batch, submitted before the next.
            Engine::Compute::CommandBatch batch(*m_context);
            m_classifier.Record(points, normals, batch);
            m_classifier.Classify(batch);
            m_classifier.Partition(batch);
            batch.Submit();

            m_classifier.ReadPartition(output.baseIndex, output.detailIndex);
            output.denseBlockCount = m_classifier.DenseBlockCount();
            output.detailSlotEstimate = m_classifier.DetailSlotEstimate();
            output.blockInsertFailureCount = m_classifier.BlockInsertFailureCount();
            output.cellInsertFailureCount = m_classifier.CellInsertFailureCount();
        }

        const char *Name() const override { return "dense"; }

    private:
        Engine::Core::Context *m_context = nullptr;
        TSDF::DenseRegionClassifier m_classifier;
    };
} // namespace TSDF

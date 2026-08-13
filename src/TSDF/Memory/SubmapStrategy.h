#pragma once

#include "Engine/Spatial/SubmapAdvancedTSDF.h"
#include "TSDF/Memory/MemoryStrategy.h"

namespace TSDF {

    // Two levels of tiled storage: a base grid plus half-voxel detail submaps over blocks the scan
    // covers densely. Costs a second full tile hierarchy, so it is the strategy where tableCount
    // and slotCapacity diverge most from occupancy -- exactly the case the A/B run is for.
    //
    // Contract deviation: the wrapped SubmapAdvancedTSDF::Integrate opens and submits its own
    // CommandBatch, so Record puts NOTHING into the caller's batch and completes the work
    // immediately. Correct, but not fused with other volumes in one submit.
    class SubmapStrategy final : public MemoryStrategy {
    public:
        void Build(Engine::Core::Context &context, const VolumeParams &params) override;
        void Reset() override;
        void Configure(const IntegrationOptions &options) override;

        void Record(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    const Eigen::Vector3f &cameraPosition,
                    Engine::Compute::CommandBatch &batch) override;

        void Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const override;

        VolumeStats Stats() const override;
        const char *Name() const override { return "submap"; }
        Engine::Core::Context *Device() const override { return m_context; }

    private:
        static constexpr int kBlockVoxels = 32;
        static constexpr float kDetailPointsPerVoxel = 4.0f;

        Engine::Core::Context *m_context = nullptr;
        Engine::Spatial::SubmapAdvancedTSDF m_tsdf;
    };

} // namespace TSDF

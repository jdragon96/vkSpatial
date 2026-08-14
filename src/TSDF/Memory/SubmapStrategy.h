#pragma once

#include "TSDF/Backends/SubmapAdvancedTSDF.h"
#include "TSDF/Memory/MemoryStrategy.h"

namespace TSDF {

    // Two levels of tiled storage: a base grid plus half-voxel detail submaps over blocks the scan
    // covers densely. Costs a second full tile hierarchy, so it is the strategy where tableCount
    // and slotCapacity diverge most from occupancy -- exactly the case the A/B run is for.
    //
    // Contract deviation: the wrapped SubmapAdvancedTSDF::IntegrateGPU opens and submits its own
    // CommandBatch, so Record puts NOTHING into the caller's batch and completes the work
    // immediately. Correct, but not fused with other volumes in one submit.
    //
    // Cost asymmetry vs TileStrategy: submap's non-clamping path uploads the WHOLE cloud once and
    // dispatches every affected tile over it, letting each tile's shader-side window filter discard
    // the points that miss it -- O(tiles x points) of GPU work. TileStrategy instead routes on the
    // CPU first and uploads only each tile's share, staying O(points). Both drop nothing; submap
    // simply pays more per frame for it, which matters when reading timings rather than memory.
    //
    // CONFIGURE ORDER: call Configure BEFORE the first integration. Both levels are lazily tiled
    // and a tile copies the current settings only when it is created, so a mid-run Configure
    // reaches neither the base nor the detail tiles that already exist (currentFrame excepted).
    //
    // occupiedEntryCount counts STORED entries across base AND detail, while Download drops base
    // entries that fall inside a dense block (detail wins there) -- so for submap
    // occupiedEntryCount != entries.size(). It is honest as a storage cost, which is what the
    // memory comparison needs, but it must not be read as a count of surface voxels.
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

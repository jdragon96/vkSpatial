#pragma once

#include "TSDF/Backends/TiledAdvancedTSDF.h"
#include "TSDF/Memory/MemoryStrategy.h"

namespace TSDF {

    // Lazily created 448^3-core tiles, so the scene is not capped by one 512^3 window. Every tile
    // owns its own hash of VolumeParams::hashCapacity slots, which is why tableCount matters:
    // a scene spread over many sparse tiles is memory-bound by tile COUNT, not by load factor.
    //
    // Record routes the cloud to tiles first and uploads only each tile's share, so its GPU work
    // is O(points) regardless of tile count. (SubmapStrategy takes the opposite trade -- see
    // SubmapStrategy.h.)
    //
    // CONFIGURE ORDER: call Configure BEFORE the first integration. Tiles are created lazily and
    // copy the current settings at creation time, and only currentFrame is pushed to tiles that
    // already exist -- so a mid-run Configure leaves older tiles on the OLD options while newer
    // tiles get the new ones, silently mixing two configurations inside one volume's numbers.
    class TileStrategy final : public MemoryStrategy {
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
        const char *Name() const override { return "tile"; }
        Engine::Core::Context *Device() const override { return m_context; }

    private:
        Engine::Core::Context *m_context = nullptr;
        Engine::Spatial::TiledAdvancedTSDF m_tsdf;
    };

} // namespace TSDF

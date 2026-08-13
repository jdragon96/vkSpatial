#pragma once

#include "Engine/Spatial/TiledAdvancedTSDF.h"
#include "TSDF/Memory/MemoryStrategy.h"

namespace TSDF {

    // Lazily created 448^3-core tiles, so the scene is not capped by one 512^3 window. Every tile
    // owns its own hash of VolumeParams::hashCapacity slots, which is why tableCount matters:
    // a scene spread over many sparse tiles is memory-bound by tile COUNT, not by load factor.
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

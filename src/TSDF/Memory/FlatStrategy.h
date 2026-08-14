#pragma once

#include "TSDF/Backends/AdvancedTSDF.h"
#include "TSDF/Memory/MemoryStrategy.h"

namespace TSDF {

    // One fixed 512^3 voxel window. The simplest strategy and the A/B baseline: whatever a tiled
    // or submap layout gains, it gains relative to this. Scenes larger than one window are clipped
    // by the kernel's bounds check rather than growing, which is exactly what makes it a baseline.
    class FlatStrategy final : public MemoryStrategy {
    public:
        void Build(Engine::Core::Context &context, const VolumeParams &params) override;
        void Reset() override;
        void Configure(const IntegrationOptions &options) override;

        void Record(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    const Eigen::Vector3f &cameraPosition,
                    Engine::Compute::CommandBatch &batch) override;

        void Download(std::vector<TSDF::AdvancedEntry> &out) const override;

        VolumeStats Stats() const override;
        const char *Name() const override { return "flat"; }
        Engine::Core::Context *Device() const override { return m_context; }

    private:
        Engine::Core::Context *m_context = nullptr;
        TSDF::AdvancedTSDF m_tsdf;
    };

} // namespace TSDF

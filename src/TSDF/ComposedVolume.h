#pragma once

#include "TSDF/Memory/MemoryStrategy.h"

#include <memory>

namespace TSDF {

    // A Volume assembled from swappable strategies. Today it holds only the Memory axis and
    // forwards to it; the Integrate and Extract axes drop in here without touching the memory
    // strategies, which is the reason this pass-through exists now rather than later.
    class ComposedVolume final : public Volume {
    public:
        explicit ComposedVolume(std::unique_ptr<MemoryStrategy> memory)
            : m_memory(std::move(memory)) {}

        void Build(Engine::Core::Context &context, const VolumeParams &params) override {
            m_memory->Build(context, params);
        }

        void Reset() override { m_memory->Reset(); }

        void Configure(const IntegrationOptions &options) override {
            m_memory->Configure(options);
        }

        void Record(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    const Eigen::Vector3f &cameraPosition,
                    Engine::Compute::CommandBatch &batch) override {
            m_memory->Record(points, normals, cameraPosition, batch);
        }

        void Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const override {
            m_memory->Download(out);
        }

        VolumeStats Stats() const override { return m_memory->Stats(); }

        // The registered name is the memory strategy's name while it is the only axis.
        const char *Name() const override { return m_memory->Name(); }

    protected:
        Engine::Core::Context *Device() const override { return m_memory->Device(); }

    private:
        std::unique_ptr<MemoryStrategy> m_memory;
    };

} // namespace TSDF

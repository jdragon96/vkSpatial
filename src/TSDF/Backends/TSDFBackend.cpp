#include "TSDF/Backends/TSDFBackend.h"

#include <limits>

#include "Engine/Core/Context.h"
#include "TSDF/Backends/AdvancedTSDF.h"
#include "TSDF/Hash/HashStrategy.h"

namespace {

    constexpr uint32_t kExtractCandidates = 1u << 19;

    class AdvancedBackend final : public TSDFBackend {
    public:
        void Build(Engine::Core::Context &context, const TSDFBackendConfig &config) override {
            const Eigen::Vector3f corner =
                    config.hasWindowMinCorner
                            ? config.windowMinCorner
                            : Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());

            m_tsdf.SetProbeStats(config.probeStats);
            m_tsdf.Build(context, config.voxelSize, config.truncation, config.hashCapacity,
                         config.maxPointPerFrame, corner, TSDF::HashStrategyByName(config.hash));
            m_tsdf.SetPointToPlane(config.pointToPlane);
            m_tsdf.SetHermitePosition(config.hermitePosition);
            m_tsdf.SetConfidenceWeight(config.confidenceWeight);
            m_tsdf.SetIntegrationQuality(
                    {config.maxDirections, config.directionExponent, config.viewAngleWeight});
        }

        void Reset() override { m_tsdf.Reset(); }

        void SetFrameIndex(int frame) override { m_tsdf.SetCurrentFrame(frame); }

        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPosition) override {
            m_tsdf.IntegrateGPU(points, normals, cameraPosition);
        }

        Engine::Core::OrientedPointCloud Extract() const override {
            return m_tsdf.ExtractPointCloud(kExtractCandidates, true);
        }

        void Download(std::vector<TSDFVoxel> &out) const override {
            for (const TSDF::AdvancedEntry &entry: m_tsdf.DownloadEntries())
                out.push_back({entry.center, entry.normal, entry.direction, entry.tsdf, entry.weight,
                               entry.firstFrame});
        }

        TSDFBackendStats Stats() const override {
            TSDFBackendStats stats;
            stats.filledCount = m_tsdf.FilledCount();
            stats.hashCapacity = m_tsdf.HashCapacity();
            stats.insertFailureCount = m_tsdf.InsertFailureCount();
            stats.growCount = m_tsdf.GrowCount();
            stats.deviceMemoryBytes = uint64_t(m_tsdf.HashCapacity()) * sizeof(TSDF::AdvDirEntry);
            stats.tableCount = 1;
            m_tsdf.ProbeStats(stats.probeSlotTotal, stats.probeQueryCount, stats.probeSlotMax);
            return stats;
        }

        const char *Name() const override { return "advanced"; }

    private:
        TSDF::AdvancedTSDF m_tsdf;
    };

} // namespace

std::unique_ptr<TSDFBackend> MakeTSDFBackend(const std::string &name) {
    if (name == "advanced") return std::make_unique<AdvancedBackend>();
    return nullptr;
}

std::vector<std::string> TSDFBackendNames() {
    return {"advanced"};
}

std::vector<std::string> TSDFHashNames() {
    return {"linear", "bucketed"};
}

std::string ResolveHashName(const std::string &name) {
    return TSDF::HashStrategyByName(name).name;
}

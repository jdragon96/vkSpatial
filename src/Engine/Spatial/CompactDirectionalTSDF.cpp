#include "Engine/Spatial/CompactDirectionalTSDF.h"

#include <algorithm>

namespace Engine::Spatial {

    static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFFu;

    namespace {
        // Must match the push_constant block in compact_directional_integrate.comp (all 4-byte
        // scalars -> tightly packed, no std430 padding).
        struct IntegratePC {
            uint32_t numPoints;
            uint32_t hashCapacity;
            float voxelSize;
            float truncation;
            float camX;
            float camY;
            float camZ;
            uint32_t maxDirections;
            uint32_t dirExponent;
            uint32_t viewAngleWeight;
        };

        // Must match the push_constant block in compact_directional_extract.comp.
        struct ExtractPC {
            float voxelSize;
            uint32_t hashCapacity;
            uint32_t maxCandidates;
        };
    } // namespace

    CompactDirectionalTSDF::CompactDirectionalTSDF() {}

    void CompactDirectionalTSDF::Build(Engine::Core::Context &ctx,
                                       float voxelSize,
                                       float truncation,
                                       uint32_t hashCapacity,
                                       uint32_t maxPoints) {
        m_ctx = &ctx;
        m_voxelSize = voxelSize;
        m_truncation = truncation;
        m_hashCapacity = hashCapacity;
        m_maxPoints = maxPoints;

        m_hashBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_pointBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_normalBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_statBuffer = std::make_unique<Engine::Core::Buffer>(ctx);

        m_hashBuffer->Allocate(hashCapacity * sizeof(DirEntry));
        m_pointBuffer->Allocate(maxPoints * 3u * sizeof(float));
        m_normalBuffer->Allocate(maxPoints * 3u * sizeof(float));
        m_statBuffer->Allocate(sizeof(uint32_t));

        m_kernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_kernel->Build("compact_directional_integrate.comp")
                .Bind(0, *m_hashBuffer)
                .Bind(1, *m_pointBuffer)
                .Bind(2, *m_normalBuffer)
                .Bind(3, *m_statBuffer);

        Reset();
    }

    void CompactDirectionalTSDF::Reset() {
        std::vector<DirEntry> empty(m_hashCapacity, {EMPTY_KEY, 0, 0u, 0u});
        m_hashBuffer->Upload(empty.data(), m_hashCapacity * sizeof(DirEntry));
        const uint32_t zero = 0;
        m_statBuffer->Upload(&zero, sizeof(uint32_t));
    }

    void CompactDirectionalTSDF::Integrate(const std::vector<Eigen::Vector3f> &points,
                                           const std::vector<Eigen::Vector3f> &normals,
                                           const Eigen::Vector3f &cameraPos) {
        if (points.empty()) return;

        const uint32_t N = std::min({static_cast<uint32_t>(points.size()),
                                     static_cast<uint32_t>(normals.size()), m_maxPoints});
        if (N == 0) return;

        m_pointBuffer->Upload(points.data(), N * 3u * sizeof(float));
        m_normalBuffer->Upload(normals.data(), N * 3u * sizeof(float));

        IntegratePC pc{
                N, m_hashCapacity, m_voxelSize, m_truncation,
                cameraPos.x(), cameraPos.y(), cameraPos.z(),
                m_quality.maxDirections, m_quality.dirExponent,
                m_quality.viewAngleWeight ? 1u : 0u};
        m_kernel->Args(pc).DispatchElements(N);
    }

    uint32_t CompactDirectionalTSDF::FilledCount() const {
        uint32_t count = 0;
        m_statBuffer->Download(&count, sizeof(uint32_t));
        return count;
    }

    OrientedPointCloud CompactDirectionalTSDF::ExtractPointCloud(uint32_t maxCandidates) const {
        OrientedPointCloud cloud;
        if (!m_ctx) return cloud;

        Engine::Core::Buffer candBuf(*m_ctx);
        Engine::Core::Buffer countBuf(*m_ctx);
        candBuf.Allocate(maxCandidates * 6u * sizeof(float)); // 6 floats/candidate: pos + normal
        countBuf.Allocate(sizeof(uint32_t));

        const uint32_t zero = 0;
        countBuf.Upload(&zero, sizeof(uint32_t));

        ExtractPC pc{m_voxelSize, m_hashCapacity, maxCandidates};

        Engine::Core::ComputePipeline kernel(*m_ctx);
        kernel.Build("compact_directional_extract.comp")
                .Bind(0, *m_hashBuffer)
                .Bind(1, candBuf)
                .Bind(2, countBuf)
                .Args(pc)
                .DispatchElements(m_hashCapacity);

        uint32_t count = 0;
        countBuf.Download(&count, sizeof(uint32_t));
        const uint32_t n = std::min(count, maxCandidates);
        if (n == 0) return cloud;

        std::vector<float> raw(n * 6u);
        candBuf.Download(raw.data(), n * 6u * sizeof(float));

        cloud.points.resize(n);
        cloud.normals.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            cloud.points[i] = Eigen::Vector3f(raw[i * 6u + 0u], raw[i * 6u + 1u], raw[i * 6u + 2u]);
            cloud.normals[i] = Eigen::Vector3f(raw[i * 6u + 3u], raw[i * 6u + 4u], raw[i * 6u + 5u]);
        }
        return cloud;
    }

} // namespace Engine::Spatial

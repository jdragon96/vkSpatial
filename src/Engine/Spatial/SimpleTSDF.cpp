#include "Engine/Spatial/SimpleTSDF.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace Engine::Spatial {

    static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFFu;

    namespace {
        struct IntegratePC {
            uint32_t numPoints;
            uint32_t hashCapacity;
            float voxelSize;
            float truncation;
            float camX;
            float camY;
            float camZ;
        };
    } // namespace

    SimpleTSDF::SimpleTSDF() {}

    void SimpleTSDF::Build(Engine::Core::Context &ctx,
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
        m_statBuffer = std::make_unique<Engine::Core::Buffer>(ctx);

        m_hashBuffer->Allocate(hashCapacity * sizeof(TSDFEntry));
        m_pointBuffer->Allocate(maxPoints * 3u * sizeof(float));
        m_statBuffer->Allocate(sizeof(uint32_t));

        m_kernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_kernel->Build("voxel_tsdf_integrate.comp")
                .Bind(0, *m_hashBuffer)
                .Bind(1, *m_pointBuffer)
                .Bind(2, *m_statBuffer);

        Reset();
    }

    void SimpleTSDF::Reset() {
        std::vector<TSDFEntry> empty(m_hashCapacity, {EMPTY_KEY, 0, 0u, 0u});
        m_hashBuffer->Upload(empty.data(), m_hashCapacity * sizeof(TSDFEntry));
        const uint32_t zero = 0;
        m_statBuffer->Upload(&zero, sizeof(uint32_t));
    }

    void SimpleTSDF::Integrate(const std::vector<Eigen::Vector3f> &points,
                               const Eigen::Vector3f &cameraPos) {
        if (points.empty()) return;

        const uint32_t N = std::min(static_cast<uint32_t>(points.size()), m_maxPoints);

        m_pointBuffer->Upload(points.data(), N * 3u * sizeof(float));

        IntegratePC pc{
                N, m_hashCapacity, m_voxelSize, m_truncation,
                cameraPos.x(), cameraPos.y(), cameraPos.z()};
        m_kernel->Args(pc).DispatchElements(N);
    }

    uint32_t SimpleTSDF::FilledCount() const {
        uint32_t count = 0;
        m_statBuffer->Download(&count, sizeof(uint32_t));
        return count;
    }


    void SimpleTSDF::ExportMC(const std::string &path, uint32_t maxTris) const {
        struct Vec4 {
            float x, y, z, w;
        };
        struct MCPushConst {
            float voxelSize;
            uint32_t hashCapacity;
            float truncation;
        };

        Engine::Core::Buffer countBuf(*m_ctx);
        Engine::Core::Buffer vertsBuf(*m_ctx);

        countBuf.Allocate(sizeof(uint32_t));
        vertsBuf.Allocate(maxTris * 3u * sizeof(Vec4));

        const uint32_t zero = 0;
        countBuf.Upload(&zero, sizeof(uint32_t));

        MCPushConst pc{m_voxelSize, m_hashCapacity, m_truncation};

        Engine::Core::ComputePipeline kernel(*m_ctx);
        kernel.Build("voxel_tsdf_mc.comp")
                .Bind(0, *m_hashBuffer)
                .Bind(1, countBuf)
                .Bind(2, vertsBuf)
                .Args(pc)
                .DispatchElements(m_hashCapacity);

        uint32_t triCount = 0;
        countBuf.Download(&triCount, sizeof(uint32_t));

        const uint32_t downloadCount = std::min(triCount, maxTris);
        std::vector<Vec4> verts(downloadCount * 3u);
        vertsBuf.Download(verts.data(), downloadCount * 3u * sizeof(Vec4));

        std::ofstream f(path);
        if (!f.is_open())
            throw std::runtime_error("SimpleTSDF::ExportMC: cannot open " + path);

        f << "ply\nformat ascii 1.0\n"
          << "element vertex " << downloadCount * 3u << "\n"
          << "property float x\nproperty float y\nproperty float z\n"
          << "element face " << downloadCount << "\n"
          << "property list uchar int vertex_indices\n"
          << "end_header\n";

        for (const auto &v: verts)
            f << v.x << ' ' << v.y << ' ' << v.z << '\n';

        for (uint32_t i = 0; i < downloadCount; i++)
            f << "3 " << i * 3 << ' ' << i * 3 + 1 << ' ' << i * 3 + 2 << '\n';
    }

} // namespace Engine::Spatial

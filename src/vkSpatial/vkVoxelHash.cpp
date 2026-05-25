#include "vkSpatial/vkVoxelHash.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace vkSpatial {

    static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFFu;

    namespace {
        struct IntegratePC {
            uint32_t numPoints;
            uint32_t hashCapacity;
            float voxelSize;
        };
    } // namespace

    vkVoxelHash::vkVoxelHash() {
    }

    void vkVoxelHash::SetAlgorithmType(EVoxelHashType type) {
        m_type = type;
    }

    void vkVoxelHash::Build(vkCommon::VkContext *ctx,
                            float voxelSize,
                            uint32_t hashCapacity,
                            uint32_t maxPoints) {
        m_ctx = ctx;
        m_voxelSize = voxelSize;
        m_hashCapacity = hashCapacity;
        m_maxPoints = maxPoints;

        m_hashBuffer = vkCommon::vkGPUMemory::MakeShared(ctx->device, ctx->physDevice);
        m_pointBuffer = vkCommon::vkGPUMemory::MakeShared(ctx->device, ctx->physDevice);
        m_statBuffer = vkCommon::vkGPUMemory::MakeShared(ctx->device, ctx->physDevice);

        if (!m_hashBuffer->Allocate(hashCapacity * sizeof(VoxelEntry)))
            throw std::runtime_error("vkVoxelHash: hashBuffer alloc failed");
        if (!m_pointBuffer->Allocate(maxPoints * 3 * sizeof(float)))
            throw std::runtime_error("vkVoxelHash: pointBuffer alloc failed");
        if (!m_statBuffer->Allocate(sizeof(uint32_t)))
            throw std::runtime_error("vkVoxelHash: statBuffer alloc failed");

        m_kernel = vkCommon::vkComputeBase::MakeUnique(
                ctx->device, ctx->physDevice, ctx->computeQueue, ctx->cmdPool);
        m_kernel->Build("voxel_wang_integrate.comp")
                .Bind(0, m_hashBuffer)
                .Bind(1, m_pointBuffer)
                .Bind(2, m_statBuffer);

        Reset();
    }

    void vkVoxelHash::Reset() {
        std::vector<VoxelEntry> empty(m_hashCapacity, {EMPTY_KEY, 0u});
        m_hashBuffer->Upload(empty.data(), m_hashCapacity * sizeof(VoxelEntry),
                             m_ctx->computeQueue, m_ctx->cmdPool);
        const uint32_t zero = 0;
        m_statBuffer->Upload(&zero, sizeof(uint32_t),
                             m_ctx->computeQueue, m_ctx->cmdPool);
    }

    void vkVoxelHash::Integrate(const std::vector<Eigen::Vector3f> &points) {
        if (points.empty()) return;

        const uint32_t N = std::min(static_cast<uint32_t>(points.size()), m_maxPoints);

        if (!m_pointBuffer->Upload(points.data(), N * 3 * sizeof(float),
                                   m_ctx->computeQueue, m_ctx->cmdPool))
            throw std::runtime_error("vkVoxelHash: pointBuffer upload failed");

        IntegratePC pc{N, m_hashCapacity, m_voxelSize};
        m_kernel->Args(pc).DispatchElements(N);
        m_kernel->Sync();
    }

    uint32_t vkVoxelHash::FilledCount() const {
        uint32_t count = 0;
        m_statBuffer->Download(&count, sizeof(uint32_t),
                               m_ctx->computeQueue, m_ctx->cmdPool);
        return count;
    }

    void vkVoxelHash::ExportMC(const std::string &path, uint32_t maxTris) const {
        struct Vec4 {
            float x, y, z, w;
        };
        struct MarchingCubePC {
            float voxelSize;
            uint32_t hashCapacity;
        };

        auto countBuf = vkCommon::vkGPUMemory::MakeUnique(m_ctx->device, m_ctx->physDevice);
        auto vertsBuf = vkCommon::vkGPUMemory::MakeUnique(m_ctx->device, m_ctx->physDevice);

        if (!countBuf->Allocate(sizeof(uint32_t)))
            throw std::runtime_error("ExportMC: countBuf alloc failed");
        if (!vertsBuf->Allocate(maxTris * 3u * sizeof(Vec4)))
            throw std::runtime_error("ExportMC: vertsBuf alloc failed");

        const uint32_t zero = 0;
        if (!countBuf->Upload(&zero, sizeof(uint32_t), m_ctx->computeQueue, m_ctx->cmdPool))
            throw std::runtime_error("ExportMC: countBuf zero-init failed");

        MarchingCubePC pc{m_voxelSize, m_hashCapacity};

        auto kernel = vkCommon::vkComputeBase::MakeUnique(
                m_ctx->device, m_ctx->physDevice, m_ctx->computeQueue, m_ctx->cmdPool);
        kernel->Build("voxel_wang_mc.comp")
                .Bind(0, m_hashBuffer)
                .Bind(1, *countBuf)
                .Bind(2, *vertsBuf)
                .Args(pc)
                .DispatchElements(m_hashCapacity);
        kernel->Sync();

        uint32_t triCount = 0;
        if (!countBuf->Download(&triCount, sizeof(uint32_t), m_ctx->computeQueue, m_ctx->cmdPool))
            throw std::runtime_error("ExportMC: count download failed");

        const uint32_t downloadCount = std::min(triCount, maxTris);
        std::vector<Vec4> verts(downloadCount * 3u);
        if (!vertsBuf->Download(verts.data(), downloadCount * 3u * sizeof(Vec4),
                                m_ctx->computeQueue, m_ctx->cmdPool))
            throw std::runtime_error("ExportMC: verts download failed");

        std::ofstream f(path);
        if (!f.is_open())
            throw std::runtime_error("ExportMC: cannot open " + path);

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

} // namespace vkSpatial

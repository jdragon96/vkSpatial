#include "TSDF/Backends/SimpleTSDF.h"

#include <Eigen/Geometry> // Vector3f::cross
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace TSDF {

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
            uint32_t useNormalWeight;
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
        m_normalBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_statBuffer = std::make_unique<Engine::Core::Buffer>(ctx);

        m_hashBuffer->Allocate(hashCapacity * sizeof(TSDFEntry));
        m_pointBuffer->Allocate(maxPoints * 3u * sizeof(float));
        // Always allocated (even when the unweighted Integrate() overload is used, which
        // never uploads to it) so binding 3 is always a valid descriptor.
        m_normalBuffer->Allocate(maxPoints * 3u * sizeof(float));
        m_statBuffer->Allocate(sizeof(uint32_t));

        m_kernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_kernel->Build("TSDF/Backends/kernel_voxel_tsdf_integrate.comp.glsl")
                .Bind(0, *m_hashBuffer)
                .Bind(1, *m_pointBuffer)
                .Bind(2, *m_statBuffer)
                .Bind(3, *m_normalBuffer);

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

        // useNormalWeight=0: the shader never reads g_normals in this path, so no upload
        // is required here -- byte-identical to the pre-normal-weighting behaviour.
        IntegratePC pc{
                N, m_hashCapacity, m_voxelSize, m_truncation,
                cameraPos.x(), cameraPos.y(), cameraPos.z(), 0u};
        m_kernel->Args(pc).DispatchElements(N);
    }

    void SimpleTSDF::Integrate(const std::vector<Eigen::Vector3f> &points,
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
                cameraPos.x(), cameraPos.y(), cameraPos.z(), 1u};
        m_kernel->Args(pc).DispatchElements(N);
    }

    uint32_t SimpleTSDF::FilledCount() const {
        uint32_t count = 0;
        m_statBuffer->Download(&count, sizeof(uint32_t));
        return count;
    }

    std::vector<VoxelStat> SimpleTSDF::DownloadVoxels() const {
        std::vector<TSDFEntry> entries(m_hashCapacity);
        m_hashBuffer->Download(entries.data(), m_hashCapacity * sizeof(TSDFEntry));

        // Same occupancy gate as voxel_tsdf_mc.comp's MIN_WEIGHT (TSDF_SCALE / 2).
        const uint32_t kMinWeight = static_cast<uint32_t>(TSDF_FIXED_SCALE) / 2u;

        std::vector<VoxelStat> out;
        out.reserve(entries.size());
        for (const TSDFEntry &e : entries) {
            if (e.key == EMPTY_KEY) continue;
            if (e.sumW < kMinWeight) continue;

            // Unpack the same way voxel_tsdf_mc.comp does (packCoord in voxel_common.glsl).
            const int kx = static_cast<int>((e.key >> 20u) & 0x3FFu) - 512;
            const int ky = static_cast<int>((e.key >> 10u) & 0x3FFu) - 512;
            const int kz = static_cast<int>(e.key & 0x3FFu) - 512;

            VoxelStat vs;
            vs.center = (Eigen::Vector3f(float(kx), float(ky), float(kz)) + Eigen::Vector3f(0.5f, 0.5f, 0.5f)) *
                        m_voxelSize;
            const float sumDW = float(e.sumDW);
            const float sumW = float(e.sumW);
            const float sumD2 = float(e.sumD2);
            vs.tsdf = sumDW / sumW;
            vs.weight = sumW / float(TSDF_FIXED_SCALE);
            const float ex2 = sumD2 / sumW;
            vs.variance = std::max(0.0f, ex2 - vs.tsdf * vs.tsdf);
            out.push_back(vs);
        }
        return out;
    }


    std::vector<Eigen::Vector3f> SimpleTSDF::downloadMCVertices(uint32_t maxTris) const {
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
        kernel.Build("TSDF/Backends/kernel_voxel_tsdf_mc.comp.glsl")
                .Bind(0, *m_hashBuffer)
                .Bind(1, countBuf)
                .Bind(2, vertsBuf)
                .Args(pc)
                .DispatchElements(m_hashCapacity);

        uint32_t triCount = 0;
        countBuf.Download(&triCount, sizeof(uint32_t));

        const uint32_t downloadCount = std::min(triCount, maxTris);
        std::vector<Vec4> raw(downloadCount * 3u);
        if (downloadCount > 0)
            vertsBuf.Download(raw.data(), downloadCount * 3u * sizeof(Vec4));

        std::vector<Eigen::Vector3f> verts(raw.size());
        for (size_t i = 0; i < raw.size(); ++i)
            verts[i] = Eigen::Vector3f(raw[i].x, raw[i].y, raw[i].z);
        return verts;
    }

    void SimpleTSDF::ExportMC(const std::string &path, uint32_t maxTris) const {
        const std::vector<Eigen::Vector3f> verts = downloadMCVertices(maxTris);
        const uint32_t triCount = static_cast<uint32_t>(verts.size() / 3u);

        std::ofstream f(path);
        if (!f.is_open())
            throw std::runtime_error("SimpleTSDF::ExportMC: cannot open " + path);

        f << "ply\nformat ascii 1.0\n"
          << "element vertex " << verts.size() << "\n"
          << "property float x\nproperty float y\nproperty float z\n"
          << "element face " << triCount << "\n"
          << "property list uchar int vertex_indices\n"
          << "end_header\n";

        for (const auto &v: verts)
            f << v.x() << ' ' << v.y() << ' ' << v.z() << '\n';

        for (uint32_t i = 0; i < triCount; i++)
            f << "3 " << i * 3 << ' ' << i * 3 + 1 << ' ' << i * 3 + 2 << '\n';
    }

    Engine::Core::OrientedPointCloud SimpleTSDF::ExtractPointCloud(uint32_t maxTris) const {
        const std::vector<Eigen::Vector3f> tri = downloadMCVertices(maxTris);
        Engine::Core::OrientedPointCloud cloud;
        if (tri.empty()) return cloud;

        // Weld coincident MC vertices (3 emitted per triangle; shared edges duplicate a
        // position) onto a fine tolerance grid, accumulating area-weighted triangle normals.
        const float weld = std::max(m_voxelSize * 1e-3f, 1e-6f);
        auto key = [weld](const Eigen::Vector3f &p) -> uint64_t {
            constexpr int64_t kBias = 1 << 20;
            constexpr uint64_t kMask = (1ull << 21) - 1;
            const int64_t qx = int64_t(std::llround(p.x() / weld)) + kBias;
            const int64_t qy = int64_t(std::llround(p.y() / weld)) + kBias;
            const int64_t qz = int64_t(std::llround(p.z() / weld)) + kBias;
            return (uint64_t(qx) & kMask) | ((uint64_t(qy) & kMask) << 21) | ((uint64_t(qz) & kMask) << 42);
        };

        std::unordered_map<uint64_t, uint32_t> lut;
        std::vector<Eigen::Vector3f> nAccum;
        for (size_t t = 0; t + 2 < tri.size(); t += 3) {
            const Eigen::Vector3f &a = tri[t], &b = tri[t + 1], &c = tri[t + 2];
            const Eigen::Vector3f fn = (b - a).cross(c - a); // area-weighted (|fn| = 2*area)
            for (const Eigen::Vector3f &p: {a, b, c}) {
                const uint64_t k = key(p);
                auto it = lut.find(k);
                uint32_t vi;
                if (it == lut.end()) {
                    vi = static_cast<uint32_t>(cloud.points.size());
                    lut.emplace(k, vi);
                    cloud.points.push_back(p);
                    nAccum.push_back(Eigen::Vector3f::Zero());
                } else {
                    vi = it->second;
                }
                nAccum[vi] += fn;
            }
        }

        cloud.normals.resize(cloud.points.size());
        for (size_t i = 0; i < cloud.points.size(); ++i) {
            const float len = nAccum[i].norm();
            cloud.normals[i] = len > 1e-12f ? Eigen::Vector3f(nAccum[i] / len) : Eigen::Vector3f(0, 0, 1);
        }
        return cloud;
    }

} // namespace TSDF

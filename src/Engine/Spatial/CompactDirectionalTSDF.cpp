#include "Engine/Spatial/CompactDirectionalTSDF.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace Engine::Spatial {

    static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFFu;
    // Matches #define TSDF_SCALE 10000.0 in compact_directional_integrate.comp /
    // compact_directional_extract.comp (extract's MIN_WEIGHT gate = TSDF_SCALE/2).
    static constexpr int32_t kTsdfScale = 10000;
    // Cluster cap per voxel bucket in MergeCandidates -- matches DirectionalTSDFTypes.h's
    // kNumDirections (6), the max number of distinct surface orientations a single voxel can
    // carry in the directional schemes this class mirrors.
    static constexpr size_t kMaxClustersPerVoxel = 6;

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
            int32_t originX;
            int32_t originY;
            int32_t originZ;
        };

        // Must match the push_constant block in compact_directional_extract.comp.
        struct ExtractPC {
            float voxelSize;
            uint32_t hashCapacity;
            uint32_t maxCandidates;
            int32_t originX;
            int32_t originY;
            int32_t originZ;
        };

        // floor(worldMinCorner / voxelSize), component-wise -- the voxel-space origin of the
        // movable 512^3 hash window (see CompactDirectionalTSDF.h's Build doc).
        Eigen::Vector3i FloorToVoxel(const Eigen::Vector3f &worldMinCorner, float voxelSize) {
            return Eigen::Vector3i(static_cast<int32_t>(std::floor(worldMinCorner.x() / voxelSize)),
                                   static_cast<int32_t>(std::floor(worldMinCorner.y() / voxelSize)),
                                   static_cast<int32_t>(std::floor(worldMinCorner.z() / voxelSize)));
        }

        // Same 21-bit-per-axis voxel-key packing as DirectionalTSDF::mergeCandidates's
        // voxelKey lambda (DirectionalTSDF.cpp ~line 339) -- world-space voxel coords, not
        // window-local, so it needs no origin bias.
        uint64_t VoxelKey(int32_t x, int32_t y, int32_t z) {
            return (uint64_t(uint32_t(x) & 0x1FFFFFu) << 42) |
                   (uint64_t(uint32_t(y) & 0x1FFFFFu) << 21) |
                   uint64_t(uint32_t(z) & 0x1FFFFFu);
        }
    } // namespace

    CompactDirectionalTSDF::CompactDirectionalTSDF() {}

    void CompactDirectionalTSDF::Build(Engine::Core::Context &ctx,
                                       float voxelSize,
                                       float truncation,
                                       uint32_t hashCapacity,
                                       uint32_t maxPoints,
                                       const Eigen::Vector3f &windowMinCorner) {
        m_ctx = &ctx;
        m_voxelSize = voxelSize;
        m_truncation = truncation;
        m_hashCapacity = hashCapacity;
        m_maxPoints = maxPoints;
        m_originVoxel = FloorToVoxel(windowMinCorner, voxelSize);

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
                m_quality.viewAngleWeight ? 1u : 0u,
                m_originVoxel.x(), m_originVoxel.y(), m_originVoxel.z()};
        m_kernel->Args(pc).DispatchElements(N);
    }

    uint32_t CompactDirectionalTSDF::FilledCount() const {
        uint32_t count = 0;
        m_statBuffer->Download(&count, sizeof(uint32_t));
        return count;
    }

    std::vector<CompactEntry> CompactDirectionalTSDF::DownloadEntries() const {
        std::vector<CompactEntry> out;
        if (!m_ctx) return out;

        std::vector<DirEntry> entries(m_hashCapacity);
        m_hashBuffer->Download(entries.data(), m_hashCapacity * sizeof(DirEntry));

        // Same occupancy gate as compact_directional_extract.comp's MIN_WEIGHT (TSDF_SCALE/2).
        const uint32_t kMinWeight = static_cast<uint32_t>(kTsdfScale) / 2u;

        out.reserve(entries.size());
        for (const DirEntry &e : entries) {
            if (e.key == EMPTY_KEY) continue;
            if (e.sumW < kMinWeight) continue;

            // Unpack with the SAME layout as packDirKey/unpackDirKey in
            // compact_directional_{integrate,extract}.comp: local (window-relative) coords,
            // no bias -- add m_originVoxel back to recover the world-space voxel.
            const uint32_t dir = e.key & 0x7u;
            const int lz = static_cast<int>((e.key >> 3u) & 0x1FFu);
            const int ly = static_cast<int>((e.key >> 12u) & 0x1FFu);
            const int lx = static_cast<int>((e.key >> 21u) & 0x1FFu);
            const int vx = lx + m_originVoxel.x();
            const int vy = ly + m_originVoxel.y();
            const int vz = lz + m_originVoxel.z();

            CompactEntry ce;
            ce.center = (Eigen::Vector3f(float(vx), float(vy), float(vz)) +
                         Eigen::Vector3f::Constant(0.5f)) * m_voxelSize;
            ce.direction = dir;
            ce.tsdf = float(e.sumDW) / float(e.sumW);
            ce.weight = float(e.sumW) / float(kTsdfScale);
            out.push_back(ce);
        }
        return out;
    }

    OrientedPointCloud CompactDirectionalTSDF::ExtractPointCloud(uint32_t maxCandidates,
                                                                 bool merge) const {
        OrientedPointCloud cloud;
        if (!m_ctx) return cloud;

        Engine::Core::Buffer candBuf(*m_ctx);
        Engine::Core::Buffer countBuf(*m_ctx);
        candBuf.Allocate(maxCandidates * 6u * sizeof(float)); // 6 floats/candidate: pos + normal
        countBuf.Allocate(sizeof(uint32_t));

        const uint32_t zero = 0;
        countBuf.Upload(&zero, sizeof(uint32_t));

        ExtractPC pc{m_voxelSize, m_hashCapacity, maxCandidates,
                     m_originVoxel.x(), m_originVoxel.y(), m_originVoxel.z()};

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

        if (!merge) return cloud;
        return MergeCandidates(cloud.points, cloud.normals, m_voxelSize);
    }

    OrientedPointCloud CompactDirectionalTSDF::MergeCandidates(
            const std::vector<Eigen::Vector3f> &points,
            const std::vector<Eigen::Vector3f> &normals, float voxelSize) {
        struct Cluster {
            Eigen::Vector3f posSum = Eigen::Vector3f::Zero();
            Eigen::Vector3f nSum = Eigen::Vector3f::Zero();
            int count = 0;
        };

        const float posThresh = 0.6f * voxelSize;    // positionMergeThreshold
        const float cosThresh = 0.866f;              // normalMergeThreshold = 30 deg
        const float strongSplitCos = 0.5f;           // 60 deg -- never merge beyond this (corner)

        std::unordered_map<uint64_t, std::vector<Cluster>> buckets;
        const size_t nCand = std::min(points.size(), normals.size());
        for (size_t i = 0; i < nCand; ++i) {
            const Eigen::Vector3f &pos = points[i];
            const Eigen::Vector3f &nrm = normals[i];
            const int vx = int(std::floor(pos.x() / voxelSize));
            const int vy = int(std::floor(pos.y() / voxelSize));
            const int vz = int(std::floor(pos.z() / voxelSize));
            auto &clusters = buckets[VoxelKey(vx, vy, vz)];

            bool merged = false;
            for (auto &cl : clusters) {
                const Eigen::Vector3f mean = cl.posSum / float(cl.count);
                const Eigen::Vector3f meanN = cl.nSum.normalized();
                if (nrm.dot(meanN) < strongSplitCos) continue; // strong split: cannot merge
                if ((pos - mean).norm() < posThresh && nrm.dot(meanN) > cosThresh) {
                    cl.posSum += pos;
                    cl.nSum += nrm;
                    cl.count++;
                    merged = true;
                    break;
                }
            }
            if (!merged && clusters.size() < kMaxClustersPerVoxel)
                clusters.push_back({pos, nrm, 1});
        }

        OrientedPointCloud out;
        for (auto &bucket : buckets)
            for (auto &cl : bucket.second) {
                out.points.push_back(cl.posSum / float(cl.count));
                out.normals.push_back(cl.nSum.normalized());
            }
        return out;
    }

} // namespace Engine::Spatial

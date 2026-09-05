#include "Features/Downsample.h"

#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace Engine::Features {

    namespace {

        // Hash-grid cell key. Mirrors the Vector3i hash style used for
        // DirectionalGroupKeyHash (Engine::Spatial), adapted to 3D (no direction axis).
        struct VoxelKey {
            int32_t x = 0, y = 0, z = 0;

            bool operator==(const VoxelKey &o) const {
                return x == o.x && y == o.y && z == o.z;
            }
        };

        struct VoxelKeyHash {
            size_t operator()(const VoxelKey &k) const {
                uint64_t h = uint64_t(uint32_t(k.x)) * 73856093ull;
                h ^= uint64_t(uint32_t(k.y)) * 19349663ull;
                h ^= uint64_t(uint32_t(k.z)) * 83492791ull;
                h ^= h >> 33;
                return size_t(h);
            }
        };

        struct VoxelAccum {
            Eigen::Vector3f pointSum = Eigen::Vector3f::Zero();
            Eigen::Vector3f normalSum = Eigen::Vector3f::Zero();
            int count = 0;
        };

        inline VoxelKey KeyOf(const Eigen::Vector3f &p, float voxelSize) {
            return VoxelKey{
                    int32_t(std::floor(p.x() / voxelSize)),
                    int32_t(std::floor(p.y() / voxelSize)),
                    int32_t(std::floor(p.z() / voxelSize))};
        }

    } // namespace

    PointCloud DownsampleVoxel(const PointCloud &in, float voxelSize) {
        PointCloud out;
        if (in.points.empty() || voxelSize <= 0.0f) return out;

        const bool hasNormals = !in.normals.empty();

        std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> cells;
        cells.reserve(in.points.size());

        for (size_t i = 0; i < in.points.size(); ++i) {
            VoxelAccum &accum = cells[KeyOf(in.points[i], voxelSize)];
            accum.pointSum += in.points[i];
            if (hasNormals) accum.normalSum += in.normals[i];
            accum.count += 1;
        }

        out.points.reserve(cells.size());
        if (hasNormals) out.normals.reserve(cells.size());

        for (const auto &entry: cells) {
            const VoxelAccum &accum = entry.second;
            out.points.push_back(accum.pointSum / float(accum.count));
            if (hasNormals) {
                const Eigen::Vector3f &n = accum.normalSum;
                const float len = n.norm();
                out.normals.push_back(len > 1e-8f ? Eigen::Vector3f(n / len) : Eigen::Vector3f::Zero());
            }
        }

        return out;
    }

} // namespace Engine::Features

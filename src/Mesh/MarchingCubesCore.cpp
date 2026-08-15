#include "Mesh/MarchingCubesCore.h"

#include <Eigen/Geometry> // Vector3f::cross
#include <algorithm>
#include <cmath>

namespace Mesh::core {

    const int kEdgeCornerPairs[12][2] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

    std::unordered_set<std::array<int, 3>, IVec3Hash> CandidateBases(const std::vector<std::array<int, 3>> &occupied) {
        std::unordered_set<std::array<int, 3>, IVec3Hash> bases;
        bases.reserve(occupied.size() * 8u);
        for (const auto &coord : occupied) {
            for (int dx = -1; dx <= 0; ++dx)
                for (int dy = -1; dy <= 0; ++dy)
                    for (int dz = -1; dz <= 0; ++dz)
                        bases.insert({coord[0] + dx, coord[1] + dy, coord[2] + dz});
        }
        return bases;
    }

    SurfaceMesh WeldAndComputeNormals(const std::vector<RawTriangle> &triangles, float weldDistance) {
        SurfaceMesh mesh;
        if (triangles.empty()) return mesh;
        weldDistance = std::max(weldDistance, 1e-6f);

        struct BinHash {
            size_t operator()(const std::array<int64_t, 3> &b) const noexcept {
                size_t h = std::hash<int64_t>()(b[0]);
                h = h * 31u + std::hash<int64_t>()(b[1]);
                h = h * 31u + std::hash<int64_t>()(b[2]);
                return h;
            }
        };
        auto binOf = [weldDistance](const Eigen::Vector3f &p) {
            return std::array<int64_t, 3>{static_cast<int64_t>(std::floor(p.x() / weldDistance)),
                                           static_cast<int64_t>(std::floor(p.y() / weldDistance)),
                                           static_cast<int64_t>(std::floor(p.z() / weldDistance))};
        };
        std::unordered_map<std::array<int64_t, 3>, std::vector<uint32_t>, BinHash> grid;
        std::vector<Eigen::Vector3f> normalAccumulator;

        auto weldVertex = [&](const Eigen::Vector3f &p) -> uint32_t {
            const auto b = binOf(p);
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        const std::array<int64_t, 3> nb{b[0] + dx, b[1] + dy, b[2] + dz};
                        const auto it = grid.find(nb);
                        if (it == grid.end()) continue;
                        for (uint32_t vi : it->second)
                            if ((mesh.vertices[vi] - p).norm() < weldDistance) return vi;
                    }
            const auto vi = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(p);
            normalAccumulator.push_back(Eigen::Vector3f::Zero());
            grid[b].push_back(vi);
            return vi;
        };

        for (const auto &t : triangles) {
            const uint32_t ia = weldVertex(t.a), ib = weldVertex(t.b), ic = weldVertex(t.c);
            if (ia == ib || ib == ic || ia == ic) continue; // collapsed to <3 verts: drop
            const Eigen::Vector3f &A = mesh.vertices[ia];
            const Eigen::Vector3f &B = mesh.vertices[ib];
            const Eigen::Vector3f &C = mesh.vertices[ic];
            const Eigen::Vector3f fn = (B - A).cross(C - A); // area-weighted (|fn|=2*area)
            if (fn.norm() <= 1e-9f) continue;                // zero-area after welding: drop
            normalAccumulator[ia] += fn;
            normalAccumulator[ib] += fn;
            normalAccumulator[ic] += fn;
            mesh.triangles.emplace_back(int(ia), int(ib), int(ic));
        }

        mesh.normals.resize(mesh.vertices.size());
        for (size_t i = 0; i < mesh.vertices.size(); ++i) {
            const float len = normalAccumulator[i].norm();
            mesh.normals[i] = len > 1e-12f ? Eigen::Vector3f(normalAccumulator[i] / len) : Eigen::Vector3f(0, 0, 1);
        }
        return mesh;
    }

} // namespace Mesh::core

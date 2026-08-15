#pragma once

// Shared CPU Marching Cubes core, reused by every isosurface extractor strategy in this
// namespace (the "mc" extractor in MarchingCubesExtractor.cpp, and AdaptiveVoxelGrid's
// multi-resolution extractor). This is a MOVE (rename-only, logic-preserving) of the
// primitives that previously lived in the anonymous namespace of AdaptiveVoxelGrid.cpp;
// see that file's ExtractMesh() for the multi-resolution caller.

#include "Mesh/SurfaceMesh.h"
#include "Mesh/MarchingCubesTables.h"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Mesh::core {

    // Hashes an integer voxel coordinate for the CPU Marching Cubes value map.
    struct IVec3Hash {
        size_t operator()(const std::array<int, 3> &v) const noexcept {
            size_t h = std::hash<int>()(v[0]);
            h = h * 31u + std::hash<int>()(v[1]);
            h = h * 31u + std::hash<int>()(v[2]);
            return h;
        }
    };

    using VoxelValueMap = std::unordered_map<std::array<int, 3>, float, IVec3Hash>;

    struct RawTriangle {
        Eigen::Vector3f a, b, c;
    };

    // Mirrors voxel_common.glsl's vertInterp: linear interpolation to the TSDF zero-crossing.
    inline Eigen::Vector3f VertexInterpolate(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2,
                                              float value1, float value2) {
        const float dv = value2 - value1;
        if (std::abs(dv) < 1e-6f) return (p1 + p2) * 0.5f;
        const float t = -value1 / dv;
        return p1 + t * (p2 - p1);
    }

    // MC edge index -> corner-index pair (edge e connects CORNER[a] and CORNER[b]); matches
    // voxel_tsdf_mc.comp's explicit ev[0..11] = vertInterp(p[a],p[b],...) list.
    extern const int kEdgeCornerPairs[12][2];

    // Builds the 8-neighbour-sweep candidate cube bases for a set of occupied coordinates
    // (mirrors voxel_tsdf_mc.comp main()'s dx,dy,dz in {-1,0} sweep over each occupied voxel):
    // for each occupied coordinate, the 8 cubes that could have it as one of their corners.
    std::unordered_set<std::array<int, 3>, IVec3Hash> CandidateBases(const std::vector<std::array<int, 3>> &occupied);

    // Generates raw (unwelded) MC triangles for every candidate cube in `bases` (own-level
    // integer coordinates) whose 8 corners all resolve via `sampleAt`. `cellSize` is the
    // world-space size of one voxel at this level; world corner positions are
    // `(base+CORNER[c]) * cellSize`. Mirrors voxel_tsdf_mc.comp's processCube/edgeTable/triTable
    // logic (tables transcribed in MarchingCubesTables.h); appends to `out` rather than
    // returning, so multiple passes (e.g. AdaptiveVoxelGrid's fine + coarse) can be combined
    // before a single weld.
    template<typename SampleFunction>
    void GenerateRawTriangles(const std::unordered_set<std::array<int, 3>, IVec3Hash> &bases,
                               SampleFunction &&sampleAt, float cellSize, std::vector<RawTriangle> &out) {
        for (const auto &base : bases) {
            float sdf[8];
            bool complete = true;
            for (int c = 0; c < 8 && complete; ++c) {
                const std::array<int, 3> k{base[0] + mc::CORNER[c][0], base[1] + mc::CORNER[c][1],
                                            base[2] + mc::CORNER[c][2]};
                if (!sampleAt(k, sdf[c])) {
                    complete = false;
                    break;
                }
            }
            if (!complete) continue; // an unresolvable corner: skip this cell

            int cubeIndex = 0;
            for (int c = 0; c < 8; ++c)
                if (sdf[c] < 0.0f) cubeIndex |= (1 << c);

            const int et = mc::edgeTable[cubeIndex];
            if (et == 0) continue;

            Eigen::Vector3f p[8];
            for (int c = 0; c < 8; ++c)
                p[c] = Eigen::Vector3f(float(base[0] + mc::CORNER[c][0]), float(base[1] + mc::CORNER[c][1]),
                                       float(base[2] + mc::CORNER[c][2])) *
                       cellSize;

            Eigen::Vector3f ev[12];
            for (int e = 0; e < 12; ++e)
                if (et & (1 << e)) {
                    const int a = kEdgeCornerPairs[e][0], b = kEdgeCornerPairs[e][1];
                    ev[e] = VertexInterpolate(p[a], p[b], sdf[a], sdf[b]);
                }

            const int triBase = cubeIndex * 16;
            for (int i = 0; i < 15; i += 3) {
                const int ei0 = mc::triTable[triBase + i];
                if (ei0 == -1) break;
                const int ei1 = mc::triTable[triBase + i + 1];
                const int ei2 = mc::triTable[triBase + i + 2];
                // Swap ei1/ei2 for outward-facing winding (mirrors voxel_tsdf_mc.comp).
                out.push_back({ev[ei0], ev[ei2], ev[ei1]});
            }
        }
    }

    // Welds raw MC triangles onto a `weldDistance`-spaced grid and accumulates area-weighted
    // per-vertex normals from the WELDED positions. Unlike a naive exact-bin lookup, this checks
    // the full 3x3x3 neighbourhood of a vertex's own bin for an existing vertex within
    // `weldDistance` -- for bin size == weldDistance, that is sufficient to guarantee any two
    // points closer than `weldDistance` are found and merged regardless of where they fall
    // relative to a bin boundary (the classic edge-of-cell failure mode of single-bin grid
    // hashing). Triangles that become degenerate after welding (two corners collapse to the same
    // vertex, or the three welded positions are collinear) are dropped, satisfying the "no
    // zero-area triangles" requirement.
    SurfaceMesh WeldAndComputeNormals(const std::vector<RawTriangle> &triangles, float weldDistance);

} // namespace Mesh::core

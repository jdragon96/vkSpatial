// Shared fixture + color/coordinate helpers for the TSDF diagnostic tools
// (tsdf_slice_debug, tsdf_viewer). Factored out so the Vulkan viewer (Task 1+) and any
// future CLI diagnostics use the exact same known-good fixture math instead of
// re-deriving it. See example2/tsdf_slice_debug.cpp for the original source of these
// helpers and for scene-level usage (interproximal fixture, slice export, etc).
#pragma once

#include "Engine/Spatial/DirectionalHostStore.h"
#include "Engine/Spatial/DirectionalTSDF.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace tsdf_fixtures {

using Engine::Spatial::DirectionalGroupKey;
using Engine::Spatial::DirectionalGroupKeyHash;
using Engine::Spatial::DirectionalTSDF;
using Group = Engine::Spatial::DirectionalHostStore::Group;
using Cache = std::unordered_map<DirectionalGroupKey, Group, DirectionalGroupKeyHash>;

struct Rgb {
    uint8_t r, g, b;
};

// Diverging SDF colormap: value in [-1,1] -> blue(-1) .. white(0) .. red(+1).
inline Rgb sdfColor(float value) {
    const float v = std::clamp(value, -1.0f, 1.0f);
    if (v < 0.0f) {
        const uint8_t c = uint8_t((v + 1.0f) * 255.0f);
        return {c, c, 255};
    }
    const uint8_t c = uint8_t((1.0f - v) * 255.0f);
    return {255, c, c};
}

// Color an extracted point by its direction bitmask: +Z(bit4)=cyan, -Z(bit5)=magenta, both=yellow.
inline Rgb dirColor(uint8_t mask) {
    const bool pz = mask & (1u << 4), nz = mask & (1u << 5);
    if (pz && nz) return {230, 230, 40};
    if (pz) return {40, 220, 220};
    if (nz) return {220, 40, 220};
    return {160, 160, 160};
}

inline int floorDiv8(int a) { return a >= 0 ? a / 8 : -(((-a) + 7) / 8); }

// Value of the TSDF at world point `p` in a given direction layer. occupied=false if the
// group is not resident or the voxel has zero weight.
inline float valueAt(DirectionalTSDF &tsdf, Cache &cache, uint8_t dir, float voxelSize,
                     const Eigen::Vector3f &p, bool &occupied) {
    const int vx = int(std::floor(p.x() / voxelSize));
    const int vy = int(std::floor(p.y() / voxelSize));
    const int vz = int(std::floor(p.z() / voxelSize));
    DirectionalGroupKey key{floorDiv8(vx), floorDiv8(vy), floorDiv8(vz), dir};
    if (tsdf.DebugQueryPoolIndex(key) == Engine::Spatial::kInvalidPoolIndex) {
        occupied = false;
        return 0.f;
    }
    auto it = cache.find(key);
    if (it == cache.end())
        it = cache.emplace(key, tsdf.DebugDownloadGroupVoxels(key)).first;
    const int lx = vx - key.gx * 8, ly = vy - key.gy * 8, lz = vz - key.gz * 8;
    const auto &vox = it->second[(uint32_t(lz) * 8u + uint32_t(ly)) * 8u + uint32_t(lx)];
    occupied = vox.weight > 0.0f;
    return vox.value;
}

// One z=0 plane facing +Z, spanning [-half, half] in x and y at `step` spacing (the
// "plane" scene from tsdf_slice_debug). Ground-truth surface for integrate/extract checks.
struct PlaneFixture {
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
};

inline PlaneFixture MakePlaneFixture(float half = 2.0f, float step = 0.1f) {
    PlaneFixture f;
    for (float x = -half; x <= half + 1e-4f; x += step)
        for (float y = -half; y <= half + 1e-4f; y += step) {
            f.points.emplace_back(x, y, 0.0f);
            f.normals.emplace_back(0.0f, 0.0f, 1.0f);
        }
    return f;
}

} // namespace tsdf_fixtures

// Visual diagnostic for the DirectionalTSDF integrate -> extract POINT-CLOUD pipeline.
//
// The pipeline has a hidden middle stage:
//   input points --[integrate]--> TSDF volume (hidden) --[extract]--> extracted points
// Looking only at the extracted cloud cannot tell an integrate bug (wrong field) from an
// extract bug (wrong zero-crossing). So this tool makes the hidden volume VISIBLE as a 2D
// slice, alongside the input and extracted clouds, on a KNOWN fixture.
//
//   tsdf_slice_debug [plane|interproximal]     (default: plane)
//
// Scene "plane": one z=0 plane facing +Z. Exports
//   plane_input.ply      white  - observed samples (ground-truth surface)
//   plane_extracted.ply  green  - extracted point cloud (output)
//   plane_sdf_slice.ply  color  - x-z slice of the +Z TSDF layer at y=0, colored by signed
//                                 distance: blue(<0 behind) -> white(0 surface) -> red(>0 front)
//   Check: white zero-crossing on the plane; green points on it; symmetric band.
//
// Scene "interproximal": two thin surfaces 0.4mm apart facing each other (a +Z surface at
//   z=-0.2 seen from above, a -Z surface at z=+0.2 seen from below) - the case a single SDF
//   would MERGE. DirectionalTSDF keeps them in separate direction layers. Exports
//   interp_input.ply       white  - both surfaces
//   interp_extracted.ply   dir-colored (cyan=+Z, magenta=-Z, yellow=both) - separation shows here
//   interp_slice_posZ.ply  color  - +Z layer slice: MUST show ONLY the z=-0.2 crossing
//   interp_slice_negZ.ply  color  - -Z layer slice: MUST show ONLY the z=+0.2 crossing
//   Check: each layer holds exactly one clean surface (no mixing across the 0.4mm gap).
//
// Overlay in CloudCompare / MeshLab. Divergences localize the bug:
//   slice zero-crossing wrong => integrate; slice right but extracted points off => extract.
#include "Engine/Core/Context.h"
#include "TSDF/Backends/DirectionalHostStore.h"
#include "TSDF/Backends/DirectionalTSDF.h"
#include "TSDF/Backends/DirectionalTSDFTypes.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using Engine::Spatial::DirectionalGroupKey;
using Engine::Spatial::DirectionalGroupKeyHash;
using Engine::Spatial::DirectionalTSDF;
using Group = Engine::Spatial::DirectionalHostStore::Group;
using Cache = std::unordered_map<DirectionalGroupKey, Group, DirectionalGroupKeyHash>;

namespace {

struct Rgb { uint8_t r, g, b; };

// Diverging SDF colormap: value in [-1,1] -> blue(-1) .. white(0) .. red(+1).
Rgb sdfColor(float value) {
    const float v = std::clamp(value, -1.0f, 1.0f);
    if (v < 0.0f) { const uint8_t c = uint8_t((v + 1.0f) * 255.0f); return {c, c, 255}; }
    const uint8_t c = uint8_t((1.0f - v) * 255.0f);
    return {255, c, c};
}

// Color an extracted point by its direction bitmask: +Z(bit4)=cyan, -Z(bit5)=magenta, both=yellow.
Rgb dirColor(uint8_t mask) {
    const bool pz = mask & (1u << 4), nz = mask & (1u << 5);
    if (pz && nz) return {230, 230, 40};
    if (pz) return {40, 220, 220};
    if (nz) return {220, 40, 220};
    return {160, 160, 160};
}

int floorDiv8(int a) { return a >= 0 ? a / 8 : -(((-a) + 7) / 8); }

void writePly(const std::string &path, const std::vector<Eigen::Vector3f> &pts,
              const std::vector<Rgb> &colors) {
    std::ofstream o(path);
    o << "ply\nformat ascii 1.0\nelement vertex " << pts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (size_t i = 0; i < pts.size(); ++i)
        o << pts[i].x() << ' ' << pts[i].y() << ' ' << pts[i].z() << ' ' << int(colors[i].r)
          << ' ' << int(colors[i].g) << ' ' << int(colors[i].b) << '\n';
}

// Value of the TSDF at world point `p` in a given direction layer. occupied=false if the
// group is not resident or the voxel has zero weight.
float valueAt(DirectionalTSDF &tsdf, Cache &cache, uint8_t dir, float voxelSize,
              const Eigen::Vector3f &p, bool &occupied) {
    const int vx = int(std::floor(p.x() / voxelSize));
    const int vy = int(std::floor(p.y() / voxelSize));
    const int vz = int(std::floor(p.z() / voxelSize));
    DirectionalGroupKey key{floorDiv8(vx), floorDiv8(vy), floorDiv8(vz), dir};
    if (tsdf.DebugQueryPoolIndex(key) == Engine::Spatial::kInvalidPoolIndex) { occupied = false; return 0.f; }
    auto it = cache.find(key);
    if (it == cache.end()) it = cache.emplace(key, tsdf.DebugDownloadGroupVoxels(key)).first;
    const int lx = vx - key.gx * 8, ly = vy - key.gy * 8, lz = vz - key.gz * 8;
    const auto &vox = it->second[(uint32_t(lz) * 8u + uint32_t(ly)) * 8u + uint32_t(lx)];
    occupied = vox.weight > 0.0f;
    return vox.value;
}

// Export an x-z slice of one direction layer at y=0, sampling at voxel centers. Returns the
// interpolated zero-crossing z along the central column (or 1e9 if none) for a numeric check.
float exportSlice(DirectionalTSDF &tsdf, uint8_t dir, float voxelSize, float xHalf, float zHalf,
                  const std::string &path) {
    Cache cache;
    const int kxLo = int(std::floor(-xHalf / voxelSize)), kxHi = int(std::floor(xHalf / voxelSize));
    const int kzLo = int(std::floor(-zHalf / voxelSize)), kzHi = int(std::floor(zHalf / voxelSize));
    std::vector<Eigen::Vector3f> sp; std::vector<Rgb> sc;
    for (int kx = kxLo; kx <= kxHi; ++kx)
        for (int kz = kzLo; kz <= kzHi; ++kz) {
            bool occ = false;
            const Eigen::Vector3f p((kx + 0.5f) * voxelSize, 0.0f, (kz + 0.5f) * voxelSize);
            const float v = valueAt(tsdf, cache, dir, voxelSize, p, occ);
            if (!occ) continue;
            sp.push_back(p); sc.push_back(sdfColor(v));
        }
    writePly(path, sp, sc);
    // central-column zero-crossing
    float prevZ = 0, prevV = 0, crossZ = 1e9f; bool prevOcc = false;
    for (int kz = kzLo; kz <= kzHi; ++kz) {
        const float zc = (kz + 0.5f) * voxelSize; bool occ = false;
        const float v = valueAt(tsdf, cache, dir, voxelSize, Eigen::Vector3f(0.05f, 0, zc), occ);
        if (occ && prevOcc && ((v > 0) != (prevV > 0)) && v != prevV)
            crossZ = prevZ + (prevV / (prevV - v)) * (zc - prevZ);
        prevOcc = occ; prevZ = zc; prevV = v;
    }
    std::cout << "  slice " << path << ": " << sp.size() << " occupied voxels, zero-crossing z="
              << crossZ << " mm\n";
    return crossZ;
}

void exportExtracted(const DirectionalTSDF &tsdf, const std::string &path, bool byDir) {
    std::vector<Eigen::Vector3f> p; std::vector<Rgb> c;
    float sumAbsZ = 0, maxAbsZ = 0;
    for (const auto &e : tsdf.PointCloud()) {
        p.push_back(e.position);
        c.push_back(byDir ? dirColor(e.dirMask) : Rgb{40, 220, 40});
        sumAbsZ += std::fabs(e.position.z()); maxAbsZ = std::max(maxAbsZ, std::fabs(e.position.z()));
    }
    writePly(path, p, c);
    std::cout << "  extracted " << path << ": " << p.size() << " points; |z| mean="
              << (p.empty() ? 0.f : sumAbsZ / p.size()) << " max=" << maxAbsZ << " mm\n";
}

} // namespace

int main(int argc, char **argv) {
    const std::string scene = argc > 1 ? argv[1] : "plane";
    Engine::Core::Context ctx;
    constexpr float voxelSize = 0.1f, truncation = 0.3f, half = 2.0f, step = 0.1f;

    DirectionalTSDF tsdf;
    tsdf.Build(ctx, voxelSize, truncation);

    if (scene == "interproximal") {
        // Two surfaces 0.4mm apart, facing INTO the gap. Each seen from its own side.
        std::vector<Eigen::Vector3f> pA, nA, pB, nB;
        for (float x = -half; x <= half + 1e-4f; x += step)
            for (float y = -half; y <= half + 1e-4f; y += step) {
                pB.emplace_back(x, y, -0.2f); nB.emplace_back(0, 0, 1);   // +Z-facing, seen from above
                pA.emplace_back(x, y, 0.2f);  nA.emplace_back(0, 0, -1);  // -Z-facing, seen from below
            }
        tsdf.Integrate(pB, nB, Eigen::Vector3f(0, 0, 5), Eigen::Vector3f::Zero());   // +Z layer
        tsdf.Integrate(pA, nA, Eigen::Vector3f(0, 0, -5), Eigen::Vector3f::Zero());  // -Z layer

        std::vector<Eigen::Vector3f> allP = pB; allP.insert(allP.end(), pA.begin(), pA.end());
        std::vector<Rgb> w(allP.size(), {255, 255, 255});
        writePly("interp_input.ply", allP, w);
        std::cout << "interproximal: 2 surfaces at z=-0.2(+Z) and z=+0.2(-Z), gap 0.4mm\n";
        exportExtracted(tsdf, "interp_extracted.ply", /*byDir=*/true);
        const float cz = exportSlice(tsdf, 4, voxelSize, half, 0.5f, "interp_slice_posZ.ply");
        const float cn = exportSlice(tsdf, 5, voxelSize, half, 0.5f, "interp_slice_negZ.ply");
        std::cout << "\nEXPECT separation: +Z layer crossing ~ -0.2 (got " << cz
                  << "), -Z layer crossing ~ +0.2 (got " << cn << ").\n"
                  << "If either layer shows the OTHER surface, directional separation is broken.\n"
                  << "wrote interp_input/extracted/slice_posZ/slice_negZ .ply\n";
    } else {
        // Single plane at z=0 facing +Z.
        std::vector<Eigen::Vector3f> pts, nrm;
        for (float x = -half; x <= half + 1e-4f; x += step)
            for (float y = -half; y <= half + 1e-4f; y += step) { pts.emplace_back(x, y, 0.0f); nrm.emplace_back(0, 0, 1); }
        tsdf.Integrate(pts, nrm, Eigen::Vector3f(0, 0, 5), Eigen::Vector3f::Zero());

        std::vector<Rgb> w(pts.size(), {255, 255, 255});
        writePly("plane_input.ply", pts, w);
        std::cout << "plane: z=0, +Z-facing\n";
        exportExtracted(tsdf, "plane_extracted.ply", /*byDir=*/false);
        exportSlice(tsdf, 4, voxelSize, half, 2.0f * truncation, "plane_sdf_slice.ply");
        std::cout << "\nEXPECT: extracted |z|~0, slice zero-crossing z~0.\n"
                  << "wrote plane_input/extracted/sdf_slice .ply\n";
    }
    return 0;
}

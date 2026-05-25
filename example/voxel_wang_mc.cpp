#include "vkSpatial/common/vkContext.h"
#include "vkSpatial/vkVoxelHash.h"

#include <Eigen/Core>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

static void savePointsPLY(const std::string &path,
                          const std::vector<Eigen::Vector3f> &pts) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("savePointsPLY: cannot open " + path);
    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << pts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "end_header\n";
    for (const auto &p: pts)
        f << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
}

int main() {
    // ── Vulkan context ─────────────────────────────────────────────────────────
    vkCommon::VkContext ctx;
    ctx.init();

    constexpr float voxelSize = 0.1f;
    constexpr float pointResolution = 0.1f;
    constexpr float radius = 1.0f;
    constexpr uint32_t hashCapacity = 1u << 20; // 16 384 slots (4× load factor)
    constexpr uint32_t maxPoints = 1u << 14;    // 8 192 points per Integrate call

    {
        std::vector<Eigen::Vector3f> fullPoint;
        vkSpatial::vkVoxelHash hash;
        hash.Build(&ctx, voxelSize, hashCapacity, maxPoints);

        for (float x = -radius; x <= radius + 1e-4f; x += pointResolution) {
            std::vector<Eigen::Vector3f> points;
            for (float y = -radius; y <= radius + 1e-4f; y += pointResolution) {
                for (float z = -radius; z <= radius + 1e-4f; z += pointResolution) {
                    if (x * x + y * y + z * z <= radius * radius) {
                        points.emplace_back(x, y, z);
                        fullPoint.emplace_back(x, y, z);
                    }
                }
            }
            std::cout << "Generated " << points.size() << " sphere points\n";
            hash.Integrate(points);
        }
        std::cout << "Filled voxels : " << hash.FilledCount() << "\n";

        // ── Save input points ──────────────────────────────────────────────────
        savePointsPLY("sphere_points.ply", fullPoint);
        std::cout << "Exported points: sphere_points.ply\n";

        // ── Marching Cubes export ──────────────────────────────────────────────
        hash.ExportMC("sphere_mc.ply");
        std::cout << "Exported mesh  : sphere_mc.ply\n";
    } // hash destroyed here, before ctx.shutdown()

    ctx.shutdown();
    return 0;
}

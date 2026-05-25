#include "vkSpatial/common/vkContext.h"
#include "vkSpatial/vkTSDF.h"

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
    vkCommon::VkContext ctx;
    ctx.init();

    constexpr float    voxelSize       = 0.1f;
    constexpr float    truncation      = 0.3f;
    constexpr float    pointResolution = 0.05f;
    constexpr float    radius          = 1.0f;
    constexpr float    camRadius       = 3.0f;   // camera orbit distance
    constexpr uint32_t hashCapacity    = 1u << 20;
    constexpr uint32_t maxPoints       = 1u << 15; // 32768 — headroom for large slices

    // ── Pre-generate sphere surface points (thin shell ±voxelSize/2 around r=1) ──
    std::vector<Eigen::Vector3f> surfacePoints;
    const float rLo = (radius - voxelSize * 0.5f) * (radius - voxelSize * 0.5f);
    const float rHi = (radius + voxelSize * 0.5f) * (radius + voxelSize * 0.5f);
    for (float x = -radius - voxelSize; x <= radius + voxelSize + 1e-4f; x += pointResolution)
        for (float y = -radius - voxelSize; y <= radius + voxelSize + 1e-4f; y += pointResolution)
            for (float z = -radius - voxelSize; z <= radius + voxelSize + 1e-4f; z += pointResolution) {
                float r2 = x*x + y*y + z*z;
                if (r2 >= rLo && r2 <= rHi)
                    surfacePoints.emplace_back(x, y, z);
            }
    std::cout << "Surface points: " << surfacePoints.size() << "\n";
    savePointsPLY("tsdf_sphere_points.ply", surfacePoints);

    {
        vkSpatial::vkTSDF tsdf;
        tsdf.Build(&ctx, voxelSize, truncation, hashCapacity, maxPoints);

        // ── Spiral camera orbit ──────────────────────────────────────────────────
        // Camera sweeps numElevation latitude rings × numAzimuth positions each.
        // At every pose, only points whose outward normal faces the camera are
        // integrated — identical to a real depth sensor scanning around an object.
        constexpr int   numElevation = 9;    // rings from -80° to +80°
        constexpr int   numAzimuth   = 36;   // 10° steps per ring
        constexpr float cosThreshold = 0.15f; // reject near-grazing rays (>81°)

        int totalPoses = 0, totalPts = 0;
        for (int el = 0; el < numElevation; el++) {
            float elDeg = -80.0f + 160.0f * static_cast<float>(el) / (numElevation - 1);
            float elRad = elDeg * static_cast<float>(M_PI) / 180.0f;
            float cosEl = std::cos(elRad), sinEl = std::sin(elRad);

            for (int az = 0; az < numAzimuth; az++) {
                float azRad = 2.0f * static_cast<float>(M_PI) * az / numAzimuth;

                Eigen::Vector3f cam(
                    camRadius * cosEl * std::cos(azRad),
                    camRadius * sinEl,
                    camRadius * cosEl * std::sin(azRad));

                // Collect visible surface points from this pose.
                // Visibility: outward normal (= p/|p| for unit sphere) points toward cam.
                std::vector<Eigen::Vector3f> visible;
                visible.reserve(512);
                for (const auto &p : surfacePoints) {
                    Eigen::Vector3f dir = (cam - p).normalized();
                    if (p.normalized().dot(dir) > cosThreshold)
                        visible.push_back(p);
                }

                if (!visible.empty()) {
                    tsdf.Integrate(visible, cam);
                    totalPts += static_cast<int>(visible.size());
                    ++totalPoses;
                }
            }
        }

        std::cout << "Poses: " << totalPoses
                  << "  integrated pts (total): " << totalPts << "\n";
        std::cout << "Filled voxels: " << tsdf.FilledCount() << "\n";

        savePointsPLY("tsdf_sphere_points.ply", surfacePoints);
        std::cout << "Exported points: tsdf_sphere_points.ply\n";

        tsdf.ExportMC("tsdf_sphere_mc.ply");
        std::cout << "Exported mesh  : tsdf_sphere_mc.ply\n";
    }

    ctx.shutdown();
    return 0;
}

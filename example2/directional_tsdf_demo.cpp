// Interproximal scenario demo: two opposing thin surfaces 0.4mm apart, scanned by a
// scripted camera path sliding along z. Logs per-frame streaming stats + stage timings
// as CSV, then exports the directional point cloud and a single-SDF (SimpleTSDF)
// marching-cubes mesh of the same data for visual comparison.
#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalTSDF.h"
#include "Engine/Spatial/SimpleTSDF.h"

#include <Eigen/Core>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    Engine::Core::Context ctx;

    constexpr float voxelSize = 0.1f;
    constexpr float truncation = 0.3f;
    constexpr float gapHalf = 0.2f;   // planes at x = ±0.2 → 0.4 gap; bands overlap in ±0.1
    constexpr float extentY = 1.0f;
    constexpr float extentZ = 6.0f;
    constexpr float step = 0.05f;
    constexpr float footprint = 1.5f; // scanner footprint half-width along z
    constexpr int numFrames = 40;

    // Full synthetic surfaces (left faces +X, right faces -X).
    struct Sample {
        Eigen::Vector3f p, n;
    };
    std::vector<Sample> surface;
    for (float y = -extentY; y <= extentY + 1e-4f; y += step)
        for (float z = -extentZ; z <= extentZ + 1e-4f; z += step) {
            surface.push_back({{-gapHalf, y, z}, {1, 0, 0}});
            surface.push_back({{+gapHalf, y, z}, {-1, 0, 0}});
        }
    std::cout << "surface samples: " << surface.size() << "\n";

    Engine::Spatial::DirectionalTSDF tsdf;
    tsdf.Build(ctx, voxelSize, truncation);

    Engine::Spatial::SimpleTSDF simple;
    simple.Build(ctx, voxelSize, truncation, 1u << 20, 1u << 15);

    std::ofstream csv("directional_tsdf_stats.csv");
    csv << "frame,points,resident,missing,overlapPct,h2dKB,d2hKB,gpuSubmits,writeBack,"
           "beginMs,ensureMs,integrateMs,extractMs,mergeMs,cloudPoints\n";

    for (int f = 0; f < numFrames; ++f) {
        const float camZ = -5.0f + 0.25f * float(f);
        const Eigen::Vector3f cam(0.0f, 2.5f, camZ);

        // Frame patch: samples inside the scanner footprint.
        std::vector<Eigen::Vector3f> points, normals;
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto &s : surface) {
            if (std::fabs(s.p.z() - camZ) > footprint) continue;
            points.push_back(s.p);
            normals.push_back(s.n);
            centroid += s.p;
        }
        if (points.empty()) continue;
        centroid /= float(points.size());

        tsdf.Integrate(points, normals, cam, centroid);
        simple.Integrate(points, cam);

        const auto st = tsdf.LastFrameStats();
        csv << f << ',' << points.size() << ',' << st.residentCount << ','
            << st.missingCount << ',' << st.overlapRatio * 100.0f << ','
            << st.h2dBytes / 1024 << ',' << st.d2hBytes / 1024 << ',' << st.gpuSubmits << ','
            << st.writeBackCount << ',' << st.beginFrameMs << ',' << st.ensureResidentMs << ','
            << st.integrateMs << ',' << st.extractMs << ',' << st.mergeMs << ','
            << tsdf.PointCloud().size() << '\n';

        std::cout << "frame " << f << ": pts=" << points.size()
                  << " missing=" << st.missingCount
                  << " overlap=" << st.overlapRatio * 100.0f << "%"
                  << " submits=" << st.gpuSubmits
                  << " cloud=" << tsdf.PointCloud().size() << "\n";
    }

    tsdf.ExportPointCloud("directional_result.ply");
    simple.ExportMC("simple_result_mc.ply");
    std::cout << "wrote directional_tsdf_stats.csv, directional_result.ply, "
                 "simple_result_mc.ply\n";
    return 0;
}

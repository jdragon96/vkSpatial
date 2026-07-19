// Real-data benchmark: integrate/extract an observed multi-frame point cloud scan
// (scanData/frame_*.ply — x,y,z,nx,ny,nz, registered in a common mm frame) through
// DirectionalTSDF, comparing SINGLE-direction (baseline) vs MULTI-direction + view-angle
// confidence (the spec-§7 quality upgrade). Reports throughput + coverage and exports
// each reconstruction as a PLY for visual comparison.
//
// Usage: directional_tsdf_chair_benchmark [scanData_dir]   (default: "scanData")
//
// The scene (~560x525x827 mm) is coarse for a 50-group window, so voxelSize is scaled so
// the whole scan fits one fixed window; this benchmarks the pipeline on real observed data,
// not clinical resolution. Per-frame camera is estimated as centroid + D*meanNormal (the
// PLY carries no pose; normals face the sensor).
#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalTSDF.h"

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Frame {
    std::vector<Eigen::Vector3f> points, normals;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f meanNormal = Eigen::Vector3f::Zero();
};

// Minimal ASCII-PLY reader for the x,y,z,nx,ny,nz layout used by scanData/frame_*.ply.
bool loadFrame(const std::string &path, Frame &out) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    bool inData = false;
    while (std::getline(in, line)) {
        if (!inData) {
            if (line.rfind("end_header", 0) == 0) inData = true;
            continue;
        }
        std::istringstream ss(line);
        float x, y, z, nx, ny, nz;
        if (!(ss >> x >> y >> z >> nx >> ny >> nz)) continue;
        out.points.emplace_back(x, y, z);
        out.normals.emplace_back(nx, ny, nz);
        out.centroid += Eigen::Vector3f(x, y, z);
        out.meanNormal += Eigen::Vector3f(nx, ny, nz);
    }
    if (out.points.empty()) return false;
    out.centroid /= float(out.points.size());
    if (out.meanNormal.norm() > 1e-6f) out.meanNormal.normalize();
    else out.meanNormal = Eigen::Vector3f(0, 0, 1);
    return true;
}

struct Result {
    double integrateMs = 0, extractMs = 0, mergeMs = 0;
    size_t totalPointsIn = 0;
    size_t cloudPoints = 0;
    uint32_t residentGroups = 0;
};

Result runScan(Engine::Core::Context &ctx, const std::vector<Frame> &frames,
               const Eigen::Vector3f &sceneCenter, float voxelSize, float truncation,
               uint32_t poolCapacity, uint32_t maxPoints, uint32_t maxCandidates,
               const Engine::Spatial::IntegrationQuality &quality, const char *label,
               const std::string &outPly) {
    Engine::Spatial::DirectionalTSDF tsdf;
    // Unified (UMA) backend: the whole scan lives in one fixed window, so the first frame's
    // missing set covers most of the model at once — that exceeds the Streaming staging cap
    // (chunked upload is an unimplemented Phase-5 TODO). Unified writes voxels straight into
    // the coherent pool with no staging, bounded only by poolCapacity.
    tsdf.Build(ctx, voxelSize, truncation, poolCapacity, maxPoints, maxCandidates,
               Engine::Spatial::ResidencyMode::Unified);
    tsdf.SetIntegrationQuality(quality);

    Result r;
    const float camDist = 1000.0f; // mm, camera placed along the frame's mean normal
    for (size_t f = 0; f < frames.size(); ++f) {
        const Frame &fr = frames[f];
        const Eigen::Vector3f cam = fr.centroid + camDist * fr.meanNormal;
        tsdf.Integrate(fr.points, fr.normals, cam, sceneCenter);
        const auto st = tsdf.LastFrameStats();
        r.integrateMs += st.integrateMs;
        r.extractMs += st.extractMs;
        r.mergeMs += st.mergeMs;
        r.totalPointsIn += fr.points.size();
        r.residentGroups = st.residentCount;
        std::cout << "  [" << label << "] frame " << f << ": in=" << fr.points.size()
                  << " resident=" << st.residentCount << " cloud=" << tsdf.PointCloud().size()
                  << " (integ " << std::fixed << std::setprecision(1) << st.integrateMs
                  << "ms, extr " << st.extractMs << "ms)\n";
    }
    r.cloudPoints = tsdf.PointCloud().size();
    tsdf.ExportPointCloud(outPly);
    return r;
}

} // namespace

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "scanData";
    const int maxFrames = argc > 2 ? std::atoi(argv[2]) : 1000;

    // Load all frames (up to maxFrames).
    std::vector<Frame> frames;
    for (int i = 0; i < maxFrames; ++i) {
        std::ostringstream name;
        name << dir << "/frame_" << std::setw(4) << std::setfill('0') << i << ".ply";
        Frame fr;
        if (!loadFrame(name.str(), fr)) break;
        std::cout << "loaded " << name.str() << ": " << fr.points.size() << " points\n";
        frames.push_back(std::move(fr));
    }
    if (frames.empty()) {
        std::cerr << "no frames found under '" << dir << "' (expected frame_0000.ply ...)\n";
        return 1;
    }

    // Scene bbox → center (window hint) + a voxel size that fits the whole scan in one window.
    Eigen::Vector3f bbMin = frames[0].points[0], bbMax = frames[0].points[0];
    size_t totalIn = 0;
    for (const auto &fr : frames)
        for (const auto &p : fr.points) {
            bbMin = bbMin.cwiseMin(p);
            bbMax = bbMax.cwiseMax(p);
            ++totalIn;
        }
    const Eigen::Vector3f center = 0.5f * (bbMin + bbMax);
    const Eigen::Vector3f extent = bbMax - bbMin;
    const float maxExtent = extent.maxCoeff();
    // Window side = 50 groups * 8 voxels * voxelSize = 400 * voxelSize; leave ~15% margin.
    const float voxelSize = (maxExtent / 400.0f) * 1.15f;
    const float truncation = 3.0f * voxelSize;

    std::cout << "\nscene: " << frames.size() << " frames, " << totalIn << " points\n"
              << "  bbox extent " << extent.x() << " x " << extent.y() << " x " << extent.z()
              << " mm, center (" << center.x() << ", " << center.y() << ", " << center.z() << ")\n"
              << "  voxelSize " << voxelSize << " mm, truncation " << truncation << " mm\n\n";

    // Buffers sized for the whole scene held in one fixed window.
    const uint32_t maxPoints = 1u << 17;      // >= largest frame (~84k)
    const uint32_t maxCandidates = 1u << 21;  // whole-scene extraction
    const uint32_t poolCapacity = 1u << 18;   // all surface groups resident at once

    Engine::Core::Context ctx;
    using Q = Engine::Spatial::IntegrationQuality;

    std::cout << "=== SINGLE-direction (baseline: K=1, no view-angle) ===\n";
    Result single = runScan(ctx, frames, center, voxelSize, truncation, poolCapacity,
                            maxPoints, maxCandidates, Q{1, 4, false}, "single",
                            "chair_recon_single.ply");

    std::cout << "\n=== MULTI-direction (K=2, view-angle confidence) ===\n";
    Result multi = runScan(ctx, frames, center, voxelSize, truncation, poolCapacity,
                           maxPoints, maxCandidates, Q{2, 4, true}, "multi",
                           "chair_recon_multi.ply");

    auto row = [](const char *name, const Result &r) {
        std::cout << std::left << std::setw(10) << name << std::right << std::fixed
                  << std::setprecision(1) << std::setw(12) << r.integrateMs
                  << std::setw(12) << r.extractMs << std::setw(12) << r.mergeMs
                  << std::setw(14) << r.residentGroups << std::setw(14) << r.cloudPoints << "\n";
    };
    std::cout << "\n==================== BENCHMARK SUMMARY ====================\n"
              << std::left << std::setw(10) << "mode" << std::right << std::setw(12) << "integ_ms"
              << std::setw(12) << "extr_ms" << std::setw(12) << "merge_ms" << std::setw(14)
              << "resident" << std::setw(14) << "cloudPts" << "\n";
    row("single", single);
    row("multi", multi);
    std::cout << "\ntotal input points: " << totalIn << " over " << frames.size() << " frames\n"
              << "wrote chair_recon_single.ply, chair_recon_multi.ply\n"
              << "coverage multi/single = " << std::setprecision(3)
              << (single.cloudPoints ? double(multi.cloudPoints) / double(single.cloudPoints) : 0.0)
              << "\n";
    return 0;
}

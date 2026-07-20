// Raw-frame registration demo: drops the "scanData is already a common frame" assumption
// that directional_tsdf_chair_benchmark.cpp relies on and instead RECOVERS the world pose
// of each frame via Engine::Registration (FPFH + RANSAC + Ceres refine), composing an
// accumulating pose from consecutive-frame relative registrations before feeding the
// transformed points into Engine::Spatial::DirectionalTSDF.
//
// Usage: registration_chair_demo <scanData_dir> [maxFrames=8]
//
// Sanity signal (this is a demo, not a unit test -- there is no assertion): scanData's
// frame_*.ply files are already captured in a common registered frame (see
// directional_tsdf_chair_benchmark.cpp's header comment), so the consecutive-frame
// relative transform recovered by Estimate() below should come out close to IDENTITY
// (small rotation/translation magnitude) with a high inlier count. That is the printed
// success signal: near-identity deltas on real data means the registration pipeline
// (Tasks 1-5) works end-to-end, not just on synthetic fixtures.
#include "Engine/Core/Context.h"
#include "Engine/Registration/GlobalRegistration.h"
#include "Engine/Registration/RegistrationTypes.h"
#include "Engine/Spatial/DirectionalTSDF.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
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
};

// Minimal ASCII-PLY reader for the x,y,z,nx,ny,nz layout used by scanData/frame_*.ply
// (same loader as directional_tsdf_chair_benchmark.cpp).
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
    }
    return !out.points.empty();
}

Eigen::Vector3f centroidOf(const std::vector<Eigen::Vector3f> &pts) {
    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (const auto &p : pts) c += p;
    if (!pts.empty()) c /= float(pts.size());
    return c;
}

Eigen::Vector3f meanNormalOf(const std::vector<Eigen::Vector3f> &nrms) {
    Eigen::Vector3f m = Eigen::Vector3f::Zero();
    for (const auto &n : nrms) m += n;
    if (m.norm() > 1e-6f) m.normalize();
    else m = Eigen::Vector3f(0, 0, 1);
    return m;
}

} // namespace

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "scanData";
    const int maxFrames = argc > 2 ? std::atoi(argv[2]) : 8;

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

    // Registration config: raw frames are ~77k points at mm scale; MatchFeatures is O(N^2)
    // brute-force, so a 10mm voxel downsample (done internally by Estimate/EstimateRansac)
    // keeps the per-frame keypoint count in the low thousands. SEPARATE from the TSDF
    // integration voxelSize computed below.
    Engine::Registration::RegistrationConfig regCfg;
    regCfg.voxelSize = 10.0f;

    // Scene bbox over all loaded (raw) frames sizes the TSDF window/voxel exactly as
    // directional_tsdf_chair_benchmark.cpp does. Since the sanity expectation is that
    // recovered world poses stay near-identity, the raw-frame bbox is a fair stand-in for
    // the reconstructed (world-frame) scene extent.
    Eigen::Vector3f bbMin = frames[0].points[0], bbMax = frames[0].points[0];
    for (const auto &fr : frames)
        for (const auto &p : fr.points) {
            bbMin = bbMin.cwiseMin(p);
            bbMax = bbMax.cwiseMax(p);
        }
    const Eigen::Vector3f sceneCenter = 0.5f * (bbMin + bbMax);
    const Eigen::Vector3f extent = bbMax - bbMin;
    const float maxExtent = extent.maxCoeff();
    // Window side = 50 groups * 8 voxels/group * voxelSize = 400 * voxelSize; ~15% margin.
    const float voxelSize = (maxExtent / 400.0f) * 1.15f;
    const float truncation = 3.0f * voxelSize;

    std::cout << "\nscene: " << frames.size() << " frames\n"
              << "  bbox extent " << extent.x() << " x " << extent.y() << " x " << extent.z()
              << " mm\n"
              << "  TSDF voxelSize " << voxelSize << " mm, truncation " << truncation << " mm\n"
              << "  registration voxelSize " << regCfg.voxelSize << " mm\n\n";

    // Buffers sized for the whole scene held in one fixed (Unified) window, matching the
    // benchmark's sizing.
    const uint32_t maxPoints = 1u << 17;     // >= largest frame (~84k)
    const uint32_t maxCandidates = 1u << 21; // whole-scene extraction
    const uint32_t poolCapacity = 1u << 18;  // all surface groups resident at once

    Engine::Core::Context ctx;
    Engine::Spatial::DirectionalTSDF tsdf;
    tsdf.Build(ctx, voxelSize, truncation, poolCapacity, maxPoints, maxCandidates,
               Engine::Spatial::ResidencyMode::Unified);

    Eigen::Matrix4f Tworld = Eigen::Matrix4f::Identity();
    for (size_t f = 0; f < frames.size(); ++f) {
        if (f > 0) {
            // Register raw frame f (src) against raw frame f-1 (tgt): T maps frame f's local
            // points into frame f-1's local frame. Composing onto the running world pose
            // (which already maps frame f-1's local points into world) yields the world pose
            // for frame f.
            Engine::Registration::PointCloud src{frames[f].points, frames[f].normals};
            Engine::Registration::PointCloud tgt{frames[f - 1].points, frames[f - 1].normals};
            Engine::Registration::RegistrationResult res =
                    Engine::Registration::Estimate(src, tgt, regCfg);

            const Eigen::Matrix3f R = res.T.block<3, 3>(0, 0);
            const float rot = Eigen::AngleAxisf(Eigen::Matrix3f(R)).angle();
            const float trans = res.T.block<3, 1>(0, 3).norm();
            std::cout << "  frame " << f << " -> " << (f - 1) << ": valid=" << (res.valid ? "1" : "0")
                      << " inliers=" << res.numInliers << " fitness=" << std::fixed
                      << std::setprecision(3) << res.fitness << " | relRot=" << rot
                      << " rad relTrans=" << trans << " mm\n";

            if (res.valid) {
                Tworld = Tworld * res.T;
            } else {
                std::cout << "    WARNING: registration invalid for frame " << f << " -> "
                          << (f - 1) << "; SKIPPING integration (untrusted alignment)\n";
                continue; // gate: never integrate a frame whose alignment is not valid
                          // (a stale/wrong pose would inject misplaced geometry into the TSDF)
            }
        }

        // Transform frame f into the accumulated world frame.
        const Eigen::Matrix3f Rw = Tworld.block<3, 3>(0, 0);
        const Eigen::Vector3f tw = Tworld.block<3, 1>(0, 3);
        const Frame &fr = frames[f];
        std::vector<Eigen::Vector3f> pw(fr.points.size());
        std::vector<Eigen::Vector3f> nw(fr.normals.size());
        for (size_t i = 0; i < pw.size(); ++i) pw[i] = Rw * fr.points[i] + tw;
        for (size_t i = 0; i < nw.size(); ++i) nw[i] = (Rw * fr.normals[i]).normalized();

        const Eigen::Vector3f centroid = centroidOf(pw);
        const Eigen::Vector3f meanNormal = meanNormalOf(nw);
        const float camDist = 1000.0f; // mm, camera placed along the frame's mean normal
        const Eigen::Vector3f cam = centroid + camDist * meanNormal;

        // aabbCenterHint: the FIXED whole-scene center (computed once above from all raw
        // frames), not this frame's own (partial-view) centroid. The Unified window here is
        // sized with only a ~15% margin around the whole scan (matching the benchmark's
        // "entire scan in one window" sizing) specifically so it never needs to move;
        // recentering it per-frame on a single partial view's centroid pushes already-
        // integrated groups from earlier frames outside the window and
        // UnifiedResidencyBackend::EnsureResident throws ("required key outside window").
        // A per-frame hint would only be safe with a much larger margin (i.e. a coarser
        // voxelSize) or a streaming/multi-window backend -- out of scope for this demo.
        tsdf.Integrate(pw, nw, cam, /*aabbCenterHint=*/sceneCenter);
        const auto st = tsdf.LastFrameStats();
        std::cout << "    integrated: resident=" << st.residentCount
                  << " cloud=" << tsdf.PointCloud().size() << " (integ " << std::fixed
                  << std::setprecision(1) << st.integrateMs << "ms, extr " << st.extractMs
                  << "ms)\n";
    }

    tsdf.ExportPointCloud("chair_registered_recon.ply");
    std::cout << "\nfinal reconstructed cloud: " << tsdf.PointCloud().size() << " points\n"
              << "wrote chair_registered_recon.ply\n";
    return 0;
}

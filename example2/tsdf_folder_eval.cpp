// TSDF folder evaluator — read every frame_*.ply in a folder, integrate them into AdvancedTSDF,
// extract the surface, and score it against ground_truth.ply (RMSE). Fully separate from
// capture (object_scan_viewer) and headless.
//
//   ./tsdf_folder_eval --dir scans/bunny [--voxel v] [--trunc t] [--gt path.ply]
//        [--out extracted.ply] [--no-p2p] [--conf lambda] [--hermite]
//
// Per-frame camera position is ESTIMATED from each cloud (centroid + k·mean-normal), so this
// works on any folder of oriented-point PLYs, not just object_scan_viewer output.

#include "Engine/Core/Context.h"
#include "Engine/Spatial/AdvancedTSDF.h"
#include "Engine/Spatial/SubmapAdvancedTSDF.h"
#include "Engine/Spatial/OrientedPointCloud.h"
#include "Engine/Spatial/TiledAdvancedTSDF.h"

#include "utilities/PointCloudIO.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using Eigen::Vector3f;
namespace fs = std::filesystem;

namespace {

    std::string strArg(int argc, char **argv, const char *k, const std::string &d) {
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == k) return argv[i + 1];
        return d;
    }
    float floatArg(int argc, char **argv, const char *k, float d) {
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == k) return float(std::atof(argv[i + 1]));
        return d;
    }
    bool flag(int argc, char **argv, const char *k) {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == k) return true;
        return false;
    }

    uint32_t nextPow2(uint32_t v) {
        if (v <= 1) return 1;
        --v;
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
        return v + 1;
    }

    // Estimate the camera position for a captured cloud: the visible points face the sensor, so
    // the camera sits along the mean normal, outside the cloud.
    Vector3f estimateCamera(const std::vector<Vector3f> &pts, const std::vector<Vector3f> &nrm) {
        Vector3f centroid = Vector3f::Zero(), meanN = Vector3f::Zero();
        for (const auto &p: pts) centroid += p;
        for (const auto &n: nrm) meanN += n;
        centroid /= float(std::max<size_t>(1, pts.size()));
        Vector3f mn = pts[0], mx = pts[0];
        for (const auto &p: pts) {
            mn = mn.cwiseMin(p);
            mx = mx.cwiseMax(p);
        }
        const float diag = (mx - mn).norm();
        if (meanN.norm() > 1e-6f) meanN.normalize();
        else
            meanN = Vector3f(0, 0, 1);
        return centroid + std::max(1.0f, 3.0f * diag) * meanN;
    }

    // Uniform-grid nearest-neighbour over a reference cloud (for RMSE against ground truth).
    struct GridNN {
        float cell = 1.0f;
        std::unordered_map<int64_t, std::vector<int>> grid;
        const std::vector<Vector3f> *pts = nullptr;

        static int64_t key(int x, int y, int z) {
            return (int64_t(x) & 0x1FFFFF) | ((int64_t(y) & 0x1FFFFF) << 21) |
                   ((int64_t(z) & 0x1FFFFF) << 42);
        }
        void build(const std::vector<Vector3f> &p, float cellSize) {
            pts = &p;
            cell = std::max(1e-6f, cellSize);
            for (int i = 0; i < int(p.size()); ++i) {
                const Vector3f &q = p[i];
                grid[key(int(std::floor(q.x() / cell)), int(std::floor(q.y() / cell)),
                         int(std::floor(q.z() / cell)))]
                        .push_back(i);
            }
        }
        float nearestSq(const Vector3f &q) const {
            const int cx = int(std::floor(q.x() / cell)), cy = int(std::floor(q.y() / cell)),
                      cz = int(std::floor(q.z() / cell));
            float best = std::numeric_limits<float>::infinity();
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        auto it = grid.find(key(cx + dx, cy + dy, cz + dz));
                        if (it == grid.end()) continue;
                        for (int idx: it->second) best = std::min(best, (q - (*pts)[idx]).squaredNorm());
                    }
            // Expand the search ring until a hit is found (sparse regions).
            for (int ring = 2; !std::isfinite(best) && ring <= 32; ++ring) {
                for (int dz = -ring; dz <= ring; ++dz)
                    for (int dy = -ring; dy <= ring; ++dy)
                        for (int dx = -ring; dx <= ring; ++dx) {
                            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != ring) continue;
                            auto it = grid.find(key(cx + dx, cy + dy, cz + dz));
                            if (it == grid.end()) continue;
                            for (int idx: it->second)
                                best = std::min(best, (q - (*pts)[idx]).squaredNorm());
                        }
            }
            return best;
        }
    };

    struct Err {
        double sum = 0, sumSq = 0;
        size_t n = 0;
        void add(double d) {
            sum += d;
            sumSq += d * d;
            ++n;
        }
        double mean() const { return n ? sum / double(n) : 0; }
        double rmse() const { return n ? std::sqrt(sumSq / double(n)) : 0; }
    };

} // namespace

int main(int argc, char **argv) {
    try {
        const std::string dir = strArg(argc, argv, "--dir", "");
        if (dir.empty() || !fs::is_directory(dir)) {
            std::cerr << "usage: tsdf_folder_eval --dir <folder> [--voxel v] [--trunc t] "
                         "[--gt path] [--out extracted.ply] [--no-p2p] [--conf L] [--hermite]\n";
            return 2;
        }

        // Collect frame_*.ply (sorted); ground_truth.ply is excluded and used as GT.
        std::vector<std::string> framePaths;
        std::string gtPath = strArg(argc, argv, "--gt", "");
        for (const auto &e: fs::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            const std::string name = e.path().filename().string();
            if (name.rfind("ground_truth", 0) == 0) {
                if (gtPath.empty()) gtPath = e.path().string();
                continue;
            }
            if (e.path().extension() == ".ply" && name.rfind("frame_", 0) == 0)
                framePaths.push_back(e.path().string());
        }
        std::sort(framePaths.begin(), framePaths.end());
        if (framePaths.empty()) throw std::runtime_error("no frame_*.ply found in " + dir);

        // Read frames + accumulate a world bbox for auto voxel sizing.
        struct Frame {
            std::vector<Vector3f> pts, nrm;
            Vector3f cam;
        };
        std::vector<Frame> frames;
        Vector3f bbMin = Vector3f::Constant(1e30f), bbMax = Vector3f::Constant(-1e30f);
        size_t totalPts = 0, maxFramePts = 0;
        for (const auto &p: framePaths) {
            Frame fr;
            if (!util::LoadPly(p, fr.pts, fr.nrm) || fr.nrm.size() != fr.pts.size()) {
                std::fprintf(stderr, "skip (no normals): %s\n", p.c_str());
                continue;
            }
            fr.cam = estimateCamera(fr.pts, fr.nrm);
            for (const auto &q: fr.pts) {
                bbMin = bbMin.cwiseMin(q);
                bbMax = bbMax.cwiseMax(q);
            }
            totalPts += fr.pts.size();
            maxFramePts = std::max(maxFramePts, fr.pts.size());
            frames.push_back(std::move(fr));
        }
        if (frames.empty()) throw std::runtime_error("no usable frames (need per-point normals)");
        const float extent = (bbMax - bbMin).norm();

        const bool p2p = !flag(argc, argv, "--no-p2p");
        const float conf = floatArg(argc, argv, "--conf", 0.5f);
        const bool hermite = flag(argc, argv, "--hermite");
        const float voxel = floatArg(argc, argv, "--voxel", extent / 200.0f);
        const float trunc = floatArg(argc, argv, "--trunc", voxel * 3.0f);

        // AdvancedTSDF is a SINGLE movable 512^3-voxel window. Its defaults (origin-centred window,
        // hashCapacity 1<<20, maxPoints 1<<15) silently break for fine voxels or off-origin/large
        // objects: voxels outside the window are dropped (garbage extract) and a saturated hash
        // collides. Size and place the window to the data instead.
        const Vector3f span = bbMax - bbMin;
        const float maxSpan = span.maxCoeff();
        const int margin = int(std::ceil(trunc / voxel)) + 2;            // truncation ghost band
        const long axisVox = long(std::ceil(maxSpan / voxel)) + 2L * margin;
        const bool forceSingle = flag(argc, argv, "--single");
        if (axisVox > 512 && forceSingle) {
            const float minVoxel = maxSpan / float(512 - 2 * margin);
            std::fprintf(stderr,
                         "error: voxel %.4f too fine — object spans %ld voxels/axis but a single "
                         "AdvancedTSDF window is 512^3 (--single).\n"
                         "  drop --single to auto-tile, or use --voxel >= %.4f.\n",
                         voxel, axisVox, minVoxel);
            return 3;
        }
        const bool useTiled = axisVox > 512; // exceeds one window -> tile
        // Place the window's min corner a margin below the object; scale the hash to the expected
        // surface-shell entry count (bbox-surface proxy, x2 load headroom, clamped).
        const Vector3f windowMinCorner = bbMin - float(margin) * Vector3f::Constant(voxel);
        const double surf = 2.0 * double(span.x() * span.y() + span.y() * span.z() +
                                         span.z() * span.x());
        const double shell = 2.0 * double(trunc) / double(voxel);
        const double estEntries = (surf / (double(voxel) * double(voxel))) * shell * 1.5;
        const uint32_t hashCap =
                std::max(1u << 20, nextPow2(uint32_t(std::min(estEntries * 2.0, double(1u << 24)))));
        const uint32_t maxPts = nextPow2(uint32_t(std::max<size_t>(maxFramePts, 1u << 15)));
        // Per-tile hash capacity for the TILED path (each tile is one AdvancedTSDF window). Default
        // 1<<21 (~48MB/tile) is safe for a fully-surface-crossed tile but × many tiles can exceed
        // VRAM; lower it via --tile-hash for fine-voxel/large scenes (risks per-tile overflow).
        const uint32_t tileHash =
                nextPow2(uint32_t(floatArg(argc, argv, "--tile-hash", float(1u << 21))));

        std::printf("dir       : %s  (%zu frames, %zu total pts, extent %.4f)\n", dir.c_str(),
                    frames.size(), totalPts, extent);
        std::printf("advanced  : voxel %.4f, trunc %.4f, point-to-plane %s, conf %.2f, hermite %s\n",
                    voxel, trunc, p2p ? "on" : "off", conf, hermite ? "on" : "off");
        std::printf("window    : %ld voxels/axis, hashCap %u, maxPts %u, minCorner (%.2f,%.2f,%.2f)\n",
                    axisVox, hashCap, maxPts, windowMinCorner.x(), windowMinCorner.y(),
                    windowMinCorner.z());

        // Integrate + extract (single window if it fits, else tiled).
        Engine::Core::Context ctx;
        Engine::Spatial::OrientedPointCloud recon;
        if (flag(argc, argv, "--submap")) {
            const int blockVoxels = int(floatArg(argc, argv, "--block", 32.0f));
            const float detailK = floatArg(argc, argv, "--detail-k", 4.0f);
            Engine::Spatial::SubmapAdvancedTSDF s;
            // Detail is at half voxel -> ~4-8x more entries/tile than base; use the (larger)
            // --tile-hash size for both levels so the detail hash doesn't overflow (holes).
            s.Build(ctx, voxel, trunc, blockVoxels, detailK, tileHash, maxPts);
            s.SetIntegrationQuality({3, 4, true});
            s.SetPointToPlane(p2p);
            s.SetConfidenceWeight(conf);
            s.SetHermitePosition(hermite);
            for (const auto &fr: frames) s.AddDensity(fr.pts);   // pass 1: density
            s.FinalizeDensity();
            for (const auto &fr: frames) s.Integrate(fr.pts, fr.nrm, fr.cam); // pass 2
            recon = s.ExtractPointCloud(/*merge=*/true);
            std::printf("path      : SUBMAP (base %.4f + detail %.4f); dense blocks %u, base tiles "
                        "%u, detail tiles %u\n",
                        voxel, voxel * 0.5f, s.DenseBlockCount(), s.BaseTileCount(),
                        s.DetailTileCount());
        } else if (useTiled) {
            std::printf("path      : TILED (scene exceeds one 512^3 window); per-tile hash %u "
                        "(~%.0f MB/tile)\n",
                        tileHash, double(tileHash) * 24.0 / 1e6);
            Engine::Spatial::TiledAdvancedTSDF tiled;
            tiled.Build(ctx, voxel, trunc, /*hashCapPerTile=*/tileHash, /*maxPtsPerFrame=*/maxPts);
            tiled.SetIntegrationQuality({3, 4, true});
            tiled.SetPointToPlane(p2p);
            tiled.SetConfidenceWeight(conf);
            tiled.SetHermitePosition(hermite);
            for (const auto &fr: frames) tiled.Integrate(fr.pts, fr.nrm, fr.cam);
            recon = tiled.ExtractPointCloud(/*merge=*/true);
            std::printf("integrated: %zu frames → %u tiles, %u occupied entries (ghost-inflated)\n",
                        frames.size(), tiled.TileCount(), tiled.FilledCount());
        } else {
            std::printf("path      : SINGLE 512^3 window\n");
            Engine::Spatial::AdvancedTSDF tsdf;
            tsdf.Build(ctx, voxel, trunc, hashCap, maxPts, windowMinCorner);
            tsdf.SetIntegrationQuality({3, 4, true});
            tsdf.SetPointToPlane(p2p);
            tsdf.SetConfidenceWeight(conf);
            tsdf.SetHermitePosition(hermite);
            for (const auto &fr: frames) tsdf.Integrate(fr.pts, fr.nrm, fr.cam);
            recon = tsdf.ExtractPointCloud(1u << 21, /*merge=*/true);
            std::printf("integrated: %zu frames → %u occupied entries\n", frames.size(),
                        tsdf.FilledCount());
        }
        std::printf("extracted : %zu oriented points\n", recon.points.size());

        // RMSE vs ground truth (accuracy: recon→GT, completeness: GT→recon).
        // std::vector<Vector3f> gtP, gtN;
        // if (!gtPath.empty() && util::LoadPly(gtPath, gtP, gtN) && !gtP.empty()) {
        //     const float cell = std::max(voxel, extent / 100.0f);
        //     GridNN gGT, gRec;
        //     gGT.build(gtP, cell);
        //     gRec.build(recon.points, cell);
        //     Err acc, comp;
        //     for (const auto &p : recon.points) acc.add(std::sqrt(gGT.nearestSq(p)));
        //     for (const auto &g : gtP) comp.add(std::sqrt(gRec.nearestSq(g)));
        //     std::printf("\n=== RMSE vs %s (%zu GT pts) ===\n", gtPath.c_str(), gtP.size());
        //     std::printf("accuracy    (recon→GT): mean %.5f  rmse %.5f\n", acc.mean(), acc.rmse());
        //     std::printf("completeness(GT→recon): mean %.5f  rmse %.5f\n", comp.mean(), comp.rmse());
        //     std::printf("chamfer-L1  (mean of means): %.5f\n", 0.5 * (acc.mean() + comp.mean()));
        // } else {
        //     std::printf("\n(no ground_truth.ply found in %s and no --gt given → skipping RMSE)\n",
        //                 dir.c_str());
        // }

        const std::string outPly = strArg(argc, argv, "--out", "");
        if (!outPly.empty()) {
            util::SavePly(outPly, recon.points, recon.normals);
            std::printf("wrote extracted cloud: %s\n", outPly.c_str());
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

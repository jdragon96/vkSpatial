// Quantitative evaluation of DirectionalTSDF integrate→extract against analytic ground
// truth. For each analytic surface (sphere, plane): synthesise an orbit scan, reconstruct,
// then report accuracy RMSE (point-to-surface) and completeness RMSE (GT→recon NN).
#include "Engine/Core/Context.h"
#include "Engine/Eval/RmseMetrics.h"
#include "Engine/Eval/ScanSampler.h"
#include "Engine/Eval/SyntheticSurface.h"
#include "TSDF/Backends/DirectionalTSDF.h"

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Engine::Eval;

static void savePLY(const std::string &path, const std::vector<Eigen::Vector3f> &pts) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("savePLY: cannot open " + path);
    f << "ply\nformat ascii 1.0\nelement vertex " << pts.size()
      << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto &p : pts) f << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
}

static std::vector<Eigen::Vector3f> reconPositions(const TSDF::DirectionalTSDF &tsdf) {
    std::vector<Eigen::Vector3f> out;
    out.reserve(tsdf.PointCloud().size());
    for (const auto &pt : tsdf.PointCloud()) out.push_back(pt.position);
    return out;
}

static void evalSurface(Engine::Core::Context &ctx, const std::string &name,
                        const Surface &surface, float cameraRadius, const Eigen::Vector3f &orbitCenter) {
    OrbitParams params;
    params.surfaceSamples = surface.SampleDense(4000); // scan candidates
    params.cameraRadius = cameraRadius;
    params.orbitCenter = orbitCenter;
    auto frames = GenerateOrbitScan(surface, params);

    TSDF::DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f);
    int totalPts = 0;
    for (const auto &fr : frames) {
        tsdf.Integrate(fr.points, fr.normals, fr.cameraPos, fr.aabbCenterHint);
        totalPts += int(fr.points.size());
    }

    const auto recon = reconPositions(tsdf);
    const auto gt = surface.SampleDense(8000); // independent dense GT
    const float acc = AccuracyRMSE(recon, surface);
    const float comp = CompletenessRMSE(gt, recon);
    const float cross = ReconToGtNnRMSE(recon, gt);

    std::cout << "=== " << name << " ===\n"
              << "  frames=" << frames.size() << " integrated_pts=" << totalPts
              << " recon_pts=" << recon.size() << " gt_pts=" << gt.size() << "\n"
              << "  accuracyRMSE (point-to-surface) = " << acc << "\n"
              << "  completenessRMSE (GT->recon NN)  = " << comp << "\n"
              << "  reconToGtNnRMSE (recon->GT NN)   = " << cross << "\n";

    savePLY("recon_" + name + ".ply", recon);
    savePLY("gt_" + name + ".ply", gt);
}

int main() {
    Engine::Core::Context ctx;
    // Sphere: radius 1 at origin, camera orbit at 3.0.
    SphereSurface sphere(Eigen::Vector3f(0, 0, 0), 1.0f);
    evalSurface(ctx, "sphere", sphere, 3.0f, Eigen::Vector3f(0, 0, 0));

    // Plane: z=0 patch, camera orbit centred above it so views look down at the patch.
    PlaneSurface plane(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 1.0f, 1.0f);
    evalSurface(ctx, "plane", plane, 3.0f, Eigen::Vector3f(0, 0, 1.5f));

    std::cout << "wrote recon_*.ply / gt_*.ply\n";
    return 0;
}

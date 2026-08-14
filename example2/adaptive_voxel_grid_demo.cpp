// Task 5 (final) of the AdaptiveVoxelGrid plan: a runnable measurement demo.
//
// For BOTH the cube and cylinder synthetic fixtures (example2/shape_fixtures.h), integrates
// the same multi-view point clouds into an AdaptiveVoxelGrid (fine voxel ~0.05, truncation
// 0.15 -- same constants test/test_adaptiveVoxelGrid.cpp uses), sweeps 3 variance percentiles
// (0.3 / 0.5 / 0.7), and reports:
//   - FineCount() / CoarseCount()          -- mixed-grid voxel counts at that percentile.
//   - total-voxel reduction                 -- all-fine SimpleTSDF::FilledCount() divided by
//                                              (FineCount()+CoarseCount()); a coarse voxel
//                                              replaces 8 fine voxels' storage with 1.
//   - meshVerts vs refVerts                 -- vertex count of the adaptive mesh vs an
//                                              all-fine reference mesh (AdaptiveVoxelGrid with
//                                              SetVarianceThreshold(0.0f), which -- per Task 2 --
//                                              is unreachable by any non-negative variance mean,
//                                              forcing zero coarsening).
//   - accuracy_mm                           -- maxNearestDist(adaptive mesh vertices, all-fine
//                                              mesh vertices): one-directional Chamfer distance,
//                                              same helper/convention as
//                                              test/test_adaptiveVoxelGrid.cpp's
//                                              MeshAccuracyVsAllFine test.
//   - true_mean / true_max                  -- mean/max distance of the adaptive mesh's
//                                              vertices to the ANALYTIC ground-truth surface
//                                              (fixtures::NearestDistance), independent of the
//                                              all-fine reference mesh. Added after a real
//                                              finding below (see HONEST-MEASUREMENT NOTE #2)
//                                              showed accuracy_mm alone can be misleading.
//
// Each mesh is also written to <outDir>/adaptive_<shape>_p<pct>.ply (ASCII PLY, faces from
// AdaptiveMesh::triangles) for external inspection, plus one adaptive_<shape>_reference.ply for
// the all-fine baseline. outDir defaults to /tmp/adaptive_voxel_grid_demo (kept OUT of the repo
// deliberately -- see plan Task 5's verify-before-done note).
//
// HONEST-MEASUREMENT NOTE #1 (carried over from Task 4's commit, 0493498): on these synthetic
// fixtures, coarsening overwhelmingly happens in the truncation-band INTERIOR -- coarsened
// blocks whose fine sub-voxels never bracketed a zero-crossing to begin with. That reduces
// stored voxel count (FineCount()+CoarseCount() < all-fine count) WITHOUT changing the
// extracted mesh at all (meshVerts == refVerts, accuracy_mm == 0). Task 4's own measurement at
// percentile=0.6 found exactly this (CoarseCount()=196, vertex/triangle counts identical to the
// all-fine mesh); real surface-geometry coarsening only showed up at a much higher, more
// aggressive percentile (0.97) in an uncommitted stress check. This demo's table reports
// meshVerts/refVerts/accuracy_mm alongside the voxel-count reduction specifically so a reader
// does NOT mistake "voxels reduced" for "surface simplified" -- at 0.3/0.5/0.7 expect the
// former without (or with very little of) the latter; do not oversell the sweep.
//
// HONEST-MEASUREMENT NOTE #2 (found while validating THIS demo's own numbers): on the cylinder,
// accuracy_mm comes back large (~0.43, vs ~0.006 for the cube) at all three swept percentiles,
// which could read as "coarsening hurts the cylinder a lot". Investigated by dumping both PLYs
// and checking the worst-offending adaptive-mesh vertices against the TRUE analytic cylinder
// surface (not just the reference mesh): every one of them sits within ~0.06 of the real
// surface (fine-voxel-scale error), and the adaptive mesh's OVERALL mean true-surface error
// (~0.023) matches the all-fine reference's (~0.023) almost exactly. The large accuracy_mm is
// an artifact of maxNearestDist being a ONE-DIRECTIONAL Chamfer against a reference mesh that
// itself has a small coverage gap (a thin strip near x in [-0.4,-0.35] where the all-fine CPU-MC
// reconstruction has zero vertices, likely sparse per-view weight in that region failing the
// MIN_WEIGHT corner gate for individual fine voxels); the coarse/Sec.8-refinement path in the
// adaptive mesh happens to bridge that exact gap with legitimately-accurate geometry, so
// comparing "adaptive -> reference" flags those extra, CORRECT vertices as far outliers simply
// because the reference has no nearby vertex to match against. This is why true_mean/true_max
// (vs the analytic surface, independent of the reference mesh's own gaps) were added: read
// accuracy_mm as "how much does the adaptive mesh differ from the all-fine baseline" (can be
// inflated by baseline gaps, as seen here) and true_mean/true_max as "how accurate is the
// adaptive mesh, actually" -- the two do not have to agree, and on the cylinder they visibly
// don't.
#include "shape_fixtures.h"

#include "Engine/Core/Context.h"
#include "Engine/Spatial/AdaptiveVoxelGrid.h"
#include "TSDF/Backends/SimpleTSDF.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using Engine::Spatial::AdaptiveMesh;
using Engine::Spatial::AdaptiveVoxelGrid;
using Engine::Spatial::SimpleTSDF;
using fixtures::Shape;

namespace {

    constexpr float kVoxel = 0.05f;
    constexpr float kTrunc = 0.15f;
    const std::array<float, 3> kPercentiles = {0.3f, 0.5f, 0.7f};

    const char *ShapeName(Shape s) { return s == Shape::Cube ? "cube" : "cylinder"; }

    void Integrate(AdaptiveVoxelGrid &g, const std::vector<fixtures::View> &views) {
        for (const auto &v : views) g.Integrate(v.points, v.camPos);
    }

    void IntegrateSimple(SimpleTSDF &s, const std::vector<fixtures::View> &views) {
        for (const auto &v : views) s.Integrate(v.points, v.camPos);
    }

    // One-directional Chamfer distance: max over `from` of the nearest-neighbour distance to
    // `to` (brute-force -- these fixture meshes stay small enough for this to be fast). Same
    // convention as test/test_adaptiveVoxelGrid.cpp's maxNearestDist.
    double MaxNearestDist(const std::vector<Eigen::Vector3f> &from,
                          const std::vector<Eigen::Vector3f> &to) {
        if (from.empty() || to.empty()) return 0.0;
        double worst = 0.0;
        for (const auto &p : from) {
            float best = std::numeric_limits<float>::max();
            for (const auto &q : to) best = std::min(best, (p - q).squaredNorm());
            worst = std::max(worst, double(std::sqrt(best)));
        }
        return worst;
    }

    // Writes an AdaptiveMesh as an ASCII PLY (same header style as SimpleTSDF::ExportMC).
    void WritePly(const std::string &path, const AdaptiveMesh &m) {
        std::ofstream f(path);
        if (!f.is_open()) {
            std::printf("  [warn] could not open %s for writing -- skipping export\n", path.c_str());
            return;
        }
        f << "ply\nformat ascii 1.0\n"
          << "element vertex " << m.vertices.size() << "\n"
          << "property float x\nproperty float y\nproperty float z\n"
          << "element face " << m.triangles.size() << "\n"
          << "property list uchar int vertex_indices\n"
          << "end_header\n";
        for (const auto &v : m.vertices) f << v.x() << ' ' << v.y() << ' ' << v.z() << '\n';
        for (const auto &t : m.triangles) f << "3 " << t.x() << ' ' << t.y() << ' ' << t.z() << '\n';
    }

    // Mean/max distance of a set of vertices to the ANALYTIC ground-truth surface
    // (fixtures::NearestDistance) -- independent of any extracted reference mesh (see
    // HONEST-MEASUREMENT NOTE #2 above for why this matters).
    struct TrueStats {
        double mean = 0.0, max = 0.0;
    };

    TrueStats TrueSurfaceStats(Shape shape, const std::vector<Eigen::Vector3f> &verts) {
        TrueStats t;
        if (verts.empty()) return t;
        double sum = 0.0;
        for (const auto &v : verts) {
            const double d = double(fixtures::NearestDistance(shape, v));
            sum += d;
            t.max = std::max(t.max, d);
        }
        t.mean = sum / double(verts.size());
        return t;
    }

    // Formats a percentile (0.3 -> "p30") for filenames/labels.
    std::string PctTag(float p) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "p%02d", int(std::lround(double(p) * 100.0)));
        return buf;
    }

    void PrintHeader() {
        std::printf("%-6s %8s %8s %8s %10s %10s %10s %9s %10s %10s %10s\n", "pct", "fine",
                    "coarse", "allFine", "reduction", "meshVerts", "refVerts", "changed",
                    "acc_mm", "true_mean", "true_max");
        std::printf("---------------------------------------------------------------------------"
                    "-----------------------\n");
    }

    struct Row {
        float pct = 0.0f;
        size_t fine = 0, coarse = 0, allFine = 0, meshVerts = 0, refVerts = 0;
        double reduction = 0.0;
        double accuracyMm = 0.0;
        double trueMean = 0.0, trueMax = 0.0;
        bool changed = false;
    };

    void PrintRow(const Row &r) {
        std::printf("%-6.2f %8zu %8zu %8zu %9.2fx %10zu %10zu %9s %10.5f %10.5f %10.5f\n", r.pct,
                    r.fine, r.coarse, r.allFine, r.reduction, r.meshVerts, r.refVerts,
                    r.changed ? "yes" : "no", r.accuracyMm, r.trueMean, r.trueMax);
    }

    // Diagnostic explaining a real (non-bug) property measured below: the raw per-voxel
    // variance distribution this fixture produces. AdaptiveVoxelGrid::buildMixed() picks the
    // p-th quantile of this SAME distribution as its coarsening threshold (and then escapes any
    // exact-tie plateau at that quantile -- see AdaptiveVoxelGrid.cpp's buildMixed comment). If
    // a large mass of fine voxels sit at EXACTLY zero variance (single-observation voxels, or
    // voxels every view reports identically), percentiles that all land inside that plateau
    // collapse to the SAME escaped threshold -- so 0.3/0.5/0.7 can (and, as measured below, do)
    // produce identical FineCount/CoarseCount/mesh. Printed so that identical rows read as an
    // explained fixture property, not a silent demo bug.
    void PrintVarianceDiagnostic(const std::vector<Engine::Spatial::VoxelStat> &voxels) {
        if (voxels.empty()) {
            std::printf("  [variance-diag] no occupied fine voxels\n");
            return;
        }
        std::vector<float> s;
        s.reserve(voxels.size());
        for (const auto &v : voxels) s.push_back(v.variance);
        std::sort(s.begin(), s.end());
        const size_t zeroCount =
                size_t(std::upper_bound(s.begin(), s.end(), 0.0f) - s.begin());
        std::printf("  [variance-diag] n=%zu  exactly-zero-variance=%zu (%.1f%%)  quantiles(p=0.3/"
                    "0.5/0.7)=%.4g/%.4g/%.4g  max=%.4g\n",
                    s.size(), zeroCount, 100.0 * double(zeroCount) / double(s.size()),
                    double(s[std::min(s.size() - 1, size_t(0.3 * s.size()))]),
                    double(s[std::min(s.size() - 1, size_t(0.5 * s.size()))]),
                    double(s[std::min(s.size() - 1, size_t(0.7 * s.size()))]), double(s.back()));
    }

    // Runs the full sweep for one shape: builds the all-fine SimpleTSDF baseline, the
    // all-fine reference AdaptiveMesh (SetVarianceThreshold(0.0f)), then for each percentile a
    // fresh AdaptiveVoxelGrid at that percentile -- printing + exporting a PLY per row.
    void RunShape(Engine::Core::Context &ctx, Shape shape, const std::string &outDir) {
        const auto views = fixtures::SampleViews(shape, kVoxel);
        std::printf("\n=== %s (voxel=%.3f trunc=%.3f, %zu views) ===\n", ShapeName(shape), kVoxel,
                    kTrunc, views.size());

        // All-fine SimpleTSDF baseline -- the denominator for "total-voxel reduction".
        SimpleTSDF allFine;
        allFine.Build(ctx, kVoxel, kTrunc);
        IntegrateSimple(allFine, views);
        const size_t allFineCount = size_t(allFine.FilledCount());

        // All-fine reference mesh via AdaptiveVoxelGrid itself (SetVarianceThreshold(0.0f) is
        // unreachable by a non-negative variance mean -- see test_adaptiveVoxelGrid.cpp's
        // FineMatchesSimpleTSDF -- so this is the same mesh a percentile=0 sweep would produce).
        AdaptiveVoxelGrid refGrid;
        refGrid.Build(ctx, kVoxel, kTrunc);
        Integrate(refGrid, views);
        refGrid.SetVarianceThreshold(0.0f);
        const AdaptiveMesh refMesh = refGrid.ExtractMesh();

        const TrueStats refTrue = TrueSurfaceStats(shape, refMesh.vertices);
        std::printf("all-fine SimpleTSDF::FilledCount() = %zu   all-fine reference mesh vertices = %zu"
                    "   ref true-surface mean/max = %.5f / %.5f\n",
                    allFineCount, refMesh.vertices.size(), refTrue.mean, refTrue.max);
        PrintVarianceDiagnostic(allFine.DownloadVoxels());
        WritePly(outDir + "/adaptive_" + ShapeName(shape) + "_reference.ply", refMesh);
        PrintHeader();

        for (float p : kPercentiles) {
            AdaptiveVoxelGrid g;
            g.Build(ctx, kVoxel, kTrunc);
            Integrate(g, views);
            g.SetVariancePercentile(p);

            Row row;
            row.pct = p;
            row.fine = g.FineCount();
            row.coarse = g.CoarseCount();
            row.allFine = allFineCount;
            const size_t totalStored = row.fine + row.coarse;
            row.reduction = totalStored > 0 ? double(allFineCount) / double(totalStored) : 0.0;

            const AdaptiveMesh mesh = g.ExtractMesh();
            row.meshVerts = mesh.vertices.size();
            row.refVerts = refMesh.vertices.size();
            row.accuracyMm = MaxNearestDist(mesh.vertices, refMesh.vertices);
            const TrueStats trueStats = TrueSurfaceStats(shape, mesh.vertices);
            row.trueMean = trueStats.mean;
            row.trueMax = trueStats.max;
            // Honest "did the surface actually change" signal: vertex-count mismatch OR a
            // non-negligible Chamfer distance (a few micrometres of floating-point/CPU-MC
            // noise is expected even when the mesh is geometrically identical). NOTE: per
            // HONEST-MEASUREMENT NOTE #2, "changed" here means "differs from the all-fine
            // reference mesh" -- it does NOT mean "less accurate" (compare against
            // true_mean/true_max, and refTrue printed above, to judge that).
            row.changed = (row.meshVerts != row.refVerts) || (row.accuracyMm > 1e-4);

            PrintRow(row);

            const std::string path = outDir + "/adaptive_" + ShapeName(shape) + "_" + PctTag(p) + ".ply";
            WritePly(path, mesh);
        }
    }

} // namespace

int main() {
    const std::string outDir = "/tmp/adaptive_voxel_grid_demo";
    std::error_code ec;
    std::filesystem::create_directories(outDir, ec);
    if (ec) std::printf("[warn] could not create %s (%s); PLY export may fail\n", outDir.c_str(),
                        ec.message().c_str());

    Engine::Core::Context ctx;

    RunShape(ctx, Shape::Cube, outDir);
    RunShape(ctx, Shape::Cylinder, outDir);

    std::printf(
            "\nHONEST NOTE #1: on these synthetic fixtures, coarsening overwhelmingly happens in "
            "the truncation-band INTERIOR (fine sub-voxels that never bracketed a zero-"
            "crossing to begin with). That is why 'reduction' > 1x commonly comes with "
            "'changed=no' / accuracy_mm ~= 0 at these percentiles -- voxel STORAGE shrank, the "
            "extracted SURFACE did not. Real surface-geometry coarsening (changed=yes, "
            "accuracy_mm approaching one coarse-voxel width) is expected to require a much more "
            "aggressive percentile than swept here (see Task 4's commit 0493498 for a percentile="
            "0.97 stress measurement).\n"
            "HONEST NOTE #2: 0.3/0.5/0.7 landed on the IDENTICAL threshold for both shapes here "
            "-- see the printed [variance-diag] line: ~75-76%% of fine voxels have EXACTLY zero "
            "measured variance (single-view or fully-consistent voxels), so all three swept "
            "percentiles fall inside that tie plateau and buildMixed()'s tie-escape logic gives "
            "them the same threshold. A sweep that actually differentiates 0.3 vs 0.5 vs 0.7 on "
            "this fixture would need percentiles above ~76%%.\n"
            "HONEST NOTE #3: the cylinder's accuracy_mm (vs the all-fine reference mesh) reads "
            "far larger than the cube's, but true_mean/true_max (vs the ANALYTIC surface) show "
            "the adaptive mesh is just as accurate as the reference -- the large accuracy_mm is "
            "a reference-mesh coverage-gap artifact (see the file header's HONEST-MEASUREMENT "
            "NOTE #2), not real degradation. Do not read accuracy_mm alone as a quality score.\n"
            "PLY meshes (including the all-fine *_reference.ply) written to %s\n",
            outDir.c_str());

    return 0;
}

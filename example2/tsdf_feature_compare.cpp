// Headless correctness milestone for the SimpleTSDF-vs-DirectionalTSDF feature-preservation
// comparison (Task 2). Integrates the SAME analytic multi-view samples into BOTH TSDFs,
// extracts both point clouds, and prints per-region (flat / curved / edge) ground-truth error.
//
// The metric is the oracle: DirectionalTSDF (per-direction layers) should beat the averaged
// SimpleTSDF on EDGE error (it preserves the sharp corner / rim) while tying on flat/curved.
//
//   tsdf_feature_compare --shape cube|cylinder [--voxel <f>] [--dump]
//
// No GUI in this task; the --dump table IS the verification.
#include "shape_fixtures.h"

#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalTSDF.h"
#include "Engine/Spatial/DirectionalTSDFTypes.h"
#include "Engine/Spatial/SimpleTSDF.h"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

using fixtures::Region;
using fixtures::Shape;

namespace {

    struct RegionStat {
        double sum = 0.0;
        double maxErr = 0.0;
        std::size_t count = 0;
        void add(float e) {
            sum += double(e);
            maxErr = std::max(maxErr, double(e));
            ++count;
        }
        double mean() const { return count ? sum / double(count) : 0.0; }
    };

    // err = NearestDistance(true surface), region = ClassifyRegion; accumulate per region.
    void accumulate(Shape s, float voxel, const std::vector<Eigen::Vector3f> &pts,
                    std::array<RegionStat, 3> &stats) {
        for (const auto &p : pts) {
            const float e = fixtures::NearestDistance(s, p);
            stats[int(fixtures::ClassifyRegion(s, p, voxel))].add(e);
        }
    }

    const char *regionName(int r) {
        switch (r) {
            case int(Region::Flat): return "flat";
            case int(Region::Curved): return "curved";
            case int(Region::Edge): return "edge";
            default: return "?";
        }
    }

} // namespace

int main(int argc, char **argv) {
    Shape shape = Shape::Cube;
    float voxel = 0.1f;
    bool dump = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shape" && i + 1 < argc) {
            const std::string v = argv[++i];
            shape = (v == "cylinder") ? Shape::Cylinder : Shape::Cube;
        } else if (a == "--voxel" && i + 1 < argc) {
            voxel = std::stof(argv[++i]);
        } else if (a == "--dump") {
            dump = true;
        }
    }
    const float truncation = 0.3f;
    const char *shapeName = (shape == Shape::Cube) ? "cube" : "cylinder";

    Engine::Core::Context ctx;
    const std::vector<fixtures::View> views = fixtures::SampleViews(shape, voxel);

    std::size_t nInput = 0, maxPerView = 0;
    for (const auto &v : views) {
        nInput += v.points.size();
        maxPerView = std::max(maxPerView, v.points.size());
    }

    // ---- SimpleTSDF: single averaged field (rounds sharp features) ----
    Engine::Spatial::SimpleTSDF simple;
    simple.Build(ctx, voxel, truncation);
    for (const auto &v : views) simple.Integrate(v.points, v.camPos);
    const Engine::Spatial::OrientedPointCloud simpleCloud = simple.ExtractPointCloud();

    // ---- DirectionalTSDF: per-direction layers (preserves sharp features) ----
    Engine::Spatial::DirectionalTSDF dir;
    dir.Build(ctx, voxel, truncation);
    dir.SetIntegrationQuality({3, 4, true}); // maxDirections=3, dirExponent=4, viewAngleWeight
    for (const auto &v : views)
        dir.Integrate(v.points, v.normals, v.camPos, Eigen::Vector3f::Zero());
    std::vector<Eigen::Vector3f> dirPts;
    dirPts.reserve(dir.PointCloud().size());
    for (const auto &e : dir.PointCloud()) dirPts.push_back(e.position);

    std::array<RegionStat, 3> simpleStats{}, dirStats{};
    accumulate(shape, voxel, simpleCloud.points, simpleStats);
    accumulate(shape, voxel, dirPts, dirStats);

    if (dump) {
        std::printf("=== tsdf_feature_compare  shape=%s  voxel=%.3f  truncation=%.3f ===\n",
                    shapeName, voxel, truncation);
        std::printf("views=%zu  nInput=%zu (max/view=%zu)  nSimple=%zu  nDir=%zu\n",
                    views.size(), nInput, maxPerView, simpleCloud.points.size(), dirPts.size());
        std::printf("%-7s | %11s %11s | %11s %11s\n", "region", "Simple.mean",
                    "Simple.max", "Dir.mean", "Dir.max");
        std::printf("--------+-------------------------+-------------------------\n");
        for (int r = 0; r < 3; ++r) {
            if (simpleStats[r].count == 0 && dirStats[r].count == 0) continue;
            std::printf("%-7s | %9.4f %11.4f | %9.4f %11.4f  (S:n=%zu D:n=%zu)\n",
                        regionName(r), simpleStats[r].mean(), simpleStats[r].maxErr,
                        dirStats[r].mean(), dirStats[r].maxErr, simpleStats[r].count,
                        dirStats[r].count);
        }
        std::printf("(per-point ground-truth error in mm; 1 world unit == 1 mm)\n");
    }
    return 0;
}

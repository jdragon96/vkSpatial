#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Eval/RmseMetrics.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial::Extraction;

// A single cube carrying a face-ambiguous configuration (two diagonally-opposite
// negative corners on one face). Original 15-case MC can leave a boundary hole here;
// MC33 must produce a topologically consistent (edge-manifold) patch.
static VoxelField AmbiguousCube() {
    VoxelField f; f.SetCellSize(1.0f);
    // corner signs (MC corner order): negatives at 0 and 2 (a face diagonal), rest positive
    const float s[8] = {-1,+1,-1,+1, +1,+1,+1,+1};
    for (int c=0;c<8;++c) f.Insert({Engine::Spatial::mc::CORNER[c][0],Engine::Spatial::mc::CORNER[c][1],Engine::Spatial::mc::CORNER[c][2]}, s[c]);
    return f;
}

TEST(Isosurface, Mc33RegisteredAndSphereAccurate) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("mc33"));
    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh mesh = reg.Create("mc33")->Extract(field, ExtractParams{});
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
    EXPECT_TRUE(isotest::IsWatertight(mesh));
    std::vector<Eigen::Vector3f> on; for (auto& v:mesh.vertices) on.push_back(v.normalized()*0.7f);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(mesh.vertices, on), 0.1f);
    // Winding regression guard: IsEdgeManifold/IsWatertight are combinatorial and would pass a
    // globally winding-inverted mesh -- assert normals actually point outward.
    EXPECT_GT(isotest::MeanOutwardNormalAlignment(mesh, Eigen::Vector3f::Zero()), 0.5);
}

TEST(Isosurface, Mc33ResolvesFaceAmbiguityEdgeManifold) {
    SurfaceMesh mesh = ExtractorRegistry::Default().Create("mc33")->Extract(AmbiguousCube(), ExtractParams{});
    ASSERT_GT(mesh.triangles.size(), 0u);
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh)); // no >2-incident edges from a bad ambiguity choice
}

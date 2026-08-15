#include "Mesh/ExtractorRegistry.h"
#include "Engine/Eval/RmseMetrics.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Mesh;

TEST(Isosurface, MarchingTetrahedraSphereClosedManifold) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("mtet"));
    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh mesh = reg.Create("mtet")->Extract(field, ExtractParams{});
    ASSERT_GT(mesh.triangles.size(), 100u);
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
    EXPECT_TRUE(isotest::IsWatertight(mesh));
    EXPECT_EQ(isotest::EulerCharacteristic(mesh), 2); // topological sphere
    std::vector<Eigen::Vector3f> on; for (auto& v:mesh.vertices) on.push_back(v.normalized()*0.7f);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(mesh.vertices, on), 0.1f);
    // Winding regression guard: IsEdgeManifold/IsWatertight are combinatorial and would pass a
    // globally winding-inverted mesh -- assert normals actually point outward.
    EXPECT_GT(isotest::MeanOutwardNormalAlignment(mesh, Eigen::Vector3f::Zero()), 0.5);
}

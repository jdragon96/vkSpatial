#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Eval/RmseMetrics.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>

using namespace Engine::Spatial::Extraction;

TEST(Isosurface, RegistryCreatesMcAndRejectsUnknown) {
    auto reg = ExtractorRegistry::Default();
    EXPECT_TRUE(reg.Has("mc"));
    ASSERT_NE(reg.Create("mc"), nullptr);
    EXPECT_EQ(reg.Create("nonexistent"), nullptr);
    EXPECT_STREQ(reg.Create("mc")->Name(), "mc");
}

TEST(Isosurface, MarchingCubesSphereIsAccurateAndManifold) {
    const float cellSize = 0.1f, radius = 0.7f;
    VoxelField field = isotest::SphereField(radius, cellSize, 12);
    auto mc = ExtractorRegistry::Default().Create("mc");
    SurfaceMesh mesh = mc->Extract(field, ExtractParams{});
    ASSERT_GT(mesh.triangles.size(), 100u);
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
    EXPECT_TRUE(isotest::IsWatertight(mesh));               // closed surface
    // accuracy: every vertex lies within ~one cell of the true sphere
    std::vector<Eigen::Vector3f> onSphere;
    for (const auto& v : mesh.vertices) onSphere.push_back(v.normalized() * radius);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(mesh.vertices, onSphere), cellSize);
    // Winding regression guard: IsEdgeManifold/IsWatertight are combinatorial and would pass a
    // globally winding-inverted mesh -- assert normals actually point outward.
    EXPECT_GT(isotest::MeanOutwardNormalAlignment(mesh, Eigen::Vector3f::Zero()), 0.5);
}

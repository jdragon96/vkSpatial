#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
#include <algorithm>
using namespace Engine::Spatial::Extraction;

// Distance from a mesh's vertices to the nearest true box edge (the sharp 90-deg creases).
// Feature-preserving extractors put vertices ON the crease; rounded MC pulls them inward.
static float MinVertexToBoxEdgeGap(const SurfaceMesh& m, const Eigen::Vector3f& he) {
    // sample the 12 edges of the box [-he,he]; measure how close the closest mesh vertex gets
    float best = 1e9f;
    for (const auto& v : m.vertices) {
        Eigen::Vector3f a = v.cwiseAbs();
        // near an edge means TWO coords ~ he
        float d1 = std::abs(a.x()-he.x()), d2 = std::abs(a.y()-he.y()), d3 = std::abs(a.z()-he.z());
        std::array<float,3> d{d1,d2,d3}; std::sort(d.begin(), d.end());
        best = std::min(best, d[0]+d[1]); // two smallest ~0 on an edge
    }
    return best;
}

TEST(Isosurface, ExtendedMcPreservesBoxEdgesBetterThanMc) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("emc"));
    const Eigen::Vector3f he(0.5f,0.5f,0.5f); const float cell=0.1f;
    VoxelField field = isotest::BoxField(he, cell, 10);
    SurfaceMesh mcMesh  = reg.Create("mc")->Extract(field, ExtractParams{});
    SurfaceMesh emcMesh = reg.Create("emc")->Extract(field, ExtractParams{});
    EXPECT_TRUE(isotest::IsEdgeManifold(emcMesh));
    // emc should land a vertex closer to the true crease than mc does
    EXPECT_LT(MinVertexToBoxEdgeGap(emcMesh, he), MinVertexToBoxEdgeGap(mcMesh, he));
}

// emc's only other committed test uses BoxField (a crease fixture); it has no smooth-field
// manifold check and no winding-regression guard. Cover both with a sphere: IsEdgeManifold/
// IsWatertight are combinatorial and would pass a globally winding-inverted mesh, so also assert
// normals actually point outward.
TEST(Isosurface, ExtendedMcSphereOutwardWinding) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("emc"));
    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh mesh = reg.Create("emc")->Extract(field, ExtractParams{});
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
    EXPECT_GT(isotest::MeanOutwardNormalAlignment(mesh, Eigen::Vector3f::Zero()), 0.5);
}

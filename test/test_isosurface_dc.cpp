#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Eval/RmseMetrics.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
#include <algorithm>
using namespace Engine::Spatial::Extraction;

TEST(Isosurface, DualContouringSphereAccurateAndManifold) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("dc"));
    const float cell=0.1f, r=0.7f;
    VoxelField field = isotest::SphereField(r, cell, 12);
    SurfaceMesh mesh = reg.Create("dc")->Extract(field, ExtractParams{});
    ASSERT_GT(mesh.triangles.size(), 100u);
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
    std::vector<Eigen::Vector3f> on; for (auto& v:mesh.vertices) on.push_back(v.normalized()*r);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(mesh.vertices, on), cell);
}

TEST(Isosurface, DualContouringPreservesBoxEdges) {
    const Eigen::Vector3f he(0.5f,0.5f,0.5f); const float cell=0.1f;
    VoxelField field = isotest::BoxField(he, cell, 10);
    SurfaceMesh dcMesh = ExtractorRegistry::Default().Create("dc")->Extract(field, ExtractParams{});
    SurfaceMesh mcMesh = ExtractorRegistry::Default().Create("mc")->Extract(field, ExtractParams{});
    // DC's QEF vertices sit on the crease; compare the tightest crease approach (helper reused from emc test file is fine to duplicate locally)
    auto gap=[&](const SurfaceMesh& m){ float best=1e9f; for(auto&v:m.vertices){Eigen::Vector3f a=v.cwiseAbs();
        std::array<float,3> d{std::abs(a.x()-he.x()),std::abs(a.y()-he.y()),std::abs(a.z()-he.z())}; std::sort(d.begin(),d.end()); best=std::min(best,d[0]+d[1]);} return best; };
    EXPECT_LT(gap(dcMesh), gap(mcMesh));
}

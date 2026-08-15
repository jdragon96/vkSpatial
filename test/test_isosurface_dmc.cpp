#include "Mesh/ExtractorRegistry.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Mesh;

// Thin slab: |z| - t  intersected as two sheets at z = +/- t, spacing 2t ~ one cell.
static VoxelField ThinSlab(float halfThickness, float cell, int halfN) {
    const std::array<int,3> lo{-halfN,-halfN,-halfN}, hi{halfN,halfN,halfN};
    auto value=[halfThickness](const Eigen::Vector3f& p){ return std::abs(p.z()) - halfThickness; };
    return FromImplicit(lo,hi,cell,value,nullptr);
}

TEST(Isosurface, DualMarchingCubesRegisteredAndSphereManifold) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("dmc"));
    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh mesh = reg.Create("dmc")->Extract(field, ExtractParams{});
    ASSERT_GT(mesh.triangles.size(), 100u);
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
    // Winding regression guard: IsEdgeManifold/IsWatertight are combinatorial and would pass a
    // globally winding-inverted mesh -- assert normals actually point outward.
    EXPECT_GT(isotest::MeanOutwardNormalAlignment(mesh, Eigen::Vector3f::Zero()), 0.5);
}

TEST(Isosurface, DualMarchingCubesResolvesThinSlabAsTwoSheets) {
    VoxelField field = ThinSlab(0.06f, 0.1f, 8); // slab thinner than a cell
    SurfaceMesh mesh = ExtractorRegistry::Default().Create("dmc")->Extract(field, ExtractParams{});
    // both z=+/-0.06 sheets present: vertices exist above AND below z=0
    bool above=false, below=false;
    for (auto& v: mesh.vertices){ if (v.z()>0.02f) above=true; if (v.z()<-0.02f) below=true; }
    EXPECT_TRUE(above && below);
}

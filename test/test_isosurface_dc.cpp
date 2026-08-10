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
    // Winding coverage: welded per-vertex normals should point outward (away from the sphere's
    // centre), i.e. agree in sign with the vertex's own radial direction on average.
    ASSERT_EQ(mesh.normals.size(), mesh.vertices.size());
    double outwardDotSum = 0.0;
    for (size_t i = 0; i < mesh.vertices.size(); ++i) outwardDotSum += mesh.normals[i].dot(mesh.vertices[i].normalized());
    EXPECT_GT(outwardDotSum / double(mesh.vertices.size()), 0.5);
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

// A synthetic "checkerboard"-ambiguous VoxelField: only (0,0,0) and (0,1,1) are negative (a
// face-diagonal pair on the face shared by cell bases (0,0,0) and (-1,0,0)); every other corner
// in the surrounding neighbourhood is positive, so that face's OTHER diagonal, (0,1,0)/(0,0,1),
// ends up entirely positive too -- making all 4 of the face's boundary grid edges simultaneously
// sign-changing (see DualContouringExtractor.cpp's Pass 2 comment for why that produces a
// non-manifold edge). The neighbourhood box spans exactly the corners every cell touching either
// negative point needs, so every one of those cells resolves to a valid dual vertex in pass 1 and
// none of the 4 problem quads is silently dropped for an unresolved corner.
static VoxelField AmbiguousFaceField() {
    VoxelField field; field.SetCellSize(1.0f);
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 2; ++y)
            for (int z = -1; z <= 2; ++z) {
                const bool negative = (x == 0 && y == 0 && z == 0) || (x == 0 && y == 1 && z == 1);
                field.Insert({x, y, z}, negative ? -1.0f : 1.0f);
            }
    return field;
}

// KNOWN-FAILING regression target for a future Manifold Dual Contouring implementation
// (Schaefer, Ju, Warren, "Manifold Dual Contouring", IEEE TVCG 2007) -- this is an accepted,
// spec-documented limitation of basic DC, not a bug:
// docs/superpowers/specs/2026-08-10-isosurface-extraction-strategies-design.md's Risks section
// states "Dual-method manifoldness -- DC can produce non-manifold edges at sharp configs.
// Mitigation: assert edge-manifoldness on smooth fixtures; document the caveat (Manifold DC is a
// noted future refinement, not in scope)". DISABLED because it currently FAILS by design (the
// checkerboard face's shared cell-pair edge reaches triangle incidence 4, not <=2) -- see
// DualContouringExtractor.cpp's Pass 2 comment for the full mechanism. A future Manifold-DC
// implementer should make this pass, not delete it.
TEST(Isosurface, DISABLED_DualContouringAmbiguousFaceIsManifold_ManifoldDcFuture) {
    VoxelField field = AmbiguousFaceField();
    SurfaceMesh mesh = ExtractorRegistry::Default().Create("dc")->Extract(field, ExtractParams{});
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
}

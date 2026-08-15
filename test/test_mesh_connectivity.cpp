#include "Mesh/MeshConnectivity.h"
#include "Mesh/ExtractorRegistry.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Mesh;

TEST(MeshConnectivity, SphereIsClosedManifold) {
    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh mesh = ExtractorRegistry::Default().Create("mc")->Extract(field, ExtractParams{});
    ConnectivityReport r = AnalyzeConnectivity(mesh);
    EXPECT_TRUE(r.IsEdgeManifold());
    EXPECT_TRUE(r.IsClosed());
    EXPECT_TRUE(r.boundaryEdges.empty());
    EXPECT_TRUE(r.nonManifoldVertices.empty());
    EXPECT_EQ(r.degenerateTriangles, 0);
}

TEST(MeshConnectivity, DetectsNonManifoldEdge) {
    // 3 triangles all sharing edge (0,1) -> that edge has 3 incident faces.
    SurfaceMesh m;
    m.vertices = {{0,0,0},{1,0,0},{0,1,0},{0,-1,0},{0,0,1}};
    m.normals.assign(5, Eigen::Vector3f(0,0,1));
    m.triangles = {{0,1,2},{0,1,3},{0,1,4}};
    ConnectivityReport r = AnalyzeConnectivity(m);
    EXPECT_FALSE(r.IsEdgeManifold());
    ASSERT_EQ(r.nonManifoldEdges.size(), 1u);
    EXPECT_EQ(r.nonManifoldEdges[0], std::make_pair(0,1));
}

TEST(MeshConnectivity, DetectsBoundaryEdges) {
    SurfaceMesh m;
    m.vertices = {{0,0,0},{1,0,0},{0,1,0}};
    m.normals.assign(3, Eigen::Vector3f(0,0,1));
    m.triangles = {{0,1,2}};
    ConnectivityReport r = AnalyzeConnectivity(m);
    EXPECT_EQ(r.boundaryEdges.size(), 3u);
    EXPECT_FALSE(r.IsClosed());
    EXPECT_TRUE(r.IsEdgeManifold()); // boundary != non-manifold
}

TEST(MeshConnectivity, DetectsBowtieVertex) {
    // Two triangles sharing ONLY vertex 0.
    SurfaceMesh m;
    m.vertices = {{0,0,0},{1,0,0},{1,1,0},{-1,0,0},{-1,1,0}};
    m.normals.assign(5, Eigen::Vector3f(0,0,1));
    m.triangles = {{0,1,2},{0,3,4}};
    ConnectivityReport r = AnalyzeConnectivity(m);
    ASSERT_FALSE(r.nonManifoldVertices.empty());
    EXPECT_NE(std::find(r.nonManifoldVertices.begin(), r.nonManifoldVertices.end(), 0),
              r.nonManifoldVertices.end());
}

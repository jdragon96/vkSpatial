#pragma once

#include "Mesh/SurfaceMesh.h"

#include <utility>
#include <vector>

namespace Mesh {

    // One directed half-edge of a triangle mesh: the vertex it starts at, the face (index
    // into SurfaceMesh::triangles) that owns it, the next half-edge going CCW around that
    // same face, and the opposite half-edge in the adjacent face across this edge.
    // twin == -1 when the edge is a boundary edge or a non-manifold edge (no unique
    // adjacent face to pair with).
    struct HalfEdge {
        int origin;
        int face;
        int next;
        int twin;
    };

    // Half-edge adjacency built over a SurfaceMesh. vertexHalfEdge[vertexIndex] is the
    // index of one half-edge whose origin is that vertex (-1 if the vertex is not used by
    // any non-degenerate triangle).
    struct HalfEdgeMesh {
        std::vector<HalfEdge> halfEdges;
        std::vector<int> vertexHalfEdge;
    };

    // Builds the half-edge adjacency for `mesh`. Triangles that are degenerate (a repeated
    // vertex index, or near-zero area) are skipped entirely so they cannot corrupt edge
    // incidence counts; twin is set only for edges with exactly two incident triangles.
    HalfEdgeMesh BuildHalfEdgeMesh(const SurfaceMesh &mesh);

    // Combinatorial connectivity diagnostics for a SurfaceMesh.
    struct ConnectivityReport {
        std::vector<std::pair<int, int>> boundaryEdges;    // (v0<v1), exactly 1 incident triangle
        std::vector<std::pair<int, int>> nonManifoldEdges; // (v0<v1), more than 2 incident triangles
        std::vector<int> nonManifoldVertices;              // bowtie vertices: one-ring is >=2 fans
        int degenerateTriangles = 0;

        bool IsEdgeManifold() const { return nonManifoldEdges.empty(); }
        bool IsClosed() const { return boundaryEdges.empty() && nonManifoldEdges.empty(); }
    };

    // Analyzes `mesh` for boundary edges, non-manifold edges, non-manifold (bowtie)
    // vertices, and degenerate triangles.
    ConnectivityReport AnalyzeConnectivity(const SurfaceMesh &mesh);

} // namespace Mesh

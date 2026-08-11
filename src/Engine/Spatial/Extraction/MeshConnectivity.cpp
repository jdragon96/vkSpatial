#include "Engine/Spatial/Extraction/MeshConnectivity.h"

#include <Eigen/Geometry> // Vector3f::cross
#include <algorithm>
#include <map>

namespace Engine::Spatial::Extraction {

    namespace {

        // A triangle is degenerate if it repeats a vertex index, or if its area is so close
        // to zero that the cross-product magnitude used to weight normals is meaningless.
        bool IsDegenerateTriangle(const SurfaceMesh &mesh, int vertexIndexA, int vertexIndexB, int vertexIndexC) {
            if (vertexIndexA == vertexIndexB || vertexIndexB == vertexIndexC || vertexIndexC == vertexIndexA)
                return true;

            const Eigen::Vector3f &positionA = mesh.vertices[vertexIndexA];
            const Eigen::Vector3f &positionB = mesh.vertices[vertexIndexB];
            const Eigen::Vector3f &positionC = mesh.vertices[vertexIndexC];
            const Eigen::Vector3f crossProduct = (positionB - positionA).cross(positionC - positionA);
            return crossProduct.norm() < 1e-12f;
        }

        // Builds 3 directed half-edges per non-degenerate triangle (next = CCW within the
        // face, matching the input vertex order; twin left at -1) and groups their global
        // indices by the undirected edge key (min(origin,target), max(origin,target)) so the
        // caller can classify each edge by its incident-triangle count. Degenerate triangles
        // are skipped and counted via `degenerateTriangleCount` rather than being allowed to
        // contribute half-edges that would corrupt those counts.
        HalfEdgeMesh BuildUntwinnedHalfEdges(const SurfaceMesh &mesh,
                                              std::map<std::pair<int, int>, std::vector<int>> &edgeGroups,
                                              int &degenerateTriangleCount) {
            HalfEdgeMesh halfEdgeMesh;
            halfEdgeMesh.vertexHalfEdge.assign(mesh.vertices.size(), -1);
            edgeGroups.clear();
            degenerateTriangleCount = 0;

            for (size_t triangleIndex = 0; triangleIndex < mesh.triangles.size(); ++triangleIndex) {
                const Eigen::Vector3i &triangle = mesh.triangles[triangleIndex];
                const int cornerVertex[3] = {triangle[0], triangle[1], triangle[2]};

                if (IsDegenerateTriangle(mesh, cornerVertex[0], cornerVertex[1], cornerVertex[2])) {
                    ++degenerateTriangleCount;
                    continue;
                }

                // 1. Emit the 3 directed half-edges for this face, chained next = CCW.
                const int firstHalfEdgeIndex = static_cast<int>(halfEdgeMesh.halfEdges.size());
                for (int corner = 0; corner < 3; ++corner) {
                    HalfEdge halfEdge;
                    halfEdge.origin = cornerVertex[corner];
                    halfEdge.face = static_cast<int>(triangleIndex);
                    halfEdge.next = firstHalfEdgeIndex + (corner + 1) % 3;
                    halfEdge.twin = -1;
                    halfEdgeMesh.halfEdges.push_back(halfEdge);

                    if (halfEdgeMesh.vertexHalfEdge[halfEdge.origin] == -1)
                        halfEdgeMesh.vertexHalfEdge[halfEdge.origin] = firstHalfEdgeIndex + corner;
                }

                // 2. Group each directed half-edge under its undirected edge key.
                for (int corner = 0; corner < 3; ++corner) {
                    const int halfEdgeIndex = firstHalfEdgeIndex + corner;
                    const int originVertex = halfEdgeMesh.halfEdges[halfEdgeIndex].origin;
                    const int targetVertex = halfEdgeMesh.halfEdges[halfEdgeMesh.halfEdges[halfEdgeIndex].next].origin;
                    const std::pair<int, int> undirectedKey(std::min(originVertex, targetVertex), std::max(originVertex, targetVertex));
                    edgeGroups[undirectedKey].push_back(halfEdgeIndex);
                }
            }

            return halfEdgeMesh;
        }

        // Sets twin on both half-edges of every undirected edge with exactly 2 incident
        // triangles -- the unique manifold case. Boundary edges (1 incident triangle) and
        // non-manifold edges (>2) are left at twin = -1.
        void AssignManifoldTwins(HalfEdgeMesh &halfEdgeMesh, const std::map<std::pair<int, int>, std::vector<int>> &edgeGroups) {
            for (const auto &[undirectedKey, halfEdgeIndices] : edgeGroups) {
                if (halfEdgeIndices.size() == 2) {
                    halfEdgeMesh.halfEdges[halfEdgeIndices[0]].twin = halfEdgeIndices[1];
                    halfEdgeMesh.halfEdges[halfEdgeIndices[1]].twin = halfEdgeIndices[0];
                }
            }
        }

        // Finds bowtie (non-manifold) vertices: a vertex whose incident triangles do not
        // form a single edge-connected fan under the twin adjacency. For every vertex with
        // 2 or more incident triangles, walks the one-ring starting from a single incident
        // half-edge and flags the vertex if that walk cannot reach every other incident
        // half-edge.
        std::vector<int> FindNonManifoldVertices(const SurfaceMesh &mesh, const HalfEdgeMesh &halfEdgeMesh) {
            // 1. Collect, per vertex, every half-edge whose origin is that vertex -- exactly
            //    one per incident (non-degenerate) triangle.
            std::vector<std::vector<int>> incidentHalfEdges(mesh.vertices.size());
            for (int halfEdgeIndex = 0; halfEdgeIndex < static_cast<int>(halfEdgeMesh.halfEdges.size()); ++halfEdgeIndex)
                incidentHalfEdges[halfEdgeMesh.halfEdges[halfEdgeIndex].origin].push_back(halfEdgeIndex);

            std::vector<int> nonManifoldVertices;
            std::vector<char> visited(halfEdgeMesh.halfEdges.size(), 0);
            std::vector<int> stack;

            for (size_t vertexIndex = 0; vertexIndex < mesh.vertices.size(); ++vertexIndex) {
                const std::vector<int> &fanHalfEdges = incidentHalfEdges[vertexIndex];
                if (fanHalfEdges.size() < 2)
                    continue; // 0 or 1 incident triangle: trivially a single fan.

                // 2. Depth-first walk of the one-ring via the twin adjacency: from a
                //    half-edge `currentHalfEdge` with origin `vertexIndex`, the two
                //    neighbouring triangles around `vertexIndex` are reached via
                //    twin(next(next(currentHalfEdge))) and next(twin(currentHalfEdge)).
                //    Both, when not -1, are guaranteed to have origin == vertexIndex, i.e.
                //    to be members of fanHalfEdges.
                stack.clear();
                stack.push_back(fanHalfEdges[0]);
                visited[fanHalfEdges[0]] = 1;
                size_t visitedCount = 1;

                while (!stack.empty()) {
                    const int currentHalfEdge = stack.back();
                    stack.pop_back();

                    const int previousHalfEdge = halfEdgeMesh.halfEdges[halfEdgeMesh.halfEdges[currentHalfEdge].next].next;
                    const int forwardNeighbour = halfEdgeMesh.halfEdges[previousHalfEdge].twin;
                    const int currentTwin = halfEdgeMesh.halfEdges[currentHalfEdge].twin;
                    const int backwardNeighbour = currentTwin == -1 ? -1 : halfEdgeMesh.halfEdges[currentTwin].next;

                    for (int neighbourHalfEdge : {forwardNeighbour, backwardNeighbour}) {
                        if (neighbourHalfEdge != -1 && !visited[neighbourHalfEdge]) {
                            visited[neighbourHalfEdge] = 1;
                            ++visitedCount;
                            stack.push_back(neighbourHalfEdge);
                        }
                    }
                }

                if (visitedCount < fanHalfEdges.size())
                    nonManifoldVertices.push_back(static_cast<int>(vertexIndex));

                // 3. Reset exactly the half-edges this fan could have marked, so `visited`
                //    stays reusable for the next vertex without a full O(halfEdges) clear.
                for (int halfEdgeIndex : fanHalfEdges)
                    visited[halfEdgeIndex] = 0;
            }

            return nonManifoldVertices;
        }

    } // namespace

    HalfEdgeMesh BuildHalfEdgeMesh(const SurfaceMesh &mesh) {
        std::map<std::pair<int, int>, std::vector<int>> edgeGroups;
        int degenerateTriangleCount = 0;
        HalfEdgeMesh halfEdgeMesh = BuildUntwinnedHalfEdges(mesh, edgeGroups, degenerateTriangleCount);
        AssignManifoldTwins(halfEdgeMesh, edgeGroups);
        return halfEdgeMesh;
    }

    ConnectivityReport AnalyzeConnectivity(const SurfaceMesh &mesh) {
        ConnectivityReport report;

        std::map<std::pair<int, int>, std::vector<int>> edgeGroups;
        HalfEdgeMesh halfEdgeMesh = BuildUntwinnedHalfEdges(mesh, edgeGroups, report.degenerateTriangles);
        AssignManifoldTwins(halfEdgeMesh, edgeGroups);

        // 1. Classify every undirected edge by its incident-triangle count. `edgeGroups`
        //    (a std::map keyed on (v0<v1)) already iterates in ascending key order, so both
        //    lists come out sorted deterministically with no extra pass required.
        for (const auto &[undirectedKey, halfEdgeIndices] : edgeGroups) {
            if (halfEdgeIndices.size() == 1)
                report.boundaryEdges.push_back(undirectedKey);
            else if (halfEdgeIndices.size() > 2)
                report.nonManifoldEdges.push_back(undirectedKey);
        }

        // 2. Classify bowtie vertices via one-ring fan connectivity.
        report.nonManifoldVertices = FindNonManifoldVertices(mesh, halfEdgeMesh);

        return report;
    }

} // namespace Engine::Spatial::Extraction

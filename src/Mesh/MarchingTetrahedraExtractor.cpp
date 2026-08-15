// The "mtet" strategy: Marching Tetrahedra over a VoxelField. Reuses the shared core
// (MarchingCubesCore.h) for candidate-cube enumeration, edge-vertex placement
// (core::VertexInterpolate) and welding (core::WeldAndComputeNormals) -- only the per-cube
// triangle TOPOLOGY differs: instead of a cube's 8 corners being resolved directly against a
// 256-case table (as "mc"/"mc33" do), each candidate cube is first split into 6 tetrahedra that
// share the cube's main diagonal from corner 0 to corner 6 (see MarchingCubesTables.h's CORNER
// layout diagram), and each tetrahedron is then triangulated independently from its own 4
// corner signs.
//
// This sidesteps trilinear ambiguity entirely: a Marching Cubes cube interpolates its scalar
// field TRILINEARLY across the cube, so some corner-sign configurations admit two different,
// disagreeing triangulations of the same face/interior (the reason "mc33" exists). A
// tetrahedron has only 4 corners, so the natural interpolant across it is purely AFFINE
// (barycentric-linear); an affine scalar field's zero-set intersected with a tetrahedron is
// always exactly one planar polygon (a triangle or a planar quadrilateral), so there is only
// ever one possible triangulation per corner-sign configuration, up to (harmless) diagonal
// choice for the quadrilateral case.
//
// Algorithm reference: A. Doi, A. Koide, "An Efficient Method of Triangulating Equi-Valued
// Surfaces by Using Tetrahedral Cells", IEICE Transactions on Information and Systems,
// E74-D(1):214-224, 1991; see also P. Bourke, "Polygonising a Scalar Field Using Tetrahedra"
// (http://paulbourke.net/geometry/polygonise/) for the widely-mirrored write-up of the same 16
// (2, up to symmetry) corner-sign cases used below. Unlike that reference's fixed per-case
// vertex ordering, TriangulateTetrahedron here derives each triangle's winding dynamically
// (EmitOutwardTriangle) by comparing its raw cross-product normal against the tetrahedron's own
// inside-corners-mean -> outside-corners-mean direction, so it needs no separate hand-verified
// winding per case.

#include "Mesh/ExtractorRegistry.h"
#include "Mesh/IsoSurfaceExtractor.h"
#include "Mesh/MarchingCubesCore.h"

#include <Eigen/Geometry> // Vector3f::cross
#include <array>
#include <memory>
#include <vector>

namespace Mesh {

    namespace {

        using TetrahedronCornerPosition = std::array<Eigen::Vector3f, 4>;
        using TetrahedronCornerValue = std::array<float, 4>;

        // The 6 tetrahedra that partition a cube, all sharing the main diagonal from corner 0
        // to corner 6 (mc::CORNER order); each inner array holds 4 indices into the cube's
        // per-corner position/value arrays (see EmitCubeTriangles).
        constexpr int kCubeTetrahedra[6][4] = {
                {0, 5, 1, 6}, {0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}};

        // "mc"/"mc33" only ever place a vertex on one of a cube's 12 EDGES, so weldFraction (a
        // fraction of a whole cube edge) is a generous cross-cube-matching tolerance for them:
        // two genuinely distinct MC vertices are essentially never that close together. Splitting
        // into 6 tetrahedra additionally cuts along face and space DIAGONALS wherever those are a
        // tetrahedron's own edge, typically emitting 2-3x more (and correspondingly smaller)
        // triangles per cube -- so two genuinely DISTINCT vertices (from different tets, meant to
        // stay separate) can legitimately sit much closer together, as a fraction of cellSize,
        // than any two "mc" vertices ever do. Welding at the full weldFraction*cellSize distance
        // over-merges such pairs into one vertex, silently collapsing the triangles that
        // reference them to zero area (dropped by core::WeldAndComputeNormals) and leaving holes
        // -- this is exactly what caused the sphere fixture to fail edge-manifold/watertight/
        // Euler==2 before this divisor was introduced (confirmed by bisection: sweeping the
        // effective weldFraction on that fixture, the mesh is byte-identical and fully
        // edge-manifold/watertight/Euler==2 for every divisor >=12.5 tested (up to 2500), and
        // already breaks at divisor 5). Dividing weldFraction*cellSize by kWeldDistanceDivisor
        // keeps the tolerance generous enough for genuine cross-cube/cross-tet duplicates (which
        // differ only by float rounding, ~1e-7) while staying well clear of mtet's legitimate
        // close-vertex spacing: 32 was then re-validated (edge-manifold, watertight, Euler==2,
        // zero mismatched edges) across 5 different isotest::SphereField radius/cellSize
        // combinations, keeping >2x margin above the first known-good divisor and >6x below the
        // first known-bad one.
        constexpr float kWeldDistanceDivisor = 32.0f;

        // Appends the triangle (vertexA,vertexB,vertexC) to `out`, flipping its winding first if
        // needed so its cross-product face normal points along `outwardDirection` -- i.e. from
        // the tetrahedron's negative (inside) corners toward its positive (outside) corners,
        // matching the negative-inside/positive-outside sign convention used throughout
        // Mesh (see core::GenerateRawTriangles's cubeIndex comment).
        void EmitOutwardTriangle(const Eigen::Vector3f &vertexA, const Eigen::Vector3f &vertexB,
                                  const Eigen::Vector3f &vertexC, const Eigen::Vector3f &outwardDirection,
                                  std::vector<core::RawTriangle> &out) {
            const Eigen::Vector3f faceNormal = (vertexB - vertexA).cross(vertexC - vertexA);
            if (faceNormal.dot(outwardDirection) < 0.0f)
                out.push_back({vertexA, vertexC, vertexB}); // flip: swap two corners to reverse winding
            else
                out.push_back({vertexA, vertexB, vertexC});
        }

        // Triangulates one tetrahedron (4 world corner positions + 4 iso-shifted values) into 0,
        // 1 or 2 outward-wound triangles, appended to `out`. No trilinear ambiguity by
        // construction (see file header): the corner signs alone fully determine the cut.
        void TriangulateTetrahedron(const TetrahedronCornerPosition &cornerPosition,
                                     const TetrahedronCornerValue &cornerValue, std::vector<core::RawTriangle> &out) {
            int negativeCornerIndex[4], positiveCornerIndex[4];
            int negativeCount = 0, positiveCount = 0;
            for (int i = 0; i < 4; ++i) {
                if (cornerValue[i] < 0.0f)
                    negativeCornerIndex[negativeCount++] = i;
                else
                    positiveCornerIndex[positiveCount++] = i;
            }
            if (negativeCount == 0 || negativeCount == 4) return; // whole tetrahedron on one side: no cut here

            Eigen::Vector3f insideMean = Eigen::Vector3f::Zero();
            for (int i = 0; i < negativeCount; ++i) insideMean += cornerPosition[negativeCornerIndex[i]];
            insideMean /= float(negativeCount);

            Eigen::Vector3f outsideMean = Eigen::Vector3f::Zero();
            for (int i = 0; i < positiveCount; ++i) outsideMean += cornerPosition[positiveCornerIndex[i]];
            outsideMean /= float(positiveCount);

            const Eigen::Vector3f outwardDirection = outsideMean - insideMean;

            const auto interpolateCutEdge = [&](int cornerIndexA, int cornerIndexB) {
                return core::VertexInterpolate(cornerPosition[cornerIndexA], cornerPosition[cornerIndexB],
                                                cornerValue[cornerIndexA], cornerValue[cornerIndexB]);
            };

            if (negativeCount == 1 || negativeCount == 3) {
                // A single "lone" corner (the minority sign) is cut off by one triangular plane
                // through the three edges connecting it to the other three corners.
                const int loneCornerIndex = (negativeCount == 1) ? negativeCornerIndex[0] : positiveCornerIndex[0];
                int otherCornerIndex[3];
                int otherCount = 0;
                for (int i = 0; i < 4; ++i)
                    if (i != loneCornerIndex) otherCornerIndex[otherCount++] = i;
                EmitOutwardTriangle(interpolateCutEdge(loneCornerIndex, otherCornerIndex[0]),
                                    interpolateCutEdge(loneCornerIndex, otherCornerIndex[1]),
                                    interpolateCutEdge(loneCornerIndex, otherCornerIndex[2]), outwardDirection, out);
                return;
            }

            // negativeCount == 2: the isosurface plane separates the two negative corners
            // {negativeCornerA,negativeCornerB} from the two positive corners
            // {positiveCornerA,positiveCornerB}, crossing exactly the four edges between one
            // negative and one positive corner (the negative-negative and positive-positive
            // edges are not cut). The resulting planar quadrilateral's vertices, in cyclic
            // order, are the crossings of edges (negativeCornerA,positiveCornerA),
            // (negativeCornerA,positiveCornerB), (negativeCornerB,positiveCornerB),
            // (negativeCornerB,positiveCornerA): each consecutive pair of those four crossings
            // lies on a shared tetrahedron face (the face opposite whichever corner is excluded
            // from that pair), so this order traces the quadrilateral's actual boundary. Split
            // it into 2 triangles on the diagonal between the 1st and 3rd crossings; either
            // diagonal is valid since the quadrilateral is exactly planar (no ambiguity).
            const int negativeCornerA = negativeCornerIndex[0], negativeCornerB = negativeCornerIndex[1];
            const int positiveCornerA = positiveCornerIndex[0], positiveCornerB = positiveCornerIndex[1];
            const Eigen::Vector3f quadrilateralVertex[4] = {
                    interpolateCutEdge(negativeCornerA, positiveCornerA),
                    interpolateCutEdge(negativeCornerA, positiveCornerB),
                    interpolateCutEdge(negativeCornerB, positiveCornerB),
                    interpolateCutEdge(negativeCornerB, positiveCornerA)};
            EmitOutwardTriangle(quadrilateralVertex[0], quadrilateralVertex[1], quadrilateralVertex[2],
                                outwardDirection, out);
            EmitOutwardTriangle(quadrilateralVertex[0], quadrilateralVertex[2], quadrilateralVertex[3],
                                outwardDirection, out);
        }

        // Splits one candidate cube (base coordinate + its 8 iso-shifted corner values, in
        // mc::CORNER order) into the 6 tetrahedra of kCubeTetrahedra and triangulates each
        // independently, appending every resulting triangle to `out`.
        void EmitCubeTriangles(const std::array<int, 3> &base, const std::array<float, 8> &cornerValue,
                                float cellSize, std::vector<core::RawTriangle> &out) {
            Eigen::Vector3f cornerPosition[8];
            for (int corner = 0; corner < 8; ++corner)
                cornerPosition[corner] = Eigen::Vector3f(float(base[0] + mc::CORNER[corner][0]),
                                                          float(base[1] + mc::CORNER[corner][1]),
                                                          float(base[2] + mc::CORNER[corner][2])) *
                                          cellSize;

            for (const auto &tetrahedronCornerIndex : kCubeTetrahedra) {
                const TetrahedronCornerPosition tetrahedronCornerPosition{
                        cornerPosition[tetrahedronCornerIndex[0]], cornerPosition[tetrahedronCornerIndex[1]],
                        cornerPosition[tetrahedronCornerIndex[2]], cornerPosition[tetrahedronCornerIndex[3]]};
                const TetrahedronCornerValue tetrahedronCornerValue{
                        cornerValue[tetrahedronCornerIndex[0]], cornerValue[tetrahedronCornerIndex[1]],
                        cornerValue[tetrahedronCornerIndex[2]], cornerValue[tetrahedronCornerIndex[3]]};
                TriangulateTetrahedron(tetrahedronCornerPosition, tetrahedronCornerValue, out);
            }
        }

        // Same candidate-cube enumeration, isoLevel shift and weld as "mc"/"mc33" -- only
        // EmitCubeTriangles (above) differs.
        class MarchingTetrahedraExtractor : public IsoSurfaceExtractor {
        public:
            const char *Name() const override { return "mtet"; }

            SurfaceMesh Extract(const VoxelField &field, const ExtractParams &params) const override {
                const auto bases = core::CandidateBases(field.OccupiedCoords());

                std::vector<core::RawTriangle> raw;
                for (const auto &base : bases) {
                    std::array<float, 8> cornerValue{};
                    bool complete = true;
                    for (int corner = 0; corner < 8 && complete; ++corner) {
                        const std::array<int, 3> cornerCoord{base[0] + mc::CORNER[corner][0],
                                                               base[1] + mc::CORNER[corner][1],
                                                               base[2] + mc::CORNER[corner][2]};
                        float value = 0.0f;
                        if (!field.Sample(cornerCoord, value)) {
                            complete = false;
                            break;
                        }
                        cornerValue[corner] = value - params.isoLevel;
                    }
                    if (!complete) continue; // an unresolvable corner: skip this cube, matching "mc"/"mc33"

                    EmitCubeTriangles(base, cornerValue, field.CellSize(), raw);
                }

                return core::WeldAndComputeNormals(raw, params.weldFraction * field.CellSize() / kWeldDistanceDivisor);
            }
        };

    } // namespace

    std::unique_ptr<IsoSurfaceExtractor> CreateMarchingTetrahedraExtractor() {
        return std::make_unique<MarchingTetrahedraExtractor>();
    }

} // namespace Mesh

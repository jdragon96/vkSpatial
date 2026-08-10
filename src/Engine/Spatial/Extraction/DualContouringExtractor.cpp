// The "dc" strategy: Dual Contouring of Hermite data over a VoxelField. Unlike every other
// strategy in this namespace ("mc"/"mc33"/"mtet"/"emc"), "dc" places AT MOST ONE vertex per CELL
// (cube) rather than one vertex per crossing EDGE, and connects those cell vertices following
// the grid's dual topology instead of any Marching-Cubes case table -- so a cell straddling a
// sharp feature (crease, corner) places its one vertex at the feature itself by construction,
// without needing "emc"'s separate feature-cell classification/fan-boundary machinery. Reuses
// the shared core (MarchingCubesCore.h) for candidate-cube enumeration, edge-crossing placement
// (core::VertexInterpolate) and the final weld/normal computation (core::WeldAndComputeNormals),
// and the shared per-cell solver (QuadraticErrorFunction.h, already used by "emc") for the one
// feature vertex per cell -- but, unlike "emc", EVERY cell with any sign change gets a QEF
// vertex, and mc::edgeTable/triTable are consulted only to detect PER-CELL sign changes (which
// of a cell's 12 edges cross), never to choose a triangle topology.
//
// Two passes over the field:
//   1. For every candidate cube (core::CandidateBases) whose 8 corners are all resolvable and
//      whose corner signs aren't unanimous (edgeTable[cubeIndex] != 0): accumulate every
//      sign-changing edge's Hermite data (crossing point via core::VertexInterpolate, crossing
//      normal via VoxelField::Gradient interpolated the same way -- central-difference fallback
//      when a corner has none, mirroring "emc"'s EstimateCornerNormal/InterpolateEdgeNormal)
//      into a QuadraticErrorFunction, and Solve() it biased to the cell's centroid for exactly
//      one dual vertex. Recorded in cellDualVertexIndex, keyed by the cell's own base coordinate,
//      so pass 2 can look any cell's vertex up by coordinate alone.
//   2. For every grid edge with a sign change (both endpoints resolvable, opposite corner signs
//      -- tested directly against the field, independently of pass 1's per-cube
//      cubeIndex/edgeTable path), the 4 cells sharing that edge (QuadCellBasesAroundEdge) each
//      contributed a dual vertex in pass 1; connect those 4 into a quad and split it into 2
//      triangles. The quad's winding is corrected per-triangle from the edge's own sign
//      direction (EmitOutwardTriangle: flip iff the raw cross-product normal disagrees with the
//      direction from the edge's negative corner to its positive corner) -- the same "derive
//      winding dynamically, don't hand-verify a table" idiom "mtet" uses
//      (MarchingTetrahedraExtractor.cpp's EmitOutwardTriangle), so the 4-cell traversal order
//      only has to be a simple (non-self-intersecting) loop around the edge, not a
//      pre-proven-correct winding for each of the 3 axes and 2 sign directions separately. A
//      grid edge whose 4 sharing cells aren't all resolvable (sparse-field boundary) is skipped,
//      leaving an open boundary there -- the same boundary behaviour as "mc"/"mc33"/"mtet"/"emc"
//      at the edge of a sparse field.
//
// The final weld (core::WeldAndComputeNormals) uses a near-zero distance: DC's dual vertices are
// already unique per cell (computed once in pass 1, referenced by bit-identical copies from every
// quad incident to that cell in pass 2), so -- unlike "mc"/"mc33"/"mtet", which weld away genuine
// floating-point disagreement between independently-computed crossings from neighbouring cubes --
// the only thing left to fold together here is repeated bit-identical positions, and the weld
// pass is really only being reused for the area-weighted per-vertex normal computation that comes
// with it.
//
// Reference: T. Ju, F. Losasso, S. Schaefer, J. Warren, "Dual Contouring of Hermite Data",
// SIGGRAPH 2002.

#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Spatial/Extraction/IsoSurfaceExtractor.h"
#include "Engine/Spatial/Extraction/MarchingCubesCore.h"
#include "Engine/Spatial/Extraction/QuadraticErrorFunction.h"

#include <Eigen/Geometry> // Vector3f::cross
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace Engine::Spatial::Extraction {

    namespace {

        // A unit outward-pointing normal at one cube corner: the field's own stored gradient if
        // present, otherwise a central-difference estimate from the corner's face-neighbour
        // scalar samples (a missing neighbour falls back to the corner's own value, degrading
        // gracefully to a one-sided difference at the edge of a sparse field). Mirrors "emc"'s
        // EstimateCornerNormal (ExtendedMarchingCubesExtractor.cpp) verbatim -- duplicated here
        // rather than shared because it lives in that file's own anonymous namespace.
        Eigen::Vector3f EstimateCornerNormal(const VoxelField &field, const std::array<int, 3> &cornerCoord) {
            Eigen::Vector3f gradient;
            if (!field.Gradient(cornerCoord, gradient)) {
                float centerValue = 0.0f;
                field.Sample(cornerCoord, centerValue); // resolvable: caller already required this corner
                const auto sampleOrCenter = [&](int offsetX, int offsetY, int offsetZ) {
                    float value = centerValue;
                    const std::array<int, 3> neighborCoord{cornerCoord[0] + offsetX, cornerCoord[1] + offsetY,
                                                           cornerCoord[2] + offsetZ};
                    field.Sample(neighborCoord, value);
                    return value;
                };
                gradient = Eigen::Vector3f(sampleOrCenter(1, 0, 0) - sampleOrCenter(-1, 0, 0),
                                           sampleOrCenter(0, 1, 0) - sampleOrCenter(0, -1, 0),
                                           sampleOrCenter(0, 0, 1) - sampleOrCenter(0, 0, -1));
            }
            const float length = gradient.norm();
            return length > 1e-8f ? Eigen::Vector3f(gradient / length) : Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        }

        // Interpolates two corner normals with the exact same edge parameter
        // core::VertexInterpolate derives from (cornerValueA, cornerValueB) for the position on
        // that edge, then renormalizes. Mirrors "emc"'s InterpolateEdgeNormal verbatim (see
        // EstimateCornerNormal above for why this is a local duplicate, not a shared function).
        Eigen::Vector3f InterpolateEdgeNormal(const Eigen::Vector3f &cornerNormalA, const Eigen::Vector3f &cornerNormalB,
                                              float cornerValueA, float cornerValueB) {
            const Eigen::Vector3f interpolated = core::VertexInterpolate(cornerNormalA, cornerNormalB, cornerValueA, cornerValueB);
            const float length = interpolated.norm();
            return length > 1e-8f ? Eigen::Vector3f(interpolated / length) : cornerNormalA;
        }

        // The 4 cube bases sharing the grid edge (lowerCoord -> lowerCoord+step(axis)): fixed at
        // lowerCoord along `axis` itself, and ranging over {lowerCoord-1, lowerCoord} along the
        // other two axes (taken in cyclic order otherAxisU=axis+1, otherAxisV=axis+2). The 4 are
        // listed walking the shared edge's actual perimeter -- each consecutive pair differs by
        // one unit step along otherAxisU or otherAxisV, never both at once -- so consecutive
        // entries are always the two cube bases of a real shared FACE, not a diagonal pair, which
        // is what makes the quad built from them (in Extract, below) a simple non-self-
        // intersecting polygon. The listed direction is otherwise arbitrary: EmitOutwardTriangle
        // independently corrects each triangle's winding from the edge's own sign direction, so
        // walking this loop the other way would still produce a correct mesh.
        std::array<std::array<int, 3>, 4> QuadCellBasesAroundEdge(const std::array<int, 3> &lowerCoord, int axis) {
            const int otherAxisU = (axis + 1) % 3, otherAxisV = (axis + 2) % 3;
            const auto offsetBase = [&](int deltaU, int deltaV) {
                std::array<int, 3> base = lowerCoord;
                base[otherAxisU] += deltaU;
                base[otherAxisV] += deltaV;
                return base;
            };
            return {offsetBase(0, 0), offsetBase(-1, 0), offsetBase(-1, -1), offsetBase(0, -1)};
        }

        // Appends the triangle (vertexA,vertexB,vertexC) to `out`, flipping its winding first if
        // needed so its cross-product face normal points along `outwardDirection`. Same
        // dynamic-orientation idiom as "mtet"'s EmitOutwardTriangle
        // (MarchingTetrahedraExtractor.cpp): rather than hand-deriving and verifying a fixed
        // winding per axis/sign-direction combination, just compare the raw cross product against
        // the direction that is, by definition, outward for this edge (its negative corner to its
        // positive corner) and flip if they disagree.
        void EmitOutwardTriangle(const Eigen::Vector3f &vertexA, const Eigen::Vector3f &vertexB,
                                 const Eigen::Vector3f &vertexC, const Eigen::Vector3f &outwardDirection,
                                 std::vector<core::RawTriangle> &out) {
            const Eigen::Vector3f faceNormal = (vertexB - vertexA).cross(vertexC - vertexA);
            if (faceNormal.dot(outwardDirection) < 0.0f)
                out.push_back({vertexA, vertexC, vertexB}); // flip: swap two corners to reverse winding
            else
                out.push_back({vertexA, vertexB, vertexC});
        }

        // DC's dual vertices are already unique per cell (computed once in pass 1, referenced by
        // bit-identical copies from every quad incident to that cell in pass 2) -- unlike
        // "mc"/"mc33"/"mtet", there is no genuine floating-point disagreement between
        // independently-computed near-duplicate crossings that needs merging. This weld pass
        // exists only to fold each cell's repeated bit-identical position references back down to
        // one shared mesh vertex and to derive area-weighted per-vertex normals from the result;
        // a near-zero distance is sufficient for the former and deliberately too small to ever
        // merge two genuinely distinct dual vertices.
        constexpr float kDualVertexWeldDistance = 1e-6f;

        class DualContouringExtractor : public IsoSurfaceExtractor {
        public:
            const char *Name() const override { return "dc"; }

            SurfaceMesh Extract(const VoxelField &field, const ExtractParams &params) const override {
                const auto bases = core::CandidateBases(field.OccupiedCoords());

                // Pass 1 (file header step 1): one QEF dual vertex per cell (cube base) with a
                // sign change on >=1 of its 12 edges.
                std::map<std::array<int, 3>, uint32_t> cellDualVertexIndex;
                std::vector<Eigen::Vector3f> dualVertexPosition;

                for (const auto &base: bases) {
                    std::array<int, 3> cornerCoord[8];
                    float cornerValue[8];
                    bool complete = true;
                    for (int corner = 0; corner < 8 && complete; ++corner) {
                        const std::array<int, 3> coord{base[0] + mc::CORNER[corner][0], base[1] + mc::CORNER[corner][1],
                                                       base[2] + mc::CORNER[corner][2]};
                        cornerCoord[corner] = coord;
                        float value;
                        if (!field.Sample(coord, value)) {
                            complete = false;
                            break;
                        }
                        cornerValue[corner] = value - params.isoLevel;
                    }
                    if (!complete) continue; // an unresolvable corner: skip this cell, matching "mc"/"emc"

                    int cubeIndex = 0;
                    for (int corner = 0; corner < 8; ++corner)
                        if (cornerValue[corner] < 0.0f) cubeIndex |= (1 << corner);

                    const int edgeMask = mc::edgeTable[cubeIndex];
                    if (edgeMask == 0) continue; // this cell's own corners never cross the isosurface

                    Eigen::Vector3f cornerPosition[8];
                    for (int corner = 0; corner < 8; ++corner)
                        cornerPosition[corner] = Eigen::Vector3f(float(cornerCoord[corner][0]), float(cornerCoord[corner][1]),
                                                                 float(cornerCoord[corner][2])) *
                                                 field.CellSize();

                    QuadraticErrorFunction qef;
                    for (int edge = 0; edge < 12; ++edge) {
                        if (!(edgeMask & (1 << edge))) continue;
                        const int cornerA = core::kEdgeCornerPairs[edge][0], cornerB = core::kEdgeCornerPairs[edge][1];
                        const Eigen::Vector3f crossingPosition = core::VertexInterpolate(
                                cornerPosition[cornerA], cornerPosition[cornerB], cornerValue[cornerA], cornerValue[cornerB]);
                        const Eigen::Vector3f cornerNormalA = EstimateCornerNormal(field, cornerCoord[cornerA]);
                        const Eigen::Vector3f cornerNormalB = EstimateCornerNormal(field, cornerCoord[cornerB]);
                        const Eigen::Vector3f crossingNormal =
                                InterpolateEdgeNormal(cornerNormalA, cornerNormalB, cornerValue[cornerA], cornerValue[cornerB]);
                        qef.Add(crossingPosition, crossingNormal);
                    }

                    const Eigen::Vector3f cellCentroid = (Eigen::Vector3f(float(base[0]), float(base[1]), float(base[2])) +
                                                          Eigen::Vector3f(0.5f, 0.5f, 0.5f)) *
                                                         field.CellSize();

                    cellDualVertexIndex[base] = static_cast<uint32_t>(dualVertexPosition.size());
                    dualVertexPosition.push_back(qef.Solve(cellCentroid));
                }

                // Pass 2 (file header step 2): for every sign-changing GRID edge, connect its 4
                // sharing cells' dual vertices into a quad, oriented outward from the edge's own
                // sign direction.
                //
                // KNOWN LIMITATION (accepted, spec-documented -- intentionally NOT fixed here):
                // on an "ambiguous"/checkerboard shared FACE -- all 4 corners of some face
                // alternate sign, e.g. (0,0,0) and (0,1,1) negative, (0,1,0) and (0,0,1) positive
                // on the face shared by cell bases (0,0,0) and (-1,0,0) -- all 4 of that face's
                // boundary grid edges are simultaneously sign-changing. Every one of those 4
                // edges' 4-sharing-cell sets includes BOTH cells adjacent to the face (the other
                // 2 cells differ per edge), and in each of those 4 quads the two face-adjacent
                // cells' dual vertices end up cyclically adjacent -- so the SAME mesh edge
                // (dualVertex(0,0,0) -- dualVertex(-1,0,0)) is emitted once per boundary edge:
                // 4 times, not the usual 2, i.e. triangle incidence 4 -- non-manifold. This is
                // the DC analogue of the face ambiguity plain "mc" resolves via "mc33"'s
                // asymptotic decider; basic (non-Manifold) DC has no equivalent per-face
                // tie-break and does not resolve it. The spec's Risk section anticipates and
                // accepts exactly this failure mode: "Dual-method manifoldness -- DC can produce
                // non-manifold edges at sharp configs. Mitigation: assert edge-manifoldness on
                // smooth fixtures; document the caveat (Manifold DC is a noted future
                // refinement, not in scope)" (docs/superpowers/specs/2026-08-10-isosurface-
                // extraction-strategies-design.md, Risks). test_isosurface_dc.cpp's
                // DISABLED_DualContouringAmbiguousFaceIsManifold_ManifoldDcFuture reproduces this
                // exact configuration as a regression target: a future Manifold Dual Contouring
                // implementation (Schaefer, Ju, Warren, "Manifold Dual Contouring", IEEE TVCG
                // 2007 -- adds a face-level asymptotic-decider-style tie-break that splits a
                // checkerboard face's dual connectivity into two separate edges, one per diagonal
                // pair, instead of connecting all 4 surrounding cells through it) should make
                // that test pass, not delete it.
                std::vector<core::RawTriangle> raw;
                for (const auto &lowerCoord: field.OccupiedCoords()) {
                    float lowerValueRaw;
                    field.Sample(lowerCoord, lowerValueRaw); // resolvable: lowerCoord is drawn from OccupiedCoords()
                    const float lowerValue = lowerValueRaw - params.isoLevel;
                    const bool lowerIsNegative = lowerValue < 0.0f;

                    for (int axis = 0; axis < 3; ++axis) {
                        std::array<int, 3> upperCoord = lowerCoord;
                        upperCoord[axis] += 1;
                        float upperValueRaw;
                        if (!field.Sample(upperCoord, upperValueRaw)) continue; // neighbour outside the field
                        const float upperValue = upperValueRaw - params.isoLevel;
                        if (lowerIsNegative == (upperValue < 0.0f)) continue; // no sign change on this edge

                        const auto quadCellBase = QuadCellBasesAroundEdge(lowerCoord, axis);
                        uint32_t quadVertexIndex[4];
                        bool allCellsResolved = true;
                        for (int quadCornerIndex = 0; quadCornerIndex < 4; ++quadCornerIndex) {
                            const auto it = cellDualVertexIndex.find(quadCellBase[quadCornerIndex]);
                            if (it == cellDualVertexIndex.end()) {
                                allCellsResolved = false;
                                break;
                            }
                            quadVertexIndex[quadCornerIndex] = it->second;
                        }
                        if (!allCellsResolved) continue; // a sharing cell is outside the resolvable field: leave a gap

                        const Eigen::Vector3f quadVertex[4] = {
                                dualVertexPosition[quadVertexIndex[0]], dualVertexPosition[quadVertexIndex[1]],
                                dualVertexPosition[quadVertexIndex[2]], dualVertexPosition[quadVertexIndex[3]]};

                        const Eigen::Vector3f lowerPosition = Eigen::Vector3f(float(lowerCoord[0]), float(lowerCoord[1]),
                                                                              float(lowerCoord[2])) *
                                                              field.CellSize();
                        const Eigen::Vector3f upperPosition = Eigen::Vector3f(float(upperCoord[0]), float(upperCoord[1]),
                                                                              float(upperCoord[2])) *
                                                              field.CellSize();
                        const Eigen::Vector3f outwardDirection =
                                lowerIsNegative ? (upperPosition - lowerPosition) : (lowerPosition - upperPosition);

                        EmitOutwardTriangle(quadVertex[0], quadVertex[1], quadVertex[2], outwardDirection, raw);
                        EmitOutwardTriangle(quadVertex[0], quadVertex[2], quadVertex[3], outwardDirection, raw);
                    }
                }

                return core::WeldAndComputeNormals(raw, kDualVertexWeldDistance);
            }
        };

    } // namespace

    std::unique_ptr<IsoSurfaceExtractor> CreateDualContouringExtractor() {
        return std::make_unique<DualContouringExtractor>();
    }

} // namespace Engine::Spatial::Extraction

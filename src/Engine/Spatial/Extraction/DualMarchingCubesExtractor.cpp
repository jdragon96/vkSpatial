// The "dmc" strategy: Dual Marching Cubes over a VoxelField (G. Nielson, "Dual Marching Cubes",
// IEEE Visualization 2004) -- primal contouring of the grid DUAL to the primal voxel grid. Where
// "dc" (DualContouringExtractor.cpp) connects its per-cell dual vertices directly into quads (one
// per sign-changing primal grid edge), "dmc" instead builds an actual cubical DUAL GRID out of
// those same per-cell dual vertices and runs the exact same table-driven case lookup plain "mc"
// uses (mc::edgeTable/triTable via core::VertexInterpolate) ON the dual grid's own cubes. This is
// "dc"'s per-cell placement (reused below, see PrimalCellDualVertex/pass 1) plus a second,
// meta-level application of the shared candidate-cube machinery (core::CandidateBases, reused a
// SECOND time -- see pass 2) -- producing a crack-free polygonalization (by the same "every
// shared corner is a single global lookup" argument that makes plain "mc" crack-free) capable of
// resolving TWO surface sheets that pass through the SAME primal-cell neighbourhood (e.g. a slab
// thinner than one voxel), which a single-vertex-per-PRIMAL-cell method ("dc", or plain "mc")
// cannot.
//
// The dual grid, concretely:
//   - A DUAL-GRID VERTEX is one primal CELL's dual vertex: for a cell with a sign change on any
//     of its 12 edges, this is exactly "dc"'s QEF-solved feature point (pass 1, below, duplicated
//     from DualContouringExtractor.cpp's pass 1 rather than shared, matching how "dc" itself
//     duplicates "emc"'s EstimateCornerNormal/InterpolateEdgeNormal -- see that file's header for
//     why). For a cell with NO sign change, "dc" has nothing to place there, but "dmc" still
//     needs every primal cell in range to contribute a dual-grid vertex (a dual CUBE needs a
//     value at all 8 of its corners to run a case lookup at all -- see below), so a non-straddling
//     cell falls back to its own centroid.
//   - A DUAL-GRID CUBE is centred on one primal grid VERTEX p: its 8 corners are the dual
//     vertices of the (up to) 8 primal CELLS that share p as a corner -- i.e. exactly the
//     neighbourhood core::CandidateBases already computes, now applied a second time to the set
//     of (resolved) primal cell bases instead of field.OccupiedCoords() to enumerate dual-cube
//     bases. Because mc::CORNER/edgeTable/triTable index a cube by its own BASE corner (not a
//     geometric centre), "dual-cube base q" below means the same thing "base" means everywhere
//     else in this namespace: q's 8 corners are q+mc::CORNER[c], each one a primal CELL base
//     (primal cell bases and primal grid vertices share one integer coordinate space in this
//     codebase, since a cell is identified by its own minimum corner) -- so q itself is (one of)
//     the primal grid vertex/vertices this dual cube is built around.
//
// The one place this implementation makes a deliberate, DOCUMENTED choice beyond the plan's
// brief: the brief describes a dual cube's 8 corner SCALAR values (needed to classify the dual
// cube into an mc::edgeTable/triTable case, exactly like any primal cube's 8 corner values) as
// "the primal field value at that shared primal vertex". Read as literally one single value
// (field at p) shared by all 8 corners of one dual cube, that is degenerate -- uniform sign means
// edgeTable==0 for every dual cube, so nothing would ever be extracted -- because a dual cube's 8
// corners are 8 DIFFERENT primal cells and need 8 (potentially different) scalars, the same way a
// primal cube's 8 corners need 8 different scalars. This implementation uses, for the primal cell
// at coordinate k, "the mean of that cell's own 8 (iso-shifted) corner values" -- i.e. the field
// trilinearly evaluated at the cell's own centroid -- as its dual-grid-vertex scalar. This was
// chosen (and hand-verified against the thin-slab fixture below) over the other natural reading,
// "field.Sample(k) itself" (the value at the cell's own base/min corner), because that
// alternative is NOT symmetric under which corner of a straddling cell happens to sit at its base:
// it reproduces the sheet whose straddling cell's base sits on the INSIDE with a comfortable
// margin, but places the OTHER sheet's crossing only ~0.012 world units from z=0 (inside the
// fixture's 0.02 assertion margin) purely because that sheet's straddling cell's base happens to
// sit on the OUTSIDE. The centroid mean is symmetric (doesn't privilege either corner of a
// straddling cell) and reproduces both sheets with several times that margin (see
// DualMarchingCubesResolvesThinSlabAsTwoSheets, and progress.md/task-6-report.md for the derivation).
// Either reading satisfies "run the standard mc topology, interpolating along dual-cube edges" --
// only the per-corner scalar differs.
//
// Reference: G. M. Nielson, "Dual Marching Cubes", IEEE Visualization 2004. The per-primal-cell
// dual vertex placement follows T. Ju, F. Losasso, S. Schaefer, J. Warren, "Dual Contouring of
// Hermite Data", SIGGRAPH 2002 (see QuadraticErrorFunction.h), exactly as "dc" uses it.

#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Spatial/Extraction/IsoSurfaceExtractor.h"
#include "Engine/Spatial/Extraction/MarchingCubesCore.h"
#include "Engine/Spatial/Extraction/QuadraticErrorFunction.h"

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
        // gracefully to a one-sided difference at the edge of a sparse field). Duplicated
        // verbatim from DualContouringExtractor.cpp (itself duplicated from "emc"'s own copy) --
        // see that file's header for why this is a per-file copy rather than a shared function.
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
        // that edge, then renormalizes. Duplicated verbatim from DualContouringExtractor.cpp.
        Eigen::Vector3f InterpolateEdgeNormal(const Eigen::Vector3f &cornerNormalA, const Eigen::Vector3f &cornerNormalB,
                                              float cornerValueA, float cornerValueB) {
            const Eigen::Vector3f interpolated = core::VertexInterpolate(cornerNormalA, cornerNormalB, cornerValueA, cornerValueB);
            const float length = interpolated.norm();
            return length > 1e-8f ? Eigen::Vector3f(interpolated / length) : cornerNormalA;
        }

        // One primal cell's contribution to the dual grid: a position (this cell's DC-style QEF
        // dual vertex if it has a sign change, else its own centroid) and a representative scalar
        // (the mean of its own 8 corner values) -- see file header for why the mean, not any one
        // corner's own value, is what feeds the dual cube's own case classification.
        struct PrimalCellDualVertex {
            Eigen::Vector3f position;
            float representativeValue;
        };

        // A small, fixed fraction of one primal cell -- deliberately much smaller than "mc"'s
        // default weldFraction*cellSize (0.25*cellSize): the dual grid's own crossings, computed
        // independently by every dual cube sharing a corner, are only float-noise-close (not
        // bit-identical -- mc::triTable's per-cube-local edge numbering can visit a shared edge's
        // two corners in either order, exactly the reason plain "mc" itself needs a real, not
        // near-zero, weld tolerance), so this only needs to be large enough to absorb that noise,
        // never large enough to merge two genuinely distinct dual-cube crossings (the thin-slab
        // fixture's two sheets are ~0.12 world units apart at cellSize=0.1 -- many multiples of
        // this tolerance).
        constexpr float kDualMarchingCubesWeldDistanceFraction = 0.05f;

        class DualMarchingCubesExtractor : public IsoSurfaceExtractor {
        public:
            const char *Name() const override { return "dmc"; }

            SurfaceMesh Extract(const VoxelField &field, const ExtractParams &params) const override {
                // Pass 1: one PrimalCellDualVertex per candidate primal cell -- "dc"'s pass 1
                // (QEF dual vertex from every sign-changing edge's Hermite data), extended with
                // the representativeValue + centroid-fallback the dual grid needs (see header).
                const auto primalCellBases = core::CandidateBases(field.OccupiedCoords());
                std::map<std::array<int, 3>, PrimalCellDualVertex> primalCellDualVertexByBase;

                for (const auto &base: primalCellBases) {
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
                    if (!complete) continue; // an unresolvable corner: skip this cell, matching "mc"/"dc"/"emc"

                    int cubeIndex = 0;
                    float cornerValueSum = 0.0f;
                    Eigen::Vector3f cornerPosition[8];
                    for (int corner = 0; corner < 8; ++corner) {
                        if (cornerValue[corner] < 0.0f) cubeIndex |= (1 << corner);
                        cornerValueSum += cornerValue[corner];
                        cornerPosition[corner] = Eigen::Vector3f(float(cornerCoord[corner][0]), float(cornerCoord[corner][1]),
                                                                 float(cornerCoord[corner][2])) *
                                                 field.CellSize();
                    }
                    // The mean of this cell's own 8 corner values -- equivalently the field
                    // trilinearly evaluated at the cell's own centroid -- see file header.
                    const float representativeValue = cornerValueSum * 0.125f;

                    const Eigen::Vector3f cellCentroidPosition =
                            (Eigen::Vector3f(float(base[0]), float(base[1]), float(base[2])) + Eigen::Vector3f(0.5f, 0.5f, 0.5f)) *
                            field.CellSize();

                    const int edgeMask = mc::edgeTable[cubeIndex];
                    Eigen::Vector3f position;
                    if (edgeMask == 0) {
                        position = cellCentroidPosition; // no sign change: no Hermite data: fall back to centroid
                    } else {
                        QuadraticErrorFunction qef;
                        for (int edge = 0; edge < 12; ++edge) {
                            if (!(edgeMask & (1 << edge))) continue;
                            const int cornerA = core::kEdgeCornerPairs[edge][0], cornerB = core::kEdgeCornerPairs[edge][1];
                            const Eigen::Vector3f crossingPosition = core::VertexInterpolate(
                                    cornerPosition[cornerA], cornerPosition[cornerB], cornerValue[cornerA], cornerValue[cornerB]);
                            const Eigen::Vector3f cornerNormalA = EstimateCornerNormal(field, cornerCoord[cornerA]);
                            const Eigen::Vector3f cornerNormalB = EstimateCornerNormal(field, cornerCoord[cornerB]);
                            const Eigen::Vector3f crossingNormal = InterpolateEdgeNormal(cornerNormalA, cornerNormalB,
                                                                                         cornerValue[cornerA], cornerValue[cornerB]);
                            qef.Add(crossingPosition, crossingNormal);
                        }
                        position = qef.Solve(cellCentroidPosition);
                    }

                    primalCellDualVertexByBase[base] = PrimalCellDualVertex{position, representativeValue};
                }

                // Pass 2: the dual grid's own cubes -- one per candidate primal grid vertex,
                // enumerated by feeding the (resolved) primal CELL bases from pass 1 back into
                // core::CandidateBases exactly as field.OccupiedCoords() seeds pass 1 -- then
                // classified/triangulated exactly like core::GenerateRawTriangles classifies a
                // primal cube, just sourcing each corner's position+value from
                // primalCellDualVertexByBase instead of (base+CORNER[c])*cellSize/field.Sample.
                std::vector<std::array<int, 3>> resolvedPrimalCellCoords;
                resolvedPrimalCellCoords.reserve(primalCellDualVertexByBase.size());
                for (const auto &entry: primalCellDualVertexByBase) resolvedPrimalCellCoords.push_back(entry.first);
                const auto dualCubeBases = core::CandidateBases(resolvedPrimalCellCoords);

                std::vector<core::RawTriangle> raw;
                for (const auto &dualBase: dualCubeBases) {
                    Eigen::Vector3f dualCornerPosition[8];
                    float dualCornerValue[8];
                    bool complete = true;
                    for (int corner = 0; corner < 8 && complete; ++corner) {
                        const std::array<int, 3> primalCellCoord{dualBase[0] + mc::CORNER[corner][0],
                                                                 dualBase[1] + mc::CORNER[corner][1],
                                                                 dualBase[2] + mc::CORNER[corner][2]};
                        const auto it = primalCellDualVertexByBase.find(primalCellCoord);
                        if (it == primalCellDualVertexByBase.end()) {
                            complete = false;
                            break;
                        }
                        dualCornerPosition[corner] = it->second.position;
                        dualCornerValue[corner] = it->second.representativeValue;
                    }
                    if (!complete) continue; // a surrounding primal cell has no data: leave a gap (sparse-field boundary)

                    int dualCubeIndex = 0;
                    for (int corner = 0; corner < 8; ++corner)
                        if (dualCornerValue[corner] < 0.0f) dualCubeIndex |= (1 << corner);

                    const int edgeMask = mc::edgeTable[dualCubeIndex];
                    if (edgeMask == 0) continue; // this dual cube's own corners never cross the isosurface

                    Eigen::Vector3f edgeCrossing[12];
                    for (int edge = 0; edge < 12; ++edge)
                        if (edgeMask & (1 << edge)) {
                            const int cornerA = core::kEdgeCornerPairs[edge][0], cornerB = core::kEdgeCornerPairs[edge][1];
                            edgeCrossing[edge] = core::VertexInterpolate(dualCornerPosition[cornerA], dualCornerPosition[cornerB],
                                                                         dualCornerValue[cornerA], dualCornerValue[cornerB]);
                        }

                    const int triangleTableBase = dualCubeIndex * 16;
                    for (int i = 0; i < 15; i += 3) {
                        const int edgeIndex0 = mc::triTable[triangleTableBase + i];
                        if (edgeIndex0 == -1) break;
                        const int edgeIndex1 = mc::triTable[triangleTableBase + i + 1];
                        const int edgeIndex2 = mc::triTable[triangleTableBase + i + 2];
                        // Swap edgeIndex1/edgeIndex2 for outward-facing winding, mirroring
                        // core::GenerateRawTriangles.
                        raw.push_back({edgeCrossing[edgeIndex0], edgeCrossing[edgeIndex2], edgeCrossing[edgeIndex1]});
                    }
                }

                return core::WeldAndComputeNormals(raw, kDualMarchingCubesWeldDistanceFraction * field.CellSize());
            }
        };

    } // namespace

    std::unique_ptr<IsoSurfaceExtractor> CreateDualMarchingCubesExtractor() {
        return std::make_unique<DualMarchingCubesExtractor>();
    }

} // namespace Engine::Spatial::Extraction

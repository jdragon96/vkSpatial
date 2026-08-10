// The "emc" strategy: Extended Marching Cubes over a VoxelField -- augments plain "mc" with a
// per-cell Quadratic Error Function (QuadraticErrorFunction.h) feature vertex, so cells whose
// edge crossings disagree in normal direction (creases, corners) get one additional vertex
// placed at the least-squares intersection of their tangent planes, instead of only the
// triangle shapes plain MC's fixed case table produces there (which implicitly interpolate a
// single smooth surface and round sharp features off). Reuses the shared core
// (MarchingCubesCore.h) for candidate-cube enumeration, edge-crossing placement
// (core::VertexInterpolate) and welding (core::WeldAndComputeNormals) -- non-feature cells are
// emitted with the exact same case/edge/triangle topology and winding as "mc"
// (MarchingCubesExtractor.cpp).
//
// Per candidate cube:
//   1. Resolve the 8 corner values (iso-shifted) and, from them, the standard MC cubeIndex,
//      edge bitmask and (edgeIndex -> world position) crossings -- identical to "mc".
//   2. For every crossing edge, also compute a UNIT crossing NORMAL: the field's stored
//      gradient at each of the edge's two corners (VoxelField::Gradient), or, where a corner
//      lacks one, a central-difference estimate from neighbouring scalar samples; either way,
//      the two corner normals are interpolated with the exact same edge parameter
//      core::VertexInterpolate derives for the position (so position and normal agree on where
//      along the edge they were evaluated), then renormalized.
//   3. Classify the cube as a "feature cell" iff the minimum pairwise dot product among its
//      crossing normals is below params.featureAngleCosineThreshold -- i.e. some pair of
//      crossings disagrees by more than the configured angle, meaning the single smooth
//      surface plain MC's triangles assume would be a poor fit.
//   4. Non-feature cells: emit this cube's standard MC triangles verbatim.
//   5. Feature cells: accumulate every (crossing point, crossing normal) pair into a
//      QuadraticErrorFunction and Solve() it -- biased to the cell's centroid, so a
//      rank-deficient accumulation (an edge/crease cell constrains only 2 of 3 dimensions)
//      falls back to the centroid along its under-constrained direction -- for ONE feature
//      vertex v*. Then, instead of this cube's standard MC triangles, emit a fan from v* to the
//      BOUNDARY of the cube's own mc-triangle set: an mc-triangle edge that appears exactly
//      once among this cube's triangles lies on a cube face (shared with a neighbour cube, or
//      the field's outer boundary) and becomes one fan triangle (v*, edgeStart, edgeEnd); an
//      edge that appears twice is an interior diagonal wholly contained within this cube (never
//      shared with a neighbour) and is simply dropped, replaced by its two new connections to
//      v*. Because only interior diagonals are ever touched, every edge shared with a
//      neighbouring cube keeps exactly the same endpoints and incidence count it had under
//      plain "mc", so cross-cube edge-manifoldness is preserved by construction.
//
// Reference: L. Kobbelt, M. Botsch, U. Schwanecke, H.-P. Seidel, "Feature Sensitive Surface
// Extraction from Volume Data", SIGGRAPH 2001 (the extractor this strategy is named after); the
// QEF itself follows T. Ju, F. Losasso, S. Schaefer, J. Warren, "Dual Contouring of Hermite
// Data", SIGGRAPH 2002 (see QuadraticErrorFunction.h).

#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Spatial/Extraction/IsoSurfaceExtractor.h"
#include "Engine/Spatial/Extraction/MarchingCubesCore.h"
#include "Engine/Spatial/Extraction/QuadraticErrorFunction.h"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace Engine::Spatial::Extraction {

    namespace {

        // One cube's standard MC triangle, stored as indices (0..11) into that cube's
        // per-edge crossing-position/-normal arrays rather than world positions, so both the
        // non-feature (verbatim) and feature (boundary-fan) emission paths below can share this
        // same per-cube topology. Index order matches core::GenerateRawTriangles' outward
        // winding (triTable's [ei0,ei1,ei2] with ei1/ei2 swapped).
        using EdgeIndexTriangle = std::array<int, 3>;

        // A unit outward-pointing normal at one cube corner: the field's own stored gradient if
        // present, otherwise a central-difference estimate from the corner's face-neighbour
        // scalar samples (a missing neighbour falls back to the corner's own value, degrading
        // gracefully to a one-sided difference at the edge of a sparse field).
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
        // that edge, then renormalizes.
        Eigen::Vector3f InterpolateEdgeNormal(const Eigen::Vector3f &cornerNormalA, const Eigen::Vector3f &cornerNormalB,
                                              float cornerValueA, float cornerValueB) {
            const Eigen::Vector3f interpolated = core::VertexInterpolate(cornerNormalA, cornerNormalB, cornerValueA, cornerValueB);
            const float length = interpolated.norm();
            return length > 1e-8f ? Eigen::Vector3f(interpolated / length) : cornerNormalA;
        }

        // Unordered-edge map key, so (a,b) and (b,a) hash to the same entry.
        std::pair<int, int> UndirectedEdgeKey(int edgeIndexA, int edgeIndexB) {
            return edgeIndexA < edgeIndexB ? std::make_pair(edgeIndexA, edgeIndexB) : std::make_pair(edgeIndexB, edgeIndexA);
        }

        class ExtendedMarchingCubesExtractor : public IsoSurfaceExtractor {
        public:
            const char *Name() const override { return "emc"; }

            SurfaceMesh Extract(const VoxelField &field, const ExtractParams &params) const override {
                const auto bases = core::CandidateBases(field.OccupiedCoords());
                std::vector<core::RawTriangle> raw;

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
                    if (!complete) continue; // an unresolvable corner: skip this cell, matching "mc"

                    int cubeIndex = 0;
                    for (int corner = 0; corner < 8; ++corner)
                        if (cornerValue[corner] < 0.0f) cubeIndex |= (1 << corner);

                    const int edgeMask = mc::edgeTable[cubeIndex];
                    if (edgeMask == 0) continue;

                    Eigen::Vector3f cornerPosition[8];
                    for (int corner = 0; corner < 8; ++corner)
                        cornerPosition[corner] = Eigen::Vector3f(float(cornerCoord[corner][0]), float(cornerCoord[corner][1]),
                                                                 float(cornerCoord[corner][2])) *
                                                 field.CellSize();

                    // Per-edge crossing position + unit normal, for every edge the cube's
                    // bitmask marks as cut (see step 2 in the file header comment).
                    Eigen::Vector3f edgeCrossingPosition[12];
                    Eigen::Vector3f edgeCrossingNormal[12];
                    bool edgeIsActive[12] = {};
                    for (int edge = 0; edge < 12; ++edge) {
                        if (!(edgeMask & (1 << edge))) continue;
                        edgeIsActive[edge] = true;
                        const int cornerA = core::kEdgeCornerPairs[edge][0], cornerB = core::kEdgeCornerPairs[edge][1];
                        edgeCrossingPosition[edge] = core::VertexInterpolate(cornerPosition[cornerA], cornerPosition[cornerB],
                                                                             cornerValue[cornerA], cornerValue[cornerB]);
                        const Eigen::Vector3f cornerNormalA = EstimateCornerNormal(field, cornerCoord[cornerA]);
                        const Eigen::Vector3f cornerNormalB = EstimateCornerNormal(field, cornerCoord[cornerB]);
                        edgeCrossingNormal[edge] =
                                InterpolateEdgeNormal(cornerNormalA, cornerNormalB, cornerValue[cornerA], cornerValue[cornerB]);
                    }

                    // Feature classification (step 3): the minimum pairwise dot product among
                    // every pair of active crossing normals.
                    float minimumPairwiseNormalDot = 1.0f;
                    for (int edgeA = 0; edgeA < 12; ++edgeA) {
                        if (!edgeIsActive[edgeA]) continue;
                        for (int edgeB = edgeA + 1; edgeB < 12; ++edgeB) {
                            if (!edgeIsActive[edgeB]) continue;
                            minimumPairwiseNormalDot = std::min(minimumPairwiseNormalDot, edgeCrossingNormal[edgeA].dot(edgeCrossingNormal[edgeB]));
                        }
                    }
                    const bool isFeatureCell = minimumPairwiseNormalDot < params.featureAngleCosineThreshold;

                    // This cube's standard MC triangle list (edge-index form), identical
                    // topology/winding to core::GenerateRawTriangles / "mc".
                    std::vector<EdgeIndexTriangle> cubeTriangles;
                    cubeTriangles.reserve(4);
                    const int triangleTableBase = cubeIndex * 16;
                    for (int i = 0; i < 15; i += 3) {
                        const int edgeIndex0 = mc::triTable[triangleTableBase + i];
                        if (edgeIndex0 == -1) break;
                        const int edgeIndex1 = mc::triTable[triangleTableBase + i + 1];
                        const int edgeIndex2 = mc::triTable[triangleTableBase + i + 2];
                        const EdgeIndexTriangle triangle{edgeIndex0, edgeIndex2, edgeIndex1}; // outward winding
                        cubeTriangles.push_back(triangle);
                    }

                    if (!isFeatureCell) {
                        // Step 4: verbatim "mc" emission.
                        for (const auto &triangle: cubeTriangles)
                            raw.push_back({edgeCrossingPosition[triangle[0]], edgeCrossingPosition[triangle[1]],
                                           edgeCrossingPosition[triangle[2]]});
                        continue;
                    }

                    // Step 5: one QEF feature vertex, fanned to this cube's boundary loop.
                    QuadraticErrorFunction qef;
                    for (int edge = 0; edge < 12; ++edge)
                        if (edgeIsActive[edge]) qef.Add(edgeCrossingPosition[edge], edgeCrossingNormal[edge]);

                    const Eigen::Vector3f cellCentroid =
                            (Eigen::Vector3f(float(base[0]), float(base[1]), float(base[2])) + Eigen::Vector3f(0.5f, 0.5f, 0.5f)) *
                            field.CellSize();
                    const Eigen::Vector3f featureVertex = qef.Solve(cellCentroid);

                    // Boundary-edge incidence over this cube's own triangle set: count==1 means
                    // the edge lies on a cube face (fan it to featureVertex); count==2 means an
                    // interior diagonal wholly inside this cube (drop it -- see file header).
                    std::map<std::pair<int, int>, int> edgeIncidence;
                    for (const auto &triangle: cubeTriangles) {
                        edgeIncidence[UndirectedEdgeKey(triangle[0], triangle[1])]++;
                        edgeIncidence[UndirectedEdgeKey(triangle[1], triangle[2])]++;
                        edgeIncidence[UndirectedEdgeKey(triangle[2], triangle[0])]++;
                    }
                    for (const auto &triangle: cubeTriangles) {
                        for (int side = 0; side < 3; ++side) {
                            const int edgeStart = triangle[side], edgeEnd = triangle[(side + 1) % 3];
                            if (edgeIncidence[UndirectedEdgeKey(edgeStart, edgeEnd)] != 1) continue; // interior: skip
                            raw.push_back({featureVertex, edgeCrossingPosition[edgeStart], edgeCrossingPosition[edgeEnd]});
                        }
                    }
                }

                return core::WeldAndComputeNormals(raw, params.weldFraction * field.CellSize());
            }
        };

    } // namespace

    std::unique_ptr<IsoSurfaceExtractor> CreateExtendedMarchingCubesExtractor() {
        return std::make_unique<ExtendedMarchingCubesExtractor>();
    }

} // namespace Engine::Spatial::Extraction

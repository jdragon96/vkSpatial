// The "cms" strategy: Cubical Marching Squares over a VoxelField (Ho, Wu, Chen, Chuang,
// Ouhyoung, "Cubical Marching Squares: Adaptive Feature Preserving Surface Extraction from
// Volume Data", Eurographics 2005) -- reduces the 3-D cube-contouring problem to six
// independent 2-D marching-squares problems, one per cube face, then stitches the six
// faces' segments into closed loops around the cube and fans each loop to either a QEF
// feature vertex (crease/corner cells, exactly like "emc"'s feature classification) or its
// own centroid (smooth cells). Reuses the shared core (MarchingCubesCore.h) for candidate-
// cube enumeration, edge-crossing placement (core::VertexInterpolate) and the final weld/
// normal pass (core::WeldAndComputeNormals), and the shared per-cell solver
// (QuadraticErrorFunction.h, already used by "emc"/"dc"/"dmc") for the one feature vertex a
// feature cell needs.
//
// Per candidate cube:
//   1. Resolve the 8 corner values (iso-shifted) exactly like "mc"/"emc"/"dc"; skip the cube
//      if any corner is unresolvable. Compute cubeIndex/edgeMask exactly like "mc"
//      (mc::edgeTable's own "which of the 12 edges cross" bitmask -- a face-local edge is
//      active in the 2-D sense below iff its two corners' SIGNS differ, the exact same
//      condition mc::edgeTable already encodes for the whole cube), and every active edge's
//      crossing position (core::VertexInterpolate) and crossing normal
//      (EstimateCornerNormal/InterpolateEdgeNormal, duplicated from "emc"/"dc" per those
//      files' own established precedent for these two helpers).
//   2. FACE STAGE (the "cubical" half of "cubical marching squares"): for each of the
//      cube's 6 axis-aligned faces (kCubeFaces below), run 2-D marching squares
//      (FaceMarchingSquares) on that face's 4 corner values, producing 0, 1 or 2 SEGMENTS,
//      each a pair of the face's 4 edges. The face's 4 crossing edges either pair
//      unambiguously (0 or 2 active edges: at most one valid pairing) or, when all 4 edges
//      are active (the classic marching-squares "checkerboard" ambiguity -- diagonally
//      opposite corners share a sign, adjacent corners don't), the pairing is resolved with
//      the 2-D asymptotic decider (Nielson & Hamann, "The Asymptotic Decider", IEEE
//      Visualization 1991 -- the same bilinear-saddle test "mc33"'s own FaceTest uses,
//      MarchingCubes33Extractor.cpp, applied here directly to a 2x2 corner square instead of
//      indexing a case table with it). CRACK-FREENESS: this decision depends ONLY on the
//      face's own 4 corner values, not on which cube is asking -- a cube's "+axis" face and
//      its +axis neighbour's corresponding "-axis" face are, by construction of the standard
//      mc::CORNER numbering (hand-verified for all 3 axes when kCubeFaces below was derived:
//      both cubes visit the SAME 4 grid vertices in the SAME cyclic order), literally
//      identical, so both independently compute the identical segment(s) on that shared
//      face. Every one of a cube's 12 edges belongs to exactly 2 of its 6 faces (kCubeFaces
//      below covers each edge index exactly twice), so every active edge ends up with
//      exactly 2 face-segment connections cube-locally.
//   3. STITCH STAGE: because step 2 gives every active edge exactly 2 connections, and two
//      distinct faces of a cube never share more than the one edge that makes them adjacent
//      (so an edge's two connections can never coincide on the same neighbour -- two faces
//      of a cube meet in at most one edge), the per-cube connection graph over active edges
//      is necessarily a disjoint union of simple cycles ("loops": TraceLoops). A simple
//      crossing cube produces exactly one loop (e.g. a 3-edge loop cutting off one corner,
//      mirroring plain "mc"'s single-triangle case); a saddle-like cube can produce two
//      disjoint loops (two separate sheets through the same cube) -- CMS handles this
//      natively, unlike "mc"'s 15-case table, which has to pre-commit to one fixed
//      triangulation per case.
//   4. FEATURE STAGE: classify the whole CUBE (not per loop) as a feature cell iff the
//      minimum pairwise dot product among ALL its active edges' crossing normals is below
//      params.featureAngleCosineThreshold -- identical test to "emc"'s per-cube
//      classification (ExtendedMarchingCubesExtractor.cpp). A feature cell accumulates every
//      active edge's (crossing position, crossing normal) into ONE QuadraticErrorFunction,
//      Solve()d biased to the cell centroid -- the SAME feature vertex is then used to fan
//      EVERY loop this cube produced (matching the brief's "like emc" wording -- emc's own
//      feature classification and QEF are cube-wide, not per-triangle-fan).
//   5. FAN STAGE: for each loop, fan-triangulate it to the cube's feature vertex (feature
//      cell) or to the loop's own centroid (its vertices' average position -- NOT the whole
//      cube's centroid, so two disjoint non-feature loops in the same saddle cube each get
//      their own, geometrically appropriate, apex). Winding is resolved ONCE per loop (never
//      per triangle, so a single fan can never end up self-intersecting/inconsistently
//      wound): sum the raw (unnormalized) cross products of consecutive loop edges around
//      the apex to get the fan's own signed-area vector, and flip the whole loop's winding
//      together iff that disagrees with the loop's reference outward direction (the sum of
//      its active edges' crossing normals) -- the same "derive orientation dynamically
//      instead of hand-verifying a table" idiom "dc"'s EmitOutwardTriangle and "mtet"'s
//      identically-named helper use, applied once per loop instead of once per triangle.
//
// Reference: Y.-J. Ho, Y.-J. Wu, W.-Y. Chen, J.-H. Chuang, M. Ouhyoung, "Cubical Marching
// Squares: Adaptive Feature Preserving Surface Extraction from Volume Data", Eurographics
// 2005 (Computer Graphics Forum 24(3)). The face ambiguity test follows G. M. Nielson, H.
// Hamann, "The Asymptotic Decider: Resolving the Ambiguity in Marching Cubes", IEEE
// Visualization 1991. The feature-vertex QEF follows T. Ju, F. Losasso, S. Schaefer, J.
// Warren, "Dual Contouring of Hermite Data", SIGGRAPH 2002 (see QuadraticErrorFunction.h),
// exactly as "emc"/"dc"/"dmc" already use it.

#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Spatial/Extraction/IsoSurfaceExtractor.h"
#include "Engine/Spatial/Extraction/MarchingCubesCore.h"
#include "Engine/Spatial/Extraction/QuadraticErrorFunction.h"

#include <Eigen/Geometry> // Vector3f::cross
#include <algorithm>
#include <array>
#include <memory>
#include <vector>

namespace Engine::Spatial::Extraction {

    namespace {

        // A unit outward-pointing normal at one cube corner: the field's own stored gradient
        // if present, otherwise a central-difference estimate from the corner's face-
        // neighbour scalar samples (a missing neighbour falls back to the corner's own
        // value, degrading gracefully to a one-sided difference at the edge of a sparse
        // field). Duplicated verbatim from ExtendedMarchingCubesExtractor.cpp /
        // DualContouringExtractor.cpp / DualMarchingCubesExtractor.cpp -- see those files'
        // headers for why this is a per-file copy rather than a shared function.
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
        // core::VertexInterpolate derives from (cornerValueA, cornerValueB) for the position
        // on that edge, then renormalizes. Duplicated verbatim from the same three files as
        // EstimateCornerNormal above.
        Eigen::Vector3f InterpolateEdgeNormal(const Eigen::Vector3f &cornerNormalA, const Eigen::Vector3f &cornerNormalB,
                                              float cornerValueA, float cornerValueB) {
            const Eigen::Vector3f interpolated = core::VertexInterpolate(cornerNormalA, cornerNormalB, cornerValueA, cornerValueB);
            const float length = interpolated.norm();
            return length > 1e-8f ? Eigen::Vector3f(interpolated / length) : cornerNormalA;
        }

        // One axis-aligned cube face: its 4 corners (mc::CORNER indices, in cyclic order)
        // and the 4 cube-local edges (core::kEdgeCornerPairs indices) connecting consecutive
        // corners in that same cyclic order -- corner[i] to corner[(i+1)%4] is edge[i].
        // Hand-derived from mc::CORNER / core::kEdgeCornerPairs (see file header step 2 for
        // the crack-freeness argument this depends on) and cross-checked to cover each of
        // the 12 cube edges exactly twice across all 6 faces.
        struct CubeFace {
            int corner[4];
            int edge[4];
        };
        constexpr CubeFace kCubeFaces[6] = {
                {{0, 1, 2, 3}, {0, 1, 2, 3}},     // z = base   (corner/edge indices already agree)
                {{4, 5, 6, 7}, {4, 5, 6, 7}},     // z = base+1
                {{0, 1, 5, 4}, {0, 9, 4, 8}},     // y = base
                {{3, 2, 6, 7}, {2, 10, 6, 11}},   // y = base+1
                {{0, 3, 7, 4}, {3, 11, 7, 8}},    // x = base
                {{1, 2, 6, 5}, {1, 10, 5, 9}},    // x = base+1
        };

        // Runs 2-D marching squares on one face given its 4 corner values in the face's own
        // canonical cyclic order (cornerValue[i] and cornerValue[(i+1)%4] are joined by
        // face-local edge i). Appends 0, 1 or 2 segments (as face-local edge index pairs) to
        // `outSegments`. See file header step 2 for the crack-freeness argument.
        void FaceMarchingSquares(const float cornerValue[4], std::vector<std::array<int, 2>> &outSegments) {
            bool edgeIsActive[4];
            int activeEdgeCount = 0;
            for (int edge = 0; edge < 4; ++edge) {
                const int cornerA = edge, cornerB = (edge + 1) % 4;
                edgeIsActive[edge] = (cornerValue[cornerA] < 0.0f) != (cornerValue[cornerB] < 0.0f);
                if (edgeIsActive[edge]) ++activeEdgeCount;
            }
            if (activeEdgeCount == 0) return; // this face never crosses the isosurface

            if (activeEdgeCount == 2) {
                // Exactly one valid pairing (either "one corner differs from the other
                // three" or "two adjacent corners differ from the other two") -- no
                // ambiguity possible with only 2 active edges.
                std::array<int, 2> segment{};
                int segmentSlotIndex = 0;
                for (int edge = 0; edge < 4; ++edge)
                    if (edgeIsActive[edge]) segment[segmentSlotIndex++] = edge;
                outSegments.push_back(segment);
                return;
            }

            // activeEdgeCount == 4: the checkerboard ambiguity (corners 0,2 share a sign
            // bucket, corners 1,3 share the other) -- resolved with the 2-D asymptotic
            // decider. bilinearTwist is provably nonzero here: with corners 0,2 strictly in
            // one sign bucket and 1,3 strictly in the other, (corner0Value+corner2Value) and
            // (corner1Value+corner3Value) fall on opposite sides of zero, so their difference
            // can never be exactly zero.
            const float corner0Value = cornerValue[0], corner1Value = cornerValue[1], corner2Value = cornerValue[2],
                        corner3Value = cornerValue[3];
            const float bilinearTwist = corner0Value - corner1Value + corner2Value - corner3Value;
            const float saddleValue = (corner0Value * corner2Value - corner1Value * corner3Value) / bilinearTwist;
            const bool diagonalZeroTwoConnected = (saddleValue < 0.0f) == (corner0Value < 0.0f);
            if (diagonalZeroTwoConnected) {
                outSegments.push_back({0, 1}); // corner 1 isolated
                outSegments.push_back({2, 3}); // corner 3 isolated
            } else {
                outSegments.push_back({3, 0}); // corner 0 isolated
                outSegments.push_back({1, 2}); // corner 2 isolated
            }
        }

        // Traces the disjoint simple cycles ("loops") of a 2-regular graph over cube-local
        // edge indices 0..11: `neighborsOf[e]` holds exactly the (<=2) cube-local edges
        // connected to e by a face segment. See file header step 3 for why every active edge
        // ends up with exactly 2 neighbours in a well-formed cube.
        std::vector<std::vector<int>> TraceLoops(const std::array<std::vector<int>, 12> &neighborsOf,
                                                  const bool edgeIsActive[12]) {
            std::vector<std::vector<int>> loops;
            bool visited[12] = {};
            for (int startEdge = 0; startEdge < 12; ++startEdge) {
                if (!edgeIsActive[startEdge] || visited[startEdge]) continue;
                std::vector<int> loop;
                int previousEdge = -1, currentEdge = startEdge;
                while (true) {
                    visited[currentEdge] = true;
                    loop.push_back(currentEdge);
                    const auto &neighbors = neighborsOf[currentEdge];
                    if (neighbors.size() != 2) break; // defensive: malformed face data, abandon this loop
                    const int nextEdge = (neighbors[0] != previousEdge) ? neighbors[0] : neighbors[1];
                    previousEdge = currentEdge;
                    currentEdge = nextEdge;
                    if (currentEdge == startEdge) break;
                }
                if (loop.size() >= 3) loops.push_back(std::move(loop)); // <3: degenerate, drop defensively
            }
            return loops;
        }

        class CubicalMarchingSquaresExtractor : public IsoSurfaceExtractor {
        public:
            const char *Name() const override { return "cms"; }

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
                    if (!complete) continue; // an unresolvable corner: skip this cell, matching "mc"/"emc"/"dc"

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

                    // Step 1 (file header): per-edge crossing position + unit normal, for
                    // every edge the cube's bitmask marks as cut.
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

                    // Step 2 (file header): 6 independent 2-D marching squares, each face's
                    // segments mapped from face-local (0..3) to cube-local (0..11) edge
                    // indices via kCubeFaces, and accumulated into a per-edge adjacency list.
                    std::array<std::vector<int>, 12> neighborsOf;
                    for (const auto &face: kCubeFaces) {
                        const float faceCornerValue[4] = {cornerValue[face.corner[0]], cornerValue[face.corner[1]],
                                                          cornerValue[face.corner[2]], cornerValue[face.corner[3]]};
                        std::vector<std::array<int, 2>> faceSegments;
                        FaceMarchingSquares(faceCornerValue, faceSegments);
                        for (const auto &segment: faceSegments) {
                            const int cubeEdgeA = face.edge[segment[0]], cubeEdgeB = face.edge[segment[1]];
                            neighborsOf[cubeEdgeA].push_back(cubeEdgeB);
                            neighborsOf[cubeEdgeB].push_back(cubeEdgeA);
                        }
                    }

                    // Step 3 (file header): stitch the per-face segments into closed loops.
                    const auto loops = TraceLoops(neighborsOf, edgeIsActive);
                    if (loops.empty()) continue;

                    // Step 4 (file header): one cube-wide feature classification + QEF,
                    // mirroring "emc"'s per-cube test exactly.
                    float minimumPairwiseNormalDot = 1.0f;
                    for (int edgeA = 0; edgeA < 12; ++edgeA) {
                        if (!edgeIsActive[edgeA]) continue;
                        for (int edgeB = edgeA + 1; edgeB < 12; ++edgeB) {
                            if (!edgeIsActive[edgeB]) continue;
                            minimumPairwiseNormalDot =
                                    std::min(minimumPairwiseNormalDot, edgeCrossingNormal[edgeA].dot(edgeCrossingNormal[edgeB]));
                        }
                    }
                    const bool isFeatureCell = minimumPairwiseNormalDot < params.featureAngleCosineThreshold;

                    const Eigen::Vector3f cellCentroid =
                            (Eigen::Vector3f(float(base[0]), float(base[1]), float(base[2])) + Eigen::Vector3f(0.5f, 0.5f, 0.5f)) *
                            field.CellSize();
                    Eigen::Vector3f featureVertex = cellCentroid;
                    if (isFeatureCell) {
                        QuadraticErrorFunction qef;
                        for (int edge = 0; edge < 12; ++edge)
                            if (edgeIsActive[edge]) qef.Add(edgeCrossingPosition[edge], edgeCrossingNormal[edge]);
                        featureVertex = qef.Solve(cellCentroid);
                    }

                    // Step 5 (file header): one fan per loop, apex = the shared feature
                    // vertex (feature cell) or this loop's own centroid (otherwise), winding
                    // resolved once per loop against that loop's own reference outward
                    // direction.
                    for (const auto &loop: loops) {
                        const int loopSize = int(loop.size());
                        Eigen::Vector3f loopCentroid = Eigen::Vector3f::Zero();
                        for (int edge: loop) loopCentroid += edgeCrossingPosition[edge];
                        loopCentroid /= float(loopSize);

                        const Eigen::Vector3f apex = isFeatureCell ? featureVertex : loopCentroid;

                        Eigen::Vector3f referenceOutward = Eigen::Vector3f::Zero();
                        for (int edge: loop) referenceOutward += edgeCrossingNormal[edge];

                        Eigen::Vector3f rawFanNormal = Eigen::Vector3f::Zero();
                        for (int i = 0; i < loopSize; ++i) {
                            const Eigen::Vector3f &loopVertexA = edgeCrossingPosition[loop[i]];
                            const Eigen::Vector3f &loopVertexB = edgeCrossingPosition[loop[(i + 1) % loopSize]];
                            rawFanNormal += (loopVertexA - apex).cross(loopVertexB - apex);
                        }
                        const bool flipWinding =
                                referenceOutward.squaredNorm() > 1e-12f && rawFanNormal.dot(referenceOutward) < 0.0f;

                        for (int i = 0; i < loopSize; ++i) {
                            const Eigen::Vector3f &loopVertexA = edgeCrossingPosition[loop[i]];
                            const Eigen::Vector3f &loopVertexB = edgeCrossingPosition[loop[(i + 1) % loopSize]];
                            if (!flipWinding) raw.push_back({apex, loopVertexA, loopVertexB});
                            else raw.push_back({apex, loopVertexB, loopVertexA});
                        }
                    }
                }

                return core::WeldAndComputeNormals(raw, params.weldFraction * field.CellSize());
            }
        };

    } // namespace

    std::unique_ptr<IsoSurfaceExtractor> CreateCubicalMarchingSquaresExtractor() {
        return std::make_unique<CubicalMarchingSquaresExtractor>();
    }

} // namespace Engine::Spatial::Extraction

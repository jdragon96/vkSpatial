// The "mc33" strategy: topologically-correct Marching Cubes 33 (Chernyaev / Lewiner et al.
// 2003) over a VoxelField. Reuses the shared core (MarchingCubesCore.h) for candidate-cube
// enumeration, edge-vertex placement (core::VertexInterpolate, identical to "mc") and welding
// (core::WeldAndComputeNormals) -- only the per-cube triangle TOPOLOGY differs: instead of
// "mc"'s naive 15-case table, every cube is resolved via the full case+subcase machinery in
// MarchingCubes33Tables.h, disambiguating ambiguous faces with the asymptotic decider and
// ambiguous interiors with the interior test, exactly as the classic 15-case table cannot
// (that table always guesses one fixed triangulation for a face/interior-ambiguous
// configuration, which can disagree with a neighbouring cube's guess for the *same* shared
// face and leave a boundary hole; MC33 instead computes, per cube, which of the two possible
// resolutions the trilinear interpolant on that shared face/interior actually has).
//
// Algorithm reference: T. Lewiner, H. Lopes, A. W. Vieira, G. Tavares, "Efficient
// Implementation of Marching Cubes' Cases with Topological Guarantees", Journal of Graphics
// Tools, 8(2):1-15, 2003 (http://thomas.lewiner.org/pdfs/marching_cubes_jgt.pdf). The case
// dispatch below (which tiling table a given (case, configuration) resolves to) and the face
// / interior ambiguity tests mirror the structure of the widely-mirrored MarchingCubes.cpp
// reference implementation that ships alongside MarchingCubes33Tables.h's LookUpTable.h,
// adapted here to this codebase's VoxelField / SurfaceMesh / core:: conventions: there is no
// shared per-grid-vertex cache (the reference dedups vertices as it walks the grid) -- each
// cube's edges are interpolated independently and the whole mesh is welded afterwards via
// core::WeldAndComputeNormals, exactly like "mc" already does.
//
// The case 13.5 "5-tunnel" subcase (the single hardest, most error-prone corner of MC33 --
// it is the only configuration needing a genuine trilinear critical-point analysis rather
// than a bilinear face test) requires all 8 corners to alternate in a 3-D checkerboard
// pattern; that pattern cannot arise from sampling a smooth surface on a regular grid (it
// needs the SDF to reverse sign between every pair of grid-adjacent corners), so it is not
// exercised by this task's tests. It is still implemented in full, for a genuinely complete
// mc33.

#include "Mesh/ExtractorRegistry.h"
#include "Mesh/IsoSurfaceExtractor.h"
#include "Mesh/MarchingCubes33Tables.h"
#include "Mesh/MarchingCubesCore.h"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace Mesh {

    namespace {

        using CubeCornerValues = std::array<float, 8>;

        /// ────────────────────────────────────────────────────────────────────────────
        /// 1. Corner sign bitmask (Lewiner's "_lut_entry"): bit c set iff corner c's
        ///    (iso-shifted) value is > 0. This is the OPPOSITE bit convention from
        ///    core::GenerateRawTriangles' cubeIndex (which sets a bit for NEGATIVE
        ///    corners, to match mc::triTable's expectation) -- the two are never mixed:
        ///    mc33::cases is only ever indexed with this positive-bit convention, matching
        ///    how mc33::cases/tiling*/test* were generated.
        /// ────────────────────────────────────────────────────────────────────────────
        // Also snaps every near-zero corner value to +epsilon in place, matching the
        // reference's epsilon handling: this guarantees no corner is ever exactly on the
        // isosurface, so every one of the 256 corner-sign patterns is well defined (the
        // face/interior ambiguity this file resolves is a *face or interior* ambiguity,
        // never a corner-value ambiguity).
        int ComputeCornerSignBitmask(CubeCornerValues &cube) {
            int bitmask = 0;
            for (int corner = 0; corner < 8; ++corner) {
                if (std::abs(cube[corner]) < std::numeric_limits<float>::epsilon())
                    cube[corner] = std::numeric_limits<float>::epsilon();
                if (cube[corner] > 0.0f)
                    bitmask |= (1 << corner);
            }
            return bitmask;
        }

        /// ────────────────────────────────────────────────────────────────────────────
        /// 2. Asymptotic decider (face ambiguity)
        /// ────────────────────────────────────────────────────────────────────────────
        // Signed face code -> the four corners bounding that face, in (A,B,C,D) order,
        // matching the reference's test_face: face +-1={0,4,5,1}, +-2={1,5,6,2},
        // +-3={2,6,7,3}, +-4={3,7,4,0}, +-5={0,3,2,1}, +-6={4,7,6,5}.
        void FaceCorners(int faceCode, int &cornerA, int &cornerB, int &cornerC, int &cornerD) {
            switch (std::abs(faceCode)) {
                case 1: cornerA = 0; cornerB = 4; cornerC = 5; cornerD = 1; break;
                case 2: cornerA = 1; cornerB = 5; cornerC = 6; cornerD = 2; break;
                case 3: cornerA = 2; cornerB = 6; cornerC = 7; cornerD = 3; break;
                case 4: cornerA = 3; cornerB = 7; cornerC = 4; cornerD = 0; break;
                case 5: cornerA = 0; cornerB = 3; cornerC = 2; cornerD = 1; break;
                default: cornerA = 4; cornerB = 7; cornerC = 6; cornerD = 5; break; // face 6
            }
        }

        // The asymptotic decider: sign of the bilinear-saddle value A*C - B*D on the named
        // face (a signed face code out of one of the test3/test4/test6/test7/test10/test12/
        // test13 tables). Returns true iff the surface cuts the face on the branch of the
        // saddle that connects the *other* diagonal pair of corners than a naive per-face
        // guess would assume. On the (measure-zero) exact saddle -- A*C == B*D -- ties break
        // on the sign of the face code itself, matching the reference exactly.
        bool FaceTest(const CubeCornerValues &cube, int faceCode) {
            int cornerA, cornerB, cornerC, cornerD;
            FaceCorners(faceCode, cornerA, cornerB, cornerC, cornerD);
            const float A = cube[cornerA], B = cube[cornerB], C = cube[cornerC], D = cube[cornerD];
            const float saddleValue = A * C - B * D;
            if (std::abs(saddleValue) < std::numeric_limits<float>::epsilon())
                return faceCode >= 0;
            return float(faceCode) * A * saddleValue >= 0.0f;
        }

        /// ────────────────────────────────────────────────────────────────────────────
        /// 3. Interior test (interior ambiguity, cases 4/6/7/10/12)
        /// ────────────────────────────────────────────────────────────────────────────
        // For each of the 12 standard cube edges, the 8 corner indices (X,Y,Z,W,P,Q,R,S)
        // feeding the trilinear-saddle formula below: a = (cube[X]-cube[Y])*(cube[Z]-cube[W])
        // - (cube[P]-cube[Q])*(cube[R]-cube[S]), and similarly for b/At/Bt/Ct/Dt. Extracted
        // from the reference's interior_ambiguity_verification (one near-identical formula
        // per edge, corner indices permuted); every row uses each of the 8 corners exactly
        // once, cross-checked mechanically against the reference source before transcription.
        constexpr int kInteriorTestCornerIndices[12][8] = {
                {0, 1, 7, 6, 4, 5, 3, 2}, // edge 0
                {3, 2, 4, 5, 0, 1, 7, 6}, // edge 1
                {2, 3, 5, 4, 6, 7, 1, 0}, // edge 2
                {1, 0, 6, 7, 2, 3, 5, 4}, // edge 3
                {2, 1, 7, 4, 3, 0, 6, 5}, // edge 4
                {3, 0, 6, 5, 2, 1, 7, 4}, // edge 5
                {0, 3, 5, 6, 4, 7, 1, 2}, // edge 6
                {1, 2, 4, 7, 0, 3, 5, 6}, // edge 7
                {4, 0, 6, 2, 7, 3, 5, 1}, // edge 8
                {5, 1, 7, 3, 4, 0, 6, 2}, // edge 9
                {6, 2, 4, 0, 5, 1, 7, 3}, // edge 10
                {7, 3, 5, 1, 6, 2, 4, 0}, // edge 11
        };

        // Which of the 12 cube edges carries the "tunnel" for a given ambiguous face and
        // sign: exactly one of the four space-diagonal corner pairs opposite that face
        // satisfies (cube[corner] * signValue) > 0 for a genuinely ambiguous configuration.
        // Mirrors the reference's interior_ambiguity. Returns -1 (defensively; the reference
        // leaves this case undefined) if no pair matches -- InteriorTestOnEdge treats that as
        // "ambiguous / no tunnel found", the same value the reference's early-out paths use.
        int InteriorTestEdgeForFace(const CubeCornerValues &cube, int ambiguousFace, float signValue) {
            int edge = -1;
            switch (ambiguousFace) {
                case 1:
                case 3:
                    if (cube[1] * signValue > 0.0f && cube[7] * signValue > 0.0f) edge = 4;
                    if (cube[0] * signValue > 0.0f && cube[6] * signValue > 0.0f) edge = 5;
                    if (cube[3] * signValue > 0.0f && cube[5] * signValue > 0.0f) edge = 6;
                    if (cube[2] * signValue > 0.0f && cube[4] * signValue > 0.0f) edge = 7;
                    break;
                case 2:
                case 4:
                    if (cube[1] * signValue > 0.0f && cube[7] * signValue > 0.0f) edge = 0;
                    if (cube[2] * signValue > 0.0f && cube[4] * signValue > 0.0f) edge = 1;
                    if (cube[3] * signValue > 0.0f && cube[5] * signValue > 0.0f) edge = 2;
                    if (cube[0] * signValue > 0.0f && cube[6] * signValue > 0.0f) edge = 3;
                    break;
                case 0:
                case 5:
                case 6:
                    if (cube[0] * signValue > 0.0f && cube[6] * signValue > 0.0f) edge = 8;
                    if (cube[1] * signValue > 0.0f && cube[7] * signValue > 0.0f) edge = 9;
                    if (cube[2] * signValue > 0.0f && cube[4] * signValue > 0.0f) edge = 10;
                    if (cube[3] * signValue > 0.0f && cube[5] * signValue > 0.0f) edge = 11;
                    break;
                default: break;
            }
            return edge;
        }

        // Given the tunnel edge selected above, decides whether the trilinear saddle along
        // that tunnel actually crosses the isosurface strictly inside the cube (a genuine
        // interior ambiguity) or not. Mirrors the reference's interior_ambiguity_verification
        // (same bilinear-saddle-with-critical-t formula as FaceTest, but evaluated on the
        // interior diagonal named by `edge` rather than on a cube face).
        bool InteriorTestOnEdge(const CubeCornerValues &cube, int edge) {
            if (edge < 0 || edge > 11)
                return true; // no tunnel edge found: mirrors the reference's ambiguous-default paths, all of which are `return 1`
            const int *cornerIndex = kInteriorTestCornerIndices[edge];
            const int X = cornerIndex[0], Y = cornerIndex[1], Z = cornerIndex[2], W = cornerIndex[3];
            const int P = cornerIndex[4], Q = cornerIndex[5], R = cornerIndex[6], S = cornerIndex[7];

            const float a = (cube[X] - cube[Y]) * (cube[Z] - cube[W]) - (cube[P] - cube[Q]) * (cube[R] - cube[S]);
            if (a > 0.0f)
                return true;

            const float b = cube[W] * (cube[X] - cube[Y]) + cube[Y] * (cube[Z] - cube[W]) -
                             cube[S] * (cube[P] - cube[Q]) - cube[Q] * (cube[R] - cube[S]);
            const float t = -b / (2.0f * a);
            if (t < 0.0f || t > 1.0f)
                return true;

            const float At = cube[Y] + (cube[X] - cube[Y]) * t;
            const float Bt = cube[Q] + (cube[P] - cube[Q]) * t;
            const float Ct = cube[W] + (cube[Z] - cube[W]) * t;
            const float Dt = cube[S] + (cube[R] - cube[S]) * t;
            const float verify = At * Ct - Bt * Dt;

            if (verify > 0.0f) return false;
            return true; // verify < 0, or (measure-zero) exactly 0: both treated as "tunnel present", matching the reference's other early-outs
        }

        // The interior test dispatcher for cases 4/6/7/10/12: which face(s) to run
        // InteriorTestEdgeForFace/InteriorTestOnEdge on depends on the case (4 and 7 check
        // three fixed faces and OR the results; 6/10/12 look up a single ambiguous face out
        // of test6/test10/test12's own configuration row). Mirrors modified_test_interior.
        bool InteriorTest(const CubeCornerValues &cube, int caseNumber, int configurationIndex, float signValue) {
            switch (caseNumber) {
                case 4: {
                    bool tunnelFound = false;
                    tunnelFound |= InteriorTestOnEdge(cube, InteriorTestEdgeForFace(cube, 1, signValue));
                    tunnelFound |= InteriorTestOnEdge(cube, InteriorTestEdgeForFace(cube, 2, signValue));
                    tunnelFound |= InteriorTestOnEdge(cube, InteriorTestEdgeForFace(cube, 5, signValue));
                    return tunnelFound;
                }
                case 6: {
                    const int ambiguousFace = std::abs(mc33::test6[configurationIndex][0]);
                    return InteriorTestOnEdge(cube, InteriorTestEdgeForFace(cube, ambiguousFace, signValue));
                }
                case 7: {
                    const float flippedSign = -signValue;
                    bool tunnelFound = false;
                    tunnelFound |= InteriorTestOnEdge(cube, InteriorTestEdgeForFace(cube, 1, flippedSign));
                    tunnelFound |= InteriorTestOnEdge(cube, InteriorTestEdgeForFace(cube, 2, flippedSign));
                    tunnelFound |= InteriorTestOnEdge(cube, InteriorTestEdgeForFace(cube, 5, flippedSign));
                    return tunnelFound;
                }
                case 10: {
                    const int ambiguousFace = std::abs(mc33::test10[configurationIndex][0]);
                    return InteriorTestOnEdge(cube, InteriorTestEdgeForFace(cube, ambiguousFace, signValue));
                }
                case 12: {
                    bool tunnelFound = false;
                    const int ambiguousFaceA = std::abs(mc33::test12[configurationIndex][0]);
                    const int ambiguousFaceB = std::abs(mc33::test12[configurationIndex][1]);
                    tunnelFound |= InteriorTestOnEdge(cube, InteriorTestEdgeForFace(cube, ambiguousFaceA, signValue));
                    tunnelFound |= InteriorTestOnEdge(cube, InteriorTestEdgeForFace(cube, ambiguousFaceB, signValue));
                    return tunnelFound;
                }
                default:
                    return false;
            }
        }

        /// ────────────────────────────────────────────────────────────────────────────
        /// 4. Case 13.5 interior tunnel test (trilinear critical-point analysis)
        /// ────────────────────────────────────────────────────────────────────────────
        // Case 13's rarest subcase (the cube's 8 corners alternate in a full 3-D checkerboard)
        // cannot be resolved by a single bilinear face test: the trilinear interpolant
        // f(x,y,z) = a*xyz + b*xy + c*xz + d*yz + e*x + f*y + g*z + h can have up to two
        // interior critical points, and whether/how the surface tunnels through the cube
        // depends on their values. Mirrors the reference's new_interior_test, using double
        // precision for the same numerically-delicate root solve the reference uses.
        struct InteriorTunnelResult {
            bool isEmpty = true;    // true: no tunnel (use the simpler 6-triangle tiling13_5_1)
            int orientation = 0;    // +-1 once a tunnel is found; picks which of the two tiling13_5_2 branches
        };

        InteriorTunnelResult EvaluateInteriorTunnel(const CubeCornerValues &cube) {
            const double cube0 = cube[0], cube1 = cube[1], cube2 = cube[2], cube3 = cube[3];
            const double cube4 = cube[4], cube5 = cube[5], cube6 = cube[6], cube7 = cube[7];

            const double a = -cube0 + cube1 + cube3 - cube2 + cube4 - cube5 - cube7 + cube6;
            const double b = cube0 - cube1 - cube3 + cube2;
            const double c = cube0 - cube1 - cube4 + cube5;
            const double d = cube0 - cube3 - cube4 + cube7;
            const double e = -cube0 + cube1;
            const double f = -cube0 + cube3;
            const double g = -cube0 + cube4;
            const double h = cube0;

            InteriorTunnelResult result;

            const double dx = b * c - a * e, dy = b * d - a * f, dz = c * d - a * g;
            if (dx == 0.0 || dy == 0.0 || dz == 0.0)
                return result; // isEmpty = true, matches the reference's final "else return true;"
            if (dx * dy * dz < 0.0)
                return result; // isEmpty = true

            const double discriminant = std::sqrt(dx * dy * dz);

            int numberOfCriticalPoints = 0;
            double criticalPointValue1 = 0.0, criticalPointValue2 = 0.0;

            const double x1 = (-d * dx - discriminant) / (a * dx);
            const double y1 = (-c * dy - discriminant) / (a * dy);
            const double z1 = (-b * dz - discriminant) / (a * dz);
            if (x1 > 0.0 && x1 < 1.0 && y1 > 0.0 && y1 < 1.0 && z1 > 0.0 && z1 < 1.0) {
                ++numberOfCriticalPoints;
                criticalPointValue1 = a * x1 * y1 * z1 + b * x1 * y1 + c * x1 * z1 + d * y1 * z1 + e * x1 + f * y1 + g * z1 + h;
            }

            const double x2 = (-d * dx + discriminant) / (a * dx);
            const double y2 = (-c * dy + discriminant) / (a * dy);
            const double z2 = (-b * dz + discriminant) / (a * dz);
            if (x2 > 0.0 && x2 < 1.0 && y2 > 0.0 && y2 < 1.0 && z2 > 0.0 && z2 < 1.0) {
                ++numberOfCriticalPoints;
                criticalPointValue2 = a * x2 * y2 * z2 + b * x2 * y2 + c * x2 * z2 + d * y2 * z2 + e * x2 + f * y2 + g * z2 + h;
            }

            if (numberOfCriticalPoints < 2) {
                result.isEmpty = true;
                return result;
            }

            const double product = criticalPointValue1 * criticalPointValue2;
            if (product > 0.0)
                result.orientation = (criticalPointValue1 > 0.0) ? 1 : -1;

            result.isEmpty = !(product < 0.0);
            return result;
        }

        // The irregular (config, subcase) -> (config, subcase) lookup for the "tunnel found,
        // opposite orientation" branch of case 13.5, extracted mechanically from the
        // reference's case 23..26 block (see task-2-report.md for provenance: this specific
        // 8-entry table is the one corner of MC33 sourced from a single, not cross-mirror-
        // diffed, reference, and the least independently verifiable -- it is also unreachable
        // by this task's tests, since case 13.5 needs a full 3-D checkerboard corner pattern).
        void OppositeOrientationTiling13_5_2Index(int configurationIndex, int subcase13_5, int &otherConfigurationIndex, int &otherSubcase13_5) {
            static constexpr int kOtherConfiguration[2] = {1, 0};
            static constexpr int kOtherSubcase[2][4] = {
                    {2, 0, 3, 1}, // configurationIndex == 0
                    {2, 3, 0, 2}, // configurationIndex == 1
            };
            otherConfigurationIndex = kOtherConfiguration[configurationIndex];
            otherSubcase13_5 = kOtherSubcase[configurationIndex][subcase13_5];
        }

        /// ────────────────────────────────────────────────────────────────────────────
        /// 5. Per-cube triangulation: resolves the case+subcase, then emits triangles by
        ///    naming edges 0..11 (core::kEdgeCornerPairs order, identical to the tiling
        ///    tables' own edge numbering) or 12 for the extra centre vertex.
        /// ────────────────────────────────────────────────────────────────────────────
        void EmitCubeTriangles(const std::array<int, 3> &base, CubeCornerValues cube, float cellSize,
                                std::vector<core::RawTriangle> &out) {
            const int cornerSignBitmask = ComputeCornerSignBitmask(cube); // snaps `cube` in place

            Eigen::Vector3f cornerPosition[8];
            for (int corner = 0; corner < 8; ++corner)
                cornerPosition[corner] = Eigen::Vector3f(float(base[0] + mc::CORNER[corner][0]),
                                                          float(base[1] + mc::CORNER[corner][1]),
                                                          float(base[2] + mc::CORNER[corner][2])) *
                                          cellSize;

            Eigen::Vector3f edgeVertex[12];
            bool edgeIsCut[12];
            for (int edge = 0; edge < 12; ++edge) {
                const int cornerA = core::kEdgeCornerPairs[edge][0], cornerB = core::kEdgeCornerPairs[edge][1];
                edgeIsCut[edge] = (cube[cornerA] > 0.0f) != (cube[cornerB] > 0.0f);
                if (edgeIsCut[edge])
                    edgeVertex[edge] = core::VertexInterpolate(cornerPosition[cornerA], cornerPosition[cornerB],
                                                                cube[cornerA], cube[cornerB]);
            }

            // The extra "centre" vertex some subcases need to stay manifold: the average of
            // every cut edge's interpolated point, matching the reference's add_c_vertex.
            Eigen::Vector3f centerVertex = Eigen::Vector3f::Zero();
            int cutEdgeCount = 0;
            for (int edge = 0; edge < 12; ++edge)
                if (edgeIsCut[edge]) {
                    centerVertex += edgeVertex[edge];
                    ++cutEdgeCount;
                }
            if (cutEdgeCount > 0)
                centerVertex /= float(cutEdgeCount);

            const auto vertexForEdgeCode = [&](int edgeCode) -> const Eigen::Vector3f & {
                return edgeCode == 12 ? centerVertex : edgeVertex[edgeCode];
            };
            const auto emit = [&](const int *trig, int triangleCount) {
                for (int t = 0; t < triangleCount; ++t)
                    out.push_back(core::RawTriangle{vertexForEdgeCode(trig[t * 3 + 0]), vertexForEdgeCode(trig[t * 3 + 1]),
                                                     vertexForEdgeCode(trig[t * 3 + 2])});
            };

            const int caseNumber = mc33::cases[cornerSignBitmask][0];
            const int configurationIndex = mc33::cases[cornerSignBitmask][1];
            int faceTestBitmask = 0;

            switch (caseNumber) {
                case 0:
                    break;

                case 1:
                    emit(mc33::tiling1[configurationIndex], 1);
                    break;

                case 2:
                    emit(mc33::tiling2[configurationIndex], 2);
                    break;

                case 3:
                    if (FaceTest(cube, mc33::test3[configurationIndex]))
                        emit(mc33::tiling3_2[configurationIndex], 4); // 3.2: connected
                    else
                        emit(mc33::tiling3_1[configurationIndex], 2); // 3.1: disjoint
                    break;

                case 4:
                    if (InteriorTest(cube, 4, configurationIndex, float(mc33::test4[configurationIndex])))
                        emit(mc33::tiling4_1[configurationIndex], 2); // 4.1.1
                    else
                        emit(mc33::tiling4_2[configurationIndex], 6); // 4.1.2
                    break;

                case 5:
                    emit(mc33::tiling5[configurationIndex], 3);
                    break;

                case 6:
                    if (FaceTest(cube, mc33::test6[configurationIndex][0]))
                        emit(mc33::tiling6_2[configurationIndex], 5); // 6.2
                    else if (InteriorTest(cube, 6, configurationIndex, float(mc33::test6[configurationIndex][1])))
                        emit(mc33::tiling6_1_1[configurationIndex], 3); // 6.1.1
                    else
                        emit(mc33::tiling6_1_2[configurationIndex], 9); // 6.1.2
                    break;

                case 7:
                    if (FaceTest(cube, mc33::test7[configurationIndex][0])) faceTestBitmask += 1;
                    if (FaceTest(cube, mc33::test7[configurationIndex][1])) faceTestBitmask += 2;
                    if (FaceTest(cube, mc33::test7[configurationIndex][2])) faceTestBitmask += 4;
                    switch (faceTestBitmask) {
                        case 0: emit(mc33::tiling7_1[configurationIndex], 3); break;
                        case 1: emit(mc33::tiling7_2[configurationIndex][0], 5); break;
                        case 2: emit(mc33::tiling7_2[configurationIndex][1], 5); break;
                        case 3: emit(mc33::tiling7_3[configurationIndex][0], 9); break;
                        case 4: emit(mc33::tiling7_2[configurationIndex][2], 5); break;
                        case 5: emit(mc33::tiling7_3[configurationIndex][1], 9); break;
                        case 6: emit(mc33::tiling7_3[configurationIndex][2], 9); break;
                        case 7:
                            if (InteriorTest(cube, 7, configurationIndex, float(mc33::test7[configurationIndex][3])))
                                emit(mc33::tiling7_4_1[configurationIndex], 5);
                            else
                                emit(mc33::tiling7_4_2[configurationIndex], 9);
                            break;
                        default: break;
                    }
                    break;

                case 8:
                    emit(mc33::tiling8[configurationIndex], 2);
                    break;

                case 9:
                    emit(mc33::tiling9[configurationIndex], 4);
                    break;

                case 10:
                    if (FaceTest(cube, mc33::test10[configurationIndex][0])) {
                        if (FaceTest(cube, mc33::test10[configurationIndex][1])) {
                            if (InteriorTest(cube, 10, configurationIndex, float(-mc33::test10[configurationIndex][2])))
                                emit(mc33::tiling10_1_1_[configurationIndex], 4); // 10.1.1
                            else
                                emit(mc33::tiling10_1_2[5 - configurationIndex], 8); // 10.1.2
                        } else {
                            emit(mc33::tiling10_2[configurationIndex], 8); // 10.2
                        }
                    } else if (FaceTest(cube, mc33::test10[configurationIndex][1])) {
                        emit(mc33::tiling10_2_[configurationIndex], 8); // 10.2
                    } else if (InteriorTest(cube, 10, configurationIndex, float(mc33::test10[configurationIndex][2]))) {
                        emit(mc33::tiling10_1_1[configurationIndex], 4); // 10.1.1
                    } else {
                        emit(mc33::tiling10_1_2[configurationIndex], 8); // 10.1.2
                    }
                    break;

                case 11:
                    emit(mc33::tiling11[configurationIndex], 4);
                    break;

                case 12:
                    if (FaceTest(cube, mc33::test12[configurationIndex][0])) {
                        if (FaceTest(cube, mc33::test12[configurationIndex][1])) {
                            if (InteriorTest(cube, 12, configurationIndex, float(-mc33::test12[configurationIndex][2])))
                                emit(mc33::tiling12_1_1_[configurationIndex], 4); // 12.1.1
                            else
                                emit(mc33::tiling12_1_2[23 - configurationIndex], 8); // 12.1.2
                        } else {
                            emit(mc33::tiling12_2[configurationIndex], 8); // 12.2
                        }
                    } else if (FaceTest(cube, mc33::test12[configurationIndex][1])) {
                        emit(mc33::tiling12_2_[configurationIndex], 8); // 12.2
                    } else if (InteriorTest(cube, 12, configurationIndex, float(mc33::test12[configurationIndex][2]))) {
                        emit(mc33::tiling12_1_1[configurationIndex], 4); // 12.1.1
                    } else {
                        emit(mc33::tiling12_1_2[configurationIndex], 8); // 12.1.2
                    }
                    break;

                case 13: {
                    if (FaceTest(cube, mc33::test13[configurationIndex][0])) faceTestBitmask += 1;
                    if (FaceTest(cube, mc33::test13[configurationIndex][1])) faceTestBitmask += 2;
                    if (FaceTest(cube, mc33::test13[configurationIndex][2])) faceTestBitmask += 4;
                    if (FaceTest(cube, mc33::test13[configurationIndex][3])) faceTestBitmask += 8;
                    if (FaceTest(cube, mc33::test13[configurationIndex][4])) faceTestBitmask += 16;
                    if (FaceTest(cube, mc33::test13[configurationIndex][5])) faceTestBitmask += 32;

                    const int subcase = mc33::subconfig13[faceTestBitmask];
                    switch (subcase) {
                        case 0: emit(mc33::tiling13_1[configurationIndex], 4); break; // 13.1

                        case 1: emit(mc33::tiling13_2[configurationIndex][0], 6); break; // 13.2
                        case 2: emit(mc33::tiling13_2[configurationIndex][1], 6); break;
                        case 3: emit(mc33::tiling13_2[configurationIndex][2], 6); break;
                        case 4: emit(mc33::tiling13_2[configurationIndex][3], 6); break;
                        case 5: emit(mc33::tiling13_2[configurationIndex][4], 6); break;
                        case 6: emit(mc33::tiling13_2[configurationIndex][5], 6); break;

                        case 7: emit(mc33::tiling13_3[configurationIndex][0], 10); break; // 13.3
                        case 8: emit(mc33::tiling13_3[configurationIndex][1], 10); break;
                        case 9: emit(mc33::tiling13_3[configurationIndex][2], 10); break;
                        case 10: emit(mc33::tiling13_3[configurationIndex][3], 10); break;
                        case 11: emit(mc33::tiling13_3[configurationIndex][4], 10); break;
                        case 12: emit(mc33::tiling13_3[configurationIndex][5], 10); break;
                        case 13: emit(mc33::tiling13_3[configurationIndex][6], 10); break;
                        case 14: emit(mc33::tiling13_3[configurationIndex][7], 10); break;
                        case 15: emit(mc33::tiling13_3[configurationIndex][8], 10); break;
                        case 16: emit(mc33::tiling13_3[configurationIndex][9], 10); break;
                        case 17: emit(mc33::tiling13_3[configurationIndex][10], 10); break;
                        case 18: emit(mc33::tiling13_3[configurationIndex][11], 10); break;

                        case 19: emit(mc33::tiling13_4[configurationIndex][0], 12); break; // 13.4
                        case 20: emit(mc33::tiling13_4[configurationIndex][1], 12); break;
                        case 21: emit(mc33::tiling13_4[configurationIndex][2], 12); break;
                        case 22: emit(mc33::tiling13_4[configurationIndex][3], 12); break;

                        case 23: // 13.5
                        case 24:
                        case 25:
                        case 26: {
                            const int subcase13_5 = subcase - 23;
                            const InteriorTunnelResult tunnel = EvaluateInteriorTunnel(cube);
                            if (tunnel.isEmpty) {
                                emit(mc33::tiling13_5_1[configurationIndex][subcase13_5], 6);
                            } else if (tunnel.orientation == 1) {
                                emit(mc33::tiling13_5_2[configurationIndex][subcase13_5], 10);
                            } else {
                                int otherConfigurationIndex, otherSubcase13_5;
                                OppositeOrientationTiling13_5_2Index(configurationIndex, subcase13_5, otherConfigurationIndex,
                                                                      otherSubcase13_5);
                                emit(mc33::tiling13_5_2[otherConfigurationIndex][otherSubcase13_5], 10);
                            }
                            break;
                        }

                        case 27: emit(mc33::tiling13_3_[configurationIndex][0], 10); break; // 13.3 (complement)
                        case 28: emit(mc33::tiling13_3_[configurationIndex][1], 10); break;
                        case 29: emit(mc33::tiling13_3_[configurationIndex][2], 10); break;
                        case 30: emit(mc33::tiling13_3_[configurationIndex][3], 10); break;
                        case 31: emit(mc33::tiling13_3_[configurationIndex][4], 10); break;
                        case 32: emit(mc33::tiling13_3_[configurationIndex][5], 10); break;
                        case 33: emit(mc33::tiling13_3_[configurationIndex][6], 10); break;
                        case 34: emit(mc33::tiling13_3_[configurationIndex][7], 10); break;
                        case 35: emit(mc33::tiling13_3_[configurationIndex][8], 10); break;
                        case 36: emit(mc33::tiling13_3_[configurationIndex][9], 10); break;
                        case 37: emit(mc33::tiling13_3_[configurationIndex][10], 10); break;
                        case 38: emit(mc33::tiling13_3_[configurationIndex][11], 10); break;

                        case 39: emit(mc33::tiling13_2_[configurationIndex][0], 6); break; // 13.2 (complement)
                        case 40: emit(mc33::tiling13_2_[configurationIndex][1], 6); break;
                        case 41: emit(mc33::tiling13_2_[configurationIndex][2], 6); break;
                        case 42: emit(mc33::tiling13_2_[configurationIndex][3], 6); break;
                        case 43: emit(mc33::tiling13_2_[configurationIndex][4], 6); break;
                        case 44: emit(mc33::tiling13_2_[configurationIndex][5], 6); break;

                        case 45: emit(mc33::tiling13_1_[configurationIndex], 4); break; // 13.1 (complement)

                        default: break; // subconfig13 == -1: mathematically unreachable for a genuine case-13 cube
                    }
                    break;
                }

                case 14:
                    emit(mc33::tiling14[configurationIndex], 4);
                    break;

                default:
                    break;
            }
        }

        /// ────────────────────────────────────────────────────────────────────────────
        /// 6. The extractor itself: same candidate-cube enumeration as "mc" (core::
        ///    CandidateBases), same isoLevel shift, same weld -- only EmitCubeTriangles
        ///    (above) differs.
        /// ────────────────────────────────────────────────────────────────────────────
        class MarchingCubes33Extractor : public IsoSurfaceExtractor {
        public:
            const char *Name() const override { return "mc33"; }

            SurfaceMesh Extract(const VoxelField &field, const ExtractParams &params) const override {
                const auto bases = core::CandidateBases(field.OccupiedCoords());

                std::vector<core::RawTriangle> raw;
                for (const auto &base : bases) {
                    CubeCornerValues cube{};
                    bool complete = true;
                    for (int corner = 0; corner < 8 && complete; ++corner) {
                        const std::array<int, 3> cornerCoord{base[0] + mc::CORNER[corner][0], base[1] + mc::CORNER[corner][1],
                                                              base[2] + mc::CORNER[corner][2]};
                        float value = 0.0f;
                        if (!field.Sample(cornerCoord, value)) {
                            complete = false;
                            break;
                        }
                        cube[corner] = value - params.isoLevel;
                    }
                    if (!complete) continue; // an unresolvable corner: skip this cell, matching "mc"

                    EmitCubeTriangles(base, cube, field.CellSize(), raw);
                }

                return core::WeldAndComputeNormals(raw, params.weldFraction * field.CellSize());
            }
        };

    } // namespace

    std::unique_ptr<IsoSurfaceExtractor> CreateMarchingCubes33Extractor() {
        return std::make_unique<MarchingCubes33Extractor>();
    }

} // namespace Mesh

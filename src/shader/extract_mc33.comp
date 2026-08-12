#version 460

/// *********************************************************************************************
/// extract_mc33.comp
///
/// GPU compute twin of the CPU "mc33" strategy (MarchingCubes33Extractor.cpp): topologically
/// correct Marching Cubes 33 (Chernyaev / Lewiner et al. 2003), one invocation per candidate cube
/// -- the SAME candidate set core::CandidateBases() builds on the CPU, same dense value-grid
/// sampling as extract_mc.comp (see GpuIsoSurfaceExtractorCommon.h's file-level doc comment for
/// why a dense grid, not a hash, is this task's deliberate "simplest first cut"). Reuses
/// voxel_common.glsl's CORNER/vertInterp (the SAME 12-edge layout and interpolation formula the
/// CPU "mc33" and "mc" extractors both use) and marching_cubes_33_tables.glsl's mechanically
/// flattened case/subcase/tiling tables (see that file's header for full provenance: Lewiner et
/// al., "Efficient Implementation of Marching Cubes' Cases with Topological Guarantees", Journal
/// of Graphics Tools, 8(2):1-15, 2003).
///
/// Every helper function below (ComputeCornerSignBitmask, FaceCorners/FaceTest,
/// InteriorTestEdgeForFace/InteriorTestOnEdge/InteriorTest, EvaluateInteriorTunnel,
/// OppositeOrientationTiling13_5_2Index) and the case/subcase dispatch switch inside processCube()
/// mirror MarchingCubes33Extractor.cpp's EmitCubeTriangles function-for-function and
/// branch-for-branch, INCLUDING its quirks (e.g. test6[..][2]/test7[..][4]/test12[..][3]/
/// test13[..][6] are transcribed but deliberately unread, exactly like the CPU twin, since the
/// consistency gate is "matches THIS codebase's CPU mc33", not "reimplements the paper from
/// scratch"). One deliberate deviation: EvaluateInteriorTunnel uses `float` where the CPU uses
/// `double` for its root solve -- see that function's own comment for why (MoltenVK/Metal has NO
/// double-precision float support) and why it is inert for every test this task runs.
///
/// Field representation, dense-grid sampling and the int-atomic emit contract (binding 0 = counter,
/// binding 1 = vertex slots, binding 2 = candidate bases, binding 3 = value grid) are IDENTICAL to
/// extract_mc.comp -- see that file's header comment for the full contract. Two differences from
/// extract_mc.comp:
///
///   1. WINDING: mc's triTable is this codebase's own naive 15-case table, whose vertices need an
///      explicit (ev[ei0], ev[ei2], ev[ei1]) swap to wind outward (see extract_mc.comp's header).
///      Lewiner's mc33 tiling tables are ALREADY wound correctly by the reference -- the CPU
///      twin's `emit` lambda pushes trig[0],trig[1],trig[2] in table order with NO swap (see
///      MarchingCubes33Extractor.cpp's EmitCubeTriangles). EmitOneTriangle() below matches that:
///      no swap. Applying mc's swap here would silently flip every mc33 triangle's winding
///      (flipped normals) without failing the position-only consistency test this task's brief
///      specifies -- a real gotcha, called out here so nobody "fixes" this into a bug later.
///
///   2. ORDER-KEY STRIDE: mc uses STRIDE=8 (a cube emits at most 5 triangles). mc33 emits up to
///      12 triangles per cube (case 13.4 / tiling13_4, the documented maximum -- see
///      GpuMarchingCubes33Extractor.cpp's kMaxTrianglesPerCube33 derivation). ORDER_KEY_STRIDE
///      below is 16, which strictly exceeds that maximum (contract: every kernel sharing
///      GpuIsoSurfaceExtractorCommon's readback must pick a stride > its own kernel's max
///      triangles-per-cube, or the order-key sort silently reconstructs the WRONG emission order
///      -- see GpuIsoSurfaceExtractorCommon.h's "Order key" doc and this file's own
///      ORDER_KEY_STRIDE comment below).
/// *********************************************************************************************

layout(local_size_x = 64) in;

#include "voxel_common.glsl"              // CORNER, vertInterp (edgeTable/triTable unused here)
#include "marching_cubes_33_tables.glsl"  // cases, tiling*, test*, subconfig13 (47 flat tables)

// Keep in sync with kGpuUnresolvedFieldValue (GpuIsoSurfaceExtractorCommon.h).
#define UNRESOLVED_FIELD_VALUE 1e37

// std::numeric_limits<float>::epsilon() == 2^-23, exactly representable in float32. Matches
// MarchingCubes33Extractor.cpp's ComputeCornerSignBitmask/FaceTest epsilon guards bit-for-bit.
#define FLOAT_EPSILON 1.1920929e-7

// Every Gpu*Extractor kernel's per-cube order key is candidateIndex*STRIDE + triangleWithinCube;
// STRIDE must strictly EXCEED that kernel's maximum triangles-per-cube, or one cube's key block
// overruns the next candidate's and GpuIsoSurfaceExtractorCommon::ReadbackRawTriangles' sort
// silently reconstructs the WRONG emission order (mismatched vertex positions post-weld, not a
// crash -- this is exactly the bug Task 3 found+fixed on "mc"). mc (extract_mc.comp) uses 8 (max
// 5 triangles/cube); mc33 emits up to 12 triangles/cube (case 13.4) so uses 16 here.
#define ORDER_KEY_STRIDE 16u

layout(push_constant) uniform PushConstants
{
	int   g_originX, g_originY, g_originZ; // dense value grid's [0,0,0] cell, in field integer coords
	int   g_dimsX, g_dimsY, g_dimsZ;       // dense value grid extents
	float g_cellSize;                      // world-space size of one voxel (VoxelField::CellSize())
	float g_isoLevel;                      // ExtractParams::isoLevel, subtracted from sampled values
	uint  g_candidateCount;                // number of entries in g_candidateBases
	uint  g_maxTriangles;                  // output buffer capacity (candidateCount * 12, see .cpp)
};

layout(std430, set = 0, binding = 0)          buffer Counter        { int   g_triangleCount; };
layout(std430, set = 0, binding = 1)          buffer VertexSlots    { vec4  g_vertexSlots[]; };
layout(std430, set = 0, binding = 2) readonly buffer CandidateBases { ivec4 g_candidateBases[]; };
layout(std430, set = 0, binding = 3) readonly buffer ValueGrid      { float g_valueGrid[]; };

/// *********************************************************************************************
/// Dense value-grid lookup -- verbatim twin of extract_mc.comp's sampleFieldValue().
/// *********************************************************************************************

bool sampleFieldValue(ivec3 coordinate, out float value)
{
	ivec3 origin = ivec3(g_originX, g_originY, g_originZ);
	ivec3 dims   = ivec3(g_dimsX, g_dimsY, g_dimsZ);
	ivec3 local  = coordinate - origin;
	if (any(lessThan(local, ivec3(0))) || any(greaterThanEqual(local, dims)))
	{
		return false;
	}

	int flatIndex = (local.z * dims.y + local.y) * dims.x + local.x;
	float rawValue = g_valueGrid[flatIndex];
	if (rawValue >= UNRESOLVED_FIELD_VALUE)
	{
		return false;
	}

	value = rawValue - g_isoLevel;
	return true;
}

/// *********************************************************************************************
/// Interior-test corner-index table -- transcribed from MarchingCubes33Extractor.cpp's
/// kInteriorTestCornerIndices[12][8] (NOT from MarchingCubes33Tables.h -- this table lives beside
/// InteriorTestOnEdge in the CPU .cpp, not the .h). Flattened the same way as
/// marching_cubes_33_tables.glsl: kInteriorTestCornerIndices[edge][k] == flat[edge*8 + k].
/// *********************************************************************************************

const int kInteriorTestCornerIndices[96] = int[96](
	0, 1, 7, 6, 4, 5, 3, 2, // edge  0
	3, 2, 4, 5, 0, 1, 7, 6, // edge  1
	2, 3, 5, 4, 6, 7, 1, 0, // edge  2
	1, 0, 6, 7, 2, 3, 5, 4, // edge  3
	2, 1, 7, 4, 3, 0, 6, 5, // edge  4
	3, 0, 6, 5, 2, 1, 7, 4, // edge  5
	0, 3, 5, 6, 4, 7, 1, 2, // edge  6
	1, 2, 4, 7, 0, 3, 5, 6, // edge  7
	4, 0, 6, 2, 7, 3, 5, 1, // edge  8
	5, 1, 7, 3, 4, 0, 6, 2, // edge  9
	6, 2, 4, 0, 5, 1, 7, 3, // edge 10
	7, 3, 5, 1, 6, 2, 4, 0  // edge 11
);

/// *********************************************************************************************
/// Corner sign bitmask (Lewiner's "_lut_entry") -- mirrors ComputeCornerSignBitmask: bit c set
/// iff corner c's (iso-shifted) value is > 0 (the OPPOSITE convention from mc's cubeIndex, which
/// sets a bit for NEGATIVE corners -- the two conventions are never mixed in this file). Also
/// snaps any exactly-zero corner to +epsilon in place, so every one of the 256 sign patterns is
/// well defined.
/// *********************************************************************************************

int ComputeCornerSignBitmask(inout float sdf[8])
{
	int bitmask = 0;
	for (int corner = 0; corner < 8; corner++) {
		if (abs(sdf[corner]) < FLOAT_EPSILON)
		{
			sdf[corner] = FLOAT_EPSILON;
		}
		if (sdf[corner] > 0.0)
		{
			bitmask |= (1 << corner);
		}
	}
	return bitmask;
}

/// *********************************************************************************************
/// Asymptotic decider (face ambiguity) -- mirrors FaceCorners/FaceTest.
/// *********************************************************************************************

// Signed face code -> the four corners bounding that face, in (A,B,C,D) order: face +-1={0,4,5,1},
// +-2={1,5,6,2}, +-3={2,6,7,3}, +-4={3,7,4,0}, +-5={0,3,2,1}, +-6={4,7,6,5}.
void FaceCorners(int faceCode, out int cornerA, out int cornerB, out int cornerC, out int cornerD)
{
	int faceIndex = abs(faceCode);
	if (faceIndex == 1)
	{
		cornerA = 0; cornerB = 4; cornerC = 5; cornerD = 1;
	}
	else if (faceIndex == 2)
	{
		cornerA = 1; cornerB = 5; cornerC = 6; cornerD = 2;
	}
	else if (faceIndex == 3)
	{
		cornerA = 2; cornerB = 6; cornerC = 7; cornerD = 3;
	}
	else if (faceIndex == 4)
	{
		cornerA = 3; cornerB = 7; cornerC = 4; cornerD = 0;
	}
	else if (faceIndex == 5)
	{
		cornerA = 0; cornerB = 3; cornerC = 2; cornerD = 1;
	}
	else // face 6
	{
		cornerA = 4; cornerB = 7; cornerC = 6; cornerD = 5;
	}
}

// The asymptotic decider: sign of the bilinear-saddle value A*C - B*D on the named face. On the
// (measure-zero) exact saddle, ties break on the sign of the face code itself.
bool FaceTest(float sdf[8], int faceCode)
{
	int cornerA, cornerB, cornerC, cornerD;
	FaceCorners(faceCode, cornerA, cornerB, cornerC, cornerD);
	float A = sdf[cornerA];
	float B = sdf[cornerB];
	float C = sdf[cornerC];
	float D = sdf[cornerD];
	float saddleValue = A * C - B * D;
	if (abs(saddleValue) < FLOAT_EPSILON)
	{
		return faceCode >= 0;
	}
	return float(faceCode) * A * saddleValue >= 0.0;
}

/// *********************************************************************************************
/// Interior test (interior ambiguity, cases 4/6/7/10/12) -- mirrors InteriorTestEdgeForFace/
/// InteriorTestOnEdge/InteriorTest.
/// *********************************************************************************************

// Which of the 12 cube edges carries the "tunnel" for a given ambiguous face and sign: exactly
// one of the four space-diagonal corner pairs opposite that face satisfies (sdf[corner]*signValue)
// > 0 for a genuinely ambiguous configuration. Deliberately sequential ifs (not else-if): matches
// the CPU's "last match wins" semantics exactly.
int InteriorTestEdgeForFace(float sdf[8], int ambiguousFace, float signValue)
{
	int edge = -1;
	if (ambiguousFace == 1 || ambiguousFace == 3)
	{
		if (sdf[1] * signValue > 0.0 && sdf[7] * signValue > 0.0) edge = 4;
		if (sdf[0] * signValue > 0.0 && sdf[6] * signValue > 0.0) edge = 5;
		if (sdf[3] * signValue > 0.0 && sdf[5] * signValue > 0.0) edge = 6;
		if (sdf[2] * signValue > 0.0 && sdf[4] * signValue > 0.0) edge = 7;
	}
	else if (ambiguousFace == 2 || ambiguousFace == 4)
	{
		if (sdf[1] * signValue > 0.0 && sdf[7] * signValue > 0.0) edge = 0;
		if (sdf[2] * signValue > 0.0 && sdf[4] * signValue > 0.0) edge = 1;
		if (sdf[3] * signValue > 0.0 && sdf[5] * signValue > 0.0) edge = 2;
		if (sdf[0] * signValue > 0.0 && sdf[6] * signValue > 0.0) edge = 3;
	}
	else if (ambiguousFace == 0 || ambiguousFace == 5 || ambiguousFace == 6)
	{
		if (sdf[0] * signValue > 0.0 && sdf[6] * signValue > 0.0) edge = 8;
		if (sdf[1] * signValue > 0.0 && sdf[7] * signValue > 0.0) edge = 9;
		if (sdf[2] * signValue > 0.0 && sdf[4] * signValue > 0.0) edge = 10;
		if (sdf[3] * signValue > 0.0 && sdf[5] * signValue > 0.0) edge = 11;
	}
	return edge;
}

// Given the tunnel edge selected above, decides whether the trilinear saddle along that edge
// actually crosses the isosurface strictly inside the cube (same bilinear-saddle-with-critical-t
// formula as FaceTest, evaluated on the interior diagonal named by `edge`).
bool InteriorTestOnEdge(float sdf[8], int edge)
{
	if (edge < 0 || edge > 11)
	{
		return true; // no tunnel edge found: mirrors the reference's ambiguous-default paths
	}

	int X = kInteriorTestCornerIndices[edge * 8 + 0];
	int Y = kInteriorTestCornerIndices[edge * 8 + 1];
	int Z = kInteriorTestCornerIndices[edge * 8 + 2];
	int W = kInteriorTestCornerIndices[edge * 8 + 3];
	int P = kInteriorTestCornerIndices[edge * 8 + 4];
	int Q = kInteriorTestCornerIndices[edge * 8 + 5];
	int R = kInteriorTestCornerIndices[edge * 8 + 6];
	int S = kInteriorTestCornerIndices[edge * 8 + 7];

	float a = (sdf[X] - sdf[Y]) * (sdf[Z] - sdf[W]) - (sdf[P] - sdf[Q]) * (sdf[R] - sdf[S]);
	if (a > 0.0)
	{
		return true;
	}

	float b = sdf[W] * (sdf[X] - sdf[Y]) + sdf[Y] * (sdf[Z] - sdf[W])
	        - sdf[S] * (sdf[P] - sdf[Q]) - sdf[Q] * (sdf[R] - sdf[S]);
	float t = -b / (2.0 * a);
	if (t < 0.0 || t > 1.0)
	{
		return true;
	}

	float At = sdf[Y] + (sdf[X] - sdf[Y]) * t;
	float Bt = sdf[Q] + (sdf[P] - sdf[Q]) * t;
	float Ct = sdf[W] + (sdf[Z] - sdf[W]) * t;
	float Dt = sdf[S] + (sdf[R] - sdf[S]) * t;
	float verify = At * Ct - Bt * Dt;

	if (verify > 0.0)
	{
		return false;
	}
	return true; // verify < 0, or (measure-zero) exactly 0: both "tunnel present"
}

// The interior-test dispatcher for cases 4/6/7/10/12: which face(s) to run
// InteriorTestEdgeForFace/InteriorTestOnEdge on depends on the case (4 and 7 check three fixed
// faces and OR the results; 6/10/12 look up a single ambiguous face out of test6/test10/test12's
// own configuration row -- columns [2]/[4]/[3] of those tables are deliberately unread here,
// matching the CPU exactly).
bool InteriorTest(float sdf[8], int caseNumber, int configurationIndex, float signValue)
{
	switch (caseNumber)
	{
		case 4:
		{
			return InteriorTestOnEdge(sdf, InteriorTestEdgeForFace(sdf, 1, signValue))
			    || InteriorTestOnEdge(sdf, InteriorTestEdgeForFace(sdf, 2, signValue))
			    || InteriorTestOnEdge(sdf, InteriorTestEdgeForFace(sdf, 5, signValue));
		}
		case 6:
		{
			int ambiguousFace = abs(test6[configurationIndex * 3 + 0]);
			return InteriorTestOnEdge(sdf, InteriorTestEdgeForFace(sdf, ambiguousFace, signValue));
		}
		case 7:
		{
			float flippedSign = -signValue;
			return InteriorTestOnEdge(sdf, InteriorTestEdgeForFace(sdf, 1, flippedSign))
			    || InteriorTestOnEdge(sdf, InteriorTestEdgeForFace(sdf, 2, flippedSign))
			    || InteriorTestOnEdge(sdf, InteriorTestEdgeForFace(sdf, 5, flippedSign));
		}
		case 10:
		{
			int ambiguousFace = abs(test10[configurationIndex * 3 + 0]);
			return InteriorTestOnEdge(sdf, InteriorTestEdgeForFace(sdf, ambiguousFace, signValue));
		}
		case 12:
		{
			int ambiguousFaceA = abs(test12[configurationIndex * 4 + 0]);
			int ambiguousFaceB = abs(test12[configurationIndex * 4 + 1]);
			return InteriorTestOnEdge(sdf, InteriorTestEdgeForFace(sdf, ambiguousFaceA, signValue))
			    || InteriorTestOnEdge(sdf, InteriorTestEdgeForFace(sdf, ambiguousFaceB, signValue));
		}
		default:
			return false;
	}
}

/// *********************************************************************************************
/// Case 13.5 interior tunnel test (trilinear critical-point analysis) -- mirrors
/// EvaluateInteriorTunnel. Case 13's rarest subcase (the cube's 8 corners alternate in a full 3-D
/// checkerboard) needs a genuine trilinear critical-point analysis rather than a bilinear face
/// test.
///
/// DEVIATION FROM THE CPU: the reference (and this codebase's CPU twin) deliberately solves this
/// in `double` for numerical delicacy near-degenerate discriminants. MoltenVK/Metal has NO
/// double-precision float support at all (unlike int atomics, which MoltenVK does support), so
/// this port uses `float`. This is inert for every test this task runs: case 13.5 only triggers
/// when the SDF reverses sign between every pair of grid-adjacent corners, which cannot arise
/// from sampling a smooth analytic surface (a sphere) on a regular grid -- see
/// MarchingCubes33Extractor.cpp's own file-header comment for the identical observation about the
/// CPU side. A future caller feeding this extractor a genuinely checkerboard-adversarial field
/// could see this disagree with the CPU (float root-finding is less stable near-degenerate
/// discriminants); flagged here rather than silently accepted.
/// *********************************************************************************************

struct InteriorTunnelResult
{
	bool isEmpty;    // true: no tunnel (use the simpler 6-triangle tiling13_5_1)
	int  orientation; // +-1 once a tunnel is found; picks which of the two tiling13_5_2 branches
};

InteriorTunnelResult EvaluateInteriorTunnel(float sdf[8])
{
	float cube0 = sdf[0], cube1 = sdf[1], cube2 = sdf[2], cube3 = sdf[3];
	float cube4 = sdf[4], cube5 = sdf[5], cube6 = sdf[6], cube7 = sdf[7];

	float a = -cube0 + cube1 + cube3 - cube2 + cube4 - cube5 - cube7 + cube6;
	float b = cube0 - cube1 - cube3 + cube2;
	float c = cube0 - cube1 - cube4 + cube5;
	float d = cube0 - cube3 - cube4 + cube7;
	float e = -cube0 + cube1;
	float f = -cube0 + cube3;
	float g = -cube0 + cube4;
	float h = cube0;

	InteriorTunnelResult result;
	result.isEmpty = true;
	result.orientation = 0;

	float dx = b * c - a * e;
	float dy = b * d - a * f;
	float dz = c * d - a * g;
	if (dx == 0.0 || dy == 0.0 || dz == 0.0)
	{
		return result; // isEmpty = true, matches the reference's final "else return true;"
	}
	if (dx * dy * dz < 0.0)
	{
		return result; // isEmpty = true
	}

	float discriminant = sqrt(dx * dy * dz);

	int   numberOfCriticalPoints = 0;
	float criticalPointValue1 = 0.0;
	float criticalPointValue2 = 0.0;

	float x1 = (-d * dx - discriminant) / (a * dx);
	float y1 = (-c * dy - discriminant) / (a * dy);
	float z1 = (-b * dz - discriminant) / (a * dz);
	if (x1 > 0.0 && x1 < 1.0 && y1 > 0.0 && y1 < 1.0 && z1 > 0.0 && z1 < 1.0)
	{
		numberOfCriticalPoints++;
		criticalPointValue1 = a * x1 * y1 * z1 + b * x1 * y1 + c * x1 * z1 + d * y1 * z1 + e * x1 + f * y1 + g * z1 + h;
	}

	float x2 = (-d * dx + discriminant) / (a * dx);
	float y2 = (-c * dy + discriminant) / (a * dy);
	float z2 = (-b * dz + discriminant) / (a * dz);
	if (x2 > 0.0 && x2 < 1.0 && y2 > 0.0 && y2 < 1.0 && z2 > 0.0 && z2 < 1.0)
	{
		numberOfCriticalPoints++;
		criticalPointValue2 = a * x2 * y2 * z2 + b * x2 * y2 + c * x2 * z2 + d * y2 * z2 + e * x2 + f * y2 + g * z2 + h;
	}

	if (numberOfCriticalPoints < 2)
	{
		result.isEmpty = true;
		return result;
	}

	float product = criticalPointValue1 * criticalPointValue2;
	if (product > 0.0)
	{
		result.orientation = (criticalPointValue1 > 0.0) ? 1 : -1;
	}

	result.isEmpty = !(product < 0.0);
	return result;
}

// The irregular (config, subcase) -> (config, subcase) lookup for case 13.5's "tunnel found,
// opposite orientation" branch -- transcribed from MarchingCubes33Extractor.cpp's
// OppositeOrientationTiling13_5_2Index (the one corner of MC33 sourced from a single, not
// cross-mirror-diffed, reference -- see that function's own comment).
void OppositeOrientationTiling13_5_2Index(int configurationIndex, int subcase13_5,
                                           out int otherConfigurationIndex, out int otherSubcase13_5)
{
	int otherConfiguration[2] = int[2](1, 0);
	int otherSubcase[8] = int[8](
		2, 0, 3, 1, // configurationIndex == 0
		2, 3, 0, 2  // configurationIndex == 1
	);

	otherConfigurationIndex = otherConfiguration[configurationIndex];
	otherSubcase13_5 = otherSubcase[configurationIndex * 4 + subcase13_5];
}

/// *********************************************************************************************
/// Triangle emission -- reserves 3 vertex slots per triangle with an int atomicAdd (the SAME
/// int-atomic-counter contract every Gpu*Extractor shader uses; MoltenVK has no float atomics).
/// NO winding swap (see file header point 1): mc33's tiling tables are already wound correctly.
/// *********************************************************************************************

vec3 VertexForEdgeCode(int edgeCode, vec3 edgeVertex[12], vec3 centerVertex)
{
	return edgeCode == 12 ? centerVertex : edgeVertex[edgeCode];
}

void EmitOneTriangle(vec3 v0, vec3 v1, vec3 v2, uint candidateIndex, uint triangleWithinCube)
{
	int slot = atomicAdd(g_triangleCount, 1);
	if (uint(slot) >= g_maxTriangles)
	{
		return;
	}

	float orderKey = uintBitsToFloat(candidateIndex * ORDER_KEY_STRIDE + triangleWithinCube);
	g_vertexSlots[slot * 3 + 0] = vec4(v0, orderKey);
	g_vertexSlots[slot * 3 + 1] = vec4(v1, orderKey);
	g_vertexSlots[slot * 3 + 2] = vec4(v2, orderKey);
}

// EMIT_TRIANGLES(TABLE, BASE, COUNT): emits COUNT triangles read from the flat table TABLE
// starting at flat offset BASE, 3 edge-codes per triangle. A macro, not a function, because GLSL
// has no array-reference/pointer parameter that could take "any one of the ~40 differently-sized
// flattened tiling* tables" the way the CPU's `emit` lambda captures its table pointer by value --
// this macro plays that same role, expanding inline at each case-dispatch call site below (which
// mirror MarchingCubes33Extractor.cpp's `emit(mc33::tilingX[...], N)` calls 1:1). `t` (0-based,
// < COUNT <= 12 < ORDER_KEY_STRIDE) IS the triangleWithinCube order-key component directly: every
// case/subcase below invokes this exactly ONCE per cube (just like the CPU's EmitCubeTriangles
// calls its `emit` lambda exactly once per cube), so table row order == emission order.
#define EMIT_TRIANGLES(TABLE, BASE, COUNT)                                                  \
	for (int t = 0; t < (COUNT); t++) {                                                     \
		vec3 v0 = VertexForEdgeCode(TABLE[(BASE) + t * 3 + 0], edgeVertex, centerVertex);      \
		vec3 v1 = VertexForEdgeCode(TABLE[(BASE) + t * 3 + 1], edgeVertex, centerVertex);      \
		vec3 v2 = VertexForEdgeCode(TABLE[(BASE) + t * 3 + 2], edgeVertex, centerVertex);      \
		EmitOneTriangle(v0, v1, v2, candidateIndex, uint(t));                                  \
	}

/// *********************************************************************************************
/// Per-cube Marching Cubes 33 -- mirrors MarchingCubes33Extractor.cpp's EmitCubeTriangles
/// function-for-function; the case/subcase switch below is the SAME dispatch tree, branch-for-
/// branch (including the two "reversed configuration index" quirks in cases 10/12 -- see the
/// tiling10_1_2[5 - configurationIndex] / tiling12_1_2[23 - configurationIndex] lines below,
/// which are NOT typos: they mirror the CPU exactly).
/// *********************************************************************************************

void processCube(ivec3 cubeBase, uint candidateIndex)
{
	// 1. Sample all 8 corners; an unresolved corner means "skip this cube" (mirrors the CPU
	//    core's `complete` flag / extract_mc.comp's processCube step 1).
	float sdf[8];
	for (int corner = 0; corner < 8; corner++) {
		if (!sampleFieldValue(cubeBase + CORNER[corner], sdf[corner]))
		{
			return;
		}
	}

	// 2. World-space corner positions.
	vec3 cornerPosition[8];
	for (int corner = 0; corner < 8; corner++) {
		cornerPosition[corner] = vec3(cubeBase + CORNER[corner]) * g_cellSize;
	}

	// 3. Corner sign bitmask (Lewiner convention), snapping any exactly-zero corner to +epsilon
	//    in place -- mirrors ComputeCornerSignBitmask exactly, including the in-place mutation
	//    order (this MUST run before step 4 below reads sdf, matching the CPU).
	int cornerSignBitmask = ComputeCornerSignBitmask(sdf);

	// 4. Interpolate all 12 standard edges unconditionally + remember which are actually cut.
	//    Unlike the CPU (which computes lazily per edgeIsCut), a case/subcase's tiling table NEVER
	//    references an edge it hasn't cut, so computing every edge unconditionally is branch-free
	//    and matches every REFERENCED result exactly; only step 5's cut-only average needs the
	//    edgeIsCut flags.
	vec3 edgeVertex[12];
	bool edgeIsCut[12];
	edgeIsCut[ 0] = (sdf[0] > 0.0) != (sdf[1] > 0.0); edgeVertex[ 0] = vertInterp(cornerPosition[0], cornerPosition[1], sdf[0], sdf[1]);
	edgeIsCut[ 1] = (sdf[1] > 0.0) != (sdf[2] > 0.0); edgeVertex[ 1] = vertInterp(cornerPosition[1], cornerPosition[2], sdf[1], sdf[2]);
	edgeIsCut[ 2] = (sdf[2] > 0.0) != (sdf[3] > 0.0); edgeVertex[ 2] = vertInterp(cornerPosition[2], cornerPosition[3], sdf[2], sdf[3]);
	edgeIsCut[ 3] = (sdf[3] > 0.0) != (sdf[0] > 0.0); edgeVertex[ 3] = vertInterp(cornerPosition[3], cornerPosition[0], sdf[3], sdf[0]);
	edgeIsCut[ 4] = (sdf[4] > 0.0) != (sdf[5] > 0.0); edgeVertex[ 4] = vertInterp(cornerPosition[4], cornerPosition[5], sdf[4], sdf[5]);
	edgeIsCut[ 5] = (sdf[5] > 0.0) != (sdf[6] > 0.0); edgeVertex[ 5] = vertInterp(cornerPosition[5], cornerPosition[6], sdf[5], sdf[6]);
	edgeIsCut[ 6] = (sdf[6] > 0.0) != (sdf[7] > 0.0); edgeVertex[ 6] = vertInterp(cornerPosition[6], cornerPosition[7], sdf[6], sdf[7]);
	edgeIsCut[ 7] = (sdf[7] > 0.0) != (sdf[4] > 0.0); edgeVertex[ 7] = vertInterp(cornerPosition[7], cornerPosition[4], sdf[7], sdf[4]);
	edgeIsCut[ 8] = (sdf[0] > 0.0) != (sdf[4] > 0.0); edgeVertex[ 8] = vertInterp(cornerPosition[0], cornerPosition[4], sdf[0], sdf[4]);
	edgeIsCut[ 9] = (sdf[1] > 0.0) != (sdf[5] > 0.0); edgeVertex[ 9] = vertInterp(cornerPosition[1], cornerPosition[5], sdf[1], sdf[5]);
	edgeIsCut[10] = (sdf[2] > 0.0) != (sdf[6] > 0.0); edgeVertex[10] = vertInterp(cornerPosition[2], cornerPosition[6], sdf[2], sdf[6]);
	edgeIsCut[11] = (sdf[3] > 0.0) != (sdf[7] > 0.0); edgeVertex[11] = vertInterp(cornerPosition[3], cornerPosition[7], sdf[3], sdf[7]);

	// 5. The extra "centre" vertex some subcases need to stay manifold: the average of every cut
	//    edge's interpolated point (mirrors EmitCubeTriangles' add_c_vertex / centerVertex).
	vec3 centerVertex = vec3(0.0);
	int cutEdgeCount = 0;
	for (int edge = 0; edge < 12; edge++) {
		if (edgeIsCut[edge])
		{
			centerVertex += edgeVertex[edge];
			cutEdgeCount++;
		}
	}
	if (cutEdgeCount > 0)
	{
		centerVertex /= float(cutEdgeCount);
	}

	// 6. Case + configuration lookup, then the case/subcase dispatch tree.
	int caseNumber = cases[cornerSignBitmask * 2 + 0];
	int configurationIndex = cases[cornerSignBitmask * 2 + 1];
	int faceTestBitmask = 0;

	switch (caseNumber)
	{
		case 0:
		{
			break;
		}

		case 1: // 1 triangle
		{
			EMIT_TRIANGLES(tiling1, configurationIndex * 3, 1);
			break;
		}

		case 2: // 2 triangles
		{
			EMIT_TRIANGLES(tiling2, configurationIndex * 6, 2);
			break;
		}

		case 3: // 3.1 disjoint (2 tri) / 3.2 connected (4 tri)
		{
			if (FaceTest(sdf, test3[configurationIndex]))
			{
				EMIT_TRIANGLES(tiling3_2, configurationIndex * 12, 4);
			}
			else
			{
				EMIT_TRIANGLES(tiling3_1, configurationIndex * 6, 2);
			}
			break;
		}

		case 4: // 4.1.1 (2 tri) / 4.1.2 (6 tri)
		{
			if (InteriorTest(sdf, 4, configurationIndex, float(test4[configurationIndex])))
			{
				EMIT_TRIANGLES(tiling4_1, configurationIndex * 6, 2);
			}
			else
			{
				EMIT_TRIANGLES(tiling4_2, configurationIndex * 18, 6);
			}
			break;
		}

		case 5: // 3 triangles
		{
			EMIT_TRIANGLES(tiling5, configurationIndex * 9, 3);
			break;
		}

		case 6: // 6.2 (5 tri) / 6.1.1 (3 tri) / 6.1.2 (9 tri)
		{
			if (FaceTest(sdf, test6[configurationIndex * 3 + 0]))
			{
				EMIT_TRIANGLES(tiling6_2, configurationIndex * 15, 5);
			}
			else if (InteriorTest(sdf, 6, configurationIndex, float(test6[configurationIndex * 3 + 1])))
			{
				EMIT_TRIANGLES(tiling6_1_1, configurationIndex * 9, 3);
			}
			else
			{
				EMIT_TRIANGLES(tiling6_1_2, configurationIndex * 27, 9);
			}
			break;
		}

		case 7: // 7.1 (3) / 7.2 (5) / 7.3 (9) / 7.4.1 (5) / 7.4.2 (9)
		{
			if (FaceTest(sdf, test7[configurationIndex * 5 + 0])) faceTestBitmask += 1;
			if (FaceTest(sdf, test7[configurationIndex * 5 + 1])) faceTestBitmask += 2;
			if (FaceTest(sdf, test7[configurationIndex * 5 + 2])) faceTestBitmask += 4;
			switch (faceTestBitmask)
			{
				case 0: EMIT_TRIANGLES(tiling7_1, configurationIndex * 9, 3); break;
				case 1: EMIT_TRIANGLES(tiling7_2, (configurationIndex * 3 + 0) * 15, 5); break;
				case 2: EMIT_TRIANGLES(tiling7_2, (configurationIndex * 3 + 1) * 15, 5); break;
				case 3: EMIT_TRIANGLES(tiling7_3, (configurationIndex * 3 + 0) * 27, 9); break;
				case 4: EMIT_TRIANGLES(tiling7_2, (configurationIndex * 3 + 2) * 15, 5); break;
				case 5: EMIT_TRIANGLES(tiling7_3, (configurationIndex * 3 + 1) * 27, 9); break;
				case 6: EMIT_TRIANGLES(tiling7_3, (configurationIndex * 3 + 2) * 27, 9); break;
				case 7:
				{
					if (InteriorTest(sdf, 7, configurationIndex, float(test7[configurationIndex * 5 + 3])))
					{
						EMIT_TRIANGLES(tiling7_4_1, configurationIndex * 15, 5);
					}
					else
					{
						EMIT_TRIANGLES(tiling7_4_2, configurationIndex * 27, 9);
					}
					break;
				}
				default:
					break;
			}
			break;
		}

		case 8: // 2 triangles
		{
			EMIT_TRIANGLES(tiling8, configurationIndex * 6, 2);
			break;
		}

		case 9: // 4 triangles
		{
			EMIT_TRIANGLES(tiling9, configurationIndex * 12, 4);
			break;
		}

		case 10: // 10.1.1 (4) / 10.1.2 (8) / 10.2 (8)
		{
			if (FaceTest(sdf, test10[configurationIndex * 3 + 0]))
			{
				if (FaceTest(sdf, test10[configurationIndex * 3 + 1]))
				{
					if (InteriorTest(sdf, 10, configurationIndex, float(-test10[configurationIndex * 3 + 2])))
					{
						EMIT_TRIANGLES(tiling10_1_1_, configurationIndex * 12, 4);
					}
					else
					{
						EMIT_TRIANGLES(tiling10_1_2, (5 - configurationIndex) * 24, 8); // reversed index -- see file header
					}
				}
				else
				{
					EMIT_TRIANGLES(tiling10_2, configurationIndex * 24, 8);
				}
			}
			else if (FaceTest(sdf, test10[configurationIndex * 3 + 1]))
			{
				EMIT_TRIANGLES(tiling10_2_, configurationIndex * 24, 8);
			}
			else if (InteriorTest(sdf, 10, configurationIndex, float(test10[configurationIndex * 3 + 2])))
			{
				EMIT_TRIANGLES(tiling10_1_1, configurationIndex * 12, 4);
			}
			else
			{
				EMIT_TRIANGLES(tiling10_1_2, configurationIndex * 24, 8);
			}
			break;
		}

		case 11: // 4 triangles
		{
			EMIT_TRIANGLES(tiling11, configurationIndex * 12, 4);
			break;
		}

		case 12: // 12.1.1 (4) / 12.1.2 (8) / 12.2 (8)
		{
			if (FaceTest(sdf, test12[configurationIndex * 4 + 0]))
			{
				if (FaceTest(sdf, test12[configurationIndex * 4 + 1]))
				{
					if (InteriorTest(sdf, 12, configurationIndex, float(-test12[configurationIndex * 4 + 2])))
					{
						EMIT_TRIANGLES(tiling12_1_1_, configurationIndex * 12, 4);
					}
					else
					{
						EMIT_TRIANGLES(tiling12_1_2, (23 - configurationIndex) * 24, 8); // reversed index -- see file header
					}
				}
				else
				{
					EMIT_TRIANGLES(tiling12_2, configurationIndex * 24, 8);
				}
			}
			else if (FaceTest(sdf, test12[configurationIndex * 4 + 1]))
			{
				EMIT_TRIANGLES(tiling12_2_, configurationIndex * 24, 8);
			}
			else if (InteriorTest(sdf, 12, configurationIndex, float(test12[configurationIndex * 4 + 2])))
			{
				EMIT_TRIANGLES(tiling12_1_1, configurationIndex * 12, 4);
			}
			else
			{
				EMIT_TRIANGLES(tiling12_1_2, configurationIndex * 24, 8);
			}
			break;
		}

		case 13: // the famous ambiguous case: 46 subcases (13.1 .. 13.5, plus complements)
		{
			if (FaceTest(sdf, test13[configurationIndex * 7 + 0])) faceTestBitmask += 1;
			if (FaceTest(sdf, test13[configurationIndex * 7 + 1])) faceTestBitmask += 2;
			if (FaceTest(sdf, test13[configurationIndex * 7 + 2])) faceTestBitmask += 4;
			if (FaceTest(sdf, test13[configurationIndex * 7 + 3])) faceTestBitmask += 8;
			if (FaceTest(sdf, test13[configurationIndex * 7 + 4])) faceTestBitmask += 16;
			if (FaceTest(sdf, test13[configurationIndex * 7 + 5])) faceTestBitmask += 32;

			int subcase = subconfig13[faceTestBitmask];
			switch (subcase)
			{
				case 0: EMIT_TRIANGLES(tiling13_1, configurationIndex * 12, 4); break; // 13.1

				case 1: EMIT_TRIANGLES(tiling13_2, (configurationIndex * 6 + 0) * 18, 6); break; // 13.2
				case 2: EMIT_TRIANGLES(tiling13_2, (configurationIndex * 6 + 1) * 18, 6); break;
				case 3: EMIT_TRIANGLES(tiling13_2, (configurationIndex * 6 + 2) * 18, 6); break;
				case 4: EMIT_TRIANGLES(tiling13_2, (configurationIndex * 6 + 3) * 18, 6); break;
				case 5: EMIT_TRIANGLES(tiling13_2, (configurationIndex * 6 + 4) * 18, 6); break;
				case 6: EMIT_TRIANGLES(tiling13_2, (configurationIndex * 6 + 5) * 18, 6); break;

				case 7:  EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 +  0) * 30, 10); break; // 13.3
				case 8:  EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 +  1) * 30, 10); break;
				case 9:  EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 +  2) * 30, 10); break;
				case 10: EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 +  3) * 30, 10); break;
				case 11: EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 +  4) * 30, 10); break;
				case 12: EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 +  5) * 30, 10); break;
				case 13: EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 +  6) * 30, 10); break;
				case 14: EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 +  7) * 30, 10); break;
				case 15: EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 +  8) * 30, 10); break;
				case 16: EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 +  9) * 30, 10); break;
				case 17: EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 + 10) * 30, 10); break;
				case 18: EMIT_TRIANGLES(tiling13_3, (configurationIndex * 12 + 11) * 30, 10); break;

				case 19: EMIT_TRIANGLES(tiling13_4, (configurationIndex * 4 + 0) * 36, 12); break; // 13.4
				case 20: EMIT_TRIANGLES(tiling13_4, (configurationIndex * 4 + 1) * 36, 12); break;
				case 21: EMIT_TRIANGLES(tiling13_4, (configurationIndex * 4 + 2) * 36, 12); break;
				case 22: EMIT_TRIANGLES(tiling13_4, (configurationIndex * 4 + 3) * 36, 12); break;

				case 23: // 13.5: needs the trilinear interior-tunnel test (EvaluateInteriorTunnel)
				case 24:
				case 25:
				case 26:
				{
					int subcase13_5 = subcase - 23;
					InteriorTunnelResult tunnel = EvaluateInteriorTunnel(sdf);
					if (tunnel.isEmpty)
					{
						EMIT_TRIANGLES(tiling13_5_1, (configurationIndex * 4 + subcase13_5) * 18, 6);
					}
					else if (tunnel.orientation == 1)
					{
						EMIT_TRIANGLES(tiling13_5_2, (configurationIndex * 4 + subcase13_5) * 30, 10);
					}
					else
					{
						int otherConfigurationIndex, otherSubcase13_5;
						OppositeOrientationTiling13_5_2Index(configurationIndex, subcase13_5,
						                                      otherConfigurationIndex, otherSubcase13_5);
						EMIT_TRIANGLES(tiling13_5_2, (otherConfigurationIndex * 4 + otherSubcase13_5) * 30, 10);
					}
					break;
				}

				case 27: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 +  0) * 30, 10); break; // 13.3 (complement)
				case 28: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 +  1) * 30, 10); break;
				case 29: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 +  2) * 30, 10); break;
				case 30: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 +  3) * 30, 10); break;
				case 31: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 +  4) * 30, 10); break;
				case 32: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 +  5) * 30, 10); break;
				case 33: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 +  6) * 30, 10); break;
				case 34: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 +  7) * 30, 10); break;
				case 35: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 +  8) * 30, 10); break;
				case 36: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 +  9) * 30, 10); break;
				case 37: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 + 10) * 30, 10); break;
				case 38: EMIT_TRIANGLES(tiling13_3_, (configurationIndex * 12 + 11) * 30, 10); break;

				case 39: EMIT_TRIANGLES(tiling13_2_, (configurationIndex * 6 + 0) * 18, 6); break; // 13.2 (complement)
				case 40: EMIT_TRIANGLES(tiling13_2_, (configurationIndex * 6 + 1) * 18, 6); break;
				case 41: EMIT_TRIANGLES(tiling13_2_, (configurationIndex * 6 + 2) * 18, 6); break;
				case 42: EMIT_TRIANGLES(tiling13_2_, (configurationIndex * 6 + 3) * 18, 6); break;
				case 43: EMIT_TRIANGLES(tiling13_2_, (configurationIndex * 6 + 4) * 18, 6); break;
				case 44: EMIT_TRIANGLES(tiling13_2_, (configurationIndex * 6 + 5) * 18, 6); break;

				case 45: EMIT_TRIANGLES(tiling13_1_, configurationIndex * 12, 4); break; // 13.1 (complement)

				default:
					break; // subconfig13 == -1: mathematically unreachable for a genuine case-13 cube
			}
			break;
		}

		case 14: // 4 triangles
		{
			EMIT_TRIANGLES(tiling14, configurationIndex * 12, 4);
			break;
		}

		default:
		{
			break;
		}
	}
}

/// *********************************************************************************************
/// Entry point -- one invocation per candidate cube.
/// *********************************************************************************************

void main()
{
	// 1. One invocation per candidate cube; extra invocations from the work-group rounding up
	//    past g_candidateCount do nothing.
	uint candidateIndex = gl_GlobalInvocationID.x;
	if (candidateIndex >= g_candidateCount)
	{
		return;
	}

	// 2. Process this invocation's candidate cube.
	processCube(g_candidateBases[candidateIndex].xyz, candidateIndex);
}

#version 460

/// *********************************************************************************************
/// extract_mtet.comp
///
/// GPU compute twin of the CPU "mtet" strategy (MarchingTetrahedraExtractor.cpp): one invocation
/// per candidate cube -- the SAME candidate set core::CandidateBases() builds on the CPU, same
/// dense value-grid sampling as extract_mc.comp/extract_mc33.comp (see
/// GpuIsoSurfaceExtractorCommon.h's file-level doc comment for why a dense grid, not a hash, is
/// this task's deliberate "simplest first cut"). Reuses voxel_common.glsl's CORNER/vertInterp --
/// the SAME 8-corner layout and interpolation formula the CPU "mc"/"mc33"/"mtet" extractors all
/// share.
///
/// Unlike "mc"/"mc33" (which resolve a cube's 8 corners directly against a 256-case table), this
/// shader first splits each cube into the 6 tetrahedra of kCubeTetrahedra below, all sharing the
/// cube's main diagonal from corner 0 to corner 6 -- transcribed VERBATIM from
/// MarchingTetrahedraExtractor.cpp's own kCubeTetrahedra -- then triangulates each tetrahedron
/// independently from its own 4 corner signs (TriangulateTetrahedron). A tetrahedron's natural
/// interpolant is purely AFFINE across its 4 corners (unlike a cube's TRILINEAR interpolant), so
/// its zero-set is always exactly one planar polygon per corner-sign configuration: no
/// asymptotic-decider/interior test is needed here, unlike "mc33" -- see
/// MarchingTetrahedraExtractor.cpp's file header for the full argument. EmitOutwardTriangle
/// resolves each triangle's winding dynamically, comparing its raw cross-product normal against
/// the tetrahedron's own inside-corners-mean -> outside-corners-mean direction -- mirrors the CPU
/// twin's own EmitOutwardTriangle idiom exactly, so (like the CPU) this needs no separate
/// hand-verified per-case winding table either.
///
/// Field representation, dense-grid sampling and the int-atomic emit contract (binding 0 = counter,
/// binding 1 = vertex slots, binding 2 = candidate bases, binding 3 = value grid) are IDENTICAL to
/// extract_mc.comp -- see that file's header comment for the full contract.
///
/// ORDER-KEY STRIDE: a cube's 6 tetrahedra each emit at most 2 triangles apiece (the
/// negativeCount==2 quadrilateral case in TriangulateTetrahedron below), so 6*2 = 12 triangles/cube
/// is the exact per-cube maximum -- the SAME maximum "mc33" has, so ORDER_KEY_STRIDE below reuses
/// that file's STRIDE=16 (strictly exceeds 12; see extract_mc33.comp's own ORDER_KEY_STRIDE comment
/// for the full "why too-small silently reconstructs the wrong order" contract).
/// triangleWithinCube is a SINGLE running counter threaded through all 6 TriangulateTetrahedron
/// calls in processCube() below (0..11 across the WHOLE cube, NEVER reset per tetrahedron) --
/// this mirrors the CPU's EmitCubeTriangles, which walks kCubeTetrahedra in order and appends every
/// tetrahedron's 0/1/2 triangles to the SAME output vector in that same tet-then-triangle sequence,
/// so the order key must count the same way for the post-readback sort
/// (GpuIsoSurfaceExtractorCommon::ReadbackRawTriangles) to reconstruct that exact CPU order.
///
/// WELD DISTANCE: the CPU "mtet" extractor does NOT weld at the plain params.weldFraction*cellSize
/// distance "mc"/"mc33" use -- it divides that by kWeldDistanceDivisor==32 ("mtet"'s denser
/// diagonal-cut vertices need a tighter tolerance; see MarchingTetrahedraExtractor.cpp's own
/// extensive comment on why). This shader itself only emits RAW (unwelded) triangles, so it is
/// unaffected by that constant -- but GpuMarchingTetrahedraExtractor::Extract() MUST weld with that
/// SAME reduced distance, or the GPU/CPU meshes will not match despite this shader being correct.
/// *********************************************************************************************

layout(local_size_x = 64) in;

#include "voxel_common.glsl" // CORNER, vertInterp (edgeTable/triTable unused here -- mtet has no case table)

// Keep in sync with kGpuUnresolvedFieldValue (GpuIsoSurfaceExtractorCommon.h).
#define UNRESOLVED_FIELD_VALUE 1e37

// See the file header's "ORDER-KEY STRIDE" section: a cube emits at most 12 triangles (6
// tetrahedra * 2 triangles), so STRIDE must strictly exceed 12; reuses mc33's value of 16.
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
/// The 6 tetrahedra that partition a cube, all sharing the main diagonal from corner 0 to corner 6
/// (CORNER order) -- transcribed VERBATIM from MarchingTetrahedraExtractor.cpp's kCubeTetrahedra;
/// flattened the same way marching_cubes_33_tables.glsl flattens its tables:
/// kCubeTetrahedra[tetrahedron*4 + corner] == the CPU's kCubeTetrahedra[tetrahedron][corner].
/// *********************************************************************************************

const int kCubeTetrahedra[24] = int[24](
	0, 5, 1, 6, // tetrahedron 0
	0, 1, 2, 6, // tetrahedron 1
	0, 2, 3, 6, // tetrahedron 2
	0, 3, 7, 6, // tetrahedron 3
	0, 7, 4, 6, // tetrahedron 4
	0, 4, 5, 6  // tetrahedron 5
);

/// *********************************************************************************************
/// Triangle emission -- reserves 3 vertex slots per triangle with an int atomicAdd (the SAME
/// int-atomic-counter contract every Gpu*Extractor shader uses; MoltenVK has no float atomics).
/// *********************************************************************************************

void EmitOneTriangle(vec3 vertexA, vec3 vertexB, vec3 vertexC, uint candidateIndex, uint triangleWithinCube)
{
	int slot = atomicAdd(g_triangleCount, 1);
	if (uint(slot) >= g_maxTriangles)
	{
		return;
	}

	float orderKey = uintBitsToFloat(candidateIndex * ORDER_KEY_STRIDE + triangleWithinCube);
	g_vertexSlots[slot * 3 + 0] = vec4(vertexA, orderKey);
	g_vertexSlots[slot * 3 + 1] = vec4(vertexB, orderKey);
	g_vertexSlots[slot * 3 + 2] = vec4(vertexC, orderKey);
}

// Emits the triangle (vertexA,vertexB,vertexC), flipping its winding first if needed so its raw
// cross-product face normal points along `outwardDirection` -- mirrors
// MarchingTetrahedraExtractor.cpp's EmitOutwardTriangle exactly (same flip test, same swapped
// corners on a flip). Advances `triangleWithinCube` by one AFTER emitting, so the caller's next
// triangle (this tetrahedron's second, or the next tetrahedron's first) gets the next order-key
// slot -- see the file header's "ORDER-KEY STRIDE" section for why this must be a single running
// counter across the whole cube, not reset per tetrahedron.
void EmitOutwardTriangle(vec3 vertexA, vec3 vertexB, vec3 vertexC, vec3 outwardDirection,
                          uint candidateIndex, inout uint triangleWithinCube)
{
	vec3 faceNormal = cross(vertexB - vertexA, vertexC - vertexA);
	if (dot(faceNormal, outwardDirection) < 0.0)
	{
		EmitOneTriangle(vertexA, vertexC, vertexB, candidateIndex, triangleWithinCube); // flip: swap two corners to reverse winding
	}
	else
	{
		EmitOneTriangle(vertexA, vertexB, vertexC, candidateIndex, triangleWithinCube);
	}
	triangleWithinCube++;
}

/// *********************************************************************************************
/// Per-tetrahedron triangulation -- mirrors MarchingTetrahedraExtractor.cpp's
/// TriangulateTetrahedron function-for-function: no trilinear ambiguity by construction (see file
/// header), the 4 corner signs alone fully determine the cut.
/// *********************************************************************************************

void TriangulateTetrahedron(vec3 cornerPosition[4], float cornerValue[4], uint candidateIndex,
                             inout uint triangleWithinCube)
{
	// 1. Split the 4 corners into negative ("inside") and positive ("outside") groups.
	int negativeCornerIndex[4];
	int positiveCornerIndex[4];
	int negativeCount = 0;
	int positiveCount = 0;
	for (int i = 0; i < 4; i++) {
		if (cornerValue[i] < 0.0)
		{
			negativeCornerIndex[negativeCount] = i;
			negativeCount++;
		}
		else
		{
			positiveCornerIndex[positiveCount] = i;
			positiveCount++;
		}
	}
	if (negativeCount == 0 || negativeCount == 4)
	{
		return; // whole tetrahedron on one side: no cut here
	}

	// 2. The outward direction for this tetrahedron: from the negative corners' centroid toward
	//    the positive corners' centroid, used to orient every triangle it emits.
	vec3 insideMean = vec3(0.0);
	for (int i = 0; i < negativeCount; i++) {
		insideMean += cornerPosition[negativeCornerIndex[i]];
	}
	insideMean /= float(negativeCount);

	vec3 outsideMean = vec3(0.0);
	for (int i = 0; i < positiveCount; i++) {
		outsideMean += cornerPosition[positiveCornerIndex[i]];
	}
	outsideMean /= float(positiveCount);

	vec3 outwardDirection = outsideMean - insideMean;

	// 3. A single "lone" corner (the minority sign) is cut off by one triangular plane through the
	//    three edges connecting it to the other three corners.
	if (negativeCount == 1 || negativeCount == 3)
	{
		int loneCornerIndex = (negativeCount == 1) ? negativeCornerIndex[0] : positiveCornerIndex[0];
		int otherCornerIndex[3];
		int otherCount = 0;
		for (int i = 0; i < 4; i++) {
			if (i != loneCornerIndex)
			{
				otherCornerIndex[otherCount] = i;
				otherCount++;
			}
		}
		vec3 cutVertexA = vertInterp(cornerPosition[loneCornerIndex], cornerPosition[otherCornerIndex[0]],
		                              cornerValue[loneCornerIndex], cornerValue[otherCornerIndex[0]]);
		vec3 cutVertexB = vertInterp(cornerPosition[loneCornerIndex], cornerPosition[otherCornerIndex[1]],
		                              cornerValue[loneCornerIndex], cornerValue[otherCornerIndex[1]]);
		vec3 cutVertexC = vertInterp(cornerPosition[loneCornerIndex], cornerPosition[otherCornerIndex[2]],
		                              cornerValue[loneCornerIndex], cornerValue[otherCornerIndex[2]]);
		EmitOutwardTriangle(cutVertexA, cutVertexB, cutVertexC, outwardDirection, candidateIndex, triangleWithinCube);
		return;
	}

	// 4. negativeCount == 2: the isosurface plane separates the two negative corners from the two
	//    positive corners, crossing exactly the four edges between one negative and one positive
	//    corner (the negative-negative and positive-positive edges are not cut). The resulting
	//    planar quadrilateral's vertices, in cyclic order, are the crossings of edges
	//    (negativeA,positiveA), (negativeA,positiveB), (negativeB,positiveB), (negativeB,positiveA);
	//    split it into 2 triangles on the diagonal between the 1st and 3rd crossings -- either
	//    diagonal is valid since the quadrilateral is exactly planar (no ambiguity). Mirrors the
	//    CPU's own comment on this case in full.
	int negativeCornerA = negativeCornerIndex[0];
	int negativeCornerB = negativeCornerIndex[1];
	int positiveCornerA = positiveCornerIndex[0];
	int positiveCornerB = positiveCornerIndex[1];
	vec3 quadrilateralVertex0 = vertInterp(cornerPosition[negativeCornerA], cornerPosition[positiveCornerA],
	                                        cornerValue[negativeCornerA], cornerValue[positiveCornerA]);
	vec3 quadrilateralVertex1 = vertInterp(cornerPosition[negativeCornerA], cornerPosition[positiveCornerB],
	                                        cornerValue[negativeCornerA], cornerValue[positiveCornerB]);
	vec3 quadrilateralVertex2 = vertInterp(cornerPosition[negativeCornerB], cornerPosition[positiveCornerB],
	                                        cornerValue[negativeCornerB], cornerValue[positiveCornerB]);
	vec3 quadrilateralVertex3 = vertInterp(cornerPosition[negativeCornerB], cornerPosition[positiveCornerA],
	                                        cornerValue[negativeCornerB], cornerValue[positiveCornerA]);
	EmitOutwardTriangle(quadrilateralVertex0, quadrilateralVertex1, quadrilateralVertex2, outwardDirection,
	                     candidateIndex, triangleWithinCube);
	EmitOutwardTriangle(quadrilateralVertex0, quadrilateralVertex2, quadrilateralVertex3, outwardDirection,
	                     candidateIndex, triangleWithinCube);
}

/// *********************************************************************************************
/// Per-cube Marching Tetrahedra -- mirrors MarchingTetrahedraExtractor.cpp's EmitCubeTriangles:
/// splits the cube into the 6 tetrahedra of kCubeTetrahedra and triangulates each independently.
/// *********************************************************************************************

void processCube(ivec3 cubeBase, uint candidateIndex)
{
	// 1. Sample all 8 corners; an unresolved corner means "skip this cube" (mirrors the CPU core's
	//    `complete` flag / extract_mc.comp's processCube step 1).
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

	// 3. Split into the 6 tetrahedra sharing the 0-6 diagonal and triangulate each in order;
	//    triangleWithinCube is ONE running counter across all 6 tetrahedra (see the file header's
	//    "ORDER-KEY STRIDE" section) -- NEVER reset between tetrahedra.
	uint triangleWithinCube = 0u;
	for (int tetrahedron = 0; tetrahedron < 6; tetrahedron++) {
		vec3  tetrahedronCornerPosition[4];
		float tetrahedronCornerValue[4];
		for (int i = 0; i < 4; i++) {
			int cornerIndex = kCubeTetrahedra[tetrahedron * 4 + i];
			tetrahedronCornerPosition[i] = cornerPosition[cornerIndex];
			tetrahedronCornerValue[i] = sdf[cornerIndex];
		}
		TriangulateTetrahedron(tetrahedronCornerPosition, tetrahedronCornerValue, candidateIndex, triangleWithinCube);
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

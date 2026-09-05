#version 450

#include "Realsense/Algorithm/Common.glsl"

#define TILE_X 32
#define TILE_Y 8
#define HALO   1
#define TILE_W (TILE_X + 2 * HALO)
#define TILE_H (TILE_Y + 2 * HALO)

layout(local_size_x = TILE_X, local_size_y = TILE_Y) in;

layout(push_constant) uniform PC
{
	int   g_width;
	int   g_height;

	float g_depthScale;         // metres per Z16 unit; 0.001 on D4xx by default

	float g_subpixelRms;
	float g_focalLengthPixels;
	float g_baselineMeters;

	float g_sameSurfaceSigmaMultiplier;

	float g_nearFadeStart;
	float g_nearFadeEnd;
	float g_farFadeStart;
	float g_farFadeEnd;

	float g_infraredFloor;      // below this the match had no signal to work with
	float g_infraredReference;  // a healthy return; at or above it the term is 1
	float g_infraredSaturation; // retro-reflectors and close gloss clip the sensor and lie
};

layout(set = 0, binding = 0) uniform usampler2D g_depth;              // R16_UINT, Z16
layout(std430, set = 0, binding = 1) writeonly buffer ScoreImage { ValidationMaskProperty g_properties[]; };



layout(std430, set = 0, binding = 2) buffer Counters { ValidationScoreCounters g_counters; };

#ifdef VALIDATION_SCORE_WITH_INFRARED
layout(set = 0, binding = 3) uniform usampler2D g_infrared;           // R8_UINT, Y8
#endif

// Depth tile with a one-pixel halo, already in metres. Out-of-image reads land
// as 0.0, which is the same "no measurement" sentinel the sensor uses, so the
// border scores lower by construction and needs no special case in the stencil.
shared float s_depth[TILE_H][TILE_W];

#ifdef VALIDATION_SCORE_WITH_COUNTERS
shared uint s_scored;
shared uint s_noMeasurement;
shared uint s_range;
shared uint s_infrared;
shared uint s_neighbour;
#endif

// sigma_z and the same-surface tolerance built from it live in Common.glsl: every normal estimator
// admits samples by the same rule, and two definitions of "same surface" in one module would let
// c_nb count one set of neighbours while a plane fit uses another.


/*
/// c_range = smoothstep(n0, n1, z) * (1 - smoothstep(f0, f1, z))
/// smoothstep(e0, e1, x) = t^2 * (3 - 2t),  t = clamp((x - e0) / (e1 - e0), 0, 1)
///
/// c_range   : range confidence, [0,1]
/// z         : depth [m]
/// n0, n1    : near fade start / end [m] -- below n0 the disparity search has nothing to match
/// f0, f1    : far fade start / end [m] -- beyond f0 sigma_z has outgrown the measurement
/// t^2(3-2t) : a fade, not a step -- neither limit is a cliff, and a hard edge would seam the
///             fused surface

c_range
   1 │            ┌─────────────────┐
     │           ╱                   ╲
     │          ╱                     ╲
     │         ╱                       ╲
   0 │────────┘                         └────────
     └────────┬───┬─────────────────┬───┬───────→  z
              n0  n1                f0  f1
           	근거리 페이드         원거리 페이드

- z < n0 → 0 (너무 가까워 disparity 탐색 범위 밖)
- n1 ≤ z ≤ f0 → 1 (센서의 스위트 스팟)
- z > f1 → 0 (σ_z가 측정값 자체보다 커짐)8
*/
float RangeConfidence(float depth)
{
	return smoothstep(g_nearFadeStart, g_nearFadeEnd, depth)
	     * (1.0 - smoothstep(g_farFadeStart, g_farFadeEnd, depth));
}

/// c_ir = 0                                  if I >= I_sat
///      = smoothstep(I_min, I_ref, I * z^2)  otherwise
///
/// c_ir    : infrared confidence, [0,1]. Constant 1 when the IR stream is not compiled in
/// I       : raw left-IR intensity [Y8 count, 0..255]
/// I * z^2 : intensity corrected for the projector's 1/z^2 falloff, leaving something
///           proportional to reflectance -- raw I is not comparable across a scene, since good
///           surface far away reads darker than poor surface close up
/// I_sat   : saturation cut -- a clipped pixel is unmeasured, not bright, and the disparity
///           fitted to a flat-topped patch is meaningless
/// I_min   : floor below which the match had no signal to work with
/// I_ref   : a healthy return; at or above it the term is 1
/// z       : depth [m]
float InfraredConfidence(ivec2 pixel, float depth)
{
#ifdef VALIDATION_SCORE_WITH_INFRARED
	float raw = float(texelFetch(g_infrared, pixel, 0).x);   // Y8 count, 0..255
	if (raw >= g_infraredSaturation) return 0.0;
	return smoothstep(g_infraredFloor, g_infraredReference, raw * depth * depth);
#else
	return 1.0;
#endif
}

/// c_nb = |N| / 8,   N = { p in 3x3(c) \ {c} : z_p > 0 and |z_p - z_c| < tau }
///
/// c_nb : neighbourhood support, [0,1]
/// N    : neighbours that carry a measurement AND sit on the centre's own surface. Validity
///        alone is not enough -- a flying pixel at a boundary has eight valid neighbours, half
///        of them half a metre behind it
/// c    : the centre pixel
/// z_p  : depth at neighbour p [m], read from the LDS tile rather than from memory
/// z_c  : depth at the centre [m]
/// tau  : same-surface tolerance [m], passed in as k * sigma_z -- derived from the sensor, tuned
///        by nobody
/// <    : strict, because step(tau, d) admits only d < tau. [H3]..[H5] in Pipeline use <=, so
///        the two definitions of "same surface" differ by one ULP
/// 8    : the full 3x3 minus the centre. A neighbour outside the image is 0 in the halo, so the
///        border scores lower by construction instead of by a special case
/// |N|  : summed branchlessly -- the loop unrolls and every lane does the same eight LDS reads
///        and the same arithmetic
float NeighbourConfidence(ivec2 local, float depth, float tolerance)
{
	int lx = local.x + HALO;
	int ly = local.y + HALO;

	float supporting = 0.0;
	for (int dy = -1; dy <= 1; ++dy)
	{
		for (int dx = -1; dx <= 1; ++dx)
		{
			if (dx == 0 && dy == 0) continue;   // resolved at compile time

			float neighbourDepth = s_depth[ly + dy][lx + dx];
			float hasMeasurement = step(1e-6, neighbourDepth);
			float sameSurface    = 1.0 - step(tolerance, abs(neighbourDepth - depth));
			supporting += hasMeasurement * sameSurface;
		}
	}
	return supporting * (1.0 / 8.0);
}

/// c = [z > 0] * c_range * c_ir * c_nb
/// z = Z * depthScale
/// tau = k * sigma_z(z)
///
/// c          : per-pixel confidence, [0,1]
/// [z > 0]    : the D4 VPU's own 1-bit verdict. It scores over forty parameters internally --
///              LRC check, second-peak ratio, texture -- and then publishes only the result, as
///              a zero. It is the most informative single bit a D4xx offers, because
///              RS2_STREAM_CONFIDENCE is L515 and will not open on this device
/// Z          : raw sample [Z16 sensor unit]
/// depthScale : metres per Z16 unit -- 0.001 on D4xx by default
/// k          : same-surface sigma multiplier; 3 covers 99.7% of the axial noise
/// *          : product, not sum -- the terms are vetoes, not votes. A pixel the VPU rejected is
///              worthless however bright its infrared return, and an average would carry it
///              through
void main()
{
	const uint threadCount  = TILE_X * TILE_Y;
	const uint tileElements = TILE_W * TILE_H;

	uint  linearLocal = gl_LocalInvocationIndex;
	ivec2 tileOrigin  = ivec2(gl_WorkGroupID.xy) * ivec2(TILE_X, TILE_Y) - ivec2(HALO);

#ifdef VALIDATION_SCORE_WITH_COUNTERS
	if (linearLocal == 0u)
	{
		s_scored        = 0u;
		s_noMeasurement = 0u;
		s_range         = 0u;
		s_infrared      = 0u;
		s_neighbour     = 0u;
	}
#endif

	// --- cooperative tile load: 340 samples across 256 lanes, 2 passes ------
	// Fetch Z16, scale to metres once, park it in LDS. The bounds test lives
	// here and nowhere else -- texelFetch is undefined out of range, and the
	// stencil below reads only the halo.
	for (uint i = linearLocal; i < tileElements; i += threadCount)
	{
		int   tx = int(i % uint(TILE_W));
		int   ty = int(i / uint(TILE_W));
		ivec2 p  = tileOrigin + ivec2(tx, ty);

		float metres = 0.0;
		if (p.x >= 0 && p.y >= 0 && p.x < g_width && p.y < g_height)
			metres = float(texelFetch(g_depth, p, 0).x) * g_depthScale;

		s_depth[ty][tx] = metres;
	}

	barrier();
	// -----------------------------------------------------------------------

	ivec2 local  = ivec2(gl_LocalInvocationID.xy);
	ivec2 global = ivec2(gl_GlobalInvocationID.xy);
	bool  inside = global.x < g_width && global.y < g_height;

	// The centre is already in the tile -- do not fetch it twice.
	float depth      = inside ? s_depth[local.y + HALO][local.x + HALO] : 0.0;
	bool  hasDepth   = depth > 0.0;
	float confidence = 0.0;

	float rangeConfidence     = 0.0;
	float infraredConfidence  = 0.0;
	float neighbourConfidence = 0.0;

	uint g_id = uint(global.y * g_width + global.x);

	if (hasDepth)
	{
		rangeConfidence    = RangeConfidence(depth);
		infraredConfidence = InfraredConfidence(global, depth);

		// The tolerance is derived from the sensor, not tuned. Shared with every normal estimator
		// through Common.glsl so "same surface" has one meaning in this module.
		float tolerance     = SameSurfaceTolerance(depth, g_subpixelRms, g_focalLengthPixels,
		                                           g_baselineMeters, g_sameSurfaceSigmaMultiplier,
		                                           g_depthScale);
		neighbourConfidence = NeighbourConfidence(local, depth, tolerance);

		// Product, not sum: the terms are vetoes, not votes. Every term is
		// already in [0,1], so no clamp is needed.
		confidence = rangeConfidence * infraredConfidence * neighbourConfidence;
	}

	// Guarded, and the ONLY writes this kernel makes. Unguarded, a padding lane of a workgroup that
	// runs off the right edge computes g_id = y * width + x with x >= width, which is not out of
	// range at all -- it is the NEXT ROW's pixel, owned by another workgroup that may already have
	// scored it. 848 is not a multiple of TILE_X, so the D435's native width hits this on every row.
	//
	// `valid` is [H2], the VPU's own verdict, decided here because this is the pass that reads Z16;
	// `emitted` belongs to the threshold pass and is cleared so a stale one cannot survive a
	// re-score at a bar that never runs.
	if (inside)
	{
		g_properties[g_id].valid   = hasDepth ? 1u : 0u;
		g_properties[g_id].emitted = 0u;
		g_properties[g_id].score   = confidence;
	}

#ifdef VALIDATION_SCORE_WITH_COUNTERS
	// Diagnostics are mutually exclusive and ordered by precedence, so the
	// five counters partition the image and sum to width*height.
	if (inside)
	{
		if      (!hasDepth)                  atomicAdd(s_noMeasurement, 1u);
		else if (rangeConfidence     <= 0.0) atomicAdd(s_range,         1u);
		else if (infraredConfidence  <= 0.0) atomicAdd(s_infrared,      1u);
		else if (neighbourConfidence <= 0.0) atomicAdd(s_neighbour,     1u);
		else                                 atomicAdd(s_scored,        1u);
	}

	barrier();

	// One flush per workgroup, not one per lane.
	if (linearLocal == 0u)
	{
		atomicAdd(g_counters.scoredPixels,             s_scored);
		atomicAdd(g_counters.zeroedByNoMeasurement,    s_noMeasurement);
		atomicAdd(g_counters.zeroedByRange,            s_range);
		atomicAdd(g_counters.zeroedByInfrared,         s_infrared);
		atomicAdd(g_counters.zeroedByNeighbourSupport, s_neighbour);
	}
#endif
}
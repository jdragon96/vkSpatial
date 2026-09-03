// Shared by every kernel in this directory. Mirrored by ValidationMask.h.
//
// ADD 4-BYTE SCALARS ONLY to either struct. std430 gives a vec3 member 16-byte alignment, so
// `{ int; vec3; }` is 32 bytes here while its obvious C++ mirror is 4 -- the buffer is then sized
// eight times too small and the kernel writes past it, with nothing pointing at the layout. A vec4
// is also safe; a vec3, a mat, or a double is not.

struct ValidationMaskProperty {
	uint valid;     // [H2] the pixel carries a depth measurement
	uint emitted;   // [H3..H6] the pixel produced a trustworthy point + normal
};

// Mirrors Pipeline::DepthFilterStats field for field, so a GPU run and a CPU run report the same
// numbers under the same names. Per cause, never summed: rejectedByRange counts PIXELS (it runs
// before back-projection) while the other two count candidate POINTS, which the forward guard has
// already thinned.
struct ValidationMaskCounters {
	uint emittedPoints;
	uint rejectedByRange;
	uint rejectedByNeighbourSupport;
	uint rejectedByIncidence;
};

/// Depth-jump tolerance at range `depth`, shared by [H1] [H3] [H4] [H5]. The four must agree or
/// "same surface" means something different in each.
///
/// Relative because stereo depth error grows as z^2/(f*B) (Keselman et al. eq. 2): one fixed
/// threshold over-rejects near the camera and under-rejects far from it. The absolute floor keeps
/// it from collapsing to nothing in the near field.
///
/// The thresholds are parameters rather than push-constant reads: a shared include cannot name a
/// PC member that every including kernel is not guaranteed to declare.
float DepthJumpTolerance(float depth, float relativeDepthJump, float minimumDepthJump)
{
	return max(minimumDepthJump, relativeDepthJump * depth);
}

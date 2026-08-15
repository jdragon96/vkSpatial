/// Shared by all three DenseRegionClassifier passes (clear, accumulate, classify). Include AFTER
/// #version, following voxel_common.glsl's precedent.
///
/// BlockRecord used to be duplicated verbatim in all three shaders with nothing keeping the copies
/// in step. The clear pass is the sharp case: it addresses one field BY NAME, so a field inserted
/// above that name in the other two -- but missed there -- would silently zero the wrong field
/// every frame, with no compile error and no test failure. One definition removes that failure
/// mode for the GLSL side; the C++ mirror in DenseRegionClassifier.h is a fourth copy that still
/// has to be edited by hand, and its static_assert on sizeof is what pins the two together.

/// Per-block statistics. Every field is a 4-byte scalar, so the std430 array stride is exactly
/// sizeof(BlockRecord) and the C++ mirror needs no padding.
///
/// The "Frame" fields hold THIS frame only and are zeroed by the clear pass at the head of every
/// frame; everything else accumulates for as long as the scene lives. Which half a quantity belongs
/// in is a decision, not a convenience: see the normal-coherence note below.
struct BlockRecord
{
	uint blockKey;              // packed block coordinate; EMPTY_KEY = empty slot
	uint pointCount;            // cumulative
	uint pointCountFrame;       // THIS frame -- denominator of the normal coherence
	uint coarseOccupied;        // cumulative sum of per-frame counts
	uint fineOccupied;          // cumulative sum of per-frame counts
	uint fineOccupiedFrame;     // THIS frame -- the absolute floor, and the sizing input
	uint fineOccupiedMax;       // max over frames -- the detail table sizing input
	int  sumNormalFrameX;       // THIS frame, fixed point x NORMAL_FIXED_POINT_SCALE
	int  sumNormalFrameY;
	int  sumNormalFrameZ;
};

/// Normals are summed as fixed-point integers because GLSL has no float atomicAdd on storage
/// buffers without an extension. Accumulate writes `int(component * SCALE)`, classify divides the
/// sum back down by the same constant -- so the two must read it from here, not from two separate
/// literals that can drift apart.
///
/// The sum is PER FRAME, and that is the fix for this branch's one critical review finding. Summed
/// cumulatively it overflowed int32, and coherent normals -- the flat wall -- maximise the sum, so
/// the flat wall overflowed FIRST: wrap-around drove coherence down through the threshold, the veto
/// that exists to keep dense flat walls out inverted into the condition that let them in, and the
/// latch made it permanent (measured: frame 56 of a 4096-point plane; ~4 frames on real scan
/// densities). Per-frame is not merely a smaller number to overflow -- it is what spec section 5
/// already decided for the occupancy half, for the same reason: averaging normals ACROSS viewpoints
/// does not give a more confident estimate, it gives a mixture of differently-misregistered ones,
/// and inter-frame registration error exceeds the detail voxel.
///
/// The remaining bound is one frame, one block: 2^31 / 10000 = 214,748 points before int32 wraps.
/// A 0.32 m block receiving 214k points in a SINGLE frame is about 3.5x the worst real scan density
/// measured for this component, so there is no counter for it.
const float NORMAL_FIXED_POINT_SCALE = 10000.0;

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
struct BlockRecord
{
	uint blockKey;              // packed block coordinate; EMPTY_KEY = empty slot
	uint pointCount;            // cumulative
	uint coarseOccupied;        // cumulative sum of per-frame counts
	uint fineOccupied;          // cumulative sum of per-frame counts
	uint fineOccupiedFrame;     // THIS frame's count; zeroed by the clear pass each frame
	uint fineOccupiedMax;       // max over frames -- the detail table sizing input
	int  sumNormalX;            // fixed point, x NORMAL_FIXED_POINT_SCALE
	int  sumNormalY;
	int  sumNormalZ;
};

/// Normals are summed as fixed-point integers because GLSL has no float atomicAdd on storage
/// buffers without an extension. Accumulate writes `int(component * SCALE)`, classify divides the
/// sum back down by the same constant -- so the two must read it from here, not from two separate
/// literals that can drift apart.
const float NORMAL_FIXED_POINT_SCALE = 10000.0;

// Shared by every kernel in this directory. Mirrored by Realsense/RealSenseTypes.h.
//
// No #version here: this file is #included, and a directive that must come first in the shader
// cannot appear in the middle of one. Every kernel declares its own.
//
// ADD 4-BYTE SCALARS ONLY to either struct. std430 gives a vec3 member 16-byte alignment, so
// `{ int; vec3; }` is 32 bytes here while its obvious C++ mirror is 4 -- the buffer is then sized
// eight times too small and the kernel writes past it, with nothing pointing at the layout.

struct ValidationScoreCounters {
	uint scoredPixels;              // score > 0
	uint zeroedByNoMeasurement;     // the VPU published 0 here
	uint zeroedByRange;             // outside the fade band entirely
	uint zeroedByInfrared;          // no signal, or clipped
	uint zeroedByNeighbourSupport;  // not one neighbour on the same surface
};

struct ValidationMaskProperty {
    uint valid;
    uint emitted;
    float score;
};

struct DownSampleCounters {
	uint insertFailures;      // the probe budget ran out
	uint outOfPackableRange;  // the voxel sits outside the 11/11/10-bit key
};

// One hash slot. Both members are cleared to 0xFFFFFFFF, which is why EMPTY_KEY and NO_WINNER share
// that value -- one vkCmdFillBuffer resets the whole table.
struct DownSampleSlot {
	uint key;
	uint winner;  // lowest pixel index seen for this voxel, reduced with atomicMin
};

struct NormalEstimationCounters {
	uint outOfDomain;  // the stencil hung off the edge of the image
	uint noSupport;    // the stencil fitted but found too few same-surface samples
};

/// sigma_z = s * z^2 / (f * B)
///
/// sigma_z : axial depth noise of one sample, one standard deviation [m]
/// s       : subpixel disparity matching error, RMS [px] -- 0.08 with texture, 0.25 without
/// z       : depth [m]
/// f       : focal length [px]
/// B       : stereo baseline [m] -- D435 0.05, D455 0.095
/// z^2     : exact for active stereo, not a fit -- z = f*B/d, so a disparity error e propagates
///           as z^2*e/(f*B)
/// 0       : returned when f*B <= 0. That makes every tolerance 0, which scores the WHOLE frame
///           zero, so the C++ side must reject such a configuration rather than lean on this
///
/// The sensor constants are parameters rather than push-constant reads: a shared include cannot
/// name a PC member that every including kernel is guaranteed to declare.
float AxialNoiseSigma(float depth, float subpixelRms, float focalLengthPixels, float baselineMeters)
{
	float denominator = focalLengthPixels * baselineMeters;
	if (denominator <= 0.0) return 0.0;
	return subpixelRms * depth * depth / denominator;
}

/// tau = k * sigma_z(z)
///
/// The ONE definition of "same surface" in this module. The score kernel's c_nb term and every
/// normal estimator go through it, so a neighbour counted as support and a sample admitted to a
/// plane fit cannot come to mean two different things.
float SameSurfaceTolerance(float depth, float subpixelRms, float focalLengthPixels,
                           float baselineMeters, float sigmaMultiplier)
{
	return sigmaMultiplier * AxialNoiseSigma(depth, subpixelRms, focalLengthPixels, baselineMeters);
}

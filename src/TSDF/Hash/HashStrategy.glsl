/// *********************************************
/// Hash strategy dispatcher
///
/// The kernels include only this file. Which fragment it pulls in is decided by a preprocessor
/// definition the C++ side passes through ComputePipeline::Define.
///
/// Why a dispatcher instead of `#include HASH_STRATEGY_INCLUDE`: glslang does NOT macro-expand
/// #include ("must be followed by a header name"), so the include name has to be a literal.
/// This keeps each fragment a real file that editors and grep can follow.
///
/// The fragment includes below use the full src-relative path (not a bare filename): ComputePipeline
/// resolves an #include against a fixed root list computed once for the top-level compiled file --
/// [that file's own folder, VKBVH_SHADER_DIR, VKBVH_SRC_DIR] -- not against THIS file's folder, even
/// though this file is itself reached via #include. A bare "LinearProbe.glsl" would only resolve if
/// one of those three roots happened to contain it, which none do; the full path resolves through
/// VKBVH_SRC_DIR the same way the kernels' own `#include "TSDF/Hash/HashStrategy.glsl"` does.
/// *********************************************

// Probe accounting. Off unless the caller defines HASH_PROBE_STATS, so the shipping kernel pays
// nothing: without it countProbes() compiles away and no extra binding is required.
#ifdef HASH_PROBE_STATS
layout(std430, set = 0, binding = 9) buffer ProbeStats
{
	uint g_probeSlotTotal;   // slots examined, summed over every lookup
	uint g_probeQueryCount;  // lookups that reported
	uint g_probeSlotMax;     // worst single lookup
};
void countProbes(uint slotsExamined)
{
	atomicAdd(g_probeSlotTotal, slotsExamined);
	atomicAdd(g_probeQueryCount, 1u);
	atomicMax(g_probeSlotMax, slotsExamined);
}
#else
void countProbes(uint slotsExamined) {}
#endif

#define HASH_INSERT_FAILED 0xFFFFFFFFu
#define HASH_NOT_FOUND     0xFFFFFFFFu

#if defined(HASH_BUCKETED)
#include "TSDF/Hash/Bucketed.glsl"
#else
#include "TSDF/Hash/LinearProbe.glsl"
#endif

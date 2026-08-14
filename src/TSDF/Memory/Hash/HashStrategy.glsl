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
/// VKBVH_SRC_DIR the same way the kernels' own `#include "TSDF/Memory/Hash/HashStrategy.glsl"` does.
/// *********************************************

#define HASH_INSERT_FAILED 0xFFFFFFFFu
#define HASH_NOT_FOUND     0xFFFFFFFFu

#if defined(HASH_BUCKETED)
#include "TSDF/Memory/Hash/Bucketed.glsl"
#else
#include "TSDF/Memory/Hash/LinearProbe.glsl"
#endif

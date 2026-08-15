#version 450

// Test fixture for ComputePipeline::Define and the compile cache key. local_size_x is driven
// by a preprocessor definition, so a caller can observe through GetLocalSize() which definition
// set produced the module. If the cache were keyed on the path alone, a second build with
// different definitions would hand back the first build's module and report the wrong size.
#ifndef PROBE_LOCAL_SIZE
#define PROBE_LOCAL_SIZE 1
#endif

layout(local_size_x = PROBE_LOCAL_SIZE) in;
layout(std430, set = 0, binding = 0) buffer Out { uint g_out[]; };

void main() { g_out[0] = uint(PROBE_LOCAL_SIZE); }

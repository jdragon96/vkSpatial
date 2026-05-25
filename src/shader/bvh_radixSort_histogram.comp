#version 460
#extension GL_GOOGLE_include_directive: enable
#include "bvh_common.glsl"

// Phase 1: 워크그룹별 로컬 히스토그램 집계
// g_hist[digit * numWGs + wgID] = 이 워크그룹에서 해당 digit 원소 수

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint g_count;
    uint g_shift;   // 현재 패스의 비트 시프트 (0, 4, 8, ..., 28)
};

layout(std430, set = 0, binding = 0) readonly buffer Input   { MortonCode g_input[]; };
layout(std430, set = 0, binding = 1)          buffer HistOut { uint       g_hist[];  };

shared uint s_hist[16];

void main() {
    uint gid     = gl_GlobalInvocationID.x;
    uint localID = gl_LocalInvocationID.x;
    uint wgID    = gl_WorkGroupID.x;
    uint numWGs  = gl_NumWorkGroups.x;

    // 16: 4비트씩 값을 비교할 것이기 때문임
    if (localID < 16) s_hist[localID] = 0;
    barrier();

    // 각 스레드가 자신의 원소 digit 을 로컬 히스토그램에 집계
    if (gid < g_count) {
        uint digit = (g_input[gid].code >> g_shift) & 0xFu;
        atomicAdd(s_hist[digit], 1);
    }
    barrier();

    // 로컬 → 전역 히스토그램에 기록
    if (localID < 16)
        g_hist[localID * numWGs + wgID] = s_hist[localID];
}

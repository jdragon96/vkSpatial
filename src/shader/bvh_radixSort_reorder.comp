#version 460
#extension GL_GOOGLE_include_directive: enable
#include "bvh_common.glsl"

layout(local_size_x = 256) in;

#define WG_SIZE 256

layout(push_constant) uniform PC {
    uint g_count;
    uint g_shift;
};

layout(std430, set = 0, binding = 0) readonly buffer Input  { MortonCode g_input[];  };
layout(std430, set = 0, binding = 1)          buffer Hist   { uint       g_hist[];   }; // prefix sum
layout(std430, set = 0, binding = 2) writeonly buffer Output { MortonCode g_output[]; };

// 이 워크그룹의 digit별 전역 시작 위치
shared uint       s_base[16];

// 이 워크그룹의 원소를 원래 순서대로 로드 (stable sort를 위해 필수)
shared MortonCode s_elems[WG_SIZE];

void main() {
    uint gid     = gl_GlobalInvocationID.x;
    uint localID = gl_LocalInvocationID.x;
    uint wgID    = gl_WorkGroupID.x;
    uint numWGs  = gl_NumWorkGroups.x;

    // ── 이 WG의 digit별 전역 시작 위치 로드 ──────────────────────────────────
    if (localID < 16)
        s_base[localID] = g_hist[localID * numWGs + wgID];

    // ── 원소를 공유 메모리에 로드 (gid 순서 == localID 순서 보장) ───────────
    s_elems[localID] = (gid < g_count) ? g_input[gid] : MortonCode(0u, 0u);
    barrier();

    if (gid >= g_count) return;

    uint my_digit = (s_elems[localID].code >> g_shift) & 0xFu;

    // ── 로컬 rank: 이 WG 안에서 내 앞에 동일 digit 원소 수 ──────────────────
    // 이 카운팅이 stable sort를 보장함 (작은 localID = 원래 배열에서 앞)
    uint local_rank = 0;
    for (uint j = 0; j < localID; j++) {
        uint globalJ = wgID * uint(WG_SIZE) + j;
        if (globalJ < g_count) {
            uint other_digit = (s_elems[j].code >> g_shift) & 0xFu;
            if (other_digit == my_digit) local_rank++;
        }
    }

    // ── 최종 위치: 전역 시작 + 로컬 rank ────────────────────────────────────
    uint pos = s_base[my_digit] + local_rank;
    g_output[pos] = s_elems[localID];
}

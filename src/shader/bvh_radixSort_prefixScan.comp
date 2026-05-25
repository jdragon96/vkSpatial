#version 460

// Phase 2: 전역 Exclusive Prefix Sum
// g_hist[digit][wg] = count  →  g_hist[digit][wg] = 이 (digit, wg) 쌍이 출력에 쓸 시작 위치

layout(local_size_x = 1) in;   // 순차 스캔 (히스토그램 = 16 × numWGs, 보통 수천 개)

layout(push_constant) uniform PC {
    uint g_numWGs;
};

layout(std430, set = 0, binding = 0) buffer Hist { uint g_hist[]; };

void main() {
    uint total = 0;

    // digit 순서 우선, workgroup 순서로 순차 스캔
    // 이 순서가 scatter 단계에서 digit 별 출력 위치 연속성 보장
    for (uint digit = 0; digit < 16; digit++) {
        for (uint wg = 0; wg < g_numWGs; wg++) {
            uint idx   = digit * g_numWGs + wg;
            uint count = g_hist[idx];
            g_hist[idx] = total;   // exclusive prefix sum
            total += count;
        }
    }
}

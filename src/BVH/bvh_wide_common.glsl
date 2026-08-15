#ifndef VKBVH_WIDE_COMMON_GLSL
#define VKBVH_WIDE_COMMON_GLSL

#include "bvh_common.glsl"

#define WIDE_WIDTH 8u

struct SceneBounds {
    uint minX;
    uint minY;
    uint minZ;
    uint maxX;
    uint maxY;
    uint maxZ;
};

struct BinaryRange {
    uint firstPrimitive;
    uint primitiveCount;
    uint visitationCount;
    uint reserved;
};

struct WideNode {
    float originX;
    float originY;
    float originZ;
    float scaleX;
    float scaleY;
    float scaleZ;
    uint child[8];
    uint qBounds[12];
    uint childCount;
    uint leafMask;
};

struct LeafRange {
    uint firstPrimitive;
    uint primitiveCount;
};

struct WideBuildState {
    uint nodeCount;
    uint leafCount;
    uint status;
    uint reserved;
};

/*
float를 uint 값으로 비교하여, 모든 경우에 정렬 가능하도록 한다.

1. 음수는 bit를 반대로 뒤집어, 거꾸로 값이 인식되게 함
2. 양수는 sign bit를 1로 바꿔, 음수보다 반드시 크도록 설정
*/
uint floatToOrdered(float value) {
    uint bits = floatBitsToUint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : bits ^ 0x80000000u;
}

float orderedToFloat(uint value) {
    uint bits = (value & 0x80000000u) != 0u
                    ? value ^ 0x80000000u
                    : ~value;
    return uintBitsToFloat(bits);
}

#endif

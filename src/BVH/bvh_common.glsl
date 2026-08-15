#ifndef VKBVH_COMMON_GLSL
#define VKBVH_COMMON_GLSL

#define INVALID_POINTER 0x0

struct MortonCode {
    uint code;
    uint index;
};

struct Primitive {
    uint index;
    float aabbMinX;
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
};

struct Node {
    int left;
    int right;
    uint primitiveIdx;
    float aabbMinX;
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
};

struct LBVHConstructionInfo {
    uint parent;
    int visitationCount;
};

#endif
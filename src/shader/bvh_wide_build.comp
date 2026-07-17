#version 460
#extension GL_GOOGLE_include_directive: enable

#include "bvh_wide_common.glsl"

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint g_binaryNodeCount;
    uint g_maxLeafPrimitives;
};

layout(std430, set = 0, binding = 0) readonly buffer BinaryNodes {
    Node g_binaryNodes[];
};

layout(std430, set = 0, binding = 1) readonly buffer Ranges {
    BinaryRange g_ranges[];
};

layout(std430, set = 0, binding = 2) buffer Queue {
    uint g_queue[];
};

layout(std430, set = 0, binding = 3) buffer State {
    WideBuildState g_state;
};

layout(std430, set = 0, binding = 4) writeonly buffer WideNodes {
    WideNode g_wideNodes[];
};

layout(std430, set = 0, binding = 5) writeonly buffer Leaves {
    LeafRange g_leaves[];
};

shared uint s_batchBegin;
shared uint s_batchEnd;
shared uint s_finished;

bool isBinaryLeaf(uint nodeIndex) {
    return g_binaryNodes[nodeIndex].left == INVALID_POINTER;
}

float nodePriority(uint nodeIndex) {
    Node node = g_binaryNodes[nodeIndex];
    vec3 extent = max(
        vec3(node.aabbMaxX, node.aabbMaxY, node.aabbMaxZ) -
        vec3(node.aabbMinX, node.aabbMinY, node.aabbMinZ),
        vec3(0.0));
    float area = 2.0 * (
        extent.x * extent.y +
        extent.y * extent.z +
        extent.z * extent.x);
    return area * float(g_ranges[nodeIndex].primitiveCount);
}

float nextPositiveFloat(float value) {
    return value > 0.0
        ? uintBitsToFloat(floatBitsToUint(value) + 1u)
        : value;
}

void setPackedByte(
        inout WideNode node, uint plane, uint slot, uint value) {
    uint wordIndex = plane * 2u + slot / 4u;
    uint shift = (slot & 3u) * 8u;
    node.qBounds[wordIndex] |= (value & 0xFFu) << shift;
}

float axisMin(Node node, uint axis) {
    if (axis == 0u)
        return node.aabbMinX;
    if (axis == 1u)
        return node.aabbMinY;
    return node.aabbMinZ;
}

float axisMax(Node node, uint axis) {
    if (axis == 0u)
        return node.aabbMaxX;
    if (axis == 1u)
        return node.aabbMaxY;
    return node.aabbMaxZ;
}

void encodeChildBounds(
        inout WideNode outputNode,
        Node root,
        Node child,
        uint slot) {
    vec3 rootMin = vec3(root.aabbMinX, root.aabbMinY, root.aabbMinZ);
    vec3 rootMax = vec3(root.aabbMaxX, root.aabbMaxY, root.aabbMaxZ);
    vec3 extent = max(rootMax - rootMin, vec3(0.0));
    vec3 scale = vec3(
        nextPositiveFloat(extent.x / 253.0),
        nextPositiveFloat(extent.y / 253.0),
        nextPositiveFloat(extent.z / 253.0));
    vec3 origin = rootMin - scale;

    outputNode.originX = origin.x;
    outputNode.originY = origin.y;
    outputNode.originZ = origin.z;
    outputNode.scaleX = scale.x;
    outputNode.scaleY = scale.y;
    outputNode.scaleZ = scale.z;

    for (uint axis = 0u; axis < 3u; ++axis) {
        uint qMin = 0u;
        uint qMax = 0u;
        float axisScale = scale[axis];
        if (axisScale > 0.0) {
            float normalizedMin =
                (axisMin(child, axis) - origin[axis]) / axisScale;
            float normalizedMax =
                (axisMax(child, axis) - origin[axis]) / axisScale;
            qMin = uint(clamp(floor(normalizedMin) - 1.0, 0.0, 255.0));
            qMax = uint(clamp(ceil(normalizedMax) + 1.0, 0.0, 255.0));
        }
        setPackedByte(outputNode, axis, slot, qMin);
        setPackedByte(outputNode, axis + 3u, slot, qMax);
    }
}

void buildWideNode(uint wideIndex) {
    uint rootIndex = g_queue[wideIndex];
    uint frontier[8];
    uint frontierCount = 1u;
    frontier[0] = rootIndex;

    while (frontierCount < WIDE_WIDTH) {
        uint bestPosition = WIDE_WIDTH;
        float bestPriority = -1.0;

        for (uint i = 0u; i < frontierCount; ++i) {
            uint candidate = frontier[i];
            if (isBinaryLeaf(candidate) ||
                g_ranges[candidate].primitiveCount <= g_maxLeafPrimitives)
                continue;

            float priority = nodePriority(candidate);
            if (priority > bestPriority) {
                bestPriority = priority;
                bestPosition = i;
            }
        }

        if (bestPosition == WIDE_WIDTH)
            break;

        Node expanded = g_binaryNodes[frontier[bestPosition]];
        for (uint i = frontierCount; i > bestPosition + 1u; --i)
            frontier[i] = frontier[i - 1u];
        frontier[bestPosition] = uint(expanded.left);
        frontier[bestPosition + 1u] = uint(expanded.right);
        ++frontierCount;
    }

    WideNode outputNode;
    outputNode.originX = 0.0;
    outputNode.originY = 0.0;
    outputNode.originZ = 0.0;
    outputNode.scaleX = 0.0;
    outputNode.scaleY = 0.0;
    outputNode.scaleZ = 0.0;
    outputNode.childCount = frontierCount;
    outputNode.leafMask = 0u;
    for (uint i = 0u; i < 8u; ++i)
        outputNode.child[i] = 0u;
    for (uint i = 0u; i < 12u; ++i)
        outputNode.qBounds[i] = 0u;

    Node root = g_binaryNodes[rootIndex];
    for (uint slot = 0u; slot < frontierCount; ++slot) {
        uint childIndex = frontier[slot];
        Node child = g_binaryNodes[childIndex];
        BinaryRange range = g_ranges[childIndex];
        bool makeLeaf =
            isBinaryLeaf(childIndex) ||
            range.primitiveCount <= g_maxLeafPrimitives;

        if (makeLeaf) {
            uint leafIndex = atomicAdd(g_state.leafCount, 1u);
            if (leafIndex >= g_binaryNodeCount) {
                atomicOr(g_state.status, 2u);
                continue;
            }
            g_leaves[leafIndex] =
                LeafRange(range.firstPrimitive, range.primitiveCount);
            outputNode.child[slot] = leafIndex;
            outputNode.leafMask |= 1u << slot;
        } else {
            uint childWideIndex = atomicAdd(g_state.nodeCount, 1u);
            if (childWideIndex >= g_binaryNodeCount) {
                atomicOr(g_state.status, 1u);
                continue;
            }
            g_queue[childWideIndex] = childIndex;
            outputNode.child[slot] = childWideIndex;
        }

        encodeChildBounds(outputNode, root, child, slot);
    }

    g_wideNodes[wideIndex] = outputNode;
}

void main() {
    uint lane = gl_LocalInvocationID.x;
    if (lane == 0u)
        s_batchBegin = 0u;
    barrier();

    while (true) {
        if (lane == 0u) {
            memoryBarrierBuffer();
            uint currentCount = g_state.nodeCount;
            s_finished = s_batchBegin >= currentCount ? 1u : 0u;
            s_batchEnd = min(s_batchBegin + gl_WorkGroupSize.x, currentCount);
        }
        barrier();

        if (s_finished != 0u)
            break;

        uint wideIndex = s_batchBegin + lane;
        if (wideIndex < s_batchEnd)
            buildWideNode(wideIndex);

        memoryBarrierBuffer();
        barrier();
        if (lane == 0u)
            s_batchBegin = s_batchEnd;
        barrier();
    }
}

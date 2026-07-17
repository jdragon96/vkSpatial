#include "utilities/SimpleResource.h"

Primitives SimpleResource::CreateCube(float size) {
    Primitives cube;

    float halfSize = size * 0.5f;

    // Define the vertices of the cube
    cube.vertices = {
            {-halfSize, -halfSize, -halfSize},
            {halfSize, -halfSize, -halfSize},
            {halfSize, halfSize, -halfSize},
            {-halfSize, halfSize, -halfSize},
            {-halfSize, -halfSize, halfSize},
            {halfSize, -halfSize, halfSize},
            {halfSize, halfSize, halfSize},
            {-halfSize, halfSize, halfSize}};

    // Define the normals for each face of the cube
    cube.normals = {
            {0.0f, 0.0f, -1.0f}, // Back face
            {0.0f, 0.0f, 1.0f},  // Front face
            {-1.0f, 0.0f, 0.0f}, // Left face
            {1.0f, 0.0f, 0.0f},  // Right face
            {0.0f, -1.0f, 0.0f}, // Bottom face
            {0.0f, 1.0f, 0.0f}   // Top face
    };

    // Define colors for each vertex (optional)
    cube.colors = {
            {1.0f, 0.0f, 0.0f}, // Red
            {0.0f, 1.0f, 0.0f}, // Green
            {0.0f, 0.0f, 1.0f}, // Blue
            {1.0f, 1.0f, 1.0f}, // White
            {1.0f, 1.0f, 1.0f}, // White
            {1.0f, 1.0f, 1.0f}, // White
            {1.0f, 1.0f, 1.0f}, // White
            {1.0f, 1.0f, 1.0f}  // White
    };

    // Define the indices for the cube (two triangles per face)
    cube.indices = {
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            0, 4, 7, 7, 3, 0,
            1, 5, 6, 6, 2, 1,
            3, 7, 6, 6, 2, 3,
            0, 4, 5, 5, 1, 0};

    return cube;
}
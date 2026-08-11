#version 450 core

/// Vertex stage for EdgeOverlayPass. Passes the per-vertex highlight color straight through to
/// the fragment stage -- no lighting, no world-space output, unlike IsosurfaceMesh.vert.glsl --
/// because a connectivity-defect overlay needs to read as an EXACT color (red = non-manifold,
/// yellow = boundary; see isosurface_viewer.cpp's rebuild()/BuildEdgeOverlayGroup), not a shaded
/// one.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 vColor;

layout(push_constant) uniform TransformPushConstants
{
	mat4 modelViewProjection;
} pushConstants;

void main()
{
	// 1. Pass the highlight color through unmodified.
	vColor = inColor;

	// 2. Clip-space position.
	gl_Position = pushConstants.modelViewProjection * vec4(inPosition, 1.0);
}

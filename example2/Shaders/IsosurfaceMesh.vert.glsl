#version 450 core

/// Vertex stage for IsosurfaceMeshPass. Outputs WORLD-space position/normal (not view-space) so
/// the fragment stage can shade against a world-space light/camera with no per-space mismatch --
/// see IsosurfaceMesh.frag.glsl and IsosurfaceMeshPass.cpp for why.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 vWorldPosition;
layout(location = 1) out vec3 vWorldNormal;
layout(location = 2) out vec3 vColor;

layout(push_constant) uniform TransformPushConstants
{
	mat4 modelViewProjection;
	mat4 model;
} pushConstants;

void main()
{
	// 1. World-space position + normal, for lighting in the fragment stage.
	vec4 worldPosition = pushConstants.model * vec4(inPosition, 1.0);
	vWorldPosition = worldPosition.xyz;
	vWorldNormal = normalize(mat3(pushConstants.model) * inNormal);
	vColor = inColor;

	// 2. Clip-space position.
	gl_Position = pushConstants.modelViewProjection * vec4(inPosition, 1.0);
}

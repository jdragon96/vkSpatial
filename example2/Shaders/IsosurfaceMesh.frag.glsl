#version 450 core

/// Fragment stage for IsosurfaceMeshPass: Blinn-Phong shading (ambient + diffuse + specular),
/// same lighting MODEL as BlinnPhong.frag.glsl, but evaluated entirely in WORLD space so the
/// light position, camera position and surface position are always in the same frame (the
/// BlinnPhong.frag.glsl this is adapted from mixes a view-space light with a world-space camera
/// constant against a view-space surface position -- harmless there because the two demo shapes
/// never separate enough to expose it, but this viewer's job IS to show subtle shading
/// differences between extractors, so it is worth avoiding).

layout(location = 0) in vec3 vWorldPosition;
layout(location = 1) in vec3 vWorldNormal;
layout(location = 2) in vec3 vColor;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform LightingPushConstants
{
	vec3  lightPosition;
	float shininess;
	vec3  lightColor;
	float specularStrength;
	vec3  cameraPosition;
	float padding;
} pushConstants;

// Extracted into a named function per house style: ambient + Blinn-Phong diffuse/specular.
vec3 EvaluateBlinnPhong(vec3 baseColor, vec3 normal, vec3 surfacePosition)
{
	vec3 unitNormal       = normalize(normal);
	vec3 surfaceToLight   = normalize(pushConstants.lightPosition - surfacePosition);
	vec3 surfaceToCamera  = normalize(pushConstants.cameraPosition - surfacePosition);
	vec3 halfwayDirection = normalize(surfaceToLight + surfaceToCamera);

	float diffuseTerm  = max(dot(unitNormal, surfaceToLight), 0.0);
	float specularTerm = pow(max(dot(unitNormal, halfwayDirection), 0.0), pushConstants.shininess)
	                      * pushConstants.specularStrength;
	vec3  ambientTerm  = baseColor * 0.12;

	return ambientTerm + baseColor * diffuseTerm * pushConstants.lightColor
	       + specularTerm * pushConstants.lightColor;
}

void main()
{
	// 1. Shade.
	vec3 shaded = EvaluateBlinnPhong(vColor, vWorldNormal, vWorldPosition);

	// 2. Reinhard tonemap + 2.2 gamma (matches BlinnPhong.frag.glsl) so specular highlights don't
	//    clip to flat white.
	shaded = shaded / (shaded + vec3(1.0));
	shaded = pow(shaded, vec3(1.0 / 2.2));
	outColor = vec4(shaded, 1.0);
}

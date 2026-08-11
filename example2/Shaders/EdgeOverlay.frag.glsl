#version 450 core

/// Fragment stage for EdgeOverlayPass: flat (unlit) output of the per-vertex highlight color --
/// deliberately skips the Blinn-Phong shading / tonemap-and-gamma pass IsosurfaceMesh.frag.glsl
/// uses, so the overlay's red/yellow read as pure, unambiguous red/yellow regardless of the
/// scene's lighting or view angle (see EdgeOverlay.vert.glsl for the matching vertex-stage
/// reasoning).

layout(location = 0) in vec3 vColor;

layout(location = 0) out vec4 outColor;

void main()
{
	// 1. Output the highlight color unmodified, fully opaque.
	outColor = vec4(vColor, 1.0);
}

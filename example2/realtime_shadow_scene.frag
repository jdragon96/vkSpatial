#version 450

layout(set = 0, binding = 0) uniform sampler2D shadowMap;

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec3 vColor;
layout(location = 3) in vec4 vLightClip;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    mat4 lightViewProj;
    vec4 lightDir;
} pc;

float shadowFactor(vec3 normal, vec3 lightToSceneDir) {
    vec3 proj = vLightClip.xyz / vLightClip.w;
    vec2 uv = proj.xy * 0.5 + 0.5;
    float currentDepth = proj.z;

    if (uv.x < 0.0 || uv.x > 1.0 ||
        uv.y < 0.0 || uv.y > 1.0 ||
        currentDepth < 0.0 || currentDepth > 1.0) {
        return 1.0;
    }

    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    float ndotl = max(dot(normal, -lightToSceneDir), 0.0);
    float bias = max(0.0025 * (1.0 - ndotl), 0.0007);

    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float closestDepth = texture(shadowMap, uv + vec2(x, y) * texelSize).r;
            visibility += currentDepth - bias <= closestDepth ? 1.0 : 0.28;
        }
    }
    return visibility / 9.0;
}

void main() {
    vec3 n = normalize(vNormal);
    vec3 lightToSceneDir = normalize(pc.lightDir.xyz);
    vec3 toLight = -lightToSceneDir;

    float diffuse = max(dot(n, toLight), 0.0);
    float visibility = shadowFactor(n, lightToSceneDir);
    vec3 ambient = vColor * 0.18;
    vec3 lit = vColor * diffuse * visibility * 0.92;

    vec3 viewDir = normalize(vec3(0.0, 3.0, 7.0) - vWorldPos);
    vec3 halfDir = normalize(toLight + viewDir);
    float specular = pow(max(dot(n, halfDir), 0.0), 48.0) * visibility * 0.18;

    outColor = vec4(ambient + lit + vec3(specular), 1.0);
}

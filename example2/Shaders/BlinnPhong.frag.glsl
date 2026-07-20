#version 450 core

layout(location = 0) in vec3 vViewPosition;
layout(location = 1) in vec3 vViewNormal;
layout(location = 2) in vec3 vColor;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform BlinnPhongConstants {
    vec3 lightPosition;
    float shininess;
    vec3 lightColor;
    float specularStrength;
    vec3 cameraPosition;
    float dummy1;
} pc;

vec3 BlinnPhong(vec3 baseColor, vec3 normal, vec3 surfacePosition) {
    vec3 lightPosition = pc.lightPosition.xyz;
    float shininess = pc.shininess;
    vec3 lightColor = pc.lightColor;
    float specularStrength = pc.specularStrength;
    vec3 cameraPosition = pc.cameraPosition;

    vec3 n = normalize(normal);
    vec3 surf2light = normalize(lightPosition - surfacePosition);
    vec3 surf2camera = normalize(cameraPosition - surfacePosition);
    vec3 halfway = normalize(surf2light + surf2camera);

    float diffuse = max(dot(n, surf2light), 0.0);
    float specular = pow(max(dot(n, halfway), 0.0), shininess) * specularStrength;
    vec3 ambient = baseColor * 0.12;

    return ambient + baseColor * diffuse * lightColor + specular * lightColor;
}

void main() {
    vec3 color = BlinnPhong(vColor, vViewNormal, vViewPosition);
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}

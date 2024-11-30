#version 330

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float time;

out vec3 position;
out vec3 normal;

void main() {
    // Add quantum fluctuation effect
    vec3 pos = vertexPosition;
    float fluctuation = sin(time * 2.0 + length(pos) * 5.0) * 0.02;
    pos += vertexNormal * fluctuation;

    // Transform position
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);

    // Pass to fragment shader
    position = pos;
    normal = normalize(mat3(modelViewMatrix) * vertexNormal);
}

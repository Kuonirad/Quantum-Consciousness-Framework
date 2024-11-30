#version 330

uniform float time;
uniform vec3 quantum_state;
uniform sampler2D interference_pattern;

in vec3 position;
in vec3 normal;
out vec4 fragColor;

// Quantum wave function visualization
vec3 quantumWave(vec3 pos, float t) {
    float phase = dot(quantum_state, pos);
    float amplitude = exp(-length(pos - quantum_state));

    vec3 interference = texture(interference_pattern, pos.xy * 0.5 + 0.5).rgb;
    float wave = sin(10.0 * phase + t) * amplitude;

    return mix(vec3(0.2, 0.4, 1.0), vec3(1.0, 0.8, 0.2), wave) + interference * 0.2;
}

void main() {
    vec3 color = quantumWave(position, time);
    float alpha = smoothstep(0.0, 1.0, 1.0 - length(position));

    // Add quantum probability cloud effect
    vec3 cloud = vec3(0.1, 0.2, 0.8) * exp(-length(position) * 2.0);
    color += cloud * (0.5 + 0.5 * sin(time * 2.0));

    fragColor = vec4(color, alpha);
}

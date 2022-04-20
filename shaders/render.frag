#version 430

layout (location=0) uniform vec4 boidColor;
layout (location=1) uniform bool useVertexColor = true;

in vec3 vertexColor;
out vec4 outputColor;

void main() {
	outputColor = (useVertexColor) ? vec4(vertexColor, 1) : boidColor;
}
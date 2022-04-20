#version 430

in vec3 point;
in vec3 color;

out vec3 vertexColor;

uniform mat4 modelview;
uniform mat4 persp;

void main() {
	vertexColor = color;
	gl_Position = persp * modelview * vec4(point, 1);
}
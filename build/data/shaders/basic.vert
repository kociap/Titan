#version 450 core

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec2 iTexCoords;
layout (location = 2) in vec3 iNormal;

layout(location = 0) uniform mat4 model;
layout(location = 1) uniform mat4 view;
layout(location = 2) uniform mat4 projection;

out vec2 TexCoords;
out vec3 Normal;

void main() {
    TexCoords = iTexCoords;
    Normal = iNormal;
    gl_Position = projection * view * model * vec4(iPos, 1.0);
}
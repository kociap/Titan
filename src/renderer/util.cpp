#include "renderer/util.hpp"

#include <glad/glad.h>
#include <stb/stb_image.h>

#include <fstream>
#include <string>
#include <stdexcept>

namespace titan::renderer {

static std::string read_file(const char* path) {
    using namespace std::literals::string_literals;
    std::ifstream f(path);
    if (!f.good()) {
        throw std::runtime_error("Failed to open file: "s + path);
    }
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static unsigned int create_shader_stage(GLenum stage, const char* source) {
    using namespace std::literals::string_literals;
    unsigned int shader = glCreateShader(stage);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    char infolog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infolog);
        throw std::runtime_error("Failed to compile shader:\n"s + source + "\nReason: "s + infolog);
    }

    return shader;
}

void set_wireframe(bool wireframe) {
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
}

unsigned int load_shader(const char* vtx_path, const char* frag_path, const char* geom_path /* = nullptr*/) {
    using namespace std::literals::string_literals;
    std::string vertex = read_file(vtx_path);
    std::string fragment = read_file(frag_path);
    unsigned int vtx = create_shader_stage(GL_VERTEX_SHADER, vertex.c_str());
    unsigned int frag = create_shader_stage(GL_FRAGMENT_SHADER, fragment.c_str());

    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vtx);
    glAttachShader(prog, frag);

    unsigned int geom = 0;

    if (geom_path) {
        std::string geometry = read_file(geom_path);
        geom = create_shader_stage(GL_GEOMETRY_SHADER, geometry.c_str());
        glAttachShader(prog, geom);
    }

    glLinkProgram(prog);
    int success;
    char infolog[512];
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(prog, 512, nullptr, infolog);
        throw std::runtime_error("Failed to link shader.\nReason: "s + infolog);
        return 0;
    }

    glDeleteShader(vtx);
    glDeleteShader(frag);

    if (geom_path) {
        glDeleteShader(geom);
    }

    return prog;
}

unsigned int load_texture(const char* path) {
    stbi_set_flip_vertically_on_load(true);
    int w, h, channels;
    unsigned char* data = stbi_load(path, &w, &h, &channels, 4);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(data);
    
    return texture;
}

unsigned int texture_from_buffer(unsigned char* buf, size_t w, size_t h) {
    unsigned int tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, w, h);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, buf);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    return tex;
}


}
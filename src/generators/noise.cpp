#include "generators/noise.hpp"

#include <cmath>

#include <stdexcept>

#include <iostream>

namespace titan {

    using namespace math;

    PerlinNoise::gradient_grid::gradient_grid(size_t w, size_t h, std::mt19937& engine) {
        ++w;
        ++h;
        size.x = w;
        size.y = h;
        gradients.resize(w * h);
        for (int i = 0; i < w * h; ++i) {
            gradients[i] = normalize(vec2::random(engine));
        }
    }

    vec2& PerlinNoise::gradient_grid::at(size_t x, size_t y) {
        return gradients[y * (size_t)size.x + x];
    }

    vec2 const& PerlinNoise::gradient_grid::at(size_t x, size_t y) const {
        return gradients[y * (size_t)size.x + x];
    }

    PerlinNoise::PerlinNoise(size_t seed) : seed(seed), random_engine(seed) {}

    std::vector<unsigned char> PerlinNoise::get_buffer(size_t w, size_t h, size_t octaves) {
        std::vector<unsigned char> buffer(w * h, 0);
        get_buffer(buffer.data(), w, h, octaves);
        return buffer;
    }

    using u64 = unsigned long long;
    using i64 = long long;
    using u32 = unsigned int;
    using i32 = int;
    using f32 = float;

    struct Gradient_Grid {
        std::vector<vec2> gradients;
        size_t width;
        size_t height;

        vec2 at(size_t const x, size_t const y) const {
            if (y * width + x >= gradients.size()) {
                throw std::runtime_error("out of bounds");
            }
            return gradients[y * width + x];
        }
    };

    static void generate_gradients(Gradient_Grid& grid, size_t const size, std::mt19937& random_engine) {
        size_t const w = size + 1;
        size_t const h = size + 1;
        grid.width = w;
        grid.height = h;
        grid.gradients.clear();
        for (int i = 0; i < w * h; ++i) {
            grid.gradients.push_back(normalize(vec2::random(random_engine)));
        }
    }

    static float perlin_noise(float const x, float const y, vec2 const g00, vec2 const g10, vec2 const g01, vec2 const g11) {
        i64 const x0 = (i64)x;
        i64 const y0 = (i64)y;

        float const x_fractional = x - x0;
        float const y_fractional = y - y0;

        float const fac00 = dot(g00, {x_fractional, y_fractional});
        float const fac10 = dot(g10, {x_fractional - 1.0f, y_fractional});
        float const fac01 = dot(g01, {x_fractional, y_fractional - 1.0f});
        float const fac11 = dot(g11, {x_fractional - 1.0f, y_fractional - 1.0f});

        float const x_lerp_factor = x_fractional * x_fractional * x_fractional * (x_fractional * (x_fractional * 6 - 15) + 10);
        float const y_lerp_factor = y_fractional * y_fractional * y_fractional * (y_fractional * (y_fractional * 6 - 15) + 10);

        float const lerped_x0 = lerp(fac00, fac10, x_lerp_factor);
        float const lerped_x1 = lerp(fac01, fac11, x_lerp_factor);
        float noise = lerp(lerped_x0, lerped_x1, y_lerp_factor);
        return 1.4142135f * noise;
    }

    // size guaranteed to be a power of 2
    static void generate_noise(unsigned char* const buffer, u32 const size, u32 const octaves, std::mt19937& random_engine) {
        float amplitude = 1.0f;
        float const persistence = 0.5f;

        Gradient_Grid grid;
        grid.gradients.reserve((1 << (octaves - 1) + 1) * (1 << (octaves - 1) + 1));

        for (size_t octave = 0; octave < octaves; ++octave) {
            generate_gradients(grid, 1 << octave, random_engine);
            amplitude *= persistence;
            u64 const noise_scale = 1 << octave;
            f32 const noise_scale_f32 = noise_scale;
            float const increment = 1.0f / size * noise_scale_f32;

            for (size_t y = 0; y < size; ++y) {
                u64 const y_coord = y * noise_scale / size;
                u64 const y_coord_next = (y + 1) * noise_scale / size;
                float const y_coord_f32 = (float)y / size * noise_scale_f32;

                u64 const size4 = size >> 2;
                for (u64 i = 0; i < size4; ++i) {
                    u64 const x = i << 2;
                    float const xf32 = (float)x / size * noise_scale_f32;
                    vec2 const g00 = grid.at(xf32, y_coord);
                    vec2 const g10 = grid.at(xf32 + 1 * increment, y_coord);
                    vec2 const g20 = grid.at(xf32 + 2 * increment, y_coord);
                    vec2 const g30 = grid.at(xf32 + 3 * increment, y_coord);
                    vec2 const g40 = grid.at(xf32 + 4 * increment, y_coord);
                    vec2 const g01 = grid.at(xf32, y_coord_next);
                    vec2 const g11 = grid.at(xf32 + 1 * increment, y_coord_next);
                    vec2 const g21 = grid.at(xf32 + 2 * increment, y_coord_next);
                    vec2 const g31 = grid.at(xf32 + 3 * increment, y_coord_next);
                    vec2 const g41 = grid.at(xf32 + 4 * increment, y_coord_next);
                    // std::cout << xf32 << ' ' << y_coord_f32 << '\n';
                    float const val0 = perlin_noise(xf32, y_coord_f32, g00, g10, g01, g11);
                    float const val1 = perlin_noise(xf32 + 1 * increment, y_coord_f32, g10, g20, g11, g21);
                    float const val2 = perlin_noise(xf32 + 2 * increment, y_coord_f32, g20, g30, g21, g31);
                    float const val3 = perlin_noise(xf32 + 3 * increment, y_coord_f32, g30, g40, g31, g41);
                    // std::cout << "perlin: " << val0 << ' ' << val1 << ' ' << val2 << ' ' << val3 << '\n';
                    buffer[y * size + x] += 255.0f * amplitude * (0.5f + 0.5f * val0);
                    buffer[y * size + x + 1] += 255.0f * amplitude * (0.5f + 0.5f * val1);
                    buffer[y * size + x + 2] += 255.0f * amplitude * (0.5f + 0.5f * val2);
                    buffer[y * size + x + 3] += 255.0f * amplitude * (0.5f + 0.5f * val3);
                }

                for (size_t x = size4 << 2; x < size; ++x) {
                    float const xf32 = (float)x / size * noise_scale_f32;
                    vec2 const g00 = grid.at(xf32, y_coord);
                    vec2 const g10 = grid.at(xf32 + increment, y_coord);
                    vec2 const g01 = grid.at(xf32, y_coord_next);
                    vec2 const g11 = grid.at(xf32 + increment, y_coord_next);
                    float const val = perlin_noise(xf32, y_coord_f32, g00, g10, g01, g11);
                    buffer[y * size + x] += 255.0f * amplitude * (0.5f + 0.5f * val);
                }
            }
        }
    }

    void PerlinNoise::get_buffer(unsigned char* buffer, size_t w, size_t h, size_t octaves) {
        // w and h are the same because noise is always square.
        // TODO: Rework interface.
        generate_noise(buffer, w, octaves, random_engine);
    }

    static float perlin_lerp(float a0, float a1, float x) {
        //    float w = 6 * std::pow(x, 5) - 15 * std::pow(x, 4) + 10 * std::pow(x, 3); ==>  SLOW, line below is 10 times faster
        float w = x * x * x * (x * (x * 6 - 15) + 10);
        return lerp(a0, a1, w);
    }

    // x and y are the point's position, ix and iy are the cell coordinates
    static float dot_grid_gradient(int ix, int iy, float x, float y, PerlinNoise::gradient_grid const& gradients) {
        // distance vector
        float dx = x - (float)ix;
        float dy = y - (float)iy;

        vec2 const gradient = gradients.at(ix, iy);
        return (dx * gradient.x + dy * gradient.y);
    }

    float PerlinNoise::value(float x, float y) const {
        // get grid cell coordinates
        int x0 = (int)x;
        int x1 = x0 + 1;
        int y0 = (int)y;
        int y1 = y0 + 1;

        // interpolation weights
        float sx = x - (float)x0;
        float sy = y - (float)y0;

        // interpolate between grid point gradients
        float n0, n1, ix0, ix1, value;
        n0 = dot_grid_gradient(x0, y0, x, y, gradients);
        n1 = dot_grid_gradient(x1, y0, x, y, gradients);
        ix0 = perlin_lerp(n0, n1, sx);

        n0 = dot_grid_gradient(x0, y1, x, y, gradients);
        n1 = dot_grid_gradient(x1, y1, x, y, gradients);
        ix1 = perlin_lerp(n0, n1, sx);

        value = perlin_lerp(ix0, ix1, sy);
        return value;
    }

    void PerlinNoise::regenerate_gradients() {
        regenerate_gradients(scale);
    }

    void PerlinNoise::regenerate_gradients(size_t new_scale) {
        gradients = gradient_grid(new_scale, new_scale, random_engine);
        scale = new_scale;
    }

} // namespace titan

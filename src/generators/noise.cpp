#include "generators/noise.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <smmintrin.h>
#include <xmmintrin.h>

namespace titan {
    using namespace math;

    struct Gradient_Grid {
        vec2 gradients[16];
        u8 perm_table[128];

        vec2 at(u64 const x, u64 const y) const {
            u8 const index = (y % 128 + x) % 128;
            return gradients[perm_table[index] % 16];
        }
    };

    static Gradient_Grid create_gradient_grid(u64 const size, std::mt19937& random_engine) {
        Gradient_Grid grid{{{0.0f, 1.0f}, {0.382683f, 0.923879f}, {0.707107f, 0.707107f}, {0.923879f, 0.382683f}, {1.0f, 0.0f}, {0.923879f, -0.382683f}, {0.707107f, -0.707107f}, {0.382683f, -0.923879f}, {0.0f, -1.0f}, {-0.382683f, -0.923879f}, {-0.707107f, -0.707107f}, {-0.923879f, -0.382683f}, {-1.0f, 0.0f}, {-0.923879f, 0.382683f}, {-0.707107f, 0.707107f}, {-0.382683f, 0.923879f}}};

        std::uniform_int_distribution<u32> d(0, 255);
        for (int i = 0; i < 128; ++i) {
            grid.perm_table[i] = d(random_engine);
        }

        return grid;
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
        return 1.4142135f * lerp(lerped_x0, lerped_x1, y_lerp_factor);
    }

    void generate_perlin_noise_texture(float* const buffer, u64 const seed, u32 const size, u32 const octaves) {
        std::mt19937 random_engine(seed);
        Gradient_Grid const grid = create_gradient_grid(1 << (octaves - 1), random_engine);

        f32 amplitude = 1.0f;
        f32 const persistence = 0.5f;
        f32 const size_f32 = size;
        for (u32 octave = 0; octave < octaves; ++octave) {
            amplitude *= persistence;
            u64 const noise_scale = 1 << octave;
            f32 const noise_scale_f32 = noise_scale;
            __m128 scale_factor = _mm_set1_ps(noise_scale_f32 / size_f32);
            u64 const resample_period = size / noise_scale;
            if (resample_period >= 4) {
                for (u64 y = 0; y < size; ++y) {
                    f32 const y_coord = (f32)y / size_f32 * noise_scale_f32;
                    u64 const sample_offset_y = y_coord;
                    i64 const y_floor = (i64)y_coord;
                    f32 const y_fractional = y_coord - y_floor;
                    f32 const y_lerp_factor = y_fractional * y_fractional * y_fractional * (y_fractional * (y_fractional * 6 - 15) + 10);
                    for (u64 x = 0, sample_offset_x = 0; sample_offset_x < noise_scale; ++sample_offset_x) {
                        vec2 const g00 = grid.at(sample_offset_x, sample_offset_y);
                        vec2 const g10 = grid.at(sample_offset_x + 1, sample_offset_y);
                        vec2 const g01 = grid.at(sample_offset_x, sample_offset_y + 1);
                        vec2 const g11 = grid.at(sample_offset_x + 1, sample_offset_y + 1);
                        for (u64 i = 0; i < resample_period; i += 4, x += 4) {
                            __m128 x_base_xmm = _mm_mul_ps(_mm_add_ps(_mm_set1_ps(x), _mm_set_ps(3, 2, 1, 0)), scale_factor);
                            __m128 x_floor = _mm_floor_ps(_mm_shuffle_ps(x_base_xmm, x_base_xmm, _MM_SHUFFLE(0, 0, 0, 0)));
                            __m128 x_fractional = _mm_sub_ps(x_base_xmm, x_floor);
                            __m128 x_fractional_less_one = _mm_add_ps(x_fractional, _mm_set1_ps(-1.0f));

                            __m128 g00x = _mm_set1_ps(g00.x);
                            __m128 fac00x = _mm_mul_ps(x_fractional, g00x);
                            __m128 fac00y = _mm_set1_ps(g00.y * y_fractional);
                            __m128 fac00 = _mm_add_ps(fac00x, fac00y);

                            __m128 g10x = _mm_set1_ps(g10.x);
                            __m128 fac10x = _mm_mul_ps(x_fractional_less_one, g10x);
                            __m128 fac10y = _mm_set1_ps(g10.y * y_fractional);
                            __m128 fac10 = _mm_add_ps(fac10x, fac10y);

                            __m128 g01x = _mm_set1_ps(g01.x);
                            __m128 fac01x = _mm_mul_ps(x_fractional, g01x);
                            __m128 fac01y = _mm_set1_ps(g01.y * (y_fractional - 1.0f));
                            __m128 fac01 = _mm_add_ps(fac01x, fac01y);

                            __m128 g11x = _mm_set1_ps(g11.x);
                            __m128 fac11x = _mm_mul_ps(x_fractional_less_one, g11x);
                            __m128 fac11y = _mm_set1_ps(g11.y * (y_fractional - 1.0f));
                            __m128 fac11 = _mm_add_ps(fac11x, fac11y);

                            __m128 x_fractional_cube = _mm_mul_ps(x_fractional, _mm_mul_ps(x_fractional, x_fractional));
                            __m128 lerp_factor = _mm_mul_ps(x_fractional_cube, _mm_add_ps(_mm_mul_ps(x_fractional, _mm_sub_ps(_mm_mul_ps(x_fractional, _mm_set1_ps(6)), _mm_set1_ps(15))), _mm_set1_ps(10)));

                            __m128 lerp_factor_compl_xmm = _mm_sub_ps(_mm_set1_ps(1.0f), lerp_factor);
                            __m128 lerpx0_c1_xmm = _mm_mul_ps(lerp_factor_compl_xmm, fac00);
                            __m128 lerpx1_c1_xmm = _mm_mul_ps(lerp_factor_compl_xmm, fac01);
                            __m128 lerpx0_c0_xmm = _mm_mul_ps(lerp_factor, fac10);
                            __m128 lerpx1_c0_xmm = _mm_mul_ps(lerp_factor, fac11);
                            __m128 lerpx0_xmm = _mm_add_ps(lerpx0_c0_xmm, lerpx0_c1_xmm);
                            __m128 lerpx1_xmm = _mm_add_ps(lerpx1_c0_xmm, lerpx1_c1_xmm);

                            __m128 lerp_factor_y_xmm = _mm_set1_ps(y_lerp_factor);
                            __m128 lerpy_c0_xmm = _mm_mul_ps(lerpx1_xmm, lerp_factor_y_xmm);
                            __m128 lerp_factor_y_compl_xmm = _mm_set1_ps(1.0f - y_lerp_factor);
                            __m128 lerpy_c1_xmm = _mm_mul_ps(lerpx0_xmm, lerp_factor_y_compl_xmm);
                            __m128 noise_r0 = _mm_add_ps(lerpy_c0_xmm, lerpy_c1_xmm);
                            __m128 noise_r1 = _mm_add_ps(noise_r0, _mm_set1_ps(0.7071067f));
                            __m128 noise_r2 = _mm_mul_ps(noise_r1, _mm_set1_ps(amplitude * 0.5f * 1.4142135f));
                            __m128 current = _mm_loadu_ps(buffer + y * size + x);
                            __m128 noise = _mm_add_ps(noise_r2, current);
                            _mm_storeu_ps(buffer + y * size + x, noise);
                        }
                    }
                }
            } else {
                for (u64 y = 0; y < size; ++y) {
                    f32 const y_coord = (f32)y / size_f32 * noise_scale_f32;
                    u64 const sample_offset_y = (u64)y_coord % size;
                    for (u64 x = 0; x < size; ++x) {
                        f32 const x_coord = (f32)x / size_f32 * noise_scale_f32;
                        u64 const sample_offset_x = x_coord;
                        vec2 const g00 = grid.at(sample_offset_x, sample_offset_y);
                        vec2 const g10 = grid.at(sample_offset_x + 1, sample_offset_y);
                        vec2 const g01 = grid.at(sample_offset_x, sample_offset_y + 1);
                        vec2 const g11 = grid.at(sample_offset_x + 1, sample_offset_y + 1);
                        f32 const val = perlin_noise(x_coord, y_coord, g00, g10, g01, g11);
                        buffer[y * size + x] += amplitude * (0.5f + 0.5f * val);
                    }
                }
            }
        }
    }
} // namespace titan

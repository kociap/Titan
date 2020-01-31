#ifndef TITAN_TERRAINS_NOISE_HPP_
#define TITAN_TERRAINS_NOISE_HPP_

#include "math.hpp"

#include <random>
#include <vector>

namespace titan {
    using u8 = unsigned char;
    using i32 = int;
    using u32 = unsigned int;
    using i64 = long long;
    using u64 = unsigned long long;
    using f32 = float;

    // size must be a power of 2
    void generate_perlin_noise_texture(float* buffer, u64 seed, u32 size, u32 octaves = 1);
} // namespace titan

#endif
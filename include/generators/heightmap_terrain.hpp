#ifndef TITAN_HEIGHTMAP_TERRAIN_GENERATOR_HPP_
#define TITAN_HEIGHTMAP_TERRAIN_GENERATOR_HPP_

#include "config.hpp"
#include "grid_mesh.hpp"

#include <vector>

namespace titan {

struct HeightmapTerrain {
    GridMesh mesh;
    std::vector<unsigned char> height_map;

    // Misc  info

    float width;
    float length;
    float height_scale;

    size_t heightmap_width;
    size_t heightmap_height;
};

struct HeightmapTerrainInfo {
    // Terrain mesh options

    // The width of the terrain mesh, in worldspace coordinates
    float width;
    // The length of the terrain mesh, in worldspace coordinates
    float length;
    // The maximum height of the terrain mesh, in worldspace coordinates
    float height_scale;
    // The amount of vertices in each row/column.
    size_t resolution;
    // Controls how texture coordinates are calculated
    TextureMode texture_mode = TextureMode::Stretch;

    // Noise options
    size_t noise_seed;
    size_t noise_size = 256;
    size_t noise_layers = 8;
};

HeightmapTerrain create_heightmap_terrain(HeightmapTerrainInfo const& info);

}

#endif
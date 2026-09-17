#pragma once

#include<cstdint>
#include<vector>

#include<core/math/vec3.hpp>

namespace engine::rsrc{

struct Vertex{
	engine::math::Vec3 position;
};

struct Mesh{
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

} // engine::render

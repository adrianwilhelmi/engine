#pragma once

#include<cstdint>

#include<core/math/mat4.hpp>

namespace engine::render{

using MeshHandle = uint32_t;
using MaterialHandle = uint32_t;

struct RenderObject{
	engine::math::Mat4 model;

	MeshHandle mesh;
	MaterialHandle material;
};

} // namespace engine::render

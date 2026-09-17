#pragma once

#include<cstdint>

#include<core/math/mat4.hpp>

namespace engine::render{

struct CameraData{
	engine::math::Mat4 view;
	engine::math::Mat4 projection;

	uint32_t viewport_width = 0;
	uint32_t viewport_height = 0;
};

} // engine::render

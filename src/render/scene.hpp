#pragma once

#include<cstdint>
#include<vector>

#include<render/camera_data.hpp>
#include<render/render_object.hpp>
#include<render/color.hpp>

namespace engine::render {

struct Scene{
	CameraData camera;
	std::vector<RenderObject> objects;
	uint32_t clear_color = pack_argb8888(engine::render::Color::black());
};

} // namespace engine::render

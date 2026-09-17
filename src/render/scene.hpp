#pragma once

#include<cstdint>
#include<vector>

#include<render/camera_data.hpp>
#include<render/render_object.hpp>

namespace engine::render {

struct Scene{
	CameraData camera;
	std::vector<RenderObject> objects;
	uint32_t clear_color = 0xFF000000;
};

} // namespace engine::render

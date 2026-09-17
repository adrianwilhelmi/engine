#pragma once

#include<cstdint>
#include<memory>

#include<render/frame/frame_buffer.hpp>
#include<render/scene.hpp>

namespace engine::render{

class Renderer{
public:
	virtual void render_frame(
		engine::render::FrameBuffer& frame_buffer,
		const engine::render::Scene& scene
	) = 0;
};

std::unique_ptr<Renderer> create_software_renderer();

} // namespace engine::render

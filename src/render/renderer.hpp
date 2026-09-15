#pragma once

#include<cstdint>
#include<memory>

#include<render/frame/frame_buffer.hpp>

namespace engine::render{

class Renderer{
public:
	virtual void render_frame(engine::render::FrameBuffer& frame_buffer) = 0;
};

std::unique_ptr<Renderer> create_software_renderer();

} // namespace engine::render

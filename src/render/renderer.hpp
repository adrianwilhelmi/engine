#pragma once

#include<cstdint>
#include<memory>

#include<core/frame/frame_buffer.hpp>

namespace render{

class Renderer{
public:
	virtual void render_frame(engine::FrameBuffer& frame_buffer) = 0;
};

std::unique_ptr<Renderer> create_vulkan_renderer();
std::unique_ptr<Renderer> create_software_renderer();

} // namespace render

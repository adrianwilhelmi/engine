#pragma once

#include<memory>

#include<render/renderer.hpp>
#include<core/frame/frame_buffer.hpp>

namespace render{

class SoftwareRenderer : public Renderer{
public:
	void render_frame(engine::FrameBuffer& frame_buffer) override;
};

} // namespace render

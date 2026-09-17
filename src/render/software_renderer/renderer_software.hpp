#pragma once

#include<memory>

#include<render/renderer.hpp>
#include<render/frame/frame_buffer.hpp>
#include<render/scene.hpp>

namespace engine::render{

class SoftwareRenderer : public Renderer{
public:
	void render_frame(
		engine::render::FrameBuffer& frame_buffer,
		const engine::render::Scene& scene
	) override;
};

} // namespace render

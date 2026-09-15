#include<render/renderer.hpp>
#include<render/software_renderer/renderer_software.hpp>
#include<render/color.hpp>

namespace engine::render{

void SoftwareRenderer::render_frame(engine::render::FrameBuffer& frame_buffer){
	for(uint32_t i = 0; i < frame_buffer.height(); ++i){
		for(uint32_t j = 0; j < frame_buffer.width(); ++j){
			frame_buffer.pixel(j,i) = pack_argb8888(Color::white());
		}
	}
}

} // namespace render

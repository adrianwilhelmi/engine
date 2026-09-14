
#include"render/renderer.hpp"
#include"render/software_renderer/renderer_software.hpp"

namespace render{

bool SoftwareRenderer::init(const RenderInitInfo& info){
	this->window_ptr_ = info.window_handle;
	this->width_ = info.width;
	this->height_ = info.height;

	return true;
}

void SoftwareRenderer::render_frame(){

}

void SoftwareRenderer::shutdown(){

}

} // namespace render

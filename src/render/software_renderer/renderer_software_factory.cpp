#include<memory>

#include"render/software_renderer/renderer_software.hpp"
#include"render/renderer.hpp"

namespace render{

std::unique_ptr<Renderer> create_software_renderer(){
	return std::make_unique<SoftwareRenderer>();
}

} //namespace rander

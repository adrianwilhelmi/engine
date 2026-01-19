#include<memory>

#include"render/vulkan/renderer_vulkan.hpp"
#include"render/renderer.hpp"

namespace render{

std::unique_ptr<Renderer> create_vulkan_renderer(){
	return std::make_unique<VulkanRenderer>();
}

} //namespace rander
